#!/bin/bash
# ============================================================================
# qMRLab QMT fitting on WAND data — slice-parallel on DGX Spark
# ============================================================================
# Splits 3D QMT volume into individual slices, processes each in parallel
# via GNU parallel + Octave/qMRLab, then reassembles output maps.
#
# Usage:
#   bash scripts/wand_processing/38_qmrlab_qmt_parallel.sh sub-08033
#   NPROC=16 bash scripts/wand_processing/38_qmrlab_qmt_parallel.sh sub-08033
# ============================================================================

set -euo pipefail

SUBJECT="${1:?Usage: $0 <subject_id>}"
WAND_ROOT="${WAND_ROOT:-/data/raw/wand}"
NPROC="${NPROC:-16}"
QMRLAB_DIR="/home/mhough/dev/qMRLab"
DERIV="${WAND_ROOT}/derivatives"
QMRI_OUT="${DERIV}/qmri/${SUBJECT}/ses-02"
ANAT="${WAND_ROOT}/${SUBJECT}/ses-02/anat"
WORK="${QMRI_OUT}/qmrlab_work"

mkdir -p "${WORK}/slices" "${WORK}/results" "${QMRI_OUT}/qmrlab"

echo "=== qMRLab QMT Fitting (slice-parallel): ${SUBJECT} ==="
echo "Processors: ${NPROC}"

# ---------------------------------------------------------------
# 1. Prepare inputs
# ---------------------------------------------------------------

# Check required files
MTOFF="${ANAT}/${SUBJECT}_ses-02_mt-off_part-mag_QMT.nii.gz"
T1MAP="${QMRI_OUT}/T1map.nii.gz"
MASK="${QMRI_OUT}/brain_mask.nii.gz"

if [ ! -f "${MTOFF}" ] || [ ! -f "${T1MAP}" ]; then
    echo "ERROR: Need MT-off and T1map. Run 29_qmri_vfa_qmt.py first."
    exit 1
fi

# Get QMT volume shape
NSLICES=$(python3 -c "
import nibabel as nib
img = nib.load('${MTOFF}')
print(img.shape[2])
")
echo "Slices: ${NSLICES}"

# ---------------------------------------------------------------
# 2. Merge QMT volumes and split into per-slice files
# ---------------------------------------------------------------
echo ""
echo "--- Preparing QMT data ---"

python3 << 'PYEOF'
import nibabel as nib
import numpy as np
import json, glob, os, sys

subject = os.environ.get('SUBJECT', 'sub-08033')
anat = os.environ.get('ANAT')
work = os.environ.get('WORK')
qmri_out = os.environ.get('QMRI_OUT')

# Load MT-off reference
mtoff_img = nib.load(f'{anat}/{subject}_ses-02_mt-off_part-mag_QMT.nii.gz')
mtoff = mtoff_img.get_fdata()
affine = mtoff_img.affine
shape = mtoff.shape
print(f'MT-off shape: {shape}')

# Load all MT-on volumes in order (matching qMRLab protocol)
qmt_tags = [
    'flip-1_mt-1_part-mag', 'flip-1_mt-2_part-mag',
    'flip-2_mt-1_part-mag', 'flip-2_mt-2_part-mag',
    'flip-2_mt-3_part-mag', 'flip-2_mt-4_part-mag',
    'flip-2_mt-5_part-mag', 'flip-2_mt-6_part-mag',
    'flip-2_mt-7_part-mag_run-1', 'flip-2_mt-7_part-mag_run-2',
    'flip-3_mt-1_part-mag',
]

# MT parameters: (flip_angle_deg, offset_hz) for each volume
mt_params = [
    (332, 56360), (332, 1000),
    (628, 47180), (628, 12060), (628, 2750), (628, 2770),
    (628, 2790), (628, 2890), (628, 1000), (628, 1000),
    (333, 1000),
]

mt_vols = []
for tag in qmt_tags:
    fname = f'{anat}/{subject}_ses-02_{tag}_QMT.nii.gz'
    if os.path.exists(fname):
        mt_vols.append(nib.load(fname).get_fdata())

if not mt_vols:
    print('ERROR: No QMT volumes found')
    sys.exit(1)

mt_stack = np.stack(mt_vols, axis=-1)  # (X, Y, Z, N_mt)
n_mt = mt_stack.shape[-1]
print(f'MT-on volumes: {n_mt}')

# Compute MTdata: ratio S_mt / S_ref for each MT volume
# qMRLab qmt_spgr expects MTdata as the MT-weighted signal divided by reference
mtdata = np.zeros_like(mt_stack)
for i in range(n_mt):
    valid = mtoff > 10
    mtdata[..., i][valid] = mt_stack[..., i][valid] / mtoff[valid]

# Load R1 map (= 1/T1) for qMRLab
t1 = nib.load(f'{qmri_out}/T1map.nii.gz').get_fdata()

# Reorient T1 if needed (SPGR is PSR, QMT is RAS)
if t1.shape != shape:
    from nibabel.orientations import io_orientation, ornt_transform, apply_orientation
    t1_img = nib.load(f'{qmri_out}/T1map.nii.gz')
    qmt_ornt = io_orientation(affine)
    t1_ornt = io_orientation(t1_img.affine)
    transform = ornt_transform(t1_ornt, qmt_ornt)
    t1 = apply_orientation(t1, transform)
    print(f'Reoriented T1 to QMT space: {t1.shape}')

r1map = np.zeros_like(t1)
valid_t1 = t1 > 0.1
r1map[valid_t1] = 1.0 / t1[valid_t1]

# Load or create mask in QMT space
mask_file = f'{qmri_out}/brain_mask.nii.gz'
if os.path.exists(mask_file):
    mask = nib.load(mask_file).get_fdata()
    if mask.shape != shape:
        from nibabel.orientations import io_orientation, ornt_transform, apply_orientation
        mask_img = nib.load(mask_file)
        transform = ornt_transform(io_orientation(mask_img.affine), io_orientation(affine))
        mask = apply_orientation(mask, transform)
else:
    mask = (mtoff > np.percentile(mtoff[mtoff > 0], 15)).astype(float)

# Save full volumes for reference
nib.save(nib.Nifti1Image(mtdata, affine), f'{work}/MTdata_full.nii.gz')
nib.save(nib.Nifti1Image(r1map, affine), f'{work}/R1map_full.nii.gz')
nib.save(nib.Nifti1Image(mask, affine), f'{work}/Mask_full.nii.gz')

# Split into per-slice files
n_slices = shape[2]
for k in range(n_slices):
    sl_mtdata = mtdata[:, :, k:k+1, :]
    sl_r1 = r1map[:, :, k:k+1]
    sl_mask = mask[:, :, k:k+1]

    # Modify affine for single-slice (shift origin to slice k)
    sl_affine = affine.copy()
    sl_affine[:3, 3] += affine[:3, 2] * k

    nib.save(nib.Nifti1Image(sl_mtdata, sl_affine), f'{work}/slices/MTdata_slice{k:03d}.nii.gz')
    nib.save(nib.Nifti1Image(sl_r1, sl_affine), f'{work}/slices/R1map_slice{k:03d}.nii.gz')
    nib.save(nib.Nifti1Image(sl_mask, sl_affine), f'{work}/slices/Mask_slice{k:03d}.nii.gz')

print(f'Split into {n_slices} slices')

# Save protocol JSON for the Octave script
protocol = {
    'mt_angles_deg': [p[0] for p in mt_params[:n_mt]],
    'mt_offsets_hz': [p[1] for p in mt_params[:n_mt]],
    'n_mt': n_mt,
    'n_slices': n_slices,
}
with open(f'{work}/protocol.json', 'w') as f:
    json.dump(protocol, f, indent=2)
print(f'Protocol saved: {n_mt} MT volumes')
PYEOF

# ---------------------------------------------------------------
# 3. Write per-slice Octave fitting script
# ---------------------------------------------------------------
echo ""
echo "--- Writing Octave fitting script ---"

cat > "${WORK}/fit_slice.m" << 'MEOF'
% fit_slice.m — qMRLab qmt_spgr fitting for a single slice
% Usage: octave --no-gui --eval "slice_idx=0; run('fit_slice.m')"

addpath(genpath('/home/mhough/dev/qMRLab'));

% Read arguments
work_dir = getenv('WORK');
if isempty(work_dir); work_dir = '.'; end

% Load slice data
mt_file = fullfile(work_dir, 'slices', sprintf('MTdata_slice%03d.nii.gz', slice_idx));
r1_file = fullfile(work_dir, 'slices', sprintf('R1map_slice%03d.nii.gz', slice_idx));
mask_file = fullfile(work_dir, 'slices', sprintf('Mask_slice%03d.nii.gz', slice_idx));

if ~exist(mt_file, 'file')
    fprintf('Slice %d: no data\n', slice_idx);
    return;
end

% Load NIfTI
mt_nii = load_nii_data(mt_file);
r1_nii = load_nii_data(r1_file);
mask_nii = load_nii_data(mask_file);

% Check if slice has brain voxels
if sum(mask_nii(:) > 0) < 10
    fprintf('Slice %d: too few brain voxels (%d), skipping\n', slice_idx, sum(mask_nii(:) > 0));
    return;
end

% Configure qmt_spgr model
m = qmt_spgr;

% Load protocol from JSON
proto_file = fullfile(work_dir, 'protocol.json');
proto = jsondecode(fileread(proto_file));

% Set MT protocol: [Angle, Offset] for each volume
m.Prot.MTdata.Mat = [proto.mt_angles_deg(:), proto.mt_offsets_hz(:)];

% Set timing (CUBRIC tfl_qMT_v09 protocol)
m.Prot.TimingTable.Mat = [0.015; 0.003; 0.0018; 0.010; 0.055];
% [Tmt=15ms, Ts=3ms, Tp=1.8ms, Tr=10ms, TR=55ms]

% Options: SuperLorentzian lineshape, use R1 map
m.options.Lineshape = 'SuperLorentzian';
m.options.Model = 'SledPikeRP';
m.options.fittingconstraints_UseR1maptoconstrainR1f = 1;
m.options.Readpulsealpha = 5;  % CUBRIC readout FA
m.options.MT_Pulse_Shape = 'gausshann';

% Set data
data.MTdata = double(mt_nii);
data.R1map = double(r1_nii);
data.Mask = double(mask_nii > 0);
data.B1map = ones(size(r1_nii));  % assume B1=1 (no B1 map available)
data.B0map = zeros(size(r1_nii)); % assume on-resonance

% Fit
fprintf('Slice %d: fitting %d voxels... ', slice_idx, sum(data.Mask(:)));
tic;
try
    FitResults = FitData(data, m, 0);  % 0 = no progress bar
    t = toc;
    fprintf('done (%.1fs)\n', t);

    % Save results
    out_dir = fullfile(work_dir, 'results');
    save_nii_data(FitResults.F, fullfile(out_dir, sprintf('BPF_slice%03d.nii.gz', slice_idx)), mt_file);
    save_nii_data(FitResults.kf, fullfile(out_dir, sprintf('kf_slice%03d.nii.gz', slice_idx)), mt_file);
    save_nii_data(FitResults.resnorm, fullfile(out_dir, sprintf('resnorm_slice%03d.nii.gz', slice_idx)), mt_file);
catch e
    t = toc;
    fprintf('FAILED (%.1fs): %s\n', t, e.message);
end
MEOF

# Also write helper functions for NIfTI I/O in Octave
cat > "${WORK}/load_nii_data.m" << 'MEOF'
function data = load_nii_data(filename)
    % Load NIfTI data using Octave's built-in or qMRLab's loader
    try
        nii = nii_tool('load', filename);
        data = nii.img;
    catch
        % Fallback: use system gunzip + raw read
        [~, ~, ext] = fileparts(filename);
        if strcmp(ext, '.gz')
            tmpfile = [tempname '.nii'];
            system(sprintf('gunzip -c %s > %s', filename, tmpfile));
            nii = nii_tool('load', tmpfile);
            data = nii.img;
            delete(tmpfile);
        else
            error('Cannot load %s', filename);
        end
    end
end
MEOF

cat > "${WORK}/save_nii_data.m" << 'MEOF'
function save_nii_data(data, filename, ref_file)
    % Save data as NIfTI using reference file for header
    try
        ref = nii_tool('load', ref_file);
        ref.img = single(data);
        nii_tool('save', ref, filename);
    catch e
        warning('NIfTI save failed: %s', e.message);
    end
end
MEOF

# ---------------------------------------------------------------
# 4. Run in parallel
# ---------------------------------------------------------------
echo ""
echo "--- Running ${NPROC} parallel fits ---"

NSLICES=$(python3 -c "import json; print(json.load(open('${WORK}/protocol.json'))['n_slices'])")

# Generate slice indices
seq 0 $((NSLICES - 1)) | \
    xargs -P "${NPROC}" -I {} \
    octave --no-gui --path "${WORK}" --eval "slice_idx={}; run('${WORK}/fit_slice.m')" \
    2>&1 | tee "${WORK}/fitting.log"

# ---------------------------------------------------------------
# 5. Reassemble output volumes
# ---------------------------------------------------------------
echo ""
echo "--- Reassembling output maps ---"

python3 << 'PYEOF'
import nibabel as nib
import numpy as np
import os, glob

work = os.environ['WORK']
qmri_out = os.environ['QMRI_OUT']

# Load reference for shape/affine
ref = nib.load(f'{work}/MTdata_full.nii.gz')
shape = ref.shape[:3]
affine = ref.affine

for param in ['BPF', 'kf', 'resnorm']:
    vol = np.zeros(shape, dtype=np.float32)
    n_found = 0
    for k in range(shape[2]):
        fname = f'{work}/results/{param}_slice{k:03d}.nii.gz'
        if os.path.exists(fname):
            sl = nib.load(fname).get_fdata()
            if sl.ndim == 3:
                vol[:, :, k] = sl[:, :, 0]
            elif sl.ndim == 2:
                vol[:, :, k] = sl
            n_found += 1

    if n_found > 0:
        out = f'{qmri_out}/qmrlab/QMT_{param}.nii.gz'
        nib.save(nib.Nifti1Image(vol, affine), out)
        brain = vol[vol > 0.001]
        print(f'{param}: {n_found}/{shape[2]} slices, '
              f'median={np.median(brain):.4f}' if len(brain) > 0 else f'{param}: empty')
    else:
        print(f'{param}: no slices found')
PYEOF

echo ""
echo "=== qMRLab QMT fitting complete ==="
echo "Outputs: ${QMRI_OUT}/qmrlab/"
ls "${QMRI_OUT}/qmrlab/"*.nii.gz 2>/dev/null
