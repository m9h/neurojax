#!/bin/bash
# Parallel qMRLab QMT fitting by splitting volume into slabs
# Runs N_PARALLEL Octave processes, each fitting a subset of slices

set -euo pipefail

SUBJECT="${1:-sub-08033}"
N_PARALLEL="${2:-4}"
WAND_ROOT="/Users/mhough/dev/wand"
QMRI="${WAND_ROOT}/derivatives/qmri/${SUBJECT}/ses-02"
OUT_DIR="${QMRI}/qmrlab"
mkdir -p "${OUT_DIR}"

# Get total number of slices
N_SLICES=$(/Users/mhough/fsl/bin/python3 -c "
import nibabel as nib
img = nib.load('${QMRI}/quit/D1_T1_in_QMT.nii.gz')
print(img.shape[2])
")
echo "Total slices: ${N_SLICES}, parallel jobs: ${N_PARALLEL}"

SLICES_PER_JOB=$(( (N_SLICES + N_PARALLEL - 1) / N_PARALLEL ))
echo "Slices per job: ${SLICES_PER_JOB}"

# Launch parallel jobs
PIDS=()
for ((i=0; i<N_PARALLEL; i++)); do
    START=$((i * SLICES_PER_JOB + 1))
    END=$(( (i+1) * SLICES_PER_JOB ))
    if [ $END -gt $N_SLICES ]; then END=$N_SLICES; fi

    echo "Job $i: slices ${START}-${END}"

    octave-cli --eval "
        addpath(genpath('~/dev/qMRLab'));
        pkg load struct; pkg load io; pkg load statistics; pkg load optim; pkg load image;

        subject = '${SUBJECT}';
        wand = '${WAND_ROOT}';
        anat_dir = fullfile(wand, subject, 'ses-02', 'anat');
        quit_dir = '${QMRI}/quit';
        out_dir = '${OUT_DIR}';

        % Load data
        mtoff_nii = load_nii(fullfile(anat_dir, [subject '_ses-02_mt-off_part-mag_QMT.nii.gz']));
        mt_off = double(mtoff_nii.img);
        tags = {'flip-1_mt-1','flip-1_mt-2','flip-2_mt-1','flip-2_mt-2','flip-2_mt-3','flip-2_mt-4','flip-2_mt-5','flip-2_mt-6','flip-3_mt-1'};
        mt_data = zeros([size(mt_off), 9]);
        for k = 1:9
            tmp = load_nii(fullfile(anat_dir, [subject '_ses-02_' tags{k} '_part-mag_QMT.nii.gz']));
            mt_data(:,:,:,k) = double(tmp.img);
        end
        t1_nii = load_nii(fullfile(quit_dir, 'D1_T1_in_QMT.nii.gz'));
        t1_map = double(t1_nii.img);

        % Slab mask: only slices ${START} to ${END}
        mask = (t1_map > 0.2) & (t1_map < 4.0) & (mt_off > 50);
        slab_mask = false(size(mask));
        slab_mask(:,:,${START}:${END}) = mask(:,:,${START}:${END});
        fprintf('Job $i: %d voxels in slices ${START}-${END}\n', sum(slab_mask(:)));

        % Setup model
        Model = qmt_spgr;
        Model.Prot.MTdata.Mat = [332 56360; 332 1000; 628 47180; 628 12060; 628 2750; 628 2770; 628 2790; 628 2890; 333 1000];
        Model.Prot.TimingTable.Mat = [5, 0.055];

        data.MTdata = mt_data;
        data.R1map = 1.0 ./ max(t1_map, 0.01);
        data.Mask = slab_mask;

        tic;
        FitResults = FitData(data, Model, 0);
        fprintf('Job $i done in %.1f min\n', toc/60);

        save(fullfile(out_dir, sprintf('FitResults_slab%02d.mat', $i)), '-struct', 'FitResults');
        fprintf('Saved slab %d\n', $i);
    " > "${OUT_DIR}/log_slab${i}.txt" 2>&1 &

    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} parallel jobs: ${PIDS[*]}"
echo "Logs in ${OUT_DIR}/log_slab*.txt"
echo "Waiting..."

# Wait for all jobs
FAILED=0
for PID in "${PIDS[@]}"; do
    wait $PID || ((FAILED++))
done

echo "All jobs complete. Failed: ${FAILED}"

# Merge slab results
if [ $FAILED -eq 0 ]; then
    echo "Merging slabs..."
    octave-cli --eval "
        out_dir = '${OUT_DIR}';
        files = dir(fullfile(out_dir, 'FitResults_slab*.mat'));
        if isempty(files); error('No slab files found'); end

        % Load first slab to get field names
        first = load(fullfile(out_dir, files(1).name));
        fields = fieldnames(first);
        merged = first;

        % Merge remaining slabs (non-zero values overwrite)
        for i = 2:length(files)
            slab = load(fullfile(out_dir, files(i).name));
            for j = 1:length(fields)
                f = fields{j};
                if isnumeric(slab.(f)) && ndims(slab.(f)) >= 3
                    nz = slab.(f) ~= 0 & isfinite(slab.(f));
                    merged.(f)(nz) = slab.(f)(nz);
                end
            end
        end

        save(fullfile(out_dir, 'FitResults_merged.mat'), '-struct', 'merged');
        fprintf('Merged %d slabs into FitResults_merged.mat\n', length(files));

        % Print summary stats
        for j = 1:length(fields)
            f = fields{j};
            if isnumeric(merged.(f)) && ndims(merged.(f)) >= 3
                v = merged.(f)(merged.(f) ~= 0 & isfinite(merged.(f)));
                if ~isempty(v)
                    fprintf('  %s: median=%.4g, n=%d\n', f, median(v(:)), length(v(:)));
                end
            end
        end
    " 2>&1
fi

echo "=== Parallel qMRLab complete ==="
