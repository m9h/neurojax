% qMRLab qmt_spgr fitting for WAND ses-02 QMT data
% Run in Octave: octave-cli 35_qmrlab_qmt.m

addpath(genpath('~/dev/qMRLab'));
pkg load struct; pkg load io; pkg load statistics; pkg load optim; pkg load image;

subject = 'sub-08033';
wand_root = '/Users/mhough/dev/wand';
anat_dir = fullfile(wand_root, subject, 'ses-02', 'anat');
out_dir = fullfile(wand_root, 'derivatives', 'qmri', subject, 'ses-02', 'qmrlab');
mkdir(out_dir);

fprintf('=== qMRLab qmt_spgr fitting: %s ===\n', subject);

% Create qmt_spgr model
Model = qmt_spgr;

% Display default protocol
fprintf('Default protocol:\n');
disp(Model.Prot);

% Set WAND QMT protocol parameters
% MT pulse: offset frequencies and flip angles from BIDS sidecars
offsets_hz = [56360, 1000, 47180, 12060, 2750, 2770, 2790, 2890, 1000];
mt_fa_deg  = [332, 332, 628, 628, 628, 628, 628, 628, 333];
n_mt = length(offsets_hz);

% Readout: FA=5deg, TR=55ms
readout_fa = 5;  % degrees
TR = 0.055;      % seconds

% Set protocol
% qmt_spgr expects: Angle(deg), Offset(Hz)
prot_data = [mt_fa_deg(:), offsets_hz(:)];
Model.Prot.MTdata.Mat = prot_data;

% Timing protocol
Model.Prot.TimingTable.Mat = [readout_fa, TR];

fprintf('Protocol set: %d MT volumes\n', n_mt);
fprintf('Offsets (Hz): '); disp(offsets_hz);
fprintf('MT FA (deg): '); disp(mt_fa_deg);

% Load data
fprintf('\nLoading data...\n');

% MT-off reference
mtoff_file = fullfile(anat_dir, [subject '_ses-02_mt-off_part-mag_QMT.nii.gz']);
nii = load_nii(mtoff_file); mtoff = double(nii.img);
fprintf('MT-off: %s\n', mat2str(size(mtoff)));

% MT-on volumes (in protocol order)
tags = {'flip-1_mt-1', 'flip-1_mt-2', ...
        'flip-2_mt-1', 'flip-2_mt-2', 'flip-2_mt-3', ...
        'flip-2_mt-4', 'flip-2_mt-5', 'flip-2_mt-6', ...
        'flip-3_mt-1'};

mt_data = zeros([size(mtoff), n_mt]);
for i = 1:n_mt
    fname = fullfile(anat_dir, [subject '_ses-02_' tags{i} '_part-mag_QMT.nii.gz']);
    if exist(fname, 'file')
        tmp = load_nii(fname); mt_data(:,:,:,i) = double(tmp.img);
        fprintf('  Loaded %s\n', tags{i});
    else
        fprintf('  MISSING: %s\n', tags{i});
    end
end

% Load T1 map (from QUIT DESPOT1, reoriented to QMT space)
t1_file = fullfile(wand_root, 'derivatives', 'qmri', subject, 'ses-02', 'quit', 'D1_T1_in_QMT.nii.gz');
if exist(t1_file, 'file')
    tmp = load_nii(t1_file); t1_map = double(tmp.img);
    fprintf('T1 map loaded: %s\n', mat2str(size(t1_map)));
else
    fprintf('WARNING: T1 map not found, using default T1=1.0s\n');
    t1_map = ones(size(mtoff));
end

% Load mask
mask_file = fullfile(wand_root, 'derivatives', 'qmri', subject, 'ses-02', 'brain_mask.nii.gz');
if exist(mask_file, 'file')
    tmp = load_nii(mask_file); mask = logical(tmp.img);
    fprintf('Mask: %d voxels\n', sum(mask(:)));
else
    mask = mtoff > prctile(mtoff(mtoff>0), 20);
end

% Build data structure for qMRLab
data.MTdata = mt_data;
data.R1map = 1 ./ max(t1_map, 0.01);  % R1 = 1/T1
data.Mask = mask;

% Set fitting options
Model.options.FitOpt = Model.st;

fprintf('\nFitting qmt_spgr (this may take a while)...\n');
tic;

try
    FitResults = FitData(data, Model, 0);  % 0 = no wait bar
    elapsed = toc;
    fprintf('Fitting complete in %.1f minutes\n', elapsed/60);

    % Save results as NIfTI using load_nii template
    ref_nii = load_nii(mtoff_file);

    fields = fieldnames(FitResults);
    for i = 1:length(fields)
        f = fields{i};
        val = FitResults.(f);
        if isnumeric(val) && ndims(val) >= 3
            out_nii = ref_nii;
            out_nii.img = single(val);
            out_nii.hdr.dime.datatype = 16;  % float32
            out_nii.hdr.dime.bitpix = 32;
            outfile = fullfile(out_dir, ['qMRLab_' f '.nii.gz']);
            save_nii(out_nii, outfile);

            valid = val(mask & isfinite(val) & val ~= 0);
            if ~isempty(valid)
                fprintf('  %s: median=%.4g, IQR=[%.4g, %.4g]\n', ...
                    f, median(valid(:)), prctile(valid(:), 25), prctile(valid(:), 75));
            end
        end
    end
catch e
    elapsed = toc;
    fprintf('ERROR after %.1f min: %s\n', elapsed/60, e.message);
end

fprintf('\n=== qMRLab fitting complete ===\n');
fprintf('Outputs: %s/\n', out_dir);
