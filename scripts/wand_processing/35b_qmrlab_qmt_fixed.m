% qMRLab qmt_spgr fitting — FIXED output + masked for speed
% Run: octave-cli --eval "run('35b_qmrlab_qmt_fixed.m')"

addpath(genpath('~/dev/qMRLab'));
pkg load struct; pkg load io; pkg load statistics; pkg load optim; pkg load image;

subject = 'sub-08033';
wand_root = '/Users/mhough/dev/wand';
anat_dir = fullfile(wand_root, subject, 'ses-02', 'anat');
quit_dir = fullfile(wand_root, 'derivatives', 'qmri', subject, 'ses-02', 'quit');
out_dir = fullfile(wand_root, 'derivatives', 'qmri', subject, 'ses-02', 'qmrlab');
mkdir(out_dir);

fprintf('=== qMRLab qmt_spgr (fixed I/O) ===\n');

% Create model
Model = qmt_spgr;

% Set protocol (same as before)
offsets_hz = [56360, 1000, 47180, 12060, 2750, 2770, 2790, 2890, 1000];
mt_fa_deg  = [332, 332, 628, 628, 628, 628, 628, 628, 333];
Model.Prot.MTdata.Mat = [mt_fa_deg(:), offsets_hz(:)];
Model.Prot.TimingTable.Mat = [5, 0.055];  % [readout_FA, TR]

% Load data with load_nii
fprintf('Loading data...\n');
mtoff_nii = load_nii(fullfile(anat_dir, [subject '_ses-02_mt-off_part-mag_QMT.nii.gz']));
mt_off = double(mtoff_nii.img);

tags = {'flip-1_mt-1', 'flip-1_mt-2', 'flip-2_mt-1', 'flip-2_mt-2', ...
        'flip-2_mt-3', 'flip-2_mt-4', 'flip-2_mt-5', 'flip-2_mt-6', 'flip-3_mt-1'};
n_mt = length(tags);
mt_data = zeros([size(mt_off), n_mt]);
for i = 1:n_mt
    fname = fullfile(anat_dir, [subject '_ses-02_' tags{i} '_part-mag_QMT.nii.gz']);
    tmp = load_nii(fname);
    mt_data(:,:,:,i) = double(tmp.img);
    fprintf('  %s loaded\n', tags{i});
end

% Load T1 map (reoriented to QMT space)
t1_nii = load_nii(fullfile(quit_dir, 'D1_T1_in_QMT.nii.gz'));
t1_map = double(t1_nii.img);

% Create a tighter brain mask to speed up fitting
mask = (t1_map > 0.2) & (t1_map < 4.0) & (mt_off > 50);
fprintf('Mask voxels: %d (of %d total)\n', sum(mask(:)), numel(mask));

% Build data structure
data.MTdata = mt_data;
data.R1map = 1.0 ./ max(t1_map, 0.01);
data.Mask = mask;

fprintf('Fitting (this will take a while)...\n');
tic;

try
    FitResults = FitData(data, Model, 0);
    elapsed = toc;
    fprintf('Fitting complete in %.1f minutes\n', elapsed/60);

    % Save each field as NIfTI using mat2nii approach
    fields = fieldnames(FitResults);
    for i = 1:length(fields)
        f = fields{i};
        val = FitResults.(f);
        if isnumeric(val) && ndims(val) >= 3 && all(size(val) == size(mask))
            % Create NIfTI from the reference
            out_nii = mtoff_nii;
            out_nii.img = single(val);
            out_nii.hdr.dime.datatype = 16;
            out_nii.hdr.dime.bitpix = 32;
            out_nii.hdr.dime.dim(5) = 1;
            outfile = fullfile(out_dir, ['qMRLab_' f '.nii.gz']);

            try
                save_nii(out_nii, outfile);

                valid = val(mask & isfinite(val) & val ~= 0);
                if ~isempty(valid)
                    fprintf('  %s: median=%.4g, n=%d, saved\n', ...
                        f, median(valid(:)), length(valid(:)));
                end
            catch me
                fprintf('  %s: save failed (%s), trying mat save\n', f, me.message);
                save(fullfile(out_dir, ['qMRLab_' f '.mat']), 'val');
            end
        end
    end

    % Also save complete results as .mat
    save(fullfile(out_dir, 'FitResults.mat'), '-struct', 'FitResults');
    fprintf('Saved FitResults.mat\n');

catch me
    elapsed = toc;
    fprintf('ERROR after %.1f min: %s\n', elapsed/60, me.message);
    fprintf('%s\n', me.stack(1).name);
end

fprintf('\n=== Done ===\n');
