%EXPORT_MATLAB_GROUND_TRUTH Re-export DYNAM-O segment_out.mat for Python validation.
%
% Run this from MATLAB in the DYNAMO_dev directory:
%   cd ~/code/toolboxes/DYNAMO_dev
%   run('~/code/toolboxes/pydynamo/scripts/export_matlab_ground_truth.m')
%
% Why:
%   `segment_out.mat` was saved with the full `runDYNAMO('segment')` output,
%   including the peak `stats_table` which is a MATLAB `table` object. Python's
%   `scipy.io.loadmat` cannot decode MATLAB tables (they show up as
%   `MatlabOpaque`). This script extracts the table to:
%     - segment_stats.csv       (human-readable, all scalar/numeric columns)
%     - segment_stats_struct.mat (table2struct, v7.3 so Python's h5py can read it)
%
% Outputs are written to ~/code/toolboxes/pydynamo/data_cache/

clear; clc

out_dir = fullfile(getenv('HOME'), 'code', 'toolboxes', 'pydynamo', 'data_cache');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

in_path = fullfile(getenv('HOME'), 'code', 'toolboxes', 'DYNAMO_dev', 'segment_out.mat');
fprintf('Loading %s ...\n', in_path);
S = load(in_path);

% Find the stats_table variable. It should be named 'stats_table' but in case
% the user used a different name, fall back to scanning for any table object.
stats_var = '';
if isfield(S, 'stats_table') && istable(S.stats_table)
    stats_var = 'stats_table';
else
    fn = fieldnames(S);
    for k = 1:numel(fn)
        if istable(S.(fn{k}))
            stats_var = fn{k};
            break
        end
    end
end

if isempty(stats_var)
    error(['export_matlab_ground_truth:no_table', ...
           ' — no table found in segment_out.mat. Did runDYNAMO finish?']);
end

T = S.(stats_var);
fprintf('Found stats table `%s` with %d peaks, %d columns.\n', ...
        stats_var, height(T), width(T));

% CSV: scalar-column-only subset.
% `writetable` writes whatever it can; multi-element cell columns (e.g.
% HeightData / Boundaries / BoundingBox) get serialized as strings or skipped
% depending on type, which is fine — we only need per-peak scalars for
% SOPH validation (PeakTime, PeakFrequency, SOpower, SOphase, PeakStage).
csv_path = fullfile(out_dir, 'segment_stats.csv');
try
    writetable(T, csv_path);
    fprintf('Wrote %s (%d rows)\n', csv_path, height(T));
catch ME
    warning('writetable failed (%s); writing scalar subset instead.', ME.message);
    scalar_vars = {};
    for k = 1:width(T)
        v = T{1, k};
        if isnumeric(v) && isscalar(v)
            scalar_vars{end+1} = T.Properties.VariableNames{k}; %#ok<AGROW>
        end
    end
    T_scalar = T(:, scalar_vars);
    writetable(T_scalar, csv_path);
    fprintf('Wrote scalar subset to %s (%d rows, %d cols)\n', ...
            csv_path, height(T_scalar), width(T_scalar));
end

% MAT (v7.3 so h5py can read it). Convert to struct-of-arrays.
stats_struct = table2struct(T, 'ToScalar', true); %#ok<NASGU>
mat_path = fullfile(out_dir, 'segment_stats_struct.mat');
save(mat_path, 'stats_struct', '-v7.3');
fprintf('Wrote %s\n', mat_path);

% Also re-save the SOPHs / spect / artifacts in a Python-friendly layout.
% segment_out.mat itself is already Python-readable for these fields, but
% copying them into data_cache/ keeps all validation inputs in one place.
spect = S.spect;             %#ok<NASGU>
stimes = S.stimes;           %#ok<NASGU>
sfreqs = S.sfreqs;           %#ok<NASGU>
artifacts = S.artifacts;     %#ok<NASGU>
data_time_range = S.data_time_range;   %#ok<NASGU>
t_time_range = S.t_time_range;         %#ok<NASGU>

% Flatten SOPHs struct (drop param/spline fits which have MATLAB fitobj that
% won't round-trip through h5py cleanly).
SOPHs_flat = struct();
keep = {'SOpower_mat','SOphase_mat','SOpower_bins','SOphase_bins','freq_bins', ...
        'num_peaks_at_freq','SOpower_TIB','SOphase_TIB', ...
        'SOpower_norm','SOpower_times'};
for k = 1:numel(keep)
    if isfield(S.SOPHs, keep{k})
        SOPHs_flat.(keep{k}) = S.SOPHs.(keep{k});
    end
end

comp_path = fullfile(out_dir, 'segment_out_compat.mat');
save(comp_path, 'spect', 'stimes', 'sfreqs', 'artifacts', ...
     'data_time_range', 't_time_range', 'SOPHs_flat', '-v7.3');
fprintf('Wrote %s\n', comp_path);

fprintf('Done. Python tests can now load from %s\n', out_dir);
