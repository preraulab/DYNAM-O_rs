function results = compare_matlab_vs_rust(dataset, varargin)
%COMPARE_MATLAB_VS_RUST  Run MATLAB runDYNAMO and the Rust-accelerated
%   runDYNAMO_rs on the same example data, then print a side-by-side
%   timing + peak-count + SOPH-similarity comparison.
%
%   Usage:
%       results = compare_matlab_vs_rust()              % segment
%       results = compare_matlab_vs_rust('segment')
%       results = compare_matlab_vs_rust('night')
%
%   Name-value kwargs (passed to both pipelines):
%       'merge_thresh' (default 11)
%       'trim_vol'     (default 0.8)
%       'seg_time'     (default 30)
%       'min_time_in_bin'     (default 5 for segment / 10 for night)
%       'min_peak_at_freq'    (default 10 for segment / 0 for night)
%
%   Returns a struct `results` with fields:
%       .matlab.{timings, stats_table, SOPHs, wall_s}
%       .rust.{timings, stats_table, SOPHs, wall_s}
%       .comparison table
%
%   Requires:
%       - DYNAM-O_dev on the MATLAB path (`which runDYNAMO` returns a path)
%       - setup_dynamo_py has been run (so runDYNAMO_rs works)
%
%   See also runDYNAMO, runDYNAMO_rs, setup_dynamo_py.

    if nargin < 1 || isempty(dataset)
        dataset = 'segment';
    end

    % ---- Resolve the runExampleData defaults for each preset ----
    switch lower(dataset)
        case 'segment'
            time_range_default = [8420 13446];
            min_time_in_bin_default = 5;
            min_peak_at_freq_default = 10;
        case 'night'
            time_range_default = [];  % computed from stages below
            min_time_in_bin_default = 10;
            min_peak_at_freq_default = 0;
        otherwise
            error('compare_matlab_vs_rust:BadDataset', ...
                'dataset must be ''segment'' or ''night'', got %s', dataset);
    end

    p = inputParser;
    p.addOptional('time_range', time_range_default);
    p.addParameter('merge_thresh', 11);
    p.addParameter('trim_vol', 0.8);
    p.addParameter('seg_time', 30);
    p.addParameter('min_time_in_bin', min_time_in_bin_default);
    p.addParameter('min_peak_at_freq', min_peak_at_freq_default);
    p.parse(varargin{:});
    args = p.Results;

    % ---- Locate & load the example data ----
    example_path = which('example_data.mat');
    if isempty(example_path)
        % Try a common location
        candidate = fullfile(fileparts(which('runDYNAMO')), ...
                              '..', 'example_data', 'example_data.mat');
        if exist(candidate, 'file') == 2
            example_path = candidate;
        else
            error('compare_matlab_vs_rust:NoExampleData', ...
                'example_data.mat not on path. Add DYNAM-O_dev/example_data to addpath.');
        end
    end
    fprintf('Loading %s\n', example_path);
    S = load(example_path, 'data', 'Fs', 'stage_times', 'stage_vals');
    data = double(S.data);
    Fs = double(S.Fs);
    stage_times = double(S.stage_times);
    stage_vals = double(S.stage_vals);

    % Derive night time_range if not given
    time_range = args.time_range;
    if strcmpi(dataset, 'night') && isempty(time_range)
        wake_buffer = 5 * 60;
        nw = find(stage_vals < 5 & stage_vals > 0);
        time_range = [stage_times(nw(1)) - wake_buffer, ...
                       stage_times(nw(end)) + wake_buffer];
    end
    fprintf('Dataset: %s, time_range = [%.1f, %.1f] s\n\n', ...
             dataset, time_range(1), time_range(2));

    %% ========== MATLAB run ==========
    fprintf('=== Running MATLAB runDYNAMO ===\n');
    t0 = tic;
    [st_m, sp_m, t_m, f_m, dtr_m, ttr_m, art_m, SOPHs_m, tim_m] = ...
        runDYNAMO(data, Fs, stage_times, stage_vals, time_range, ...
            baseline_opts(), detection_opts(), ...
            set_SOPH_opts(args.min_time_in_bin, args.min_peak_at_freq), ...
            'verbose', false, 'plot_on', false);
    wall_m = toc(t0);
    fprintf('  MATLAB wall: %.1f s\n', wall_m);
    fprintf('  MATLAB peak count: %d\n', height(st_m));

    %% ========== Rust run ==========
    fprintf('\n=== Running runDYNAMO_rs (Rust) ===\n');
    t0 = tic;
    [st_r, sp_r, t_r, f_r, dtr_r, ttr_r, art_r, SOPHs_r, tim_r] = ...
        runDYNAMO_rs(data, Fs, stage_times, stage_vals, time_range, ...
            'merge_thresh', args.merge_thresh, ...
            'trim_vol', args.trim_vol, ...
            'seg_time', args.seg_time, ...
            'min_time_in_bin', args.min_time_in_bin, ...
            'min_peak_at_freq', args.min_peak_at_freq);
    wall_r = toc(t0);
    fprintf('  Rust wall: %.1f s\n', wall_r);
    fprintf('  Rust peak count: %d\n', height(st_r));

    %% ========== Compare ==========
    fprintf('\n=== Comparison ===\n\n');
    print_header();

    % Per-stage timings
    stages = { ...
        'artifact',           'artifacts'; ...
        'spect_pass1',        'spect pass-1'; ...
        'baseline_pass1',     'baseline pass-1'; ...
        'extract_pass1',      'extract pass-1'; ...
        'spect_pass2',        'spect pass-2'; ...
        'baseline_pass2',     'baseline pass-2'; ...
        'extract_pass2',      'extract pass-2'; ...
        'refine',             'Hann refinement'; ...
        'peak_sopower',       'assign SO-power'; ...
        'peak_sophase',       'assign SO-phase'; ...
        'soph_sopower_hist',  'SO-power histogram'; ...
        'soph_sophase_hist',  'SO-phase histogram'; ...
        'total',              'TOTAL'; ...
    };
    for i = 1:size(stages, 1)
        key = stages{i, 1}; label = stages{i, 2};
        tm = getfield_or_nan(tim_m, key);
        tr = getfield_or_nan(tim_r, key);
        print_row(label, tm, tr);
    end

    fprintf('\n%-26s  %-13s  %-13s  %-12s\n', '', 'MATLAB', 'Rust', 'diff');
    print_sep();
    fprintf('%-26s  %13d  %13d  %+12d\n', ...
        'peak count (final)', height(st_m), height(st_r), ...
        height(st_r) - height(st_m));

    fprintf('\n%-26s  %-13s  %-13s\n', '', 'cos vs MATLAB', '');
    print_sep();
    sp_cos = cos_sim(SOPHs_r.SOpower_mat, SOPHs_m.SOpower_mat);
    ph_cos = cos_sim(SOPHs_r.SOphase_mat, SOPHs_m.SOphase_mat);
    fprintf('%-26s  %13.4f\n', 'SOpower cos', sp_cos);
    fprintf('%-26s  %13.4f\n', 'SOphase cos', ph_cos);

    fprintf('\nSpeedup: Rust is %.2fx faster than MATLAB end-to-end\n\n', ...
             wall_m / wall_r);

    %% ========== Return ==========
    results.matlab = struct('timings', tim_m, 'stats_table', st_m, ...
                             'SOPHs', SOPHs_m, 'wall_s', wall_m);
    results.rust   = struct('timings', tim_r, 'stats_table', st_r, ...
                             'SOPHs', SOPHs_r, 'wall_s', wall_r);
    results.time_range = time_range;
    results.dataset = dataset;
    results.speedup = wall_m / wall_r;
    results.SOpower_cos = sp_cos;
    results.SOphase_cos = ph_cos;
end


function v = getfield_or_nan(s, key)
    if isstruct(s) && isfield(s, key)
        v = double(s.(key));
    elseif isa(s, 'py.dict')
        try
            v = double(s{key});
        catch
            v = NaN;
        end
    else
        v = NaN;
    end
end


function c = cos_sim(A, B)
    A = double(A(:)); B = double(B(:));
    m = isfinite(A) & isfinite(B);
    if ~any(m), c = NaN; return; end
    c = dot(A(m), B(m)) / (norm(A(m)) * norm(B(m)) + 1e-20);
end


function print_header()
    fprintf('%-22s %10s %10s %10s  %s\n', ...
        'stage', 'MATLAB(s)', 'Rust(s)', 'speedup', '');
    print_sep();
end


function print_row(label, tm, tr)
    spd_str = '';
    if isfinite(tm) && isfinite(tr) && tr > 0
        spd_str = sprintf('%.2fx', tm / tr);
    end
    tm_str = num_or_dash(tm);
    tr_str = num_or_dash(tr);
    fprintf('%-22s %10s %10s %10s\n', label, tm_str, tr_str, spd_str);
end


function s = num_or_dash(x)
    if ~isfinite(x)
        s = '   —';
    else
        s = sprintf('%.3f', x);
    end
end


function print_sep()
    fprintf('%s\n', repmat('-', 1, 62));
end


function opts = set_SOPH_opts(min_tib, min_paf)
    opts = SOpowerphasehist_opts();
    opts.SOpower_min_time_in_bin = min_tib;
    opts.SOphase_min_peak_at_freq = min_paf;
end
