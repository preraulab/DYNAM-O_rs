function export_validation_segment(out_dir)
%EXPORT_VALIDATION_SEGMENT  Run MATLAB runDYNAMO on the 'segment' example
%   and dump (a) the pass-2 spectrogram + baseline + artifacts + axes, and
%   (b) the final stats_table as CSV. These two files are the ground truth
%   for validating the Rust `dynamo_extract_tfpeaks` C ABI (Phase 4 code)
%   and, later, the MEX wrappers (Phase 5).
%
%   Usage (defaults to DYNAM-O_rs/data_cache/):
%       export_validation_segment()
%       export_validation_segment('/tmp/validation')
%
%   Writes:
%       <out_dir>/segment_spect.mat   containing: spect, stimes, sfreqs,
%                                     baseline, seg_time, merge_thresh,
%                                     trim_vol, downsample_spect,
%                                     dur_min, dur_max, bw_min, bw_max
%       <out_dir>/segment_stats.csv   MATLAB's final peak stats_table
%
%   Quick sanity: stats.csv should have ~5700 rows. .mat should be ~250 MB
%   for 1-second pass-1 or ~500 MB for 2-second pass-2.

    if nargin < 1 || isempty(out_dir)
        this_dir = fileparts(mfilename('fullpath'));          % DYNAM-O_rs/scripts
        out_dir = fullfile(fileparts(this_dir), 'data_cache');
    end
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    % --- Load example_data.mat directly so we can call runDYNAMO with the
    %     full 4-arg signature (which returns timings as the 9th output).
    %     The 'segment' shortcut path routes through runExampleData which
    %     only returns 8 outputs. ---
    % Force the DYNAMO_dev copy — there are other example_data.mat files
    % on the user's MATLAB path (e.g. TF_sigma_peaks_SLEEP2021) that use
    % different variable names (EEG/stages vs data/stage_vals).
    here = fileparts(which('runDYNAMO'));
    assert(~isempty(here), 'runDYNAMO not on MATLAB path');
    % Ensure the full toolbox/ tree is on the path (hann_event_spectra
    % lives in toolbox/helper_functions/dynamo_helpers/; MATLAB's stale
    % `which` cache may miss it when runDYNAMO's own auto-addpath is
    % skipped because computeTFPeaks is already on path).
    addpath(genpath(fullfile(here, 'toolbox')));
    addpath(fullfile(here, 'example_data'));
    ed_path = fullfile(here, 'example_data', 'example_data.mat');
    assert(exist(ed_path, 'file') == 2, ...
        'example_data.mat not found at %s', ed_path);
    ed = load(ed_path);
    assert(isfield(ed, 'data') && isfield(ed, 'Fs') && ...
           isfield(ed, 'stage_times') && isfield(ed, 'stage_vals'), ...
        'Unexpected example_data.mat schema at %s: fields are %s', ...
        ed_path, strjoin(fieldnames(ed), ', '));
    fprintf('Loaded %s\n', ed_path);
    data = ed.data; Fs = ed.Fs;
    stage_times = ed.stage_times; stage_vals = ed.stage_vals;

    % Segment default time range (matches runExampleData's 'segment' preset)
    time_range = [8420, 13446];

    fprintf('Running runDYNAMO on segment [%g, %g]...\n', time_range(1), time_range(2));
    t0 = tic;
    try
        [stats, spect, stimes, sfreqs, ~, t_time_range, artifacts, ~, timings] = ...
            runDYNAMO(data, Fs, stage_times, stage_vals, time_range, ...
                      'verbose', false, 'plot_on', false);
        total_s = timings.total;
    catch ME
        if strcmp(ME.identifier, 'MATLAB:TooManyOutputs')
            % older runDYNAMO: only 8 outputs, no timings field
            [stats, spect, stimes, sfreqs, ~, t_time_range, artifacts, ~] = ...
                runDYNAMO(data, Fs, stage_times, stage_vals, time_range, ...
                          'verbose', false, 'plot_on', false);
            total_s = toc(t0);
        else
            rethrow(ME);
        end
    end
    fprintf('Done. Peaks: %d   wall: %.1f s\n', height(stats), total_s);

    % --- Recompute pass-2 baseline the same way computeTFPeaks does, so
    %     the Rust test can feed (spect_pass2, baseline_pass2) and replay
    %     the final extract pass. ---
    baseline_opts_s = baseline_opts();
    baseline_exclude = artifacts(:);       % matches computeTFPeaks line 296
    baseline_range = baseline_opts_s.baseline_trim;
    if isscalar(baseline_range); baseline_range = [-inf, inf]; end
    excl = logical(interp1(t_time_range, single(baseline_exclude), stimes, 'nearest'));
    in_range = stimes >= baseline_range(1) & stimes <= baseline_range(2);
    valid = ~excl & in_range;
    spect_bl = spect(:, valid);
    spect_bl(spect_bl == 0) = NaN;
    baseline = prctile(spect_bl, baseline_opts_s.baseline_ptile, 2);   %#ok<NASGU>

    % --- Detection options (so the Rust test uses identical params). ---
    d = detection_opts();
    seg_time = d.seg_time;                     %#ok<NASGU>
    merge_thresh = d.merge_thresh;             %#ok<NASGU>
    trim_vol = d.trim_vol;                     %#ok<NASGU>
    downsample_spect = d.downsample_spect;     %#ok<NASGU>
    dur_max = d.dur_max;                       %#ok<NASGU>
    bw_max = d.bw_max;                         %#ok<NASGU>

    % min_duration / min_bandwidth are derived inside computeSpectrogram.
    mtm_window_length_2 = d.mtm_window_length_2;
    df = d.mtm_taper_params(1) / mtm_window_length_2 * 2;
    dur_min = mtm_window_length_2 / 2;         %#ok<NASGU>
    bw_min = df / 2;                           %#ok<NASGU>

    % --- Save spectrogram + params. ---
    spect_path = fullfile(out_dir, 'segment_spect.mat');
    fprintf('Writing %s ... ', spect_path);
    save(spect_path, 'spect', 'stimes', 'sfreqs', 'baseline', ...
        'seg_time', 'merge_thresh', 'trim_vol', 'downsample_spect', ...
        'dur_min', 'dur_max', 'bw_min', 'bw_max', '-v7');
    d_info = dir(spect_path);
    fprintf('%.1f MB\n', d_info.bytes / 1e6);

    % --- Save stats_table as CSV. ---
    stats_path = fullfile(out_dir, 'segment_stats.csv');
    fprintf('Writing %s ... ', stats_path);
    writetable(stats, stats_path);
    s_info = dir(stats_path);
    fprintf('%.1f KB, %d peaks\n', s_info.bytes / 1e3, height(stats));

    fprintf('\nDone. Feed these two files to:\n');
    fprintf('  python scripts/validate_c_api_extract.py\n');
end
