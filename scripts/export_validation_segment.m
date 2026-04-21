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

    % --- Run the full MATLAB pipeline on segment data. ---
    fprintf('Running runDYNAMO(''segment'')...\n');
    [stats, spect, stimes, sfreqs, ~, t_time_range, artifacts, ~, timings] = ...
        runDYNAMO('segment');
    fprintf('Done. Peaks: %d   wall: %.1f s\n', height(stats), timings.total);

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
