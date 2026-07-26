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
    % Force the DYNAM-O copy — there are other example_data.mat files
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

    % Replicate computeTFPeaks's pass-1 -> mask step to capture the MASKED
    % pass-2 spectrogram — that's what MATLAB's final extract runs on, and
    % what MEX/pydynamo need to see for apples-to-apples peak comparison.
    fprintf('Computing pass-1 spectrogram + regions for masking ...\n');
    d = detection_opts();
    b = baseline_opts();
    t_data  = (0:length(data)-1) / Fs;
    data_tr = data(t_data >= time_range(1) & t_data <= time_range(2));
    t_tr    = t_time_range;
    % Pass-1 multitaper spectrogram (1-s window)
    nfft1 = 2^(nextpow2(Fs / d.mtm_dsfreqs));
    [spect1, stimes1, sfreqs_check] = multitaper_spectrogram(...
        data_tr, Fs, d.mtm_freq_range, d.mtm_taper_params, ...
        [d.mtm_window_length_1, d.mtm_window_stepsize], ...
        nfft1, 'constant', 'unity', false, false);
    stimes1 = stimes1 + t_tr(1);
    % Pass-1 baseline (same recipe as computeTFPeaks)
    excl_stages = ~ismember(stage_vals, b.baseline_stages);
    excl_re = interp1(stage_times, single(excl_stages), t_tr, 'previous') ~= 0;
    bl_exclude = artifacts(:) | excl_re(:);
    bl_excl_st = logical(interp1(t_tr, single(bl_exclude), stimes1, 'nearest'));
    spect1_bl = spect1(:, ~bl_excl_st);
    spect1_bl(spect1_bl == 0) = NaN;
    baseline1 = prctile(spect1_bl, b.baseline_ptile, 2);
    % Pass-1 extract with regions + borders
    [downsample_spect1, seg_time1, merge_thresh1] = deal(d.downsample_spect, d.seg_time, d.merge_thresh);
    if isempty(downsample_spect1); downsample_spect1 = [2 2]; end
    if isempty(seg_time1); seg_time1 = 30; end
    if isempty(merge_thresh1); merge_thresh1 = 11; end
    df1 = d.mtm_taper_params(1) / d.mtm_window_length_1 * 2;
    dur_min1 = d.mtm_window_length_1 / 2;
    bw_min1  = df1 / 2;
    fprintf('  pass-1 runSegmentedData ...\n');
    [~, regions1, borders1] = runSegmentedData( ...
        spect1, stimes1, sfreqs, baseline1, seg_time1, downsample_spect1, 'all', ...
        dur_min1, bw_min1, merge_thresh1, inf, d.trim_vol, -1, false, false);
    % Mask the pass-2 spectrogram using pass-1 regions + borders.
    % Inlined from the private maskSpectrogram helper in computeTFPeaks.m
    % (it's a nested function there, not on the path).
    dt_shift = stimes(2) - stimes(1);
    idx_shift = round((stimes(1) - stimes1(1)) / dt_shift) * size(spect, 1);
    all_region_inds = cat(1, regions1{:}) - idx_shift;
    all_region_inds = all_region_inds(all_region_inds >= 1 & all_region_inds <= numel(spect));
    spect_masked = zeros(size(spect));
    spect_masked(all_region_inds) = spect(all_region_inds);
    all_border_inds = cat(1, borders1{:}) - idx_shift;
    all_border_inds = all_border_inds(all_border_inds >= 1 & all_border_inds <= numel(spect));
    spect_masked(all_border_inds) = 0;
    % Pass-2 baseline (runDYNAMO already applied it; recompute for save)
    bl_excl_st2 = logical(interp1(t_tr, single(bl_exclude), stimes, 'nearest'));
    spect2_bl = spect(:, ~bl_excl_st2);
    spect2_bl(spect2_bl == 0) = NaN;
    baseline2 = prctile(spect2_bl, b.baseline_ptile, 2);
    fprintf('  saved spect_masked [%d x %d], baseline2 [%d]\n', ...
        size(spect_masked, 1), size(spect_masked, 2), numel(baseline2));

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

    % min_duration / min_bandwidth for the FINAL pass-2 extract+filter.
    % computeTFPeaks.m line 275 sets dur_min from pass-1 and NEVER
    % recomputes it at line 342, so pass-2 uses the pass-1 value.
    % bw_min IS recomputed at line 342 for pass-2.
    dur_min = d.mtm_window_length_1 / 2;       %#ok<NASGU>  pass-1 value stays
    df_pass2 = d.mtm_taper_params(1) / d.mtm_window_length_2 * 2;
    bw_min   = df_pass2 / 2;                   %#ok<NASGU>  pass-2

    % ht_db_min matches computeTFPeaks.m:395-397:
    %   chi2_df = 2 * taper_params(2), alpha = 0.95
    %   ht_db_min = -pow2db(chi2_df / chi2inv(alpha/2 + 0.5, chi2_df)) * 2
    chi2_df = 2 * d.mtm_taper_params(2);
    alpha = 0.95;
    ht_db_min = -pow2db(chi2_df / chi2inv(alpha/2 + 0.5, chi2_df)) * 2; %#ok<NASGU>

    % --- Save spectrogram + params. ---
    spect_path = fullfile(out_dir, 'segment_spect.mat');
    fprintf('Writing %s ... ', spect_path);
    save(spect_path, 'spect', 'spect_masked', 'stimes', 'sfreqs', ...
        'baseline', 'baseline2', ...
        'seg_time', 'merge_thresh', 'trim_vol', 'downsample_spect', ...
        'dur_min', 'dur_max', 'bw_min', 'bw_max', 'ht_db_min', '-v7');
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
