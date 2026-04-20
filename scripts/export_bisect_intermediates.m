%EXPORT_BISECT_INTERMEDIATES Rerun runDYNAMO with instrumented intermediates
%                            captured, for the pydynamo backwards-bisection.
%
% Run in MATLAB from the DYNAMO_dev directory:
%   cd ~/code/toolboxes/DYNAMO_dev
%   run('~/code/toolboxes/pydynamo/scripts/export_bisect_intermediates.m')
%
% Writes one file per dataset:
%   data_cache/bisect_intermediates_segment.mat
%   data_cache/bisect_intermediates_night.mat
%
% Fields saved per dataset:
%   spect1_raw          - pass-1 spectrogram (1s window, pre-baseline-divide)
%   baseline1           - pass-1 baseline vector (freq,)
%   stimes1, sfreqs     - pass-1 time/freq axes (absolute time)
%   spect2_masked       - pass-2 post-baseline-divide AND post-pass-1-mask
%   stimes2             - pass-2 stimes (absolute)
%   SOphase_norm        - SOphase timeseries, WRAPPED to (-π, π] (what MATLAB
%                         uses for its SOphase histogram)
%   SOphase_times       - timestamps for SOphase_norm
%   SOphase_stages      - stage per sample (same length as SOphase_times)
%   slope_test_mask     - bad_slope boolean from detect_artifacts (pre-merge
%                         with other bad_inds); useful to verify pydynamo's
%                         slope_test port
%   stats_pre_refine    - stats_table contents as a struct, BEFORE
%                         refinePeakFrequency is applied (so we can test
%                         refinement in isolation)
%   SOphase_filter_sos  - SOS matrix of the MATLAB elliptic bandpass filter
%                         used for SOphase. Allows pydynamo to bypass its
%                         scipy iirdesign and use MATLAB's exact filter.
%   SOphase_filter_ba   - num/den of same filter (fallback representation)
%
% This is meant as a ONE-TIME rerun; the pipeline already runs in ~60-140 s.

clear; clc

addpath(genpath(fullfile(getenv('HOME'), 'code', 'toolboxes', 'DYNAMO_dev', 'toolbox')));

out_dir = fullfile(getenv('HOME'), 'code', 'toolboxes', 'pydynamo', 'data_cache');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

% Load bundled example data
load(fullfile(getenv('HOME'), 'code', 'toolboxes', 'DYNAMO_dev', ...
              'example_data', 'example_data.mat'));

% The bundled .mat stores stage_vals as int8 and data as single;
% DYNAM-O pipeline functions (computeSOphase, designfilt, etc.) validate
% for {double,single}. Cast once up front to keep the rest of this script
% simple.
stage_vals  = double(stage_vals);
stage_times = double(stage_times);
data        = double(data);
Fs          = double(Fs);

segments = {
    'segment', [8420 13446];
    'night',   [];         % computed from stage times per runExampleData.m
};
for s = 1:size(segments, 1)
    name = segments{s, 1};
    tr = segments{s, 2};
    if isempty(tr)
        % 'night': 5-min buffer around first/last non-wake stage
        wake_buffer = 5 * 60;
        nonwake = find(stage_vals < 5 & stage_vals > 0);
        tr = [stage_times(nonwake(1)) - wake_buffer, ...
              stage_times(nonwake(end)) + wake_buffer];
    end
    fprintf('\n=== Capturing intermediates for "%s" (t_range=[%g %g]) ===\n', ...
             name, tr(1), tr(2));

    capture_intermediates(data, Fs, stage_times, stage_vals, tr, ...
        fullfile(out_dir, sprintf('bisect_intermediates_%s.mat', name)));
end

fprintf('\nAll intermediates saved to %s\n', out_dir);


function capture_intermediates(data, Fs, stage_times, stage_vals, time_range, out_path)
    % Minimal reimplementation of computeTFPeaks that ALSO saves the
    % in-between arrays we need for pydynamo bisection.

    % Slice data to time_range (inclusive, matches MATLAB convention)
    t_all = (0:length(data)-1) / Fs;
    time_range_inds = t_all >= time_range(1) & t_all <= time_range(2);
    data_tr = data(time_range_inds);
    t_tr = t_all(time_range_inds);

    % ==== artifacts + slope_test ====
    % Match detect_artifacts.m defaults (slope_test on).
    [artifacts, slope_test_mask] = detect_artifacts_with_slope(data_tr, Fs);

    % ==== pass-1 spectrogram (1 s window) + baseline ====
    freq_range = [0, 30];
    taper_params = [2, 3];
    win1 = [1.0, 0.05];
    dsfreqs = 0.1;
    nfft = 2^nextpow2(Fs / dsfreqs);
    [spect1, stimes1, sfreqs] = multitaper_spectrogram_mex( ...
        data_tr, Fs, freq_range, taper_params, win1, nfft, ...
        'constant', 'unity', false, false);
    stimes1 = stimes1 + t_tr(1);

    % baseline_exclude per computeTFPeaks.m
    baseline_stages = [1,2,3,4,5];
    exclude_stages = ~ismember(stage_vals, baseline_stages);
    exc_resamp = interp1(stage_times, single(exclude_stages), t_tr, 'previous') ~= 0;
    baseline_exclude = artifacts(:) | exc_resamp(:);
    % MATLAB baseline fn inlined:
    bl_ex_stimes = logical(interp1(t_tr, single(baseline_exclude), stimes1, 'nearest'));
    bl_valid = ~bl_ex_stimes;
    sp_bl = spect1(:, bl_valid);
    sp_bl(sp_bl == 0) = NaN;
    baseline1 = prctile(sp_bl, 2, 2);

    % ==== pass-1 extract to get regions/borders for the mask ====
    seg_time = 30;
    downsample_spect = [2, 2];
    merge_thresh = 11;
    trim_vol = 0.8;
    dur_min = 0.5;
    bw_min = 2;
    compute_features = {'PeakFrequency','PeakTime','Duration','Bandwidth','Height','Boundaries'};
    [~, regions, borders] = runSegmentedData( ...
        spect1, stimes1, sfreqs, baseline1, seg_time, downsample_spect, ...
        compute_features, dur_min, bw_min, merge_thresh, inf, trim_vol, ...
        0, false, true, true);

    % ==== pass-2 spectrogram (2 s window) + baseline + mask ====
    win2 = [2.0, 0.05];
    [spect2, stimes2, ~] = multitaper_spectrogram_mex( ...
        data_tr, Fs, freq_range, taper_params, win2, nfft, ...
        'constant', 'unity', false, false);
    stimes2 = stimes2 + t_tr(1);
    bl_ex2 = logical(interp1(t_tr, single(baseline_exclude), stimes2, 'nearest'));
    sp2_bl = spect2(:, ~bl_ex2); sp2_bl(sp2_bl == 0) = NaN;
    baseline2 = prctile(sp2_bl, 2, 2);
    % Divide and mask
    spect2_norm = spect2 ./ baseline2;
    % MATLAB's maskSpectrogram inlined:
    dt = stimes2(2) - stimes2(1);
    indshift = round((stimes2(1) - stimes1(1)) / dt) * size(spect2, 1);
    region_inds = cat(1, regions{:}) - indshift;
    region_inds = region_inds(region_inds >= 1 & region_inds <= numel(spect2));
    border_inds = cat(1, borders{:}) - indshift;
    border_inds = border_inds(border_inds >= 1 & border_inds <= numel(spect2));
    spect2_masked = zeros(size(spect2));
    spect2_masked(region_inds) = spect2_norm(region_inds);
    spect2_masked(border_inds) = 0;

    % ==== pass-2 extract → stats_pre_refine ====
    % Use the REAL pipeline's pass-2 filter params (computeTFPeaks.m line 342
    % ignores pass-2's dur_min return, so pass-2 uses pass-1's dur_min=0.5;
    % bw_min = df/2 where df = TW/T*2 = 2 Hz for pass-2 → bw_min = 1.0).
    p2_dur_min_real = win1(1) / 2;                      % 0.5 (pass-1 window_length/2)
    p2_bw_min_real  = (taper_params(1)/win2(1)*2) / 2;  % 1.0 for pass-2
    p2_dur_max = 5;
    p2_bw_max  = 15;
    % ht_db_min from MATLAB's computeSpectrogram formula (pass-2 uses
    % num_tapers = taper_params(2) = 3 → chi2_df = 6, alpha = 0.95):
    chi2_df_p2 = 2 * taper_params(2);
    ht_db_min_p2 = -pow2db(chi2_df_p2 / chi2inv(0.95/2 + 0.5, chi2_df_p2)) * 2;
    [stats_pre_refine_t, ~, ~] = runSegmentedData( ...
        spect2_masked, stimes2, sfreqs, ones(size(baseline2)), seg_time, downsample_spect, ...
        compute_features, p2_dur_min_real, p2_bw_min_real, merge_thresh, inf, trim_vol, 0, false, true, true);
    % Apply the SAME filterStatsTable that runDYNAMO applies after
    % runSegmentedData (computeTFPeaks.m:369) so stats_pre_refine represents
    % what the real pipeline has immediately before refinePeakFrequency.
    filter_idx_pre_refine = filterStatsTable(stats_pre_refine_t, ...
        [p2_dur_min_real, p2_dur_max], [p2_bw_min_real, p2_bw_max], ...
        [-inf inf], ht_db_min_p2, false);
    stats_pre_refine_t = stats_pre_refine_t(filter_idx_pre_refine, :);
    stats_pre_refine = table2struct(stats_pre_refine_t, 'ToScalar', true);

    % ==== SOphase timeseries ====
    [SOphase_unwrapped, SOphase_times, SOphase_stages, SOphase_filtdata] = ...
        computeSOphase(data_tr, Fs, 'stage_times', stage_times, ...
                       'stage_vals', stage_vals, 'EEG_times', t_tr, ...
                       'isexcluded', artifacts);
    SOphase_norm = wrapToPi(SOphase_unwrapped);

    % ==== SOphase filter coefficients (extract from precomputed or re-design) ====
    SO_freqrange = [0.3, 1.5];
    SOphase_filter_path = fullfile(fileparts(which('computeSOphase')), 'SOphase_filters.mat');
    filter_name = ['filter_' num2str(Fs) 'Hz_' strrep(num2str(SO_freqrange(1)),'.','dot') '_' strrep(num2str(SO_freqrange(2)),'.','dot')];
    if exist(SOphase_filter_path, 'file') && ~isempty(who('-file', SOphase_filter_path, filter_name))
        tmp = load(SOphase_filter_path, filter_name);
        d = tmp.(filter_name);
    else
        d = designfilt('bandpassiir', ...
            'StopbandFrequency1', SO_freqrange(1)-0.1, ...
            'PassbandFrequency1', SO_freqrange(1), ...
            'PassbandFrequency2', SO_freqrange(2), ...
            'StopbandFrequency2', SO_freqrange(2)+0.1, ...
            'StopbandAttenuation1', 60, 'PassbandRipple', 1, ...
            'StopbandAttenuation2', 60, 'DesignMethod', 'ellip', ...
            'MatchExactly', 'passband', 'SampleRate', Fs);
    end
    % Extract filter coefficients. For IIR digitalFilter, d.Coefficients is
    % already the SOS matrix (n_sections x 6). For safety, also apply the
    % filter to a test impulse in MATLAB and save the impulse response so
    % pydynamo can cross-check by filter-by-convolution.
    try
        SOphase_filter_sos = d.Coefficients;  % digitalFilter SOS direct
        SOphase_filter_g = 1.0;
    catch
        SOphase_filter_sos = zeros(0, 6);
        SOphase_filter_g = 1.0;
    end
    % Also save b, a (num, den) via tf()
    try
        [SOphase_filter_b, SOphase_filter_a] = tf(d);
    catch
        SOphase_filter_b = [];
        SOphase_filter_a = [];
    end
    % Impulse response: filter an impulse to get a reference signal that
    % pydynamo can convolve with to emulate MATLAB's filter exactly.
    impulse = [1; zeros(1000, 1)];
    SOphase_impulse_response = filtfilt(d, impulse);

    % Save
    save(out_path, ...
        'spect1', 'stimes1', 'sfreqs', 'baseline1', ...
        'spect2', 'stimes2', 'baseline2', 'spect2_masked', ...
        'artifacts', 'slope_test_mask', ...
        'SOphase_norm', 'SOphase_times', 'SOphase_stages', ...
        'stats_pre_refine', ...
        'SOphase_filter_sos', 'SOphase_filter_g', ...
        'SOphase_filter_b', 'SOphase_filter_a', ...
        'SOphase_impulse_response', ...
        '-v7.3');
    fprintf('Wrote %s\n', out_path);
end


function [artifacts, slope_test_mask] = detect_artifacts_with_slope(data, Fs)
    % Wrapper: run detect_artifacts but also capture the slope_test mask
    % (which is an internal variable). Re-runs the slope test logic from
    % detect_artifacts.m:142-152 here so we can save the mask independently.

    % Main artifacts (default params — includes slope_test)
    artifacts = detect_artifacts(data, Fs);

    % Independent slope_test computation (same as detect_artifacts.m:143-152)
    [spect, stimes, sfreqs] = multitaper_spectrogram_mex( ...
        data, Fs, [1, min(55, Fs/2)], [10, 19], [10, 5], [], [], [], false, false);
    B = zeros(length(stimes), 2);
    for ii = 1:length(stimes)
        B(ii, :) = polyfit(log(sfreqs)', log(spect(:, ii)), 1);
    end
    t = (0:length(data)-1) / Fs;
    slope_crit = -0.5;
    slope_test_mask = interp1(stimes, double(B(:,1) > slope_crit)', t, 'nearest')';
    slope_test_mask(isnan(slope_test_mask)) = 1;
    slope_test_mask = logical(slope_test_mask);
end
