%EXPORT_MERGE_DIAGNOSTICS Export MATLAB data that pydynamo is missing for
% the bisect. This is a ONE-TIME rerun — the heavy lifting is pass-2 extract
% on the segment dataset, which runs in ~45 s.
%
% Run from MATLAB:
%   cd ~/code/toolboxes/DYNAMO_dev
%   run('~/code/toolboxes/pydynamo/scripts/export_merge_diagnostics.m')
%
% Writes:
%   data_cache/night_stats.csv                     (full MATLAB stats_table, night)
%   data_cache/segment_stats.csv                   (re-exported in case format drifted)
%   data_cache/merge_diagnostics_segment.mat       (per-segment merge counts +
%                                                   one representative segment's
%                                                   pre-merge, post-merge, post-trim
%                                                   data and label images)
%
% The representative-segment dump (default: segment index 40, a mid-night
% NREM-rich segment) lets pydynamo replay MATLAB's merge+trim on MATLAB's
% exact pre-merge input and diff region-by-region at each step — this is
% what we need to localize the 9 % pass-2 over-detect.

clear; clc

addpath(genpath(fullfile(getenv('HOME'), 'code', 'toolboxes', 'DYNAMO_dev', 'toolbox')));

out_dir = fullfile(getenv('HOME'), 'code', 'toolboxes', 'pydynamo', 'data_cache');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

% Load bundled example data (same path as export_bisect_intermediates.m)
load(fullfile(getenv('HOME'), 'code', 'toolboxes', 'DYNAMO_dev', ...
              'example_data', 'example_data.mat'));

% Cast so DYNAM-O validators accept the inputs
stage_vals  = double(stage_vals);
stage_times = double(stage_times);
data        = double(data);
Fs          = double(Fs);

% Load the default opts
baseline_options = baseline_opts();
detection_options = detection_opts();
SOPH_options = SOpowerphasehist_opts();

%% ==== 1. Run full pipeline for BOTH datasets and save final stats_table CSVs ====
% Re-run runDYNAMO so the CSV columns (including Volume/SOphase/SOpower/PeakStage)
% match exactly what pydynamo's compare script expects.

tr_segment = [8420 13446];

nonwake = find(stage_vals < 5 & stage_vals > 0);
wake_buffer = 5 * 60;
tr_night = [stage_times(nonwake(1)) - wake_buffer, ...
            stage_times(nonwake(end)) + wake_buffer];

for run_case = {'segment', tr_segment, 5,  10; ...
                'night',   tr_night,   10, 10}'
    name = run_case{1};
    tr   = run_case{2};
    SOPH_options.SOpower_min_time_in_bin = run_case{3};
    SOPH_options.SOphase_min_peak_at_freq = run_case{4};
    fprintf('\n=== Re-running runDYNAMO(%s) for stats_table CSV ===\n', name);
    [stats_table, ~, ~, ~, ~, ~, ~, ~, ~] = runDYNAMO( ...
        data, Fs, stage_times, stage_vals, tr, ...
        baseline_options, detection_options, SOPH_options, ...
        'verbose', false, 'plot_on', false);
    csv_path = fullfile(out_dir, sprintf('%s_stats.csv', name));
    % Drop cell-valued columns (HeightData, Boundaries) so the CSV stays
    % loadable by pandas. Keep all numeric/logical columns (including
    % multi-column ones like BoundingBox — writetable auto-expands those
    % into BoundingBox_1..BoundingBox_4).
    keep_cols = false(1, width(stats_table));
    for ci = 1:width(stats_table)
        v = stats_table{:, ci};
        keep_cols(ci) = ~iscell(v) && (isnumeric(v) || islogical(v));
    end
    stats_small = stats_table(:, keep_cols);
    writetable(stats_small, csv_path);
    fprintf('  wrote %s (%d rows, %d cols)\n', csv_path, height(stats_small), width(stats_small));
end

%% ==== 2. Per-segment merge diagnostics (segment dataset, pass-2) ====
% Re-produce the pass-2 input (spect2_masked) the same way
% export_bisect_intermediates.m does, then walk segments manually,
% recording pre-merge / post-merge / post-dur-bw-filter / post-trim /
% post-final-filter region counts. Also save ONE representative segment's
% full intermediate arrays for pydynamo to replay.

fprintf('\n=== Capturing per-segment merge diagnostics (segment dataset) ===\n');

% Slice data
t_all = (0:length(data)-1) / Fs;
tr = tr_segment;
time_range_inds = t_all >= tr(1) & t_all <= tr(2);
data_tr = data(time_range_inds);
t_tr = t_all(time_range_inds);

% Re-compute pass-1 spect, baseline, extract regions, mask -> spect2_masked
freq_range = [0, 30];
taper_params = [2, 3];
dsfreqs = 0.1;
nfft = 2^nextpow2(Fs / dsfreqs);
win1 = [1.0, 0.05];
win2 = [2.0, 0.05];
seg_time = 30;
downsample_spect = [2, 2];
merge_thresh = 11;
trim_vol = 0.8;

% Pass-1
[spect1, stimes1, sfreqs] = multitaper_spectrogram_mex( ...
    data_tr, Fs, freq_range, taper_params, win1, nfft, ...
    'constant', 'unity', false, false);
stimes1 = stimes1 + t_tr(1);
% Load artifacts from the prior bisect export (so we don't re-run artifact
% detection — which has a private wrapper in export_bisect_intermediates.m).
bisect_mat = fullfile(out_dir, 'bisect_intermediates_segment.mat');
assert(exist(bisect_mat, 'file') > 0, ...
    'Expected %s to exist from a prior export_bisect_intermediates.m run.', ...
    bisect_mat);
tmp_bisect = load(bisect_mat, 'artifacts');
artifacts = logical(tmp_bisect.artifacts(:));
clear tmp_bisect;
exclude_stages = ~ismember(stage_vals, [1,2,3,4,5]);
exc_resamp = interp1(stage_times, single(exclude_stages), t_tr, 'previous') ~= 0;
baseline_exclude = artifacts(:) | exc_resamp(:);
bl_ex1 = logical(interp1(t_tr, single(baseline_exclude), stimes1, 'nearest'));
sp_bl = spect1(:, ~bl_ex1); sp_bl(sp_bl == 0) = NaN;
baseline1 = prctile(sp_bl, 2, 2);
% Pass-1 extract — match REAL pipeline: dur_min = window_length/2,
% bw_min = df/2 where df = taper_params(1)/window_length*2 (multitaper
% spectral resolution, NOT bin width).
df_pass1 = taper_params(1)/win1(1)*2;   % = 4.0 Hz for pass-1
dur_min  = win1(1) / 2;                 % = 0.5
bw_min   = df_pass1 / 2;                % = 2.0  (was wrongly (sfreqs(2)-sfreqs(1))/2)
compute_features = {'PeakFrequency','PeakTime','Duration','Bandwidth','Height','Boundaries'};
[~, regions1, borders1] = runSegmentedData( ...
    spect1, stimes1, sfreqs, baseline1, seg_time, downsample_spect, ...
    compute_features, dur_min, bw_min, merge_thresh, inf, trim_vol, ...
    0, false, true, true);

% Pass-2 spect + baseline + mask
[spect2, stimes2, ~] = multitaper_spectrogram_mex( ...
    data_tr, Fs, freq_range, taper_params, win2, nfft, ...
    'constant', 'unity', false, false);
stimes2 = stimes2 + t_tr(1);
bl_ex2 = logical(interp1(t_tr, single(baseline_exclude), stimes2, 'nearest'));
sp2_bl = spect2(:, ~bl_ex2); sp2_bl(sp2_bl == 0) = NaN;
baseline2 = prctile(sp2_bl, 2, 2);
spect2_norm = spect2 ./ baseline2;
dt = stimes2(2) - stimes2(1);
indshift = round((stimes2(1) - stimes1(1)) / dt) * size(spect2, 1);
region_inds = cat(1, regions1{:}) - indshift;
region_inds = region_inds(region_inds >= 1 & region_inds <= numel(spect2));
border_inds = cat(1, borders1{:}) - indshift;
border_inds = border_inds(border_inds >= 1 & border_inds <= numel(spect2));
spect2_masked = zeros(size(spect2));
spect2_masked(region_inds) = spect2_norm(region_inds);
spect2_masked(border_inds) = 0;

% === walk segments manually and record region counts at each stage ===
% Replicate segmentData's windowing
[data_segs, x_segs, x_inds] = segmentData(spect2_masked, stimes2, sfreqs, seg_time, 0, '');
n_seg = numel(data_segs);

% Per-segment counts + a few summary statistics
seg_counts = struct( ...
    'n_regions_postwshed',  zeros(n_seg, 1), ...
    'n_regions_postmerge',  zeros(n_seg, 1), ...
    'n_regions_postdurbw',  zeros(n_seg, 1), ...
    'n_regions_posttrim',   zeros(n_seg, 1), ...
    'n_regions_postfilter', zeros(n_seg, 1));

% Representative segment to fully dump (pick a mid-night segment likely to
% have many NREM peaks; seg index 40 is ~30*40/3600 ≈ 20 min into segment)
rep_seg = min(40, n_seg);
rep_dump = struct();

% pass-2 dur/bw/trim config — match the REAL pipeline (computeTFPeaks.m
% line 342 keeps dur_min=0.5 from pass-1; bw_min = df/2 = TW/T = 1.0 for
% pass-2 with taper_params=[2,3] and window=[2,0.05]).
p2_dur_min = 0.5;
p2_bw_min  = 1.0;
p2_conn_wshed = 8;
p2_conn_trim = 8;
p2_merge_rule = 'default';
p2_bl_thresh = 0;
p2_trim_shift = false;

for ii = 1:n_seg
    img = data_segs{ii};
    x = x_segs{ii};
    y = sfreqs;
    if all(img(:) == 0) || all(isnan(img(:))) || length(x) <= 1
        continue
    end

    % Downsample (matches extractTFPeaks.m:207)
    img_LR = img(1:downsample_spect(2):end, 1:downsample_spect(1):end);

    % Watershed
    Ldata = runWatershed(img_LR, p2_conn_wshed, p2_bl_thresh, 0, '', 0);
    [regions, region_lbls, borders, adj_list] = Ldata2graph(Ldata, [], 0);
    seg_counts.n_regions_postwshed(ii) = numel(regions);

    % Save representative segment pre-merge state
    if ii == rep_seg
        rep_dump.seg_num = ii;
        rep_dump.img_full = img;
        rep_dump.img_LR = img_LR;
        rep_dump.x_segment = x(:);
        rep_dump.sfreqs = y(:);
        rep_dump.Ldata_wshed = Ldata;
        rep_dump.regions_pre = regions;
        rep_dump.borders_pre = borders;
        rep_dump.adj_list_pre = adj_list;
        rep_dump.region_lbls_pre = region_lbls;
    end

    % Merge
    if ~isempty(adj_list)
        [regions, borders] = mergeWshedSegment(img_LR, regions, region_lbls, ...
            borders, adj_list, merge_thresh, inf, p2_merge_rule, 0, '', 0);
    end
    seg_counts.n_regions_postmerge(ii) = numel(regions);

    if ii == rep_seg
        rep_dump.regions_postmerge_LR = regions;
        rep_dump.borders_postmerge_LR = borders;
    end

    % Upscale labels to HR
    Ldata_HR = zeros(size(img_LR));
    for kk = 1:numel(regions)
        if ~isempty(regions{kk})
            Ldata_HR(regions{kk}) = kk;
        end
    end
    LdataHR = imresize(Ldata_HR, size(img), 'nearest');
    regions_HR = cell(1, numel(regions));
    for kk = 1:numel(regions)
        regions_HR{kk} = find(LdataHR == kk);
    end
    regions = regions_HR;

    % Pre-trim dur/bw filter
    dt_img = x(2) - x(1); df_img = y(2) - y(1);
    [f_inds, t_inds] = cellfun(@(r) ind2sub(size(img), r), regions, 'UniformOutput', false);
    good = cellfun(@(tt) ~isempty(tt) && (max(tt)-min(tt))*dt_img > p2_dur_min, t_inds) & ...
           cellfun(@(ff) ~isempty(ff) && (max(ff)-min(ff))*df_img > p2_bw_min, f_inds);
    regions = regions(good);
    borders = borders(good);
    seg_counts.n_regions_postdurbw(ii) = numel(regions);

    if ii == rep_seg
        rep_dump.regions_postdurbw_HR = regions;
        rep_dump.borders_postdurbw_HR = borders;
    end

    % Trim
    if ~isempty(regions) && trim_vol < 1
        [trim_regions, trim_borders] = trimWshedRegions(img, regions, trim_vol, ...
            p2_trim_shift, p2_conn_trim, 0, '', 0, true);
        % post-trim dur/bw
        [f_inds, t_inds] = cellfun(@(r) ind2sub(size(img), r), trim_regions, 'UniformOutput', false);
        good = cellfun(@(tt) ~isempty(tt) && (max(tt)-min(tt))*dt_img > p2_dur_min, t_inds) & ...
               cellfun(@(ff) ~isempty(ff) && (max(ff)-min(ff))*df_img > p2_bw_min, f_inds);
        trim_regions = trim_regions(good);
        trim_borders = trim_borders(good);
        seg_counts.n_regions_posttrim(ii) = numel(trim_regions);
        % final (redundant) filter — MATLAB does this twice
        [f_inds, t_inds] = cellfun(@(r) ind2sub(size(img), r), trim_regions, 'UniformOutput', false);
        good = cellfun(@(tt) ~isempty(tt) && (max(tt)-min(tt))*dt_img > p2_dur_min, t_inds) & ...
               cellfun(@(ff) ~isempty(ff) && (max(ff)-min(ff))*df_img > p2_bw_min, f_inds);
        trim_regions = trim_regions(good);
        seg_counts.n_regions_postfilter(ii) = numel(trim_regions);

        if ii == rep_seg
            rep_dump.regions_posttrim_HR = trim_regions;
            rep_dump.borders_posttrim_HR = trim_borders;
        end
    end
end

fprintf('Per-segment region counts (summary over %d segments):\n', n_seg);
for fld = fieldnames(seg_counts)'
    v = seg_counts.(fld{1});
    fprintf('  %-22s sum=%6d  median=%5.0f  max=%5d\n', fld{1}, sum(v), median(v), max(v));
end

% Save
save(fullfile(out_dir, 'merge_diagnostics_segment.mat'), ...
    'seg_counts', 'rep_dump', 'rep_seg', 'n_seg', ...
    'downsample_spect', 'merge_thresh', 'trim_vol', ...
    'p2_dur_min', 'p2_bw_min', 'p2_conn_wshed', 'p2_conn_trim', ...
    '-v7.3');
fprintf('Wrote %s\n', fullfile(out_dir, 'merge_diagnostics_segment.mat'));

fprintf('\nALL DONE — pydynamo can now consume:\n');
fprintf('  - %s\n', fullfile(out_dir, 'segment_stats.csv'));
fprintf('  - %s\n', fullfile(out_dir, 'night_stats.csv'));
fprintf('  - %s\n', fullfile(out_dir, 'merge_diagnostics_segment.mat'));
