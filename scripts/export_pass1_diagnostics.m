%EXPORT_PASS1_DIAGNOSTICS Export MATLAB pass-1 intermediates so pydynamo
% can diagnose why its pass-1 produces ~60% more regions than MATLAB
% (pydynamo mask covers 24% of pass-2 spect, MATLAB only 14%).
%
% Run from MATLAB:
%   cd ~/code/toolboxes/DYNAMO_dev
%   run('~/code/toolboxes/pydynamo/scripts/export_pass1_diagnostics.m')
%
% Writes:
%   data_cache/pass1_diagnostics_segment.mat   — per-segment pass-1 region
%     counts at each stage (wshed / post-merge / post-dur-bw / post-trim /
%     post-filter), full pre-merge + post-merge + post-trim state for one
%     representative segment (seg 40), AND the final pass-1 regions/borders
%     cell arrays for the WHOLE segment dataset (so pydynamo can use them
%     directly to mask pass-2 and isolate pass-2 divergence).
%   data_cache/pass1_stats_segment.csv         — MATLAB pass-1 final
%     stats_table (pre pass-2) for direct peak-by-peak comparison.
%
% Runtime ~90 s (pass-1 extract dominates).

clear; clc
addpath(genpath(fullfile(getenv('HOME'), 'code', 'toolboxes', 'DYNAMO_dev', 'toolbox')));

out_dir = fullfile(getenv('HOME'), 'code', 'toolboxes', 'pydynamo', 'data_cache');
load(fullfile(getenv('HOME'), 'code', 'toolboxes', 'DYNAMO_dev', ...
              'example_data', 'example_data.mat'));
stage_vals  = double(stage_vals);
stage_times = double(stage_times);
data        = double(data);
Fs          = double(Fs);

tr = [8420 13446];

% Slice data
t_all = (0:length(data)-1) / Fs;
time_range_inds = t_all >= tr(1) & t_all <= tr(2);
data_tr = data(time_range_inds);
t_tr = t_all(time_range_inds);

% ==== pass-1 spectrogram (1s window) + baseline ====
freq_range = [0, 30];
taper_params = [2, 3];
dsfreqs = 0.1;
nfft = 2^nextpow2(Fs / dsfreqs);
win1 = [1.0, 0.05];
seg_time = 30;
downsample_spect = [2, 2];
merge_thresh = 11;
trim_vol = 0.8;

fprintf('Running pass-1 spectrogram...\n');
[spect1, stimes1, sfreqs] = multitaper_spectrogram_mex( ...
    data_tr, Fs, freq_range, taper_params, win1, nfft, ...
    'constant', 'unity', false, false);
stimes1 = stimes1 + t_tr(1);

% Load artifacts from prior bisect export (keep consistent)
bisect_mat = fullfile(out_dir, 'bisect_intermediates_segment.mat');
tmp = load(bisect_mat, 'artifacts');
artifacts = logical(tmp.artifacts(:));
clear tmp;

exclude_stages = ~ismember(stage_vals, [1,2,3,4,5]);
exc_resamp = interp1(stage_times, single(exclude_stages), t_tr, 'previous') ~= 0;
baseline_exclude = artifacts(:) | exc_resamp(:);

bl_ex1 = logical(interp1(t_tr, single(baseline_exclude), stimes1, 'nearest'));
sp_bl = spect1(:, ~bl_ex1); sp_bl(sp_bl == 0) = NaN;
baseline1 = prctile(sp_bl, 2, 2);
spect1_norm = spect1 ./ baseline1;

% Pass-1 params (MATLAB computeTFPeaks: window/2, df/2)
df = taper_params(1)/win1(1)*2;  % = 4.0 Hz for pass-1
dur_min = win1(1) / 2;           % = 0.5
bw_min = df / 2;                  % = 2.0

fprintf('Pass-1 params: dur_min=%.3f bw_min=%.3f\n', dur_min, bw_min);
fprintf('Running pass-1 extract + capturing per-segment diagnostics...\n');

% Walk segments manually (same as pass-2 diag script)
[data_segs, x_segs, x_inds] = segmentData(spect1_norm, stimes1, sfreqs, seg_time, 0, '');
n_seg = numel(data_segs);

seg_counts = struct( ...
    'n_regions_postwshed',  zeros(n_seg, 1), ...
    'n_regions_postmerge',  zeros(n_seg, 1), ...
    'n_regions_postdurbw',  zeros(n_seg, 1), ...
    'n_regions_posttrim',   zeros(n_seg, 1), ...
    'n_regions_postfilter', zeros(n_seg, 1));

rep_seg = min(40, n_seg);
rep_dump = struct();

% All final regions/borders across segments, concatenated into LINEAR
% indices of the FULL spect1 (same convention as computeTFPeaks stores them)
all_regions = cell(1, 0);
all_borders = cell(1, 0);
all_segment_nums = [];

p1_conn_wshed = 8;
p1_conn_trim = 8;
p1_merge_rule = 'default';
p1_bl_thresh = 0;
p1_trim_shift = false;

tic;
for ii = 1:n_seg
    img = data_segs{ii};
    x = x_segs{ii};
    y = sfreqs;
    xinds = x_inds{ii};
    if all(img(:) == 0) || all(isnan(img(:))) || length(x) <= 1
        continue
    end

    img_LR = img(1:downsample_spect(2):end, 1:downsample_spect(1):end);
    Ldata = runWatershed(img_LR, p1_conn_wshed, p1_bl_thresh, 0, '', 0);
    [regions, region_lbls, borders, adj_list] = Ldata2graph(Ldata, [], 0);
    seg_counts.n_regions_postwshed(ii) = numel(regions);

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

    if ~isempty(adj_list)
        [regions, borders] = mergeWshedSegment(img_LR, regions, region_lbls, ...
            borders, adj_list, merge_thresh, inf, p1_merge_rule, 0, '', 0);
    end
    seg_counts.n_regions_postmerge(ii) = numel(regions);

    if ii == rep_seg
        rep_dump.regions_postmerge_LR = regions;
        rep_dump.borders_postmerge_LR = borders;
    end

    % Upscale to HR
    Ldata_HR = zeros(size(img_LR));
    for kk = 1:numel(regions)
        if ~isempty(regions{kk}); Ldata_HR(regions{kk}) = kk; end
    end
    LdataHR = imresize(Ldata_HR, size(img), 'nearest');
    regions_HR = cell(1, numel(regions));
    for kk = 1:numel(regions)
        regions_HR{kk} = find(LdataHR == kk);
    end
    regions = regions_HR;

    dt_img = x(2) - x(1); df_img = y(2) - y(1);
    [f_inds, t_inds] = cellfun(@(r) ind2sub(size(img), r), regions, 'UniformOutput', false);
    good = cellfun(@(tt) ~isempty(tt) && (max(tt)-min(tt))*dt_img > dur_min, t_inds) & ...
           cellfun(@(ff) ~isempty(ff) && (max(ff)-min(ff))*df_img > bw_min, f_inds);
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
            p1_trim_shift, p1_conn_trim, 0, '', 0, true);
        [f_inds, t_inds] = cellfun(@(r) ind2sub(size(img), r), trim_regions, 'UniformOutput', false);
        good = cellfun(@(tt) ~isempty(tt) && (max(tt)-min(tt))*dt_img > dur_min, t_inds) & ...
               cellfun(@(ff) ~isempty(ff) && (max(ff)-min(ff))*df_img > bw_min, f_inds);
        trim_regions = trim_regions(good);
        trim_borders = trim_borders(good);
        seg_counts.n_regions_posttrim(ii) = numel(trim_regions);
        % Extra filter (matches MATLAB's triple filter)
        [f_inds, t_inds] = cellfun(@(r) ind2sub(size(img), r), trim_regions, 'UniformOutput', false);
        good = cellfun(@(tt) ~isempty(tt) && (max(tt)-min(tt))*dt_img > dur_min, t_inds) & ...
               cellfun(@(ff) ~isempty(ff) && (max(ff)-min(ff))*df_img > bw_min, f_inds);
        trim_regions = trim_regions(good);
        trim_borders = trim_borders(good);
        seg_counts.n_regions_postfilter(ii) = numel(trim_regions);

        if ii == rep_seg
            rep_dump.regions_posttrim_HR = trim_regions;
            rep_dump.borders_posttrim_HR = trim_borders;
        end

        % Convert per-segment HR regions back to indices in FULL spect1
        % (column offset = xinds(1)-1) so pydynamo can use them directly.
        col_offset = xinds(1) - 1;
        [H_seg, ~] = size(img);
        full_H = size(spect1, 1);  % = H_seg (same freq axis)
        for kk = 1:numel(trim_regions)
            r = trim_regions{kk};
            if isempty(r); continue; end
            [frow, tcol] = ind2sub(size(img), r);
            full_inds = sub2ind(size(spect1), frow, tcol + col_offset);
            all_regions{end+1} = full_inds(:);
            b = trim_borders{kk};
            if ~isempty(b)
                [fr_b, tc_b] = ind2sub(size(img), b);
                full_inds_b = sub2ind(size(spect1), fr_b, tc_b + col_offset);
                all_borders{end+1} = full_inds_b(:);
            else
                all_borders{end+1} = zeros(0, 1);
            end
            all_segment_nums(end+1) = ii;
        end
    end
end
fprintf('Pass-1 per-segment walk took %.1f s\n', toc);

% Also run the full runSegmentedData to get the stats_table (for direct
% peak-by-peak comparison) — faster to just re-run than reconstruct.
compute_features = {'PeakFrequency','PeakTime','Duration','Bandwidth','Height','BoundingBox','Volume'};
fprintf('Re-running runSegmentedData to get pass-1 stats_table...\n');
tic;
[stats_table_p1, regions_p1, borders_p1] = runSegmentedData( ...
    spect1_norm, stimes1, sfreqs, ones(size(baseline1)), seg_time, downsample_spect, ...
    compute_features, dur_min, bw_min, merge_thresh, inf, trim_vol, ...
    0, false, true, true);
fprintf('runSegmentedData took %.1f s, %d pass-1 peaks\n', toc, height(stats_table_p1));

% Save stats_table as CSV (drop cell cols)
keep_cols = false(1, width(stats_table_p1));
for ci = 1:width(stats_table_p1)
    v = stats_table_p1{:, ci};
    keep_cols(ci) = ~iscell(v) && (isnumeric(v) || islogical(v));
end
stats_small = stats_table_p1(:, keep_cols);
writetable(stats_small, fullfile(out_dir, 'pass1_stats_segment.csv'));
fprintf('Wrote %s (%d rows, %d cols)\n', ...
    fullfile(out_dir, 'pass1_stats_segment.csv'), ...
    height(stats_small), width(stats_small));

% Save merge diagnostics
fprintf('Saving pass1_diagnostics_segment.mat...\n');
save(fullfile(out_dir, 'pass1_diagnostics_segment.mat'), ...
    'seg_counts', 'rep_dump', 'rep_seg', 'n_seg', ...
    'downsample_spect', 'merge_thresh', 'trim_vol', ...
    'dur_min', 'bw_min', 'p1_conn_wshed', 'p1_conn_trim', ...
    'regions_p1', 'borders_p1', ...
    '-v7.3');
fprintf('Wrote %s\n', fullfile(out_dir, 'pass1_diagnostics_segment.mat'));

fprintf('\nPer-segment pass-1 region counts:\n');
for fld = fieldnames(seg_counts)'
    v = seg_counts.(fld{1});
    fprintf('  %-22s sum=%6d  median=%5.0f  max=%5d\n', fld{1}, sum(v), median(v), max(v));
end

fprintf('\nALL DONE\n');
