function gen_matlab_pipeline_fixtures(toolbox_dir)
%GEN_MATLAB_PIPELINE_FIXTURES  Write the MATLAB reference outputs for the kernel conformance test
%
%   Usage:
%       gen_matlab_pipeline_fixtures(toolbox_dir)
%
%   Inputs:
%       toolbox_dir : char - path to the DYNAM-O MATLAB toolbox checkout -- required
%
%   Outputs:
%       none (writes rust/tests/fixtures/matlab_pipeline/*.f64 and *.u16)
%
%   Notes:
%       Runs the pure-MATLAB pipeline (backend = 'matlab') on a ten-minute
%       synthetic recording and saves the input and the MATLAB output of
%       each stage. `rust/tests/matlab_pipeline.rs` feeds the kernel the
%       same inputs and compares stage by stage, so an error in one stage
%       cannot hide behind or leak into another.
%
%       Every `.f64` file is a flat list of little-endian doubles and
%       every `.u16` file a flat list of little-endian 16-bit integers.
%       Matrices are written row by row.
%
%       Generate from a MATLAB tree that includes the BoundingBox pixel
%       extent fix (preraulab/DYNAM-O #11).
%
%       Regenerate only when the MATLAB reference changes on purpose. A
%       difference in the regenerated files is a decision to review.
%
%   See also: detect_artifacts, computeSOphase, computeSOpower,
%   computeTFPeaks, extractTFPeaks, refinePeakFrequency, computePeakStage
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿

addpath(genpath(toolbox_dir));
this_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(this_dir, '..', 'rust', 'tests', 'fixtures', 'matlab_pipeline');
if ~isfolder(out_dir)
    mkdir(out_dir);
end

%% Synthetic recording
Fs = 100;
duration_s = 600;
[data, stage_times, stage_vals] = make_recording(Fs, duration_s);
t = (0:numel(data) - 1) / Fs;
write_f64(out_dir, 'fs', Fs);
write_f64(out_dir, 'signal', data);
write_f64(out_dir, 'stage_times', stage_times);
write_f64(out_dir, 'stage_vals', stage_vals);

%% Stage 1: artifact mask (detect_artifacts.m with its defaults)
artifacts = logical(detect_artifacts(data, Fs));
write_f64(out_dir, 'artifacts', double(artifacts(:)));
fprintf('artifacts: %d of %d samples\n', sum(artifacts), numel(artifacts));

%% Stage 2: slow-oscillation phase (computeSOphase.m)
so_band = [0.3 1.5];
[so_phase, so_phase_times, ~, so_filtered] = computeSOphase(data, Fs, [], [], [], logical([]), so_band);
write_f64(out_dir, 'so_band', so_band);
write_f64(out_dir, 'so_phase', so_phase(:));
% The phase is on the EEG sample grid, so its time axis is not saved.
assert(isequal(so_phase_times(:), (0:numel(data) - 1)' / Fs), 'SO-phase is not on the EEG sample grid');
write_f64(out_dir, 'so_filtered', so_filtered(:));

%% Stage 3: TF-peak extraction on single segments (extractTFPeaks.m)
% computeTFPeaks supplies the spectrogram and the baseline exactly as the
% pipeline builds them. Each saved segment is the baseline-divided
% spectrogram that runSegmentedData hands to extractTFPeaks.
[~, spect, stimes, sfreqs, ~, ~, ~, ~, baseline] = computeTFPeaks(data, Fs, stage_times, stage_vals, ...
    'backend', 'matlab', 'verbose', false);
% MATLAB carries the spectrogram in single precision. The segment is
% converted to double before extraction so the comparison is of the
% algorithm on identical numbers. Left in single, MATLAB's sums differ
% from the kernel's by about 1e-7 relative, which is rounding.
divided = double(spect) ./ double(baseline(:));
trim_shift = min(divided, [], 'all');

merge_thresh = 8;
downsample_spect = [2 2];
dur_min = 1;
bw_min = 1;
trim_vol = 0.8;
seg_time = 30;
features = {'PeakTime', 'PeakFrequency', 'Height', 'Volume', 'Duration', 'Bandwidth', 'BoundingBox'};

dt = stimes(2) - stimes(1);
columns_per_segment = floor(seg_time / dt);
n_segments_total = floor(numel(stimes) / columns_per_segment);
% Two segments, one early and one late, to keep the fixture small.
chosen = round(linspace(2, n_segments_total - 1, 2));
all_peak_times = [];
all_peaks = zeros(0, 10);
for k = 1:numel(chosen)
    first_column = (chosen(k) - 1) * columns_per_segment + 1;
    columns = first_column:(first_column + columns_per_segment - 1);
    img = divided(:, columns);
    seg_times = stimes(columns);
    stats = extractTFPeaks(img, seg_times(:)', sfreqs(:), features, k, 8, merge_thresh, inf, ...
        downsample_spect, dur_min, bw_min, trim_vol, trim_shift, 8, [], [], 0, '', 0);
    if isempty(stats)
        peaks = zeros(0, 10);
    else
        peaks = [stats.PeakTime, stats.PeakFrequency, stats.Height, stats.Volume, stats.Duration, ...
            stats.Bandwidth, stats.BoundingBox];
    end
    prefix = sprintf('seg_%d', k - 1);
    write_f64(out_dir, [prefix '_dims'], size(img));
    write_f64(out_dir, [prefix '_values'], img');     % row by row
    write_f64(out_dir, [prefix '_stimes'], seg_times);
    write_f64(out_dir, [prefix '_peaks'], peaks');    % one peak after another
    all_peak_times = [all_peak_times; peaks(:, 1)]; %#ok<AGROW>
    all_peaks = [all_peaks; peaks]; %#ok<AGROW>
    fprintf('segment %d: %d x %d, %d peaks\n', k, size(img, 1), size(img, 2), size(peaks, 1));
end
write_f64(out_dir, 'sfreqs', sfreqs);
write_f64(out_dir, 'extract_params', [merge_thresh, downsample_spect, dur_min, bw_min, trim_vol, trim_shift]);
write_f64(out_dir, 'n_segments', numel(chosen));

%% Stage 4: peak frequency refinement (refinePeakFrequency.m)
% Events are the extracted peaks that sit at least half a refinement
% window (2 s) inside the recording. Columns 7 to 10 are the BoundingBox.
refine_freq_range = [sfreqs(1), sfreqs(end)];
inside = all_peaks(:, 1) >= 2 & all_peaks(:, 1) <= t(end) - 2;
event_time = all_peaks(inside, 1);
event_bbox = all_peaks(inside, 7:10);
event_table = table(event_time, nan(size(event_time)), event_bbox, ...
    'VariableNames', {'PeakTime', 'PeakFrequency', 'BoundingBox'});
refined_table = refinePeakFrequency(data, Fs, event_table, refine_freq_range, [], false, 'spline_interp', true, 'matlab');
write_f64(out_dir, 'refine_freq_range', refine_freq_range);
write_f64(out_dir, 'refine_event_time', event_time);
write_f64(out_dir, 'refine_event_bbox', event_bbox');   % one event after another
write_f64(out_dir, 'refine_frequency', refined_table.PeakFrequency);
fprintf('refinement: %d events, %d removed\n', numel(event_time), sum(~isfinite(refined_table.PeakFrequency)));

%% Stage 5: the pass-two mask
% The first pass finds regions on the short-window spectrogram. The second
% pass keeps only the pixels inside those regions and off their traced
% contours. This stage saves the first-pass regions as a label image and
% the keep/zero decision for every second-pass pixel, over the first two
% segments of the recording.
[~, spect_first, stimes_first, sfreqs_first, ~, ~, ~, ~, baseline_first] = computeTFPeaks(data, Fs, stage_times, stage_vals, ...
    'backend', 'matlab', 'verbose', false, 'double_watershed', false);
assert(isequal(sfreqs_first(:), sfreqs(:)), 'the two passes do not share a frequency axis');
dt_first = stimes_first(2) - stimes_first(1);
first_columns = 1:(2 * floor(seg_time / dt_first));
spect_first = double(spect_first(:, first_columns));
stimes_first = stimes_first(first_columns);

% First-pass floors for a 1 s window with a time half-bandwidth of 2:
% half the window, and half the 4 Hz spectral resolution.
first_dur_min = 0.5;
first_bw_min = 2;
first_features = {'PeakTime', 'PeakFrequency', 'Duration', 'Bandwidth', 'Height'};
[~, regions, borders] = runSegmentedData(spect_first, stimes_first, sfreqs, double(baseline_first), seg_time, downsample_spect, ...
    first_features, first_dur_min, first_bw_min, merge_thresh, inf, trim_vol, 0, false, false, 'backend', 'matlab');

label_image = zeros(size(spect_first));
for k = 1:numel(regions)
    label_image(regions{k}) = k;
end

% Second-pass columns that fall inside the stretch the first pass covered.
second_columns = find(stimes >= stimes_first(1) & stimes <= stimes_first(end));
stimes_second = stimes(second_columns);
n_rows = size(spect_first, 1);
n_second = numel(second_columns);

% The masking rule, as computeTFPeaks applies it: region and border pixel
% indices are shifted from the first-pass grid onto the second-pass grid.
dt_second = stimes_second(2) - stimes_second(1);
index_shift = round((stimes_second(1) - stimes_first(1)) / dt_second) * n_rows;
region_indices = cat(1, regions{:}) - index_shift;
region_indices = region_indices(region_indices >= 1 & region_indices <= n_rows * n_second);
border_indices = cat(1, borders{:}) - index_shift;
border_indices = border_indices(border_indices >= 1 & border_indices <= n_rows * n_second);
kept = false(n_rows, n_second);
kept(region_indices) = true;
kept(border_indices) = false;

write_f64(out_dir, 'mask_dims', [n_rows, numel(first_columns), n_second]);
write_f64(out_dir, 'mask_stimes_first', stimes_first);
write_f64(out_dir, 'mask_stimes_second', stimes_second);
write_u16(out_dir, 'mask_labels', label_image');   % row by row
write_u16(out_dir, 'mask_kept', double(kept)');    % row by row
fprintf('mask: %d regions, %d of %d second-pass pixels kept\n', numel(regions), sum(kept(:)), numel(kept));

%% Stage 6: slow-oscillation power (computeSOpower.m)
% Every option is given explicitly so the fixture does not move when a
% toolbox default does. The slow-oscillation spectrogram is recomputed
% here with the same call computeSOpower makes, because computeSOpower
% does not return it.
so_power_band = [0.3 1.5];
so_power_tapers = [15 29];
so_power_window = [30 15];
so_power_outlier = 3;
so_power_norm = 'p2shift1234';
[so_power_full, so_power_full_times] = computeSOpower(data, Fs, stage_times, stage_vals, ...
    'isexcluded', artifacts(:), 'SO_freqrange', so_power_band, 'tapers', so_power_tapers, ...
    'window_params', so_power_window, 'SOpower_outlier_threshold', so_power_outlier, ...
    'norm_method', so_power_norm, 'retain_Fs', true);
assert(numel(so_power_full) == numel(data), 'SO-power is not on the EEG sample grid');
excluded_data = data;
excluded_data(artifacts) = nan;
[so_spect, so_spect_times, so_spect_freqs] = multitaper_spectrogram_dynamo(excluded_data, Fs, so_power_band, ...
    so_power_tapers, so_power_window, [], 'linear', [], false, false);
write_f64(out_dir, 'sopower_spect_dims', size(so_spect));
write_f64(out_dir, 'sopower_spect', double(so_spect)');   % row by row
write_f64(out_dir, 'sopower_spect_times', so_spect_times);
write_f64(out_dir, 'sopower_spect_freqs', so_spect_freqs);
write_f64(out_dir, 'sopower_outlier', so_power_outlier);
write_f64(out_dir, 'sopower_norm', double(so_power_full(:)));
fprintf('SO-power: %d x %d spectrogram, %d NaN of %d output samples\n', size(so_spect, 1), size(so_spect, 2), ...
    sum(isnan(so_power_full)), numel(so_power_full));

%% Stage 7: per-peak stage, SO-power, and SO-phase (computePeak*.m)
[so_power, so_power_times] = computeSOpower(data, Fs);
% Both series go to double for the same reason as the segments above
% (computeSOphase returns single precision).
so_power = double(so_power(:))';
so_power_times = so_power_times(:)';
so_phase_row = double(so_phase(:))';
so_phase_times_row = so_phase_times(:)';

% Queries: the extracted peak times, plus probes placed where the rules
% differ between implementations: on grid samples, exactly halfway
% between samples, and around both ends of each grid.
eeg_step = 1 / Fs;
probes = [t(1:997:end), t(1:1501:end) + eeg_step / 2, ...
    so_power_times(1) + (-2:0.5:2) * (so_power_times(2) - so_power_times(1)), ...
    so_power_times(end) + (-2:0.5:2) * (so_power_times(2) - so_power_times(1)), ...
    t(1) + (0:0.5:2) * eeg_step, t(end) - (0:0.5:2) * eeg_step];
query_times = sort([all_peak_times(:)', probes]);
query_times = query_times(query_times >= t(1) & query_times <= t(end));

stage_table = table(query_times(:), 'VariableNames', {'PeakTime'});
stage_table = computePeakStage(stage_table, stage_times, stage_vals, t, artifacts(:)');
peak_stage = stage_table.PeakStage;

power_step = so_power_times(2) - so_power_times(1);
peak_so_power = interp1([so_power_times(1) - power_step, so_power_times, so_power_times(end) + power_step], ...
    [so_power(1), so_power, so_power(end)], query_times);
phase_step = so_phase_times_row(2) - so_phase_times_row(1);
peak_so_phase = interp1([so_phase_times_row(1) - phase_step, so_phase_times_row, so_phase_times_row(end) + phase_step], ...
    [so_phase_row(1), so_phase_row, so_phase_row(end)], query_times);
peak_so_phase = wrapToPi(peak_so_phase);

write_f64(out_dir, 'so_power', so_power);
write_f64(out_dir, 'so_power_times', so_power_times);
write_f64(out_dir, 'query_times', query_times);
write_f64(out_dir, 'query_stage', peak_stage);
write_f64(out_dir, 'query_so_power', peak_so_power);
write_f64(out_dir, 'query_so_phase', peak_so_phase);
fprintf('queries: %d (%d artifact-stage, %d NaN SO-power)\n', numel(query_times), sum(peak_stage == 6), sum(isnan(peak_so_power)));
end

function [data, stage_times, stage_vals] = make_recording(Fs, duration_s)
% A sleep-like signal with a fixed seed: 1/f background, a slow
% oscillation whose depth changes by stage, sigma bursts, and three
% artifacts (a flat run, a high-frequency burst, a large slow swing).
rng(20261117, 'twister');
n = Fs * duration_s;
t = (0:n - 1)' / Fs;

white = randn(n, 1);
background = cumsum(white);
background = background - movmean(background, 4 * Fs);
background = 6 * background / std(background) + 3 * randn(n, 1);

depth = 0.5 + 0.5 * sin(2 * pi * t / duration_s - pi / 2) .^ 2;
slow_oscillation = 35 * depth .* sin(2 * pi * 0.8 * t + 0.6 * sin(2 * pi * 0.05 * t));

sigma = zeros(n, 1);
burst_starts = 5:9:(duration_s - 5);
for k = 1:numel(burst_starts)
    burst_frequency = 12 + 3 * rand;
    burst_length = 0.8 + 0.8 * rand;
    in_burst = t >= burst_starts(k) & t < burst_starts(k) + burst_length;
    window = hann(sum(in_burst));
    sigma(in_burst) = 12 * window .* sin(2 * pi * burst_frequency * t(in_burst));
end

data = background + slow_oscillation + sigma;

flat_run = t >= 120 & t < 124;
data(flat_run) = data(find(flat_run, 1));
noisy = t >= 300 & t < 302;
data(noisy) = data(noisy) + 150 * randn(sum(noisy), 1);
swing = t >= 450 & t < 453;
data(swing) = data(swing) + 400 * sin(2 * pi * 0.3 * (t(swing) - 450));

% Thirty-second epochs. Stage codes: 5 wake, 4 REM, 3 N1, 2 N2, 1 N3.
stage_times = 0:30:(duration_s - 30);
pattern = [5 3 2 2 1 1 1 2 2 4 4 2 1 1 2 3 5 2 2 4];
stage_vals = pattern(1:numel(stage_times));
end

function write_u16(out_dir, name, values)
% Small whole numbers (labels, yes/no masks) as little-endian 16-bit
% integers, a quarter of the size of doubles.
assert(all(values(:) >= 0 & values(:) <= 65535 & values(:) == round(values(:))), '%s does not fit 16-bit integers', name);
file_id = fopen(fullfile(out_dir, [name '.u16']), 'w', 'ieee-le');
assert(file_id > 0, 'cannot open %s for writing', name);
fwrite(file_id, values(:), 'uint16');
fclose(file_id);
end

function write_f64(out_dir, name, values)
file_id = fopen(fullfile(out_dir, [name '.f64']), 'w', 'ieee-le');
assert(file_id > 0, 'cannot open %s for writing', name);
fwrite(file_id, double(values(:)), 'double');
fclose(file_id);
end
