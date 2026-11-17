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
%       none (writes rust/tests/fixtures/matlab_pipeline/*.f64)
%
%   Notes:
%       Runs the pure-MATLAB pipeline (backend = 'matlab') on a ten-minute
%       synthetic recording and saves the input and the MATLAB output of
%       each stage. `rust/tests/matlab_pipeline.rs` feeds the kernel the
%       same inputs and compares stage by stage, so an error in one stage
%       cannot hide behind or leak into another.
%
%       Every file is a flat list of little-endian doubles. Matrices are
%       written row by row.
%
%       Generate from a MATLAB tree that includes the BoundingBox pixel
%       extent fix (preraulab/DYNAM-O #11).
%
%       Regenerate only when the MATLAB reference changes on purpose. A
%       difference in the regenerated files is a decision to review.
%
%   See also: detect_artifacts, computeSOphase, computeSOpower,
%   computeTFPeaks, extractTFPeaks, computePeakStage
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
    fprintf('segment %d: %d x %d, %d peaks\n', k, size(img, 1), size(img, 2), size(peaks, 1));
end
write_f64(out_dir, 'sfreqs', sfreqs);
write_f64(out_dir, 'extract_params', [merge_thresh, downsample_spect, dur_min, bw_min, trim_vol, trim_shift]);
write_f64(out_dir, 'n_segments', numel(chosen));

%% Stage 4: per-peak stage, SO-power, and SO-phase (computePeak*.m)
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

function write_f64(out_dir, name, values)
file_id = fopen(fullfile(out_dir, [name '.f64']), 'w', 'ieee-le');
assert(file_id > 0, 'cannot open %s for writing', name);
fwrite(file_id, double(values(:)), 'double');
fclose(file_id);
end
