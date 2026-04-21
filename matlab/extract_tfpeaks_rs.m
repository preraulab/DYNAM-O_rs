function [stats_table, labels] = extract_tfpeaks_rs(spect, stimes, sfreqs, varargin)
%EXTRACT_TFPEAKS_RS  Drop-in faster replacement for runSegmentedData /
%   extractTFPeaks. Calls the Rust-accelerated extraction pipeline
%   (watershed → merge → trim → regionprops) via pydynamo.
%
%   [stats_table, labels] = extract_tfpeaks_rs(spect, stimes, sfreqs, ...)
%
%   Required:
%       spect    [F x T] double  — RAW spectrogram (will be divided by
%                                  baseline if you supply one)
%       stimes   [1 x T] double  — time axis (s)
%       sfreqs   [1 x F] double  — frequency axis (Hz)
%
%   Name-value parameters (defaults match MATLAB extractTFPeaks):
%       'baseline'      [F x 1] double or []     (default [])
%       'seg_time'      scalar                    (default 30)
%       'downsample'    [1 x 2] int               (default [2 2])
%       'merge_thresh'  scalar                    (default 11)
%       'trim_vol'      scalar                    (default 0.8)
%       'dur_min'       scalar                    (default 0.5)
%       'dur_max'       scalar                    (default 5)
%       'bw_min'        scalar                    (default 1)
%       'bw_max'        scalar                    (default 15)
%
%   Returns:
%       stats_table  MATLAB table with columns PeakTime, PeakFrequency,
%                    Duration, Bandwidth, Height, Volume, SegmentNum, and
%                    BoundingBox (Nx4 numeric: [time_tl freq_tl width_s
%                    height_Hz]).
%       labels       (F x T) uint32 — pass-1 label image, suitable for
%                    passing to mask_spectrogram_rs to mask a pass-2
%                    spectrogram.
%
%   Typical speedup vs MATLAB runSegmentedData on full-night data:
%       pass-1: ~3.6x (84s -> 23s)
%       pass-2: ~1.9x (32s -> 17s)
%
%   See also runDYNAMO_rs, mask_spectrogram_rs, refine_peaks_rs, setup_dynamo_py.

    p = inputParser;
    p.addParameter('baseline', []);
    p.addParameter('seg_time', 30);
    p.addParameter('downsample', [2, 2]);
    p.addParameter('merge_thresh', 11);
    p.addParameter('trim_vol', 0.8);
    p.addParameter('dur_min', 0.5);
    p.addParameter('dur_max', 5);
    p.addParameter('bw_min', 1);
    p.addParameter('bw_max', 15);
    p.parse(varargin{:});
    a = p.Results;

    if isempty(a.baseline)
        baseline_py = py.None;
    else
        baseline_py = py.numpy.array(double(a.baseline(:)));
    end

    out = py.pydynamo.matlab_api.extract_tfpeaks_rs( ...
        py.numpy.array(double(spect)), ...
        py.numpy.array(double(stimes(:))), ...
        py.numpy.array(double(sfreqs(:))), ...
        pyargs( ...
            'baseline',     baseline_py, ...
            'seg_time',     double(a.seg_time), ...
            'downsample',   py.tuple({int32(a.downsample(1)), int32(a.downsample(2))}), ...
            'merge_thresh', double(a.merge_thresh), ...
            'trim_vol',     double(a.trim_vol), ...
            'dur_min',      double(a.dur_min), ...
            'dur_max',      double(a.dur_max), ...
            'bw_min',       double(a.bw_min), ...
            'bw_max',       double(a.bw_max)));

    n = double(out{'n_peaks'});
    if n == 0
        stats_table = table();
    else
        pt = double(out{'PeakTime'});       pt = pt(:);
        pf = double(out{'PeakFrequency'});  pf = pf(:);
        du = double(out{'Duration'});       du = du(:);
        bw = double(out{'Bandwidth'});      bw = bw(:);
        ht = double(out{'Height'});         ht = ht(:);
        vo = double(out{'Volume'});         vo = vo(:);
        sn = double(out{'SegmentNum'});     sn = sn(:);
        bb = double(out{'BoundingBox'});    % N x 4, keep as-is
        if size(bb, 2) ~= 4 && numel(bb) == 4 * n
            bb = reshape(bb, n, 4);
        end
        stats_table = table(pt, pf, du, bw, ht, vo, sn, bb, ...
            'VariableNames', {'PeakTime', 'PeakFrequency', 'Duration', ...
                              'Bandwidth', 'Height', 'Volume', ...
                              'SegmentNum', 'BoundingBox'});
    end

    if nargout > 1
        labels = uint32(out{'labels'});
    end
end
