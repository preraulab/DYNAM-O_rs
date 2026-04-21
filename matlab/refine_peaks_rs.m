function stats_table_refined = refine_peaks_rs(stats_table, data, Fs, varargin)
%REFINE_PEAKS_RS  Drop-in replacement for refinePeakFrequency, calling
%   the Rust Hann-window + cubic-spline argmax routine.
%
%   stats_table_refined = refine_peaks_rs(stats_table, data, Fs, ...)
%
%   stats_table must contain columns PeakTime, PeakFrequency, BoundingBox
%   (where BoundingBox is Nx4: [time_tl freq_tl width_s height_Hz]).
%
%   Name-value parameters (defaults match MATLAB refinePeakFrequency):
%       't'                 (default 0:1/Fs:(N-1)/Fs)
%       'freq_range'        (default [0 30])
%       'window_size'       (default 4)
%       'dsfreqs'           (default 0.05)
%       'refine_method'     (default 'spline_interp')
%       'remove_edge_peaks' (default true)
%
%   Typical speedup: ~18x vs MATLAB on full-night data.
    p = inputParser;
    p.addParameter('t', []);
    p.addParameter('freq_range', [0, 30]);
    p.addParameter('window_size', 4);
    p.addParameter('dsfreqs', 0.05);
    p.addParameter('refine_method', 'spline_interp');
    p.addParameter('remove_edge_peaks', true);
    p.parse(varargin{:});
    a = p.Results;

    if isempty(a.t)
        t_py = py.None;
    else
        t_py = py.numpy.array(double(a.t(:)));
    end

    out = py.pydynamo.matlab_api.refine_peaks_rs( ...
        py.numpy.array(double(stats_table.PeakTime)), ...
        py.numpy.array(double(stats_table.PeakFrequency)), ...
        py.numpy.array(double(stats_table.BoundingBox)), ...
        py.numpy.array(double(data(:))), ...
        double(Fs), ...
        pyargs( ...
            't',                 t_py, ...
            'freq_range',        py.tuple({double(a.freq_range(1)), double(a.freq_range(2))}), ...
            'window_size',       double(a.window_size), ...
            'dsfreqs',           double(a.dsfreqs), ...
            'refine_method',     a.refine_method, ...
            'remove_edge_peaks', logical(a.remove_edge_peaks)));

    keep = logical(out{'keep_mask'});
    keep = keep(:);
    stats_table_refined = stats_table(keep, :);
    if nnz(keep) > 0
        pf = double(out{'PeakFrequency'});
        stats_table_refined.PeakFrequency = pf(:);
    end
end
