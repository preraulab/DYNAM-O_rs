function out = read_edf_rs(path, label)
%READ_EDF_RS  Thin MATLAB shim around the Rust port of read_EDF.
%
%   out = read_edf_rs(path)            % all signals
%   out = read_edf_rs(path, label)     % single channel ('A-B' rereference ok)
%
%   Calls py.pydynamo.io_edf.read_edf. Returns a struct whose fields mirror
%   the Python dict (header as struct, labels as cellstr, data as double
%   row vector when `label` is given, signals as a struct array otherwise).
%
%   Requires `setup_dynamo_py` to have been run so pydynamo is importable.

if nargin < 2
    py_res = py.pydynamo.io_edf.read_edf(path);
else
    py_res = py.pydynamo.io_edf.read_edf(path, label);
end

out = struct();
out.header = struct(py_res{'header'});

lbl_py = py_res{'labels'};
out.labels = cellfun(@char, cell(lbl_py), 'UniformOutput', false);

fs_py = py_res{'sampling_frequencies'};
out.sampling_frequencies = double(py.array.array('d', fs_py));

if nargin >= 2
    out.data  = double(py.numpy.asarray(py_res{'data'}));
    out.fs    = double(py_res{'fs'});
    out.label = char(py_res{'label'});
    out.signal_header = struct(py_res{'signal_header'});
else
    sigs = cell(py_res{'signals'});
    for k = 1:numel(sigs)
        s = struct(sigs{k});
        s.data = double(py.numpy.asarray(s.data));
        out.signals(k) = s;
    end
end
end
