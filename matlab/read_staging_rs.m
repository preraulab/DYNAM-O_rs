function [times, vals] = read_staging_rs(path, varargin)
%READ_STAGING_RS  Thin MATLAB shim around the Rust port of read_staging.
%
%   [times, vals] = read_staging_rs(path, ...)
%
%   Name-value pairs (defaults match read_staging.m / Rust defaults):
%       'time_col'     (default 1)
%       'stage_col'    (default 2)
%       'header_lines' (default 0)
%       'delimiter'    (default ',')
%       'epoch_dur'    (default 30)
%       'start_time'   (default [])   'HH:MM:SS [AM|PM]' string
%
%   Requires `setup_dynamo_py` so pydynamo is importable.

p = inputParser;
addRequired(p,  'path');
addParameter(p, 'time_col', 1);
addParameter(p, 'stage_col', 2);
addParameter(p, 'header_lines', 0);
addParameter(p, 'delimiter', ',');
addParameter(p, 'epoch_dur', 30);
addParameter(p, 'start_time', []);
parse(p, path, varargin{:});
r = p.Results;

if isempty(r.start_time)
    start_time = py.None;
else
    start_time = r.start_time;
end

tup = py.pydynamo.io_edf.read_staging( ...
    r.path, ...
    int64(r.time_col), ...
    int64(r.stage_col), ...
    int64(r.header_lines), ...
    r.delimiter, ...
    double(r.epoch_dur), ...
    start_time);

times = double(py.numpy.asarray(tup{1}));
vals  = double(py.numpy.asarray(tup{2}));
times = times(:);
vals  = vals(:);
end
