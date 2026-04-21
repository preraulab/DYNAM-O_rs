function [stats_table, spect, stimes, sfreqs, data_time_range, ...
          t_time_range, artifacts, SOPHs, timings] = ...
    runDYNAMO_rs(data, Fs, stage_times, stage_vals, time_range, varargin)
%RUNDYNAMO_RS  Drop-in faster replacement for runDYNAMO using the Rust-
%              accelerated pydynamo pipeline.
%
%   [stats_table, spect, stimes, sfreqs, data_time_range, t_time_range, ...
%       artifacts, SOPHs, timings] = runDYNAMO_rs(data, Fs, stage_times, ...
%       stage_vals, time_range, varargin)
%
%   Argument conventions match MATLAB's runDYNAMO. Name-value pairs in
%   varargin are forwarded to py.pydynamo.run_dynamo (merge_thresh,
%   trim_vol, seg_time, min_time_in_bin, min_peak_at_freq, refinement,
%   double_watershed, ...).
%
%   Requires a Python environment with DYNAM-O_py installed. Configure once:
%       pyenv('Version', '/path/to/venv/bin/python')
%       py.importlib.import_module('pydynamo');   % sanity check
%
%   Typical speedup vs MATLAB runDYNAMO on an 8-hour recording: ~2.4x.
%
%   Example:
%       load example_data.mat
%       [st, ~, ~, ~, ~, ~, ~, SOPHs] = runDYNAMO_rs( ...
%           data, Fs, stage_times, stage_vals, [8420 13446]);
%
%   See also: runDYNAMO, pyenv.

    if nargin < 5 || isempty(time_range)
        time_range_py = py.None;
    else
        time_range_py = py.tuple({double(time_range(1)), double(time_range(2))});
    end

    % --- Validate that pydynamo is importable ---
    try
        py.importlib.import_module('pydynamo');
    catch ME
        error('runDYNAMO_rs:PyImport', ...
            ['Failed to import the ''pydynamo'' Python module.\n' ...
             'Set pyenv to a venv where DYNAM-O_py is installed:\n' ...
             '  pyenv(''Version'', ''/path/to/.venv/bin/python'')\n' ...
             'Original error: %s'], ME.message);
    end

    % --- Force column vectors / row-vector shapes the Python side expects ---
    data        = double(data(:));
    Fs          = double(Fs);
    stage_times = double(stage_times(:));
    stage_vals  = double(stage_vals(:));

    % --- Build the pyargs kwargs ---
    kv = varargin;
    % Always suppress Python-side plotting (we don't want matplotlib GUI),
    % and verbose (we manage that here).
    kv = [kv, {'plot', false, 'verbose', false, 'time_range', time_range_py}];

    % --- Call ---
    tic;
    out = py.pydynamo.run_dynamo( ...
        py.numpy.array(data), Fs, ...
        py.numpy.array(stage_times), ...
        py.numpy.array(stage_vals), ...
        pyargs(kv{:}));
    total_wall = toc;

    % ---------- Convert outputs back to MATLAB types ----------
    % stats_table: pandas DataFrame → MATLAB table. Use the Python-side
    % helper to flatten to a plain dict of numpy arrays (MATLAB brace
    % indexing on py.pandas.DataFrame is unreliable).
    stats_dict = py.pydynamo.matlab_api.stats_table_to_dict(out.stats_table);
    stats_table = dict_to_table(stats_dict);

    % Spectrograms + axes
    spect   = double(out.spect);
    stimes  = double(out.stimes);
    sfreqs  = double(out.sfreqs);

    % Time ranges: pydynamo doesn't store these on the output; derive
    % from the user's input (or the full-recording span if not given).
    if nargin < 5 || isempty(time_range)
        data_time_range = [0, (numel(data) - 1) / Fs];
    else
        data_time_range = double(time_range(:))';
    end
    t_time_range = data_time_range;

    % Artifacts (numpy bool → MATLAB logical)
    artifacts = logical(out.artifacts);

    % SOPHs struct
    SOPHs = struct();
    sp = out.SOPHs;
    fields = {'SOpower_mat','SOphase_mat','SOpower_bins','SOphase_bins', ...
              'freq_bins','SOpower_TIB','SOphase_TIB', ...
              'peak_at_freq_SOpower','peak_at_freq_SOphase', ...
              'SOpower_norm','SOpower_times','SOphase','SOphase_times'};
    for k = 1:numel(fields)
        f = fields{k};
        if py.hasattr(sp, f)
            val = sp.(f);
            SOPHs.(f) = double(val);
        end
    end

    % Timings: Python dict → MATLAB struct
    timings = dict_to_struct(out.timings);
    timings.matlab_wall = total_wall;
end


function T = dict_to_table(d)
%DICT_TO_TABLE  Convert a Python dict {col_name: numpy_array} into a MATLAB
%   table. Keys starting with '_' are treated as metadata and skipped.
    keys = cell(py.list(d.keys()));
    varargs = {};
    var_names = {};
    for i = 1:numel(keys)
        nm = char(keys{i});
        if startsWith(nm, '_')
            continue
        end
        val = py.operator.getitem(d, nm);
        try
            vec = double(py.numpy.asarray(val));
        catch
            vec = string(cellfun(@char, cell(val), 'UniformOutput', false));
        end
        if isnumeric(vec)
            if size(vec, 2) > 1 && size(vec, 1) == 1
                vec = vec(:);
            elseif ndims(vec) == 2 && size(vec, 2) ~= 1 && size(vec, 1) ~= 1
                % Keep multi-column (e.g. BoundingBox Nx4) intact.
            elseif isvector(vec)
                vec = vec(:);
            end
        end
        varargs{end+1} = vec; %#ok<AGROW>
        var_names{end+1} = nm; %#ok<AGROW>
    end
    T = table(varargs{:}, 'VariableNames', var_names);
end


function T = df_to_matlab_table(df) %#ok<DEFNU>  — kept for back-compat
%DF_TO_MATLAB_TABLE  Convert a pandas DataFrame to a MATLAB table. Uses
%   py.getattr for column access (MATLAB's df.(nm) dot-syntax fails on
%   some column names because MATLAB only whitelists known DataFrame
%   attributes).
    cols = cell(py.list(df.columns));
    n = double(df.shape{1});
    varargs = {};
    var_names = {};
    for i = 1:numel(cols)
        nm = char(cols{i});
        % Python-side column access: df[nm] via __getitem__
        col = df{nm};
        vals_py = col.values;

        if strcmp(nm, 'BoundingBox')
            % BoundingBox is a column of 4-tuples. Convert to Nx4 numeric.
            if n == 0
                bb = zeros(0, 4);
            else
                bb = double(py.numpy.asarray(py.numpy.stack(vals_py)));
            end
            varargs{end+1} = bb; %#ok<AGROW>
            var_names{end+1} = nm; %#ok<AGROW>
        else
            % All other columns are scalar → 1-D numeric. numpy array
            % → MATLAB double.
            try
                vec = double(py.numpy.asarray(vals_py));
            catch
                % Fallback for object-dtype columns: cast to string array.
                vec = string(cellfun(@char, cell(col.astype('str').tolist()), ...
                    'UniformOutput', false));
            end
            if isnumeric(vec)
                vec = vec(:);
            end
            varargs{end+1} = vec; %#ok<AGROW>
            var_names{end+1} = nm; %#ok<AGROW>
        end
    end
    T = table(varargs{:}, 'VariableNames', var_names);
end


function s = dict_to_struct(d)
%DICT_TO_STRUCT  Convert a Python dict of scalars to a MATLAB struct.
    s = struct();
    keys = cell(py.list(d.keys()));
    for i = 1:numel(keys)
        k = char(keys{i});
        v = d{keys{i}};
        try
            s.(matlab.lang.makeValidName(k)) = double(v);
        catch
            s.(matlab.lang.makeValidName(k)) = v;
        end
    end
end
