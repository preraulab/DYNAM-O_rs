function setup_dynamo_py(python_exe)
%SETUP_DYNAMO_PY  One-time MATLAB setup to point pyenv at a Python env
%   that has the DYNAM-O_py (pydynamo + dynamo_rs) package installed.
%
%   Usage:
%       setup_dynamo_py('/path/to/DYNAM-O_rs/.venv/bin/python')
%       setup_dynamo_py()          % uses ~/.venv/bin/python heuristic
%
%   Verifies the install and prints the dynamo_rs version.
%
%   After running this once per MATLAB session, `runDYNAMO_rs` is ready
%   to use:
%       [st, ~, ~, ~, ~, ~, ~, SOPHs, timings] = runDYNAMO_rs(...);

    if nargin < 1 || isempty(python_exe)
        % Best-effort auto-detect: sibling .venv of the DYNAM-O_rs source.
        this_dir = fileparts(mfilename('fullpath'));
        candidates = {
            fullfile(this_dir, '..', '.venv', 'bin', 'python');
        };
        for i = 1:numel(candidates)
            if exist(candidates{i}, 'file') == 2
                python_exe = candidates{i};
                break;
            end
        end
        if nargin < 1 || isempty(python_exe)
            error('setup_dynamo_py:NotFound', ...
                'No python_exe provided and no .venv found at expected paths.');
        end
    end

    try
        pe = pyenv('Version', python_exe);
    catch ME
        error('setup_dynamo_py:PyenvFailed', ...
            'pyenv() failed: %s', ME.message);
    end
    fprintf('pyenv: Python %s at %s\n', char(pe.Version), char(pe.Executable));

    % Sanity-check imports
    try
        py.importlib.import_module('pydynamo');
        py.importlib.import_module('dynamo_rs');
    catch ME
        error('setup_dynamo_py:ImportFailed', ...
            ['Required Python modules not found.\n' ...
             '  %s\n\n' ...
             'Install via:\n' ...
             '  cd /path/to/DYNAM-O_rs\n' ...
             '  python -m venv .venv && source .venv/bin/activate\n' ...
             '  pip install maturin\n' ...
             '  maturin develop --release -m rust/Cargo.toml\n' ...
             '  pip install -e .\n'], ME.message);
    end

    % Print the Rust feature set we have available
    attrs = cell(py.list(py.dir(py.importlib.import_module('dynamo_rs'))));
    rust_fns = {};
    for i = 1:numel(attrs)
        a = char(attrs{i});
        if ~startsWith(a, '_'), rust_fns{end+1} = a; end %#ok<AGROW>
    end
    fprintf('dynamo_rs exposes %d functions: %s\n', ...
        numel(rust_fns), strjoin(rust_fns, ', '));
    fprintf('Ready. Call: runDYNAMO_rs(data, Fs, stage_times, stage_vals, time_range, varargin)\n');
end
