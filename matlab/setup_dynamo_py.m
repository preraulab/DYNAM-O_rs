function setup_dynamo_py(python_exe)
%SETUP_DYNAMO_PY  One-time MATLAB setup to point pyenv at a Python env
%   that has the DYNAM-O_py (pydynamo + dynamo_rs) package installed.
%
%   Usage:
%       setup_dynamo_py('/path/to/DYNAM-O_rs/.venv/bin/python')
%       setup_dynamo_py()          % auto-detect (sibling .venv of this file)
%
%   Verifies the install and prints the list of Rust kernels available.
%
%   After running this once per MATLAB session (or permanently via
%   `savepath` + `pyenv`), `runDYNAMO_rs` and `extract_tfpeaks_rs` are
%   ready to use.

    if nargin < 1 || isempty(python_exe)
        % Resolve the sibling .venv relative to this .m file (robust to
        % whatever the caller's pwd is, and to relative `..` that some
        % exist() variants don't auto-canonicalize).
        this_file = mfilename('fullpath');
        this_dir  = fileparts(this_file);        % .../DYNAM-O_rs/matlab
        repo_dir  = fileparts(this_dir);         % .../DYNAM-O_rs
        % Prefer .venv-matlab (built from system Python which has
        % libpython.dylib, required for MATLAB's pyenv) over a generic
        % .venv that may have been built from miniconda's static-only
        % libpython.
        candidates = { ...
            fullfile(repo_dir, '.venv-matlab', 'bin', 'python'); ...
            fullfile(repo_dir, '.venv', 'bin', 'python'); ...
            fullfile(getenv('HOME'), 'code', 'toolboxes', 'DYNAM-O_rs', '.venv-matlab', 'bin', 'python'); ...
            fullfile(getenv('HOME'), 'code', 'toolboxes', 'DYNAM-O_rs', '.venv', 'bin', 'python'); ...
        };
        python_exe = '';
        for i = 1:numel(candidates)
            % Canonicalize any `..` components so exist() is happy.
            cand = char(java.io.File(candidates{i}).getCanonicalPath());
            if exist(cand, 'file') == 2
                python_exe = cand;
                break;
            end
        end
        if isempty(python_exe)
            tried = strjoin(candidates, sprintf('\n  '));
            error('setup_dynamo_py:NotFound', ...
                ['No python_exe provided and no .venv found. Tried:\n  %s\n\n' ...
                 'Usage:  setup_dynamo_py(''/full/path/to/.venv/bin/python'')'], tried);
        end
    end

    % If pyenv is currently loaded — possibly with the wrong Python — we
    % must terminate it first (OutOfProcess Python can be terminated and
    % switched at runtime; InProcess can't, requires MATLAB restart).
    cur = pyenv;
    if cur.Status == "Loaded" && string(cur.Executable) ~= string(python_exe)
        if cur.ExecutionMode == "InProcess"
            error('setup_dynamo_py:NeedRestart', ...
                ['pyenv is locked InProcess to a different Python:\n' ...
                 '  current:   %s\n  requested: %s\n\n' ...
                 'Restart MATLAB, then run setup_dynamo_py(''%s'').'], ...
                char(cur.Executable), python_exe, python_exe);
        end
        % Terminate the current OutOfProcess Python so we can switch.
        try
            terminate(pyenv);
        catch
            % older MATLAB: terminate(pyenv) may not exist; fall through.
        end
    end
    try
        pyenv('Version', python_exe, 'ExecutionMode', 'OutOfProcess');
    catch ME
        error('setup_dynamo_py:PyenvFailed', ...
            'pyenv(''Version'', ''%s'') failed: %s', python_exe, ME.message);
    end
    pe = pyenv;
    if string(pe.Executable) ~= string(python_exe)
        error('setup_dynamo_py:SwitchFailed', ...
            ['pyenv did NOT switch interpreters.\n' ...
             '  requested: %s\n  active:    %s\n\n' ...
             'This usually means MATLAB has Python loaded InProcess.\n' ...
             'Restart MATLAB and try again.'], python_exe, char(pe.Executable));
    end
    fprintf('pyenv: Python %s at %s (%s)\n', char(pe.Version), ...
        char(pe.Executable), char(pe.ExecutionMode));

    % Force a Python op to verify the interpreter actually launches.
    try
        py.importlib.import_module('sys');
    catch ME
        error('setup_dynamo_py:LaunchFailed', ...
            'Could not start Python: %s', ME.message);
    end

    % Sanity-check imports
    try
        py.importlib.import_module('pydynamo');
        py.importlib.import_module('dynamo_rs');
    catch ME
        error('setup_dynamo_py:ImportFailed', ...
            ['Required Python modules not found in %s.\n' ...
             'Install via:\n' ...
             '  cd %s\n' ...
             '  python -m venv .venv && source .venv/bin/activate\n' ...
             '  pip install maturin\n' ...
             '  maturin develop --release -m rust/Cargo.toml\n' ...
             '  pip install -e .\n\n' ...
             'Original error: %s'], ...
            char(pe.Executable), fileparts(fileparts(python_exe)), ME.message);
    end

    % Print the Rust feature set we have available
    drs = py.importlib.import_module('dynamo_rs');
    attrs = cell(py.list(py.dir(drs)));
    rust_fns = {};
    for i = 1:numel(attrs)
        a = char(attrs{i});
        if ~startsWith(a, '_'); rust_fns{end+1} = a; end %#ok<AGROW>
    end
    fprintf('dynamo_rs exposes %d functions: %s\n', ...
        numel(rust_fns), strjoin(rust_fns, ', '));
    fprintf(['Ready. Entry points:\n' ...
             '  runDYNAMO_rs        — full pipeline (drop-in for runDYNAMO)\n' ...
             '  extract_tfpeaks_rs  — just the extraction step (drop-in for runSegmentedData)\n']);
end
