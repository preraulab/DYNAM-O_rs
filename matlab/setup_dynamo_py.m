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
        candidates = { ...
            fullfile(repo_dir, '.venv', 'bin', 'python'); ...
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

    % If pyenv is already loaded with a different Python, we need to switch
    % via OutOfProcess mode (in-process Python can't be rebound).
    cur = pyenv;
    if cur.Status == "Loaded" && cur.Executable ~= string(python_exe)
        warning('setup_dynamo_py:Reload', ...
            ['pyenv is already loaded with a different Python:\n' ...
             '  current:   %s\n  requested: %s\n' ...
             'Restart MATLAB to switch interpreters.'], ...
            char(cur.Executable), python_exe);
    else
        try
            pyenv('Version', python_exe, 'ExecutionMode', 'OutOfProcess');
        catch ME
            error('setup_dynamo_py:PyenvFailed', ...
                'pyenv() failed: %s', ME.message);
        end
    end
    pe = pyenv;
    fprintf('pyenv: Python %s at %s\n', char(pe.Version), char(pe.Executable));

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
