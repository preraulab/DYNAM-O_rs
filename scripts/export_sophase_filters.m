function export_sophase_filters(src_mat, dst_dir)
%EXPORT_SOPHASE_FILTERS  Export all MATLAB-designed SOphase SOS filters as
%   .npy files so the pydynamo Python loader and the Rust filter_cache can
%   read them bit-identically.
%
%   Usage (defaults resolve to the canonical repo layout):
%       export_sophase_filters()
%       export_sophase_filters('/path/to/SOphase_filters.mat', '/path/to/data_matlab_filters')
%
%   Source file:  DYNAM-O_dev/toolbox/SOpowphase_functions/SOphase_filters.mat
%   Destination:  DYNAM-O_rs/data_matlab_filters/
%
%   Each variable named `filter_<Fs>Hz_<loDot>_<hiDot>` (e.g.
%   `filter_100Hz_0dot3_1dot5`) becomes
%   `sophase_sos_Fs<Fs>_<lo>_<hi>.npy` (N×6 float64, little-endian).
%
%   Self-contained — no npy-matlab dependency.

    if nargin < 1 || isempty(src_mat)
        this_dir = fileparts(mfilename('fullpath'));        % DYNAM-O_rs/scripts
        repo_root = fileparts(this_dir);                    % DYNAM-O_rs
        toolbox_root = fullfile(repo_root, '..', 'DYNAM-O_dev');
        src_mat = fullfile(toolbox_root, 'toolbox', ...
                           'SOpowphase_functions', 'SOphase_filters.mat');
    end
    if nargin < 2 || isempty(dst_dir)
        this_dir = fileparts(mfilename('fullpath'));
        dst_dir = fullfile(fileparts(this_dir), 'data_matlab_filters');
    end

    assert(exist(src_mat, 'file') == 2, ...
        'Source .mat not found: %s', src_mat);
    if ~exist(dst_dir, 'dir')
        mkdir(dst_dir);
    end
    fprintf('Source: %s\n', src_mat);
    fprintf('Target: %s\n\n', dst_dir);

    info = whos('-file', src_mat);
    fprintf('Found %d variables in source .mat\n', numel(info));

    n_written = 0;
    n_skipped = 0;
    for i = 1:numel(info)
        name = info(i).name;
        parsed = parse_filter_name(name);
        if isempty(parsed)
            n_skipped = n_skipped + 1;
            continue;
        end
        S = load(src_mat, name);
        d = S.(name);
        % digitalFilter object → SOS matrix
        try
            sos = d.Coefficients;   % N×6 double
        catch
            warning('Skipping %s: cannot access .Coefficients', name);
            n_skipped = n_skipped + 1;
            continue;
        end
        if ~isa(sos, 'double') || size(sos, 2) ~= 6
            warning('Skipping %s: unexpected SOS shape %s', name, mat2str(size(sos)));
            n_skipped = n_skipped + 1;
            continue;
        end

        out_name = sprintf('sophase_sos_Fs%d_%s_%s.npy', ...
                           parsed.fs, parsed.lo_str, parsed.hi_str);
        out_path = fullfile(dst_dir, out_name);
        write_npy_float64(out_path, sos);
        fprintf('  %s -> %s  (%d sections)\n', name, out_name, size(sos, 1));
        n_written = n_written + 1;
    end

    fprintf('\nWrote %d .npy files, skipped %d.\n', n_written, n_skipped);
end


function p = parse_filter_name(name)
%PARSE_FILTER_NAME  Expects `filter_<Fs>Hz_<loDot>_<hiDot>`; returns struct
%   with fs (integer), lo_str (e.g. '0.3'), hi_str (e.g. '1.5'), or [].
    p = [];
    tokens = regexp(name, '^filter_(\d+)Hz_(\d+dot\d+|\d+)_(\d+dot\d+|\d+)$', 'tokens', 'once');
    if isempty(tokens), return; end
    p.fs = str2double(tokens{1});
    p.lo_str = strrep(tokens{2}, 'dot', '.');
    p.hi_str = strrep(tokens{3}, 'dot', '.');
end


function write_npy_float64(path, A)
%WRITE_NPY_FLOAT64  Minimal .npy writer for 2-D double arrays, little-endian.
%   NPY format v1.0 spec: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
    if ~isa(A, 'double')
        A = double(A);
    end
    shape = size(A);
    % C-order header dict (numpy default read as row-major).
    shape_str = sprintf('(%d, %d)', shape(1), shape(2));
    dict_str = sprintf("{'descr': '<f8', 'fortran_order': False, 'shape': %s, }", shape_str);
    % Pad header so that (10 + len(dict_str) + 1) is multiple of 64 (nicer
    % than the spec's 16-byte minimum and still compliant). Trailing \n
    % required inside the padded region.
    preamble_len = 10;   % \x93NUMPY + version bytes + 2-byte header length
    total_target = 64 * ceil((preamble_len + strlength(dict_str) + 1) / 64);
    pad_len = total_target - preamble_len - strlength(dict_str) - 1;
    dict_padded = [char(dict_str), repmat(' ', 1, pad_len), newline];

    fid = fopen(path, 'wb');
    if fid < 0
        error('Cannot open for write: %s', path);
    end
    cleaner = onCleanup(@() fclose(fid));
    % magic + version (1.0)
    fwrite(fid, [uint8(147), uint8('NUMPY'), uint8(1), uint8(0)], 'uint8');
    % header length (little-endian uint16)
    fwrite(fid, numel(dict_padded), 'uint16');
    % header dict
    fwrite(fid, dict_padded, 'uint8');
    % data, C-order (MATLAB is column-major — transpose before write)
    fwrite(fid, A', 'double');   % '
end
