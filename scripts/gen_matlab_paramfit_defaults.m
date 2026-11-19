function gen_matlab_paramfit_defaults()
%GEN_MATLAB_PARAMFIT_DEFAULTS  Regenerate the MATLAB fixture for the default fit bounds
%
%   Usage:
%       gen_matlab_paramfit_defaults()
%
%   Inputs:
%       none (the DYNAM-O toolbox must be on the path)
%
%   Outputs:
%       none (writes rust/tests/fixtures/parity_paramfit_default_bounds.csv)
%
%   Notes:
%       Writes the default lower and upper bounds on the three shape
%       parameters of a fitted mode (fstd0, pstd0, theta0), as
%       param_basis_opts returns them for the 'power' and 'phase' fits.
%       `rust/tests/paramfit_defaults_matlab_parity.rs` checks that the
%       kernel's paramfit::defaults holds exactly these values.
%
%       Each value is printed with 17 significant digits, which is enough
%       to read the same double back.
%
%       Regenerate only when the MATLAB defaults change on purpose. A
%       difference in the regenerated file is a decision to review.
%
%   See also: param_basis_opts, gen_matlab_parity
%
%   ∿∿∿  Prerau Laboratory · sleepEEG.org  ∿∿∿

this_dir = fileparts(mfilename('fullpath'));
out_dir  = fullfile(this_dir, '..', 'rust', 'tests', 'fixtures');
assert(isfolder(out_dir), 'fixture dir not found: %s', out_dir);

% Positions of the shape parameters in the LB_default and UB_default rows,
% which are ordered [amp0, fmean0, fstd0, pmean0, pstd0, theta0].
FSTD_COLUMN  = 3;
PSTD_COLUMN  = 5;
THETA_COLUMN = 6;

out = fopen(fullfile(out_dir, 'parity_paramfit_default_bounds.csv'), 'w');
fprintf(out, 'fit,fstd_lower,fstd_upper,pstd_lower,pstd_upper,theta_lower,theta_upper\n');
fit_types = {'power', 'phase'};
for type_idx = 1:numel(fit_types)
    fit_type = fit_types{type_idx};
    opts = param_basis_opts(fit_type);
    lower = opts.LB_default;
    upper = opts.UB_default;
    fprintf(out, '%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n', fit_type, ...
        lower(FSTD_COLUMN), upper(FSTD_COLUMN), ...
        lower(PSTD_COLUMN), upper(PSTD_COLUMN), ...
        lower(THETA_COLUMN), upper(THETA_COLUMN));
end
fclose(out);
fprintf('wrote default bounds for %d fits\n', numel(fit_types));
end
