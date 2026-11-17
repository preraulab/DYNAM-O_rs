%GEN_MATLAB_PARITY  Regenerate the MATLAB parity fixtures for the Rust stats
%
%   Usage:
%       run('scripts/gen_matlab_parity.m')
%
%   Inputs:
%       none
%
%   Outputs:
%       none (writes rust/tests/fixtures/parity_*.csv)
%
%   Notes:
%       Produces the reference p-values that
%       `rust/tests/stats_matlab_parity.rs` checks the Rust
%       port against. Cases deliberately span both sides of ranksum's
%       exact/approximate switch (n1 + n2 = 20), heavy/light/no ties,
%       samples containing NaN, and FDR families containing NaN p-values.
%
%       fdr_bh is nested inside FDR_1D.m and so is not callable from
%       outside it; a verbatim copy is appended to the bottom of this file
%       as fdr_bh_ref. If FDR_1D's copy ever changes, re-copy it here.
%
%       A diff in the regenerated fixtures means the MATLAB reference
%       moved -- treat that as a decision to make, not a file to commit.
%
%   See also: FDR_1D, FDR_2D, ranksum
%
%   ∿∿∿  Prerau Laboratory · sleepEEG.org  ∿∿∿

this_dir = fileparts(mfilename('fullpath'));
out_dir  = fullfile(this_dir, '..', 'rust', 'tests', 'fixtures');
assert(isfolder(out_dir), 'fixture dir not found: %s', out_dir);

% Generate randomized ranksum cases spanning both branches, ties, and NaNs.
rng(2026);
out = fopen(fullfile(out_dir, 'parity_ranksum.csv'),'w');
fprintf(out, 'case,n1,n2,p,x,y\n');
ncase = 0;
for trial = 1:300
    n1 = randi([1 25]); n2 = randi([1 25]);
    style = randi(4);
    switch style
        case 1, x = randn(1,n1);            y = randn(1,n2) + 0.6;      % continuous
        case 2, x = randi([1 4],1,n1);      y = randi([1 4],1,n2);      % heavy ties
        case 3, x = round(randn(1,n1),1);   y = round(randn(1,n2)+0.3,1); % light ties
        case 4, x = [randn(1,max(1,n1-2)) nan(1,min(2,n1))];            % with NaNs
                y = [randn(1,max(1,n2-2))+0.4 nan(1,min(2,n2))];
                n1 = numel(x); n2 = numel(y);
    end
    if all(isnan(x)) || all(isnan(y)), continue; end
    if sum(isfinite(x))==0 || sum(isfinite(y))==0, continue; end
    p = ranksum(x, y);
    ncase = ncase + 1;
    fprintf(out, '%d,%d,%d,%.17g,%s,%s\n', ncase, n1, n2, p, ...
        strjoin(arrayfun(@(v) sprintf('%.17g',v), x, 'uni', 0), ' '), ...
        strjoin(arrayfun(@(v) sprintf('%.17g',v), y, 'uni', 0), ' '));
end
fclose(out);
fprintf('wrote %d ranksum cases\n', ncase);

% FDR fixtures across both methods and several family shapes
out = fopen(fullfile(out_dir, 'parity_fdr.csv'),'w');
fprintf(out, 'case,method,q,p,padj\n');
nf = 0;
for trial = 1:60
    m = randi([1 200]);
    p = rand(1, m).^randi([1 4]);
    if mod(trial,3)==0                     % sprinkle NaNs
        p(randperm(m, max(1,floor(m/5)))) = nan;
    end
    for ms = {'pdep','dep'}
        q = 0.1;
        [~,~,~,padj] = fdr_bh_ref(p, q, ms{1});
        nf = nf + 1;
        fprintf(out, '%d,%s,%.17g,%s,%s\n', nf, ms{1}, q, ...
            strjoin(arrayfun(@(v) sprintf('%.17g',v), p, 'uni', 0), ' '), ...
            strjoin(arrayfun(@(v) sprintf('%.17g',v), padj, 'uni', 0), ' '));
    end
end
fclose(out);
fprintf('wrote %d fdr cases\n', nf);
function [h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh_ref(pvals,q,method,report)

if nargin<1
    error('You need to provide a vector or matrix of p-values.');
else
    if any(pvals<0)
        error('Some p-values are less than 0.');
    elseif any(pvals>1)
        error('Some p-values are greater than 1.');
    end
end

%Check for nan/inf and remove them
inds = isfinite(pvals);
if any(~inds)
    %     warning('Contains Nan/Inf values. Removing from invalid pvals from the list.')
    %     pvals_all = pvals;
    pvals = pvals(inds);
end

if nargin<2
    q=.05;
end

if nargin<3
    method='pdep';
end

if nargin<4
    report='no';
end

s=size(pvals);
if (length(s)>2) || s(1)>1
    [p_sorted, sort_ids]=sort(reshape(pvals,1,prod(s)));
else
    %p-values are already a row vector
    [p_sorted, sort_ids]=sort(pvals);
end
[~, unsort_ids]=sort(sort_ids); %indexes to return p_sorted to pvals order
m=length(p_sorted); %number of tests

if strcmpi(method,'pdep')
    %BH procedure for independence or positive dependence
    thresh=(1:m)*q/m;
    wtd_p=m*p_sorted./(1:m);

elseif strcmpi(method,'dep')
    %BH procedure for any dependency structure
    denom=m*sum(1./(1:m));
    thresh=(1:m)*q/denom;
    wtd_p=denom*p_sorted./(1:m);
    %Note, it can produce adjusted p-values greater than 1!
    %compute adjusted p-values
else
    error('Argument ''method'' needs to be ''pdep'' or ''dep''.');
end

if nargout>3
    %compute adjusted p-values; This can be a bit computationally intensive
    adj_p_samp=zeros(1,m)*NaN;
    [wtd_p_sorted, wtd_p_sindex] = sort( wtd_p );
    nextfill = 1;
    for k = 1 : m
        if wtd_p_sindex(k)>=nextfill
            adj_p_samp(nextfill:wtd_p_sindex(k)) = wtd_p_sorted(k);
            nextfill = wtd_p_sindex(k)+1;
            if nextfill>m
                break;
            end
        end
    end

    adj_p_samp=reshape(adj_p_samp(unsort_ids),s);
end

rej=p_sorted<=thresh;
max_id=find(rej,1,'last'); %find greatest significant pvalue
if isempty(max_id)
    crit_p_samp=0;
    h_samp=pvals*0;
    adj_ci_cvrg_samp=NaN;
else
    crit_p_samp=p_sorted(max_id);
    h_samp=pvals<=crit_p_samp;
    adj_ci_cvrg_samp=1-thresh(max_id);
end

if strcmpi(report,'yes')
    n_sig=sum(p_sorted<=crit_p_samp);
    if n_sig==1
        fprintf('Out of %d tests, %d is significant using a false discovery rate of %f.\n',m,n_sig,q);
    else
        fprintf('Out of %d tests, %d are significant using a false discovery rate of %f.\n',m,n_sig,q);
    end
    if strcmpi(method,'pdep')
        fprintf('FDR/FCR procedure used is guaranteed valid for independent or positively dependent tests.\n');
    else
        fprintf('FDR/FCR procedure used is guaranteed valid for independent or dependent tests.\n');
    end
end

h = nan(1,length(inds));
crit_p = nan(1,length(inds));
adj_ci_cvrg = nan(1,length(inds));
adj_p = nan(1,length(inds));

h(inds) = h_samp;
crit_p(inds) = crit_p_samp;
adj_ci_cvrg(inds) = adj_ci_cvrg_samp;
adj_p(inds) = adj_p_samp;
end
