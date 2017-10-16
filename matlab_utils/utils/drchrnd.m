function r = drchrnd(a,n)
%DRCHRND generate n random number from dirichlet distribution
%   Parameters:
%       a: vector of prior.

p = length(a);
r = gamrnd(repmat(a,n,1),1,n,p);
r = r ./ repmat(sum(r,2),1,p);

end

