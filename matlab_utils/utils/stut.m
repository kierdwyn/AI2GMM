function [prob] = stut(X, mu, Sigma, nu)
% Multivariate student T distribution, pdf
% X(i,:) is i'th case

[N, d] = size(X);
M = repmat(mu(:)', N, 1); % replicate the mean across rows
X = X-M;
mahal = sum((X*inv(Sigma)).*X,2); %#ok
logc = gammaln(nu/2 + d/2) - gammaln(nu/2) - 0.5*logdet(Sigma) ...
   - (d/2)*log(pi * nu);
logp = logc -(nu+d)/2*log1p(mahal/nu);
prob = exp(logp);

end