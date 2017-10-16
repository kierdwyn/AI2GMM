function [x, labels, mu, sigma] = guassian_mixture(mu0, sigma0, df, kappa, n_clusters, n_data)
% Generate a guassian mixture data set

if numel(n_data) == 1
    n_data = repmat(n_data, n_clusters, 1);
end

x = zeros(sum(n_data), numel(mu0));
labels = zeros(sum(n_data),1);
mu = zeros(n_clusters, numel(mu0));
sigma = zeros(numel(mu0), numel(mu0), n_clusters);
for i = 1:n_clusters
    if df == 0
        sigma(:,:,i) = sigma0;
    else
        sigma(:,:,i) = iwishrnd(sigma0, df);
    end
    mu(i, :) = mvnrnd(mu0, sigma(:,:,i)/kappa);
    x(sum(n_data(1:i-1)) + 1 : sum(n_data(1:i)),:) = mvnrnd(mu(i, :), sigma(:,:,i), n_data(i));
    labels(sum(n_data(1:i-1)) + 1 : sum(n_data(1:i)),:) = i;
end

end