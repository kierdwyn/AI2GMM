function [ label, samples, lastlabel, bestlabel, likelihood, rtime, hyperparams ] =...
    i3gmm_exe( xtest,xtrain,ytrain, fname, conf, prior, model, max_sweep, burnin, sample, init_sweep, max_n_clusters, load_result )
%I3GMM_EXE Running I3GMM/I2GMM/IGMM models.
%     Inputs:
%         Required:
%             xtest: n-by-d matrix where n is the number of data points.
%         Optional:
%             xtrain: n-by-d matrix where n is the number of data points
%                 to be classified. Default to []
%             ytrain: n-by-d matrix where n is the number of data points
%                 to be classified. Default to []
%             fname: the path where you want to save the results.
%                 Default to "./result/".
%             conf: a vector containing hyper-parameters in following order:
%                 for i2gmm:
%                     [m kappa0 kappa1 alpha gamma [specs]]
%                 for i3gmm:
%                     [m c1 c2 alpha0 beta0 alpha1 beta1 alpha gamma [specs]]
%                 specs contains some specification for model adjustment.
%             prior: [mu0;sigma0] where mu0 and sigma0 are hyper-parameters.
%             model: 1 is igmm, 2 is i2gmm, 3 is i3gmm. Default 3.
%             max_sweep: # of Gibbs sweeps, defualt to 1000
%             init_sweep: # of Gibbs sweeps performed on training data
%                 before adding testing data. Default to 100
%             burnin: # of gibbs sweeps we don't take any sample. Default to
%                 max_sweep / 2;
%             sample: take [sample] samples each at interval
%                 (max_sweep-burnin)/sample. Default to 10
%     Outputs:
%         label: n-by-1 matrix containing labels for both training and
%             testing in the form [label_train; label_test].
%         likelihood: n-by-1 vector containing the log-likelihood for all
%             data points
%         rtime: running time in seconds.

ntrain = size(xtrain,1);
d = size([xtrain;xtest],2);
m = 2+d;
mu0 = zeros(1,d); sigma0 = eye(d);
kappa0 = 0.1; kappa1 = 0.5;
c1 = 0.1; c2 = d+2;
beta0 = 20; alpha0 = kappa0*beta0 + 1;
beta1 = 20; alpha1 = kappa1*beta1 + 1;
alpha = 1; gamma = 1;
specs = [ 5 0 0 [1 0] 0 0]; % ratio, prior, train, kap1>kap0, kap1<x*kap0, all in H, table

if ~exist('xtrain','var'), xtrain = []; end
if ~exist('ytrain','var'), ytrain = []; end
if ~exist('fname','var'), fname = './result/'; end
if ~exist('conf','var'), conf = [m c1 c2 alpha0 beta0 alpha1 beta1 alpha gamma specs]; end
if ~exist('prior','var'), prior = [mu0;sigma0]; end
if ~exist('model','var'), model = 3; end
if ~exist('max_sweep','var'), max_sweep = 1000; end
if ~exist('init_sweep','var'), init_sweep = 100; end
if ~exist('burnin','var'), burnin = max_sweep/2; end
if ~exist('sample','var'), sample = 10; end
if ~exist('load_result','var'), load_result = 0; end
if ~exist('max_n_clusters','var'), max_n_clusters = []; end

[fpath,~] = fileparts(fname);
fres = [ '"' fname '"'];
save_data([fpath '\data'], xtest, conf, prior, [ytrain xtrain], []); % For exe
args = gen_arguments([fpath '\data'], fres, model, max_sweep, burnin, sample, init_sweep, max_n_clusters);
fid = fopen('cmd.txt', 'wt');
fprintf(fid, '%s', args);
fclose(fid);
rtime = 0;
if load_result == 0
    tic
    [status, ret] = system(['..\Release\i2gmm_semi.exe ' args]);
    rtime = toc;
    if status
        fprintf('Something wrong when running i2gmm\n');
        disp(ret);
        close all;
        return
    end
end

%-------read results---------
fres = fres(2:end-1);
samples = [];
likelihood = [];
hyperparams = [];
if sample~=0
    samples = readMat([fres '_samplelabels.txt'], 'text')';
end
if model > 1
    likelihood = readMat([fres '_likelihoods.matrix'], 'double'); % TODO likelihood for igmm
    hyperparams = readMat([fres '_hyperparams.txt'],'text');
    labels = samples(:,mod(1:(2*sample),2)==1);
else
    labels = samples;
end
lastlabel = readMat([fres '_lastlabels.txt'], 'text')';
bestlabel = readMat([fres '_bestlabels.txt'], 'text')';

%-------align labels--------
unique_y_s = zeros(sample,1);
for i = 1:sample
    unique_y_s(i) = numel(unique(labels(:,i)));
end
[~,ref_idx] = max(flip(unique_y_s));
ref_idx = sample-ref_idx+1;
label = align_labels(labels, labels(:,ref_idx), ntrain);

end