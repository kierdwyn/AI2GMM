% Demo of run IGMM, I2GMM, AI2GMM on the synthetic dataset.
% clear;
% startup;

%-----load data and normalize-------
fpath = '..\data\synthetic\'; % folder containing the data set
name = '1base_d2_0.03_0.5_4_10'; % file name of the data set without extension
fpath = GetFullPath(fpath);
load([fpath name '.mat'],'x','y');
if size(x,2)>50
    x = pca_wzy(x,30);
end
x = standardize(x);
d = size(x,2);

%-----model control parameters-----
max_sweep = 1000;   % # of sweeps after adding test data.
init_sweep = 100;   % # of training sweeps.
burnin = 600;       % # of burnin sweeps we don't take sample.
sample = 20;        % # of samples we take between burinin+1:max_sweep.
max_n_c = 300;      % maximum # of clusters allowed.
ins = [2 5 6 10];        % A list of training classes.
train_ratio = 5;    % 1/5 (20%) training data from trainig classes.
model = 3;          % 1 run igmm, 2 run i2gmm, 3 run ai2gmm

%-----hyper params------
m = d+2; c_s = (m-d-1);
mu0 = zeros(1,d); sigma0 = eye(d) * c_s;
kappa0 = 0.1; kappa1 = 0.5;
alpha = 1; gamma = 1;
% hyper parameters for igmm
priors{1} = [mu0;sigma0];
confs{1} = [m kappa0 alpha];
% hyper parameters for i2gmm
priors{2} = [mu0;sigma0]; specs{2} = [ 0 0 0 [0 0] 0 0];
confs{2} = [m kappa0 kappa1 alpha gamma specs{2}];
% hyper parameters for ai2gmm
c1 = 0.1; c2 = d+2;
beta0 = d; alpha0 = beta0*kappa0 + 1;
beta1 = d; alpha1 = beta1*kappa1 + 1;
priors{3} = [mu0;sigma0/(c2-d)]; specs{3} = [ 0 0 0 [0 0] 0 0 1]; % ratio, prior, train, kap1>kap0, kap1<x*kap0, all in H, table, weighted kappa
confs{3} = [m c1 c2 alpha0 beta0 alpha1 beta1 alpha gamma specs{3}];

%------split data into training and testing------
if train_ratio == 1
    in = ismember(y(:,1), ins) & y(:,2)==1;
else
    in = ismember(y(:,1), ins) & (mod(1:length(y), train_ratio)==0)';
end
xtrain = x(in, :);
ytrain = y(in, 1);
xtest = x(~in, :);
ytest = y(~in, 1);
ntrain = size(xtrain,1);
fprintf('Training classes: ');
fprintf('%d ', numel(unique(ytrain))); fprintf('\n');

%--------------run ixgmm---------------
conf = confs{model};
prior = priors{model};
fname = [fpath name '_i' num2str(model) 'gmm_' num2str(numel(ytrain)) '\result_weighted_kappa_1'];
[ label, samples, lastlabel, bestlabel, likelihood, rtime, hyperparams ] =...
    i3gmm_exe( xtest,xtrain,ytrain, fname, conf, prior, model, max_sweep, burnin, sample, init_sweep, max_n_c );
y1 = [ytrain;ytest];
ypred = align_label(y1,label,ntrain);

%-------evaluate results-----------
F1 = calc_F1(label, y1, 2, ntrain,1);
Acc = calc_acc(ypred(ntrain+1:end,1),ytest);
fprintf('mean F1: %.4f, weighted F1: %.4f, ', F1);
fprintf('mean Acc: %.4f, weighted Acc: %.4f, ', Acc);
fprintf('# class found: %d\n', numel(unique(ypred(:,1))));
