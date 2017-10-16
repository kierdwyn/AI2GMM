% Demo of using I2GMM or AI2GMM as a classifier on the synthetic dataset.
clear;
startup;

%-----load data and normalize-------
fpath = '..\data\synthetic\'; % folder containing the data set
name = '1base_d2_0.03_0.5_4_10'; % file name of the data set without extension
fpath = GetFullPath(fpath);
load([fpath name '.mat'],'x','y');
x = x(y(:,1)~=0,:);y = y(y(:,1)~=0,:);
x = standardize(x); % Normalize data to be 0 mean and unit variance for each dimension
in = ismember(y(:,1), 1:10) & ((mod(1:length(y), 10)==0)'); % A n-by-1 logical. put 1/10 data point of all classes into training.
xtrain = x(in, :); % training data
ytrain = y(in, 1); % training label
xtest = x(~in, :); % testing data
ytest = y(~in, 1); % testing label. for evaluation only
d = size(x,2);

%-----model control parameters-----
model = 2;   % 1 is igmm, 2 is i2gmm, 3 is i3gmm
train_sweep = 200; % # of Gibbs sweeps performed on training data before adding testing data.

%-----hyper params------
m = 2+d;
mu0 = zeros(1,d); sigma0 = eye(d)*(m-d-1);
kappa0 = 0.1; kappa1 = 0.5;
alpha = 1; gamma = 1;
% hyper parameters for igmm
confs{1} = [m kappa0 alpha];
priors{1} = [mu0;sigma0];
% hyper parameters for i2gmm
confs{2} = [m kappa0 kappa1 alpha gamma];
confs{2} = [confs{2} [ 0 0 0 [0 0] 0 0]]; % ratio, prior, train, kap1>kap0, kap1<x*kap0, all in H, table
priors{2} = [mu0;sigma0];
% hyper parameters for i3gmm
c1 = 0.1; c2 = d+2;
beta0 = 20; alpha0 = kappa0*beta0 + 1;
beta1 = 20; alpha1 = kappa1*beta1 + 1;
confs{3} = [m c1 c2 alpha0 beta0 alpha1 beta1 alpha gamma];
confs{3} = [confs{3} [ 5 0 0 [1 0] 0 0]]; % ratio, prior, train, kap1>kap0, kap1<x*kap0, all in H, table
priors{3} = [mu0;sigma0*(c2-d)];

%--------------train i3gmm---------------
conf = confs{model};
prior = priors{model};
fname = [fpath name '_i' num2str(model) 'gmm_' num2str(numel(ytrain)) '\result1'];
i3gmm_exe(zeros(0,d),xtrain,ytrain, fname, conf, prior, model, 0, 0, 0, train_sweep );

%--------------predict using i3gmm-----------
[label, samples] = i3gmm_continue( xtest, fname, 0, 0, 1, 0, 1 );
ntrain=size(xtrain,1);
y1 = [ytrain;ytest];
ypred = align_label(y1,label,ntrain);

%-------evaluate results-----------
F1 = calc_F1(ypred, y1, 2, ntrain,1);
Acc = calc_acc(ypred(ntrain+1:end,1),ytest);
fprintf('mean F1: %.4f, weighted F1: %.4f, ', F1);
fprintf('mean Acc: %.4f, weighted Acc: %.4f, ', Acc);
fprintf('# class found: %d\n', numel(unique(ypred(:,1))));

