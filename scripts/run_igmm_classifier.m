% Demo of using IGMM as a classifier on the synthetic dataset.
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

%-----hyper params------
m = 2+d;
mu0 = zeros(1,d); sigma0 = eye(d);
kappa0 = 0.1; kappa1 = 0.5;
alpha = 0;
% hyper parameters for igmm
conf = [m kappa0 alpha];
prior = [mu0;sigma0];

%--------------run i3gmm---------------
fname = [fpath name '_igmm_' num2str(numel(ytrain)) '\result1'];
[ypred, samples] = i3gmm_exe(xtest,xtrain,ytrain, fname, conf, prior, 1, 1, 0, 1);
ntrain=size(xtrain,1);
y1 = [ytrain;ytest];

%-------evaluate results-----------
F1 = calc_F1(ypred, y1, 2, ntrain,1);
Acc = calc_acc(ypred(ntrain+1:end,1),ytest);
fprintf('mean F1: %.4f, weighted F1: %.4f, ', F1);
fprintf('mean Acc: %.4f, weighted Acc: %.4f, ', Acc);
fprintf('# class found: %d\n', numel(unique(ypred(:,1))));

