function [ label, samples, lastlabel, bestlabel, likelihood, rtime, hyperparams ] = i3gmm_continue( test_data, fname, alpha, gamma, n_sweeps, burnin, sample )
%I3GMM_CONTINUE Continue running from last results
%     Inputs:
%         Required:
%             test_data: n-by-d matrix where n is the number of data points
%                 to be classified.
%         Optional:
%             fname: the path to load the model. Default in folder
%                 './i3gmm_results/'. Note that temporary files will also
%                 be saved in this folder.
%             alpha: hyper-parameter to control expectation of # of
%                 components. Default to 0, don't generate new components.
%             gamma: hyper-parameter to control expectation of # of
%                 classes. Default to 0, don't generate new classes.
%             n_sweeps: number of Gibbs sweeps to perform the
%                 classification. Default to 1.
%             burnin: # of gibbs sweeps we don't take any sample. Default to
%                 max_sweep / 2;
%             sample: take [sample] samples each at interval
%                 (max_sweep-burnin)/sample. Default to 10
%     Outputs:
%         label: n-by-1 matrix containing labels for both training and
%             testing in the form [label_train; label_test].
%         likelihood: n-by-1 vector containing the log-likelihood for all
%             data points


if ~exist('fname','var'), fname = './i3gmm_results/'; end
if ~exist('alpha','var'), alpha = 0;end
if ~exist('gamma','var'), gamma = 0;end
if ~exist('n_sweeps','var'), n_sweeps = 1;end
if ~exist('burnin','var'), burnin = floor(n_sweeps/2); end
if ~exist('sample','var'), sample = floor(n_sweeps/100); end

writeMat([fileparts(fname) '\data.matrix'], test_data, 'double');

args = ['continue "' fname '" ' int2str(alpha) ' ' int2str(gamma) ' '...
    int2str(n_sweeps) ' ' int2str(burnin) ' ' int2str(sample)];
fid = fopen('cmd.txt', 'wt');
fprintf(fid, '%s', args);
fclose(fid);
tic
[status, ret] = system(['..\Release\i2gmm_semi.exe ' args]);
rtime = toc;
if status
    fprintf('Something wrong when running i2gmm\n');
    disp(ret);
    close all;
    return
end


%-------read results---------
samples = [];
if sample~=0
    samples = readMat([fname '_continued_samplelabels.txt'], 'text')';
end
hyperparams = readMat([fname '_continued_hyperparams.txt'],'text');
labels = samples(:,mod(1:(2*sample),2)==1);
lastlabel = readMat([fname '_continued_lastlabels.txt'], 'text')';
bestlabel = readMat([fname '_continued_bestlabels.txt'], 'text')';
likelihood = readMat([fname '_continued_likelihoods.matrix'], 'double');

%-------align labels--------
ntrain = size(lastlabel,1)-size(test_data,1);
unique_y_s = zeros(sample,1);
for i = 1:sample
    unique_y_s(i) = numel(unique(labels(:,i)));
end
[~,ref_idx] = max(flip(unique_y_s));
ref_idx = sample-ref_idx+1;
label = align_labels(labels, labels(:,ref_idx), ntrain);

end