function [ F1, f1_classes ] = calc_F1( ypred, y, weighted, ntrain, show_results )
%CALC_F1 Calculate the weighted and mean F1 scores.
%     Parameters:
%         ypred: length N vector. predicted labels generated by models.
%         y: length N vector. ground truth labels.
%         weighted (optional, default 2): 0: return mean F1 score;
%               1: return weighted F1 score;
%               2: return both.
%         ntrain (optional, default 0): number of training labels in ypred
%               and y. We are assuming y = [ytrain; ytest]. If you are
%               doing classification or non-exhaustive learning, you need
%               to including training labels in both ypred and y.
%         show_results (optional, default 0): display the mean F1 score for
%               each class.
%     Return value:
%         F1(1): mean F1.
%         F1(2): weighted F1.
%         Default: return both

if ~exist('weighted','var'), weighted = 2; end
if ~exist('ntrain', 'var'), ntrain = 0; end
if ~exist('show_results','var'), show_results = 0;end

has0 = any([any(y==0) any(ypred==0)]);
ypred = ypred + has0;
y = y + has0;

ytrain = y(1:ntrain);
ytest = y(ntrain+1:end);
ytestp = ypred(ntrain+1:end);
uy = unique(y);
utrain = unique(ytrain);
utest = uy(~ismember(uy,utrain));

F1 = [0 0];
cfs = confusionmat(ytest, ytestp, 'order', 1:max([uy(end); ypred(:)]));
sum_trues = sum(cfs, 2);
sum_preds = sum(cfs, 1);

cfs_unknown = cfs(utest,:);
cfs_unknown(:,utrain) = 0;

TP = zeros(size(cfs,1),1);
idx = ones(size(cfs,1),1);
[TP1,idx1] = max(cfs_unknown, [], 2);
TP2 = diag(cfs(utrain,utrain));
TP(utrain) = TP2;
TP(utest) = TP1;
idx(utrain) = utrain;
idx(utest) = idx1;
denominator = sum_trues + sum_preds(idx)';
denominator = denominator + (denominator==0);
f1_classes = 2 * TP ./ denominator;
F1(1) = sum(f1_classes);
F1(2) = sum(2 * TP.*sum_trues ./ denominator);
F1 = F1 ./ [numel(unique(ytest)) sum(sum_trues)];

if weighted == 1
    F1 = F1(2);
else if weighted == 0
        F1 = F1(1);
    end
end

if show_results
    fprintf('%5g ',1:numel(uy)); fprintf('\n');
    fprintf('%5g ',sum_trues(uy)); fprintf('\n');
    fprintf('%5g ',sum_preds(idx(uy))); fprintf('\n');
    fprintf('%.3f ',f1_classes(uy)); fprintf('\n');
end

f1_classes = f1_classes(1:numel(uy),:);

end
