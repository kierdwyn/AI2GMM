function score = purity(y, ypred)
%PURITY calculate the cluster purity scores
%   Detailed explanation goes here

cfs = confusionmat(y, ypred);
score = sum(max(cfs))/numel(y);


end