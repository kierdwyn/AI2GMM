function [ acc, acc_classes ] = calc_acc( ypred, y, align )
%CALC_ACC Calculate the weighted and mean accuracies.

if ~exist('align','var'), align = 0; end
if align
    ypred = align_label(y, ypred);
end

uy = unique(y);
acc(2) = sum(ypred==y)/numel(y);
acc_classes = zeros(numel(uy),1);
for i = 1:numel(uy);
    acc_classes(i) = sum((ypred==uy(i))&(y==uy(i))) / sum(y==uy(i));
end
acc(1) = sum(acc_classes)/numel(uy);

end

