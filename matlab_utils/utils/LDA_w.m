function [ x ] = LDA_w( xtrain, ytrain, xtest, ~ )
%LDA_w the wrapped version of Linear Discriminant Analysis
%   Params:
%         xtrain: n-by-d matrix where d is the dimension.
%         ytrain: n-by-1 vector of the labels
%         xtest: nt-by-d matrix where nt is the number of test data points
%     return:
%         x: the projected data. first n rows are training data the rest is
%         testing data. the project axises are determined by training data.

[xtrain1, T] = LDA(xtrain, ytrain);
if ~exist('xtest','var')
    xtest1 = [];
else
    xtest1 = xtest * T;
end
x = [xtrain1; xtest1];

end

