function [ x ] = LFDA_w( xtrain, ytrain, xtest, d )
%LFDA_w Wrapped version of LFDA
%   Detailed explanation goes here

[T, xtrain1] = LFDA(xtrain', ytrain, d);
xtrain1 = xtrain1';
xtest1 = xtest * T;
x = [xtrain1; xtest1];

end

