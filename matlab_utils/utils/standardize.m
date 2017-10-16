function [ X1 ] = standardize( X )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

n = size(X,1);
X1 = (X - repmat(mean(X), n, 1)) ./ repmat(std(X), n, 1);

end

