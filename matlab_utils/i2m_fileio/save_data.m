function save_data( fname, x, conf, prior, x_train, labels_gt )
%SAVE_DATA Summary of this function goes here
%   Detailed explanation goes here
warning('off', 'MATLAB:MKDIR:DirectoryExists');
mkdir(fileparts(fname));

fdata = [ fname '.matrix'];
fparams = [ fname '_params.matrix'];
fprior = [ fname '_prior.matrix'];
ftrain = [ fname '_train.matrix'];
flabel = [fname '_labels_gt.matrix'];
writeMat(fdata, x, 'double');
writeMat(fparams,conf,'double');
writeMat(fprior,prior,'double');
writeMat(ftrain,x_train, 'double');
writeMat(flabel,labels_gt,'double');

end

