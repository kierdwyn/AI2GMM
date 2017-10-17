function [args] = gen_arguments(fname, fres, model, max_sweep, burnin, sample, init_sweep, max_n_clusters)
if ~exist('max_n_clusters','var'), max_n_clusters = [];end

fdata = [ fname '.matrix'];
fparams = [ fname '_params.matrix'];
fprior = [ fname '_prior.matrix'];
ftrain = [ fname '_train.matrix'];
args = ['"' fdata '" "' fprior '" "' fparams '" ' fres ' ' ...
    int2str(model) ' ' int2str(max_sweep) ' ' int2str(burnin) ' '...
    int2str(sample) ' "' ftrain '" ' int2str(init_sweep) ' ' int2str(max_n_clusters)];
end