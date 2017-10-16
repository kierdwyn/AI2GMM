function best=align_labels(labels, ref, ntrain)
[~, nsample] = size(labels); % Each column is a sampled labels

if ~exist('ntrain', 'var'), ntrain = 0; end
if ~exist('ref', 'var')
    max = 0;
    idx = 0;
    for i = 1:nsample
        nc = numel(unique(labels(:,i)));
        if nc > max
            max = nc;
            idx = i;
        end
    end
    ref = relabel(labels(:,idx));
end

for i = 1:nsample
    labels(:,i) = align_label(ref, labels(:,i), ntrain);
end

best = mode(labels,2);

end