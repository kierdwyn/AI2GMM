function label_aligned=align_label(label_gt, label, ntrain)

if ~exist('ntrain', 'var'), ntrain = 0; end

utrain_gt = unique(label_gt(1:ntrain), 'stable');
utrain = unique(label(1:ntrain), 'stable');
[label_gt1, map_gt] = relabel(label_gt);
[label, map] = relabel(label);

cfm = confusionmat(label_gt1, label);
for i = 1:numel(utrain_gt)
    cfm(map_gt(utrain_gt(i)), map(utrain(i))) = numel(label_gt1);
end

cost = numel(label_gt1)-cfm;
alignment = munkres(cost);
[~, inverseal]=sort(alignment);
label_aligned = inverseal(label)';

% Map back to original labels
remap = containers.Map(values(map_gt, keys(map_gt)), keys(map_gt));
max_val = max(cell2mat(keys(map_gt)));
for i = 1:(map.Count - map_gt.Count)
    remap(map_gt.Count + i) = max_val + i;
end
label_aligned = cell2mat(values(remap, num2cell(label_aligned)));

end