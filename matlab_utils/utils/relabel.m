function [ lnew, map ] = relabel( label )
%RELABEL Renumber labels to be continuous integers
%     Parameters:
%         label: n-by-1 or n-by-2 matrix. the original labels.
%         
%     Return:
%         lnew: the new labels re-numbered to be continuous integers, 
%             begins with 1
%         map: # unique label-by-2 matrix contains the map from old label
%             to new label.

ul = unique(label(:,1), 'stable');
lnew = zeros(size(label,1),1);
keySet = cell(1,numel(ul));
valueSet = cell(1,numel(ul));
for i = 1:numel(ul)
    if iscell(ul)
        lnew(cellfun(@(a)(strcmp(a,ul{i})),label)) = i;
        keySet{i} = ul{i};
    else
        lnew(label==ul(i),1) = i;
        keySet{i} = ul(i);
    end
    valueSet{i} = i;
end
map = containers.Map(keySet, valueSet);

end

