function yn = relabel_component(y, disp)
if ~exist('disp','var'); disp = 0; end

uy = unique(y(:,1));
yn = zeros(size(y));
for i = 1:numel(uy)
    yi = y(y(:,1)==uy(i),:);
    uyc = unique(yi(:,2));
    if disp
        fprintf('%3g ', uy(i));
        fprintf('%5g ',uyc);
        fprintf('\n');
    end
    for j = 1:numel(uyc)
        yn(y(:,1)==uy(i) & y(:,2)==uyc(j), 2) = j;
    end
end
yn(:,1) = y(:,1);

end