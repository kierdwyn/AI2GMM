function [h1] = plot_clusters_3d( X, Y )

uY = unique(Y);
colors = hsv(numel(uY));

h1 = figure;
for i = 1:numel(uY)
    Xp = X(Y==uY(i),:);
    hold on;
    scatter3(Xp(:,1), Xp(:,2), Xp(:,3), 2, colors(i,:),'filled');
end

end

