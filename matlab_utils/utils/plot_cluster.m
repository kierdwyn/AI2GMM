function h = plot_cluster( x, labels, maxlabel )
%PLOT_CLUSTER plot the dataset.
%   Only plot first 2 dimensions.

% Generate colors
labels = labels + repmat(min(labels) == 0, length(labels), 1);
ul = unique(labels(:,1));
if ~exist('maxlabel','var'); maxlabel = max(ul); end
colors = hsv(maxlabel); % See http://www.mathworks.com/help/matlab/ref/colormap.html#input_argument_name for colormap options
x_c = colors(labels(:,1), :);

h = scatter(x(:,1), x(:,2), 2, x_c, 'filled');

label = labels(:,1);
ul = unique(label);
for i = 1:numel(ul)
    data = x(label == ul(i), 1:2);
    hold on
    if (size(data, 1) > size(data, 2))
        error_ellipse(cov(data), mean(data,1), 'color', colors(ul(i),:));
    else
        data = mean(data, 1);
        scatter(data(:,1), data(:,2), 50, colors(ul(i),:));
    end
end

if size(labels,2) == 2
    for i = 1:numel(ul)
        data = x(label==ul(i), 1:2);
        datal = labels(label==ul(i), 2);
        c = colors(ul(i),:);
        ull = unique(datal);
        for j = 1:numel(ull)
            data2 = data(datal==ull(j), :);
            hold on
            if (size(data2, 1) > size(data2, 2))
                error_ellipse(cov(data2), mean(data2,1), 'style', ':', 'color', c);
            else
                error_ellipse(eye(size(data2,2))*0.01, mean(data2,1), 'style', ':', 'color', c);
            end
        end
    end
end

end

