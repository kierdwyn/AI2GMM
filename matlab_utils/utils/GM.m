classdef GM
    %GM Gaussian mixture classifier
    %   Training: build a guassian mixture model for each class.
    %   Each component is a guassian over the class in one image.
    %   The weight for component is its propotion of number of points in
    %   the class.
    
    properties
%         d       % dimension
%         k       % # of class.
        mus     % k -by- 1 cell. Each cell contains a # of images
                % -by- d matrix.
        sigmas  % k -by- 1 cell contains # of images -by- d -by- d
                % matrix.
        labels
    end
    
    methods
        function obj = train(obj, X, Y, ~, sigma0)
            d = size(X, 2);
            if ~exist('sigma0','var'), sigma0 = eye(d)*0.001; end
            uY = unique(Y);
            k = numel(uY);
            
            obj.labels = uY;
            obj.mus = cell(k, 1);
            obj.sigmas = cell(k, 1);
            for i = 1:k
                in = Y == uY(i);
                Xc = X(in,:);
                obj.mus{i} = mean(Xc,1);
                obj.sigmas{i} = cov(Xc) + sigma0;
            end
        end
        
        function ypred = predict(obj, X, ~)
            n = size(X, 1);
            k = numel(obj.mus);
            likelihood = zeros(k, n);
            for i = 1:k
                m = obj.mus{i};
                s = obj.sigmas{i};
                likelihood(i,:) = mvnpdf(X, m, s);
            end
            [~, lindex] = max(likelihood);
            ypred = obj.labels(lindex);
        end
    end
    
end
