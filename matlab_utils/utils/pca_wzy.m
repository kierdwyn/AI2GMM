function Data=pca_wzy(X,threshold)

[n,~]=size(X);
avg=mean(X,1);
X=X-repmat(avg, n, 1);
sigma=X'*X/n;

[U,S,~]=svd(sigma);

eig_value=diag(S);
s=0;
sum_eig=sum(eig_value);

if threshold < 1
    for i=1:numel(eig_value)
        s=s+eig_value(i);
        if s/sum_eig>threshold
            break;
        end
    end
else
    i = threshold;
end

Data=X*U(:,1:i);
