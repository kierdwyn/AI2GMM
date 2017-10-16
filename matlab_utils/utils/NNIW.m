function [label]=NNIW(xtrain, xtest, ytrain, P)
d=size(xtrain,2);

labels=unique(ytrain);
num_class=numel(labels);
Data_train=cell(num_class,1);
for i=1:num_class
    Data_train{i}=xtrain(ytrain==labels(i),:);
end

if exist('P', 'var')
    u0 = P(1,:);
    sigma0 = P(2:d+1, :);
    kapa=P(d+2,1);
    m=P(d+2,2);
else
    sigma0 = eye(d);
    kapa=5;
    u0=500*ones(1,d);
    m=d+3+10;
end

u=zeros(num_class,d);
n=zeros(num_class,1);
SS=cell(num_class,1);

u_stu=zeros(num_class,d);
Sigma=cell(num_class,1);
D_f=zeros(num_class,1);

for i=1:num_class
    D=Data_train{i};
    [n(i),~]=size(D);
    
    u(i,:)=mean(D);
    
    SS{i}=(n(i)-1)*cov(D);
    
    D_f(i)=n(i)+m+1-d;
    
    u_stu(i,:)=(n(i)*u(i,:)+kapa*u0)/(n(i)+kapa);
    
%     ma=diag(sigma0)+(n(i)-1)*SS{i}+(n(i)*kapa/(n(i)+kapa))*(u0-u(i,:))'*(u0-u(i,:));
    diff = u0-u(i,:);
    ma=sigma0+SS{i}+(n(i)*kapa/(n(i)+kapa))*(diff'*diff);
    Sigma{i}=((n(i)+kapa+1)/((n(i)+kapa)*D_f(i)))*ma;
    
    % Make sure C is a valid covariance matrix
    [~,err] = cholcov(Sigma{i},0);
    if err ~= 0
        error(message('stats:mvtpdf:BadCorrelationSymPos'));
    end
end

[nt,~]=size(xtest);
px = zeros(num_class, nt);
for i = 1:num_class
    px(i,:) = stut(xtest, u_stu(i,:), Sigma{i}, D_f(i))';
end
[~, lindex] = max(px);
label = labels(lindex);

end
