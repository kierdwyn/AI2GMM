function [X_dafe,vv]=LDA(XX,YY)


%% DAFE

uY=unique(YY);
mu = zeros(length(uY), size(XX,2));
Sigma = zeros(size(XX, 2), size(XX, 2), length(uY));
for i=1:length(uY)
    nn=sum(YY==uY(i));
    mu(i,:)=mean(XX(YY==uY(i),:),1);
    Sigma(:,:,i)=cov(XX(YY==uY(i),:))*(nn-1);
end
SW=mean(Sigma,3);
SB=cov(mu);

Sww=SW\SB;

[vv, ~]=eig(Sww);
vv=real(vv);
vv = vv(:,1:min([length(uY) size(vv,2)]));

X_dafe=XX*vv;




% 
% ss=hsv(length(uY));
% f1=figure
% hold
% for i=1:length(uY)
%     plot(X_dafe(YY==uY(i),1),X_dafe(YY==uY(i),2),'b.','MarkerSize',1,'Color',ss(i,:));
% end
% 


