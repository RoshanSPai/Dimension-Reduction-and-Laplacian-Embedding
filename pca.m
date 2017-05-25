function [] = pca()
[fileName1,pathName1] = uigetfile('*.txt','Select the training data file');
trainData = csvread(strcat(pathName1,fileName1),1,0);
classData = csvread(strcat(pathName1,fileName1),0,0,[0, 0, 0, size(trainData,2)-1]);

[U S V]=svd(trainData,'econ');
%[coeff,score,latent,tsquared,explained] = pca(Data');
top=[10 20 30 50 100];
ts=size(top,2);
output = [];
for i=1:ts
    k=top(i);
    Uk=U(:,1:k);
    Vk=V(:,1:k);
    Sk=S(1:k,1:k);
    Xk=Uk*Sk*Vk';
    pca_train=Uk'*trainData;
    knn=10;
    [accKnn,accLin,accCen,accSvm]=kfoldCV(pca_train,classData,knn);
    output=vertcat(output,[accKnn,accLin,accCen,accSvm]);
end
output
for i=1:4
    figure(1);
    xaxis=top;
    yaxis=output(1:ts,i);
    plot(xaxis,yaxis,'--s')
    title('Principal Component Analysis');
    xlabel('Top k features');
    ylabel('Acurracy');
    legend('KNN','Linear Regression','Centroid Clustering', 'Svm');
    hold on
end
end

