function [] = laplacianEmbedding()
[fileName1,pathName1] = uigetfile('*.txt','Select the training data file');
trainData = csvread(strcat(pathName1,fileName1),1,0);
classData = csvread(strcat(pathName1,fileName1),0,0,[0, 0, 0, size(trainData,2)-1]);
Data=pdist(trainData');
Data=exp(-(Data/max(Data)));
Data = squareform(Data);
diagonalData=diag(sum(Data,2));
Lambda=diagonalData-Data;
[eigvec,eigval] = eig(Lambda,diagonalData);
top=[5 10 20 30 50];
ts=size(top,2);
output = [];
for i=1:ts
    k=i;
    lapData=eigvec(:,2:k+1)';
    knn=10;
    [accKnn,accLin,accCen,accSvm]=kfoldCV(lapData,classData,knn);
    output=vertcat(output,[accKnn,accLin,accCen,accSvm]);
end
for i=1:4
    figure(1);
    xaxis=top;
    yaxis=output(1:ts,i);
    plot(xaxis,yaxis,'--s')
    title('Laplacian Embedding');
    xlabel('Top k features');
    ylabel('Acurracy');
    legend('KNN','Linear Regression','Centroid Clustering', 'Svm');
    hold on
end
output
end

