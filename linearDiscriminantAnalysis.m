function [] = linearDiscriminantAnalysis()
[fileName1,pathName1] = uigetfile('*.txt','Select the training data file');
trainData = csvread(strcat(pathName1,fileName1),1,0);
classData = csvread(strcat(pathName1,fileName1),0,0,[0, 0, 0, size(trainData,2)-1]);
k=size(classData,2);
[mappedData mapping]=lda(trainData', classData, k-1);
knn=10;
output=[];
[accKnn,accLin,accCen,accSvm]=kfoldCV(mappedData',classData,knn);
output=vertcat(output,[accKnn,accLin,accCen,accSvm]);

for i=1:4
    figure(1);
    xaxis=[0,50];
    yaxis=[output(i),output(i)];
    plot(xaxis,yaxis,'--')
    title('Linear Discriminant Analysis');
    xlabel('Top k features');
    ylabel('Acurracy');
    set(gca,'XTick',[]);
    legend('KNN','Linear Regression','Centroid Clustering', 'Svm');
    hold on
end
output
end

