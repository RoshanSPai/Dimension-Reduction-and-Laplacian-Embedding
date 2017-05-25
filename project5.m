clear all;
prompt = 'Choose one of the following:\n1. PCA \n2.LDA \n3.Laplacian Embedding\n >>';
choice=input(prompt);
switch choice
    case 1
        pca()
    case 2
        linearDiscriminantAnalysis()
    case 3
        laplacianEmbedding()
end

   