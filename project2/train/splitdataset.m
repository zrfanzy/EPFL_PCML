
% origin : split label, X_hog,X_cnn, prop

function [cnn_Xtrain, cnn_ytrain, cnn_Xtest, cnn_ytest, hog_Xtrain, hog_ytrain, hog_Xtest, hog_ytest] = split()
% split the data into train and test given a proportion
    load('train.mat')
    prop = 1;
		setSeed(1);
    N = size(labels,1);
		% generate random indices
		idx = randperm(N);
    Ntr = floor(prop * N);
		% select few as training and others as testing

    idxTr = idx(1:Ntr);
		idxTe = idx(Ntr+1:end);
		% create train-test split
    cnn_Xtrain = X_cnn(idxTr,:);
    cnn_ytrain = labels(idxTr);
    cnn_Xtest = X_cnn(idxTe,:);
    cnn_ytest = labels(idxTe);

    hog_Xtrain = X_hog(idxTr,:);
    hog_ytrain = labels(idxTr);
    hog_Xtest = X_hog(idxTe,:);
    hog_ytest = labels(idxTe);
    
    csvwrite('cnn_Xtrain.csv',cnn_Xtrain)
    csvwrite('cnn_ytrain.csv',cnn_ytrain)
    csvwrite('cnn_Xtest.csv',cnn_Xtest)
    csvwrite('cnn_ytest.csv',cnn_ytest)
    
    csvwrite('hog_Xtrain.csv',hog_Xtrain)
    csvwrite('hog_ytrain.csv',hog_ytrain)
    csvwrite('hog_Xtest.csv',hog_Xtest)
    csvwrite('hog_ytest.csv',hog_ytest)
end

function setSeed(seed)
% set seed
	global RNDN_STATE  RND_STATE
	RNDN_STATE = randn('state');
	randn('state',seed);
	RND_STATE = rand('state');
	%rand('state',seed);
	rand('twister',seed);
end

