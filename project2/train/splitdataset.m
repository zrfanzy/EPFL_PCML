
% origin : split label, X_hog,X_cnn, prop

function [cnn_Xtrain, cnn_ytrain, cnn_Xtest, cnn_ytest, hog_Xtrain, hog_ytrain, hog_Xtest, hog_ytest] = splitdataset()
% split the data into train and test given a proportion
    load('train.mat')
    load('modified_y.mat')
    train
%    labels = train.y;
    labels = modified_y
    X_hog = train.X_hog;
    X_cnn = train.X_cnn;

    
    prop = 0.8;
		setSeed(1);

    N = size(labels,1);
		% generate random indices
    idx = randperm(N);
    Ntr = floor(prop * N);
		% select few as training and others as testing

    idxTr = idx(1:Ntr);
		idxTe = idx(Ntr+1:end);
		% create train-test split
    cnn_Xtrain = X_cnn(idxTr,1:end-1);
    cnn_ytrain = labels(idxTr);
    cnn_Xtest = X_cnn(idxTe,1:end-1);
    cnn_ytest = labels(idxTe);

    hog_Xtrain = X_hog(idxTr,:);
    hog_ytrain = labels(idxTr);
    hog_Xtest = X_hog(idxTe,:);
    hog_ytest = labels(idxTe);
    
    size(cnn_Xtrain)
    size(cnn_ytrain)
    size(cnn_Xtest)
    size(cnn_ytest)
    size(hog_Xtrain)
    size(hog_ytrain)
    size(hog_Xtest)
    size(hog_ytest)
%    save('cnn_Xtrain.mat','cnn_Xtrain')
%    save('cnn_Xtest.mat','cnn_Xtest')
  %  save('cnn_ytrain.mat','cnn_ytrain')
   % save('cnn_ytest.mat','cnn_ytest')
%    save('hog_Xtrain.mat','hog_Xtrain')
%    save('hog_Xtest.mat','hog_Xtest')
%    save('hog_ytrain.mat','hog_ytrain')
 %   save('hog_ytest.mat','hog_ytest')


%    csvwrite('cnn_Xtrain.csv',cnn_Xtrain)
    csvwrite('cnn_ytrain.csv',cnn_ytrain)
%    csvwrite('cnn_Xtest.csv',cnn_Xtest)
    csvwrite('cnn_ytest.csv',cnn_ytest)
    
%    csvwrite('hog_Xtrain.csv',hog_Xtrain)
    csvwrite('hog_ytrain.csv',hog_ytrain)
%    csvwrite('hog_Xtest.csv',hog_Xtest)
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

