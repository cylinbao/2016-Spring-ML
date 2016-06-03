%%%%% NCTU ML HW6 %%%%%
clear;

feature_size = 15;

% randomForest(feature_size,nTrees,minSample,fracSample)
[acc rf] = randomForest(feature_size,100,1000,0.5); % The basis
