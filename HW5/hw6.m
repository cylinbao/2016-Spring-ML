%%%%% NCTU ML HW6 %%%%%
clear;

% randomForest(feature_size,nTrees,minSample,fracSample)
acc(1) = randomForest(10,10,1000,0.5);
acc(2) = randomForest(10,100,1000,0.5); % The basis
acc(3) = randomForest(10,1000,1000,0.5);

acc(4) = randomForest(10,100,10,0.5);
acc(5) = randomForest(10,100,100,0.5);
acc(6) = randomForest(10,100,10000,0.5);

acc(7) = randomForest(10,100,1000,0.25);
acc(8) = randomForest(10,100,1000,0.75);