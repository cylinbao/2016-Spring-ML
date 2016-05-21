clear;

addpath '../libsvm/matlaba';
load 'Training_data_hw4.mat';

model = svmtrain(T_train, X_train);
