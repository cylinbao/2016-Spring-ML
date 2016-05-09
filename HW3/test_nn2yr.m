clear all;
close all;

test_data = load('Test_data2_hw3.mat');
train_w = load('TwoHidNNR_W.mat');

%% Input setting
x = test_data.X_test;                 % x is an input data sample

%% Parameter 
%eta = 0.00001;     % learning rate
size_0 = size(x,2);           % size of layer 1 (input-layer)
size_1 = 30;            % size of layer 2 (hidden-layer)
size_2 = 15;            % size of layer 3 (output-layer)
size_3 = 4;            % size of layer 3 (output-layer)

%% Read out train result 
W_01 = train_w.W_01;  % inner-product weights between layer 0 and layer 1
W_12 = train_w.W_12;  % inner-product weights between layer 1 and layer 2
W_23 = train_w.W_23;  % inner-product weights between layer 1 and layer 2
b_01 = train_w.b_01;   % inner-product bias between layer 0 and layer 1
b_12 = train_w.b_12;   % inner-product bias between layer 1 and layer 2
b_23 = train_w.b_23;   % inner-product bias between layer 1 and layer 2

%% Forward-propagation
a_1 = InnerProduct_ForProp(x,W_01,b_01);
z_1 = Rectified_ForProp(a_1);
a_2 = InnerProduct_ForProp(z_1,W_12,b_12);
z_2 = Rectified_ForProp(a_2);
a_3 = InnerProduct_ForProp(z_2,W_23,b_23);
out = Softmax_ForProp(a_3);

%% calculate the accuracy
T_test = Do_Prediction(out);

save -append -mat "./TwoHidNN_T_test.mat" T_test;
