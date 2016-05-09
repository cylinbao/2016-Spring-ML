%% Classification with MNIST using sigmoid neural networks with 2 hidden layers
clear all;
close all;

train_data = load('Training_data_hw3.mat');

theta = 0.2;

%% Input setting
x = train_data.X_train;                 % x is an input data sample
t =  train_data.T_train;                % t is an label

%% Parameter 
%eta = 0.00001;     % learning rate
eta = 0.5;     % learning rate
size_0 = size(x,2);           % size of layer 1 (input-layer)
size_1 = 25;            % size of layer 2 (hidden-layer1)
size_2 = 10;            % size of layer 3 (hidden-layer2)
size_3 = 4;            % size of layer 4 (output-layer)

%% Initialization
W_01 = randn(size_0,size_1);  % inner-product weights between layer 0 and layer 1
W_12 = randn(size_1,size_2);  % inner-product weights between layer 1 and layer 2
W_23 = randn(size_2,size_3);  % inner-product weights between layer 2 and layer 3
b_01 = randn(size_1,1);       % inner-product bias between layer 0 and layer 1
b_12 = randn(size_2,1);       % inner-product bias between layer 1 and layer 2
b_23 = randn(size_3,1);       % inner-product bias between layer 2 and layer 3

%% Forward-propagation
a_1 = InnerProduct_ForProp(x,W_01,b_01);
z_1 = Sigmoid_ForProp(a_1);
a_2 = InnerProduct_ForProp(z_1,W_12,b_12);
z_2 = Sigmoid_ForProp(a_2);
a_3 = InnerProduct_ForProp(z_2,W_23,b_23);
out = Softmax_ForProp(a_3);

ite = 1
%% calculate the accuracy
p = Do_Prediction(out);
e = Energy_Function(out, t)
acc = Accuracy(p, t)
old_e = 0;

while 1
ite += 1
%% Backward-propagation
[dEda_3]                 = Softmax_BackProp(out,t);
[dEdz_2,dEdW_23,dEdb_23] = InnerProduct_BackProp(dEda_3,z_2,W_23,b_23);
[dEda_2]                 = Sigmoid_BackProp(dEdz_2,z_2);
[dEdz_1,dEdW_12,dEdb_12] = InnerProduct_BackProp(dEda_2,z_1,W_12,b_12);
[dEda_1]                 = Sigmoid_BackProp(dEdz_1,z_1);
[dEdz_0,dEdW_01,dEdb_01] = InnerProduct_BackProp(dEda_1,x,W_01,b_01);

%% Parameters Updating
W_01 = W_01-eta*dEdW_01;
W_12 = W_12-eta*dEdW_12;
W_23 = W_23-eta*dEdW_23;
b_01 = b_01-eta*dEdb_01;
b_12 = b_12-eta*dEdb_12;
b_23 = b_23-eta*dEdb_23;

%% Forward-propagation
a_1 = InnerProduct_ForProp(x,W_01,b_01);
z_1 = Sigmoid_ForProp(a_1);
a_2 = InnerProduct_ForProp(z_1,W_12,b_12);
z_2 = Sigmoid_ForProp(a_2);
a_3 = InnerProduct_ForProp(z_2,W_23,b_23);
out = Softmax_ForProp(a_3);

%% calculate the accuracy
p = Do_Prediction(out);
e = Energy_Function(out, t)
acc = Accuracy(p, t)

if (abs(old_e - e) > theta)
	old_e = e;
else
	break;
end

end

save -append -mat "./TwoHidNN_W.mat" W_01 W_12 W_23 b_01 b_12 b_23
