clear all;
close all;

train_data = load('../Training_data_hw5.mat');

ite_limit = 1000;
theta = 0.2;
tot_class = 10;
feature_size = 10;

%% Input setting
x = train_data.X_train; % x is an input data sample
c = train_data.T_train; % t is an class
t = labelize(c, tot_class); %transfer class into bit matrix

%% Parameter 
%eta = 0.00001;     % learning rate
eta = 0.5;     % learning rate
size_0 = size(x,2);           % size of layer 1 (input-layer)
size_1 = feature_size;            % size of layer 2 (hidden-layer)
size_2 = tot_class;            % size of layer 3 (output-layer)

%% Initialization
W_01 = randn(size_0,size_1);  % inner-product weights between layer 0 and layer 1
W_12 = randn(size_1,size_2);  % inner-product weights between layer 1 and layer 2
b_01 = randn(size_1,1);       % inner-product bias between layer 0 and layer 1
b_12 = randn(size_2,1);       % inner-product bias between layer 1 and layer 2

%% Forward-propagation
a_1 = InnerProduct_ForProp(x,W_01,b_01);
z_1 = Sigmoid_ForProp(a_1);
a_2 = InnerProduct_ForProp(z_1,W_12,b_12);
out = Softmax_ForProp(a_2);

ite = 1
%% calculate the accuracy
p = Do_Prediction(out);
e = Energy_Function(out, t)
acc = Accuracy(p, t)
old_e = 0;

%pause();

while ite < ite_limit
ite = ite + 1
%% Backward-propagation
[dEda_2]                 = Softmax_BackProp(out,t);
[dEdz_1,dEdW_12,dEdb_12] = InnerProduct_BackProp(dEda_2,z_1,W_12,b_12);
[dEda_1]                 = Sigmoid_BackProp(dEdz_1,z_1);
[dEdz_0,dEdW_01,dEdb_01] = InnerProduct_BackProp(dEda_1,x,W_01,b_01);

%% Parameters Updating
W_01 = W_01-eta*dEdW_01;
W_12 = W_12-eta*dEdW_12;
b_01 = b_01-eta*dEdb_01;
b_12 = b_12-eta*dEdb_12;

%% Forward-propagation
a_1 = InnerProduct_ForProp(x,W_01,b_01);
z_1 = Sigmoid_ForProp(a_1);
a_2 = InnerProduct_ForProp(z_1,W_12,b_12);
out = Softmax_ForProp(a_2);

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

save -append -mat './OneHidNN_W.mat' W_01 W_12 b_01 b_12
