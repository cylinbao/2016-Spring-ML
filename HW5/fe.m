%%%%% The Feature Extrator by One Hidden Layer Neural Network with 5 Nodes %%%%%
function [y] = fe(x,tot_class, feature_size)

addpath('./Training');
train_w = load('./Training/OneHidNN_W.mat');

%% Parameter 
size_0 = size(x,2);           % size of layer 1 (input-layer)
size_1 = feature_size;            % size of layer 2 (hidden-layer)
size_2 = tot_class;            % size of layer 3 (output-layer)

%% Read out train result 
W_01 = train_w.W_01;  % inner-product weights between layer 0 and layer 1
W_12 = train_w.W_12;  % inner-product weights between layer 1 and layer 2
b_01 = train_w.b_01;   % inner-product bias between layer 0 and layer 1
b_12 = train_w.b_12;   % inner-product bias between layer 1 and layer 2

%% Forward-propagation
a_1 = InnerProduct_ForProp(x,W_01,b_01);
y = Sigmoid_ForProp(a_1);

end
