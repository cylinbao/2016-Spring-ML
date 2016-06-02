%%%%%%%%%% Random Forest %%%%%%%%%%
clear all;

data = load('./Training_data_hw5.mat');
test_data = load('./Test_data4_hw5.mat');

x = data.X_train;
x_test = test_data.X_test;
t = data.T_train;
t_test = test_data.T_test;

exFe = fe(x);

nTrees = 1000;
randomForest = TreeBagger(nTrees, exFe, t, 'FBoot', 0.5,...
                          'MinLeaf', 1000, 'Method', 'classification');

%view(randomForest.Trees{1},'Mode','graph');
exFe_test = fe(x_test);
p_test = str2double(randomForest.predict(exFe_test));
acc = Accuracy(p_test, t_test)