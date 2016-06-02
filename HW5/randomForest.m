%%%%%%%%%% Random Forest %%%%%%%%%%
function [acc] = randomForest(feature_size,nTrees,minSample,fracSample)

% parameters for feature extrator
tot_class = 10;
%feature_size = 5;
% parameters for random forest
%nTrees = 100;
%minSample = 1000;
%fracSample = 0.5;

data = load('./Training_data_hw5.mat');
test_data = load('./Test_data4_hw5.mat');

x = data.X_train;
x_test = test_data.X_test;
t = data.T_train;
t_test = test_data.T_test;

exFe = fe(x,tot_class,feature_size);

rf = TreeBagger(nTrees, exFe, t, 'FBoot', fracSample,...
                          'MinLeaf', minSample, 'Method', 'classification');

%view(randomForest.Trees{1},'Mode','graph');
exFe_test = fe(x_test,tot_class,feature_size);
p_test = str2double(rf.predict(exFe_test));
acc = Accuracy(p_test, t_test);

end