%%%%%%%%%% Random Forest %%%%%%%%%%
function [acc rf] = randomForest(feature_size,nTrees,minSample,fracSample)

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
t = data.T_train;
x_test = test_data.X_test;
t_test = test_data.T_test;

% pre-process data with feature extractor
%exFe = fe(x,tot_class,feature_size);
%exFe_test = fe(x_test,tot_class,feature_size);

% train by whole features
rf = TreeBagger(nTrees, x, t, 'FBoot', fracSample,...
                          'MinLeaf', minSample, 'Method', 'classification');
% train by extracted features
%rf = TreeBagger(nTrees, exFe, t, 'FBoot', fracSample,...
%                          'MinLeaf', minSample, 'Method', 'classification');

% Deal with certain trees
view(rf.Trees{25},'Mode','graph');
view(rf.Trees{50},'Mode','graph');
view(rf.Trees{75},'Mode','graph');

tree_p_test25 = str2double(rf.Trees{25}.predict(x_test));
tree_p_test50 = str2double(rf.Trees{50}.predict(x_test));
tree_p_test75 = str2double(rf.Trees{75}.predict(x_test));

%tree_p_test25 = str2double(rf.Trees{25}.predict(exFe_test));
%tree_p_test50 = str2double(rf.Trees{50}.predict(exFe_test));
%tree_p_test75 = str2double(rf.Trees{75}.predict(exFe_test));

acc_tree25 = Accuracy(tree_p_test25, t_test)
acc_tree50 = Accuracy(tree_p_test50, t_test)
acc_tree75 = Accuracy(tree_p_test75, t_test)

% predict on test data by random forest and compute accuracy
% predict from whole data
p_test = str2double(rf.predict(x_test));

% predict from extracted features
%p_test = str2double(rf.predict(exFe_test));
acc = Accuracy(p_test, t_test)

end