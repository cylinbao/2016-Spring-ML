clear;

data = load('../data/Training_data_hw2.mat');

# calculate how many classes: 4 in this case
num_class = size(data.unT_train,2);
# 4 X 3 matrix, summation of data by each class
sum_by_class = data.unT_train' * data.unPhi_train;
# 1 X 4 vector, calculate how much data by each class
tot_num_class = sum(data.unT_train(:,1:num_class));
# calculate the mean of each class
mean_class = sum_by_class ./ tot_num_class';
# Total number of data
tot_num_data = sum(tot_num_class);

sigma = zeros(3,3);
for i=1:1:tot_num_data
	idx = find(data.unT_train(i,:)==1);
	row_vec = data.unPhi_train(i,:) - mean_class(idx,:);
	sigma += row_vec' * row_vec;
endfor
sigma /= tot_num_data;

for i=1:1:num_class
	w_g(:,i) = inv(sigma)*mean_class(i,:)';
	w0_g(i) = (-1/2)*mean_class(i,:)*inv(sigma)*mean_class(i,:)' + log(1/num_class);
endfor
	
for i=1:1:tot_num_data
	for j=1:1:num_class
		a(j) = exp(data.unPhi_train(i,:)*w_g(:,j) + w0_g(j));
	endfor
	predict_mtx(i,:) = all(a==max(a),1);
endfor

correct_mtx = data.unT_train - predict_mtx;
correct_vec = all(correct_mtx==0,2);
correct_num = sum(correct_vec);
error_rate = (tot_num_data - correct_num)/ tot_num_data * 100

class_1 = predict_mtx(:,1) .* data.unPhi_train;
class_1(all(class_1==0,2),:) = [];
scatter3(class_1(:,1), class_1(:,2), class_1(:,3), [], [1 0 0], 'filled')
hold on
class_2 = predict_mtx(:,2) .* data.unPhi_train;
class_2(all(class_2==0,2),:) = [];
scatter3(class_2(:,1), class_2(:,2), class_2(:,3), [], [0 1 0], 'filled')
hold on
class_3 = predict_mtx(:,3) .* data.unPhi_train;
class_3(all(class_3==0,2),:) = [];
scatter3(class_3(:,1), class_3(:,2), class_3(:,3), [], [0 0 1], 'filled')
hold on
class_4 = predict_mtx(:,4) .* data.unPhi_train;
class_4(all(class_4==0,2),:) = [];
scatter3(class_4(:,1), class_4(:,2), class_4(:,3), [], [1 1 0], 'filled')

save -append -mat "./train_result.mat" w_g w0_g;
