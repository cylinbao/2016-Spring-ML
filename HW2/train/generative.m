clear;

data = load('../data/Training_data_hw2.mat');

# calculate how many classes: 4 in this case
num_class = size(data.T_train,2);
# 4 X 3 matrix, summation of data by each class
sum_by_class = data.T_train' * data.Phi_train;
# 1 X 4 vector, calculate how much data by each class
tot_num_class = sum(data.T_train(:,1:num_class));
# calculate the mean of each class
mean_class = sum_by_class ./ tot_num_class';
# Total number of data
tot_num_data = sum(tot_num_class);

sigma = zeros(3,3);
for i=1:1:tot_num_data
	idx = find(data.T_train(i,:)==1);
	row_vec = data.Phi_train(i,:) - mean_class(idx,:);
	sigma += row_vec' * row_vec;
endfor
sigma /= tot_num_data;

for i=1:1:num_class
	w(:,i) = inv(sigma)*mean_class(i,:)';
	w0(i) = (-1/2)*mean_class(i,:)*inv(sigma)*mean_class(i,:)' + log(1/num_class);
endfor
	
for i=1:1:tot_num_data
	for j=1:1:num_class
		a(j) = exp(data.Phi_train(i,:)*w(:,j) + w0(j));
	endfor
	predict_mtx(i,:) = all(a==max(a),1);
endfor

correct_mtx = data.T_train - predict_mtx;
correct_vec = all(correct_mtx==0,2);
correct_num = sum(correct_vec);
error_rate = (tot_num_data - correct_num)/ tot_num_data * 100;
