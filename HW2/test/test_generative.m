clear;

data = load('../data/Test_data1_hw2.mat');
train_result = load('../train/train_result.mat');

# calculate how many classes: 4 in this case
num_class = 4;
# Total number of data
tot_num_data = size(data.Phi_test,1);

w = train_result.w_g;
w0 = train_result.w0_g;
	
for i=1:1:tot_num_data
	for j=1:1:num_class
		a(j) = exp(data.Phi_test(i,:)*w(:,j) + w0(j));
	endfor
	T_test(i,:) = all(a==max(a),1);
endfor

save -append -mat "./GenMod_T_test.mat" T_test;
