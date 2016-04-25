clear;

data = load('../data/Test_data1_hw2.mat');
train_result = load('../train/train_result.mat');

% calculate how many classes: 4 in this case
num_class = 4;
% Total number of data
tot_num_data = size(data.Phi_test,1);

w = train_result.w_g;
w0 = train_result.w0_g;
	
% compute test result for generative model
for i=1:1:tot_num_data
	for j=1:1:num_class
		a(j) = exp(data.Phi_test(i,:)*w(:,j) + w0(j));
	endfor
	T_test(i,:) = all(a==max(a),1);
endfor

save -append -mat "./GenMod_T_test.mat" T_test;

w_d = train_result.w_d;
phi = data.Phi_test;
phi(:,end+1) = 1;
dim_phi = size(phi,2);

% transfer w_d to 4x4 matrix w                                                   
for i=1:1:num_class                                                              
	idx = (i-1)*dim_phi+1;                                                       
	w_t(:,i) = w_d(idx:idx+dim_phi-1,1);                                             
end 

% compute test result for discriminative model
for i=1:1:tot_num_data
	for j=1:1:num_class
		a(j) = exp(phi(i,:)*w_t(:,j));
	endfor
	T_test(i,:) = all(a==max(a),1);
endfor

save -append -mat "./DisMod_T_test.mat" T_test;
