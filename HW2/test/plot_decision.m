clear;

data = load('../data/Test_data1_hw2.mat');
train_result = load('../train/train_result.mat');

% calculate how many classes: 4 in this case
num_class = 4;
% Total number of data
tot_num_data = size(data.Phi_test,1);

boundary_precision = 0.565;

w = train_result.w_g;
w0 = train_result.w0_g;
	
dis_mtx = [1 -1 0 0;1 0 -1 0;1 0 0 -1;0 1 -1 0;0 1 0 -1;0 0 1 -1];

for x1=-1:0.05:1
	for x2=-1:0.05:1
		for x3=-1:0.05:1
			for j=1:1:num_class
				a(j) = exp([x1 x2 x3]*w(:,j) + w0(j));
			end
			a ./= sum(a);
			dis = a*dis_mtx';
			T = all(a==max(a),1);
			max(abs(dis));
			if max(abs(dis)) < boundary_precision
				bound(end+1,:) = [x1 x2 x3];
			elseif T == [1 0 0 0]
				class_1(end+1,:) = [x1 x2 x3];
			elseif T == [0 1 0 0]
				class_2(end+1,:) = [x1 x2 x3];
			elseif T == [0 0 1 0]
				class_3(end+1,:) = [x1 x2 x3];
			else
				class_4(end+1,:) = [x1 x2 x3];
			end
		end
	end
end

figure(1)
check = exist('bound');
if check
	scatter3(bound(:,1), bound(:,2), bound(:,3), [], [0 0 0],'filled')
	hold on
end
check = exist('class_1');
if check
	scatter3(class_1(:,1), class_1(:,2), class_1(:,3), [], [1 0 0],'filled')
	hold on
end
check = exist('class_2');
if check
	scatter3(class_2(:,1), class_2(:,2), class_2(:,3), [], [0 1 0],'filled')
	hold on
end
check = exist('class_3');
if check
	scatter3(class_3(:,1), class_3(:,2), class_3(:,3), [], [0 0 1],'filled')
	hold on
end
check = exist('class_4');
if check
	scatter3(class_4(:,1), class_4(:,2), class_4(:,3), [], [1 1 0],'filled')
end

w_d = train_result.w_d;
phi = data.Phi_test;
phi(:,end+1) = 1;
dim_phi = size(phi,2);

% transfer w_d to 4x4 matrix w                                                   
for i=1:1:num_class
	idx = (i-1)*dim_phi+1;
	w_t(:,i) = w_d(idx:idx+dim_phi-1,1);
end 

for x1=-1:0.05:1
	for x2=-1:0.05:1
		for x3=-1:0.05:1
			for j=1:1:num_class
				a(j) = exp([x1 x2 x3 1]*w_t(:,j));
			end
			a ./= sum(a);
			dis = a*dis_mtx';
			T = all(a==max(a),1);
			if max(abs(dis)) < boundary_precision
				bound_d(end+1,:) = [x1 x2 x3];
			elseif T == [1 0 0 0]
				class_1d(end+1,:) = [x1 x2 x3];
			elseif T == [0 1 0 0]
				class_2d(end+1,:) = [x1 x2 x3];
			elseif T == [0 0 1 0]
				class_3d(end+1,:) = [x1 x2 x3];
			else
				class_4d(end+1,:) = [x1 x2 x3];
			end
		end
	end
end


figure(2)
check = exist('bound_d');
if check
	scatter3(bound_d(:,1), bound_d(:,2), bound_d(:,3), [], [0 0 0],'filled')
	hold on
end
check = exist('class_1d');
if check
	scatter3(class_1d(:,1), class_1d(:,2), class_1d(:,3), [], [1 0 0],'filled')
	hold on
end
check = exist('class_2d');
if check
	scatter3(class_2d(:,1), class_2d(:,2), class_2d(:,3), [], [0 1 0],'filled')
	hold on
end
check = exist('class_3d');
if check
	scatter3(class_3d(:,1), class_3d(:,2), class_3d(:,3), [], [0 0 1],'filled')
	hold on
end
check = exist('class_4d');
if check
	scatter3(class_4d(:,1), class_4d(:,2), class_4d(:,3), [], [1 1 0],'filled')
end
