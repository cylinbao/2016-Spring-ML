clear;

pkg load image;

total = 30;

models = load ('./cSVM_trainedModels_n100.mat');
train_data = load ('../../Training_data_hw4.mat');
pred_train = load ('./cSVM_predict_train.mat');

X_train = train_data.X_train;
T_train = train_data.T_train;
P_train = pred_train.pred_radial_train;

radial_m = models.radial_m;
sv_num = radial_m.totalSV;

sv_indices = radial_m.sv_indices;

for i=1:1:sv_num
	indx = sv_indices(i);
	t_data = T_train(indx);
	p_data = P_train(indx);
	if(t_data != p_data)
		outliers(end+1,:) = indx;
		sv_indices(i) = 0;
	end
end

sv_indices = sv_indices(sv_indices!=0);

step = uint32(size(outliers,1) / total)-1;
count = 1;
figure('Name','Outliers I','Position', [0,0,1920,1080]);
disp('Outliers');
for i=1:1:total/2
	indx = outliers(count);
	subplot(5,3,i,"align");
	imshow(mat2gray(reshape(X_train(indx,:),28,28)));
	disp(sprintf('Image number: %d',i));
	Predict = P_train(indx) - 1
	Target = T_train(indx) - 1
	count += step;
end
figure('Name','Outliers II','Position', [0,0,1920,1080]);
for i=1:1:total/2
	indx = outliers(count);
	subplot(5,3,i,"align");
	imshow(mat2gray(reshape(X_train(indx,:),28,28)));
	disp(sprintf('Image number: %d',i));
	Predict = P_train(indx) - 1
	Target = T_train(indx) - 1
	count += step;
end

step = uint32(size(sv_indices,1) / total);
count = 1;
disp('Support Vectors');
figure('Name','Support Vectors I','Position', [0,0,1920,1080]);
for i=1:1:total/2
	indx = sv_indices(count);
	subplot(5,3,i,"align");
	imshow(mat2gray(reshape(X_train(indx,:),28,28)));
	Target = T_train(indx) - 1
	count += step;
end
figure('Name','Support Vectors II','Position', [0,0,1920,1080]);
for i=1:1:total/2
	indx = sv_indices(count);
	subplot(5,3,i,"align");
	imshow(mat2gray(reshape(X_train(indx,:),28,28)));
	Target = T_train(indx) - 1
	count += step;
end
pause();
