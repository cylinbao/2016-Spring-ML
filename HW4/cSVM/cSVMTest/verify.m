clear;

pkg load image;
addpath '/home/myislin/Spring_2016/ML/libsvm/matlab';

models = load ('./cSVM_trainedModels_n100.mat');
train_data = load ('../../Training_data_hw4.mat');

T_train = train_data.T_train;
X_train = train_data.X_train;

radial_m = models.radial_m;
sv_num = radial_m.totalSV;

[pred_radial_train, acc_radial_train, est_radial] = svmpredict(T_train, X_train, radial_m, '-b 1');

sv_indices = radial_m.sv_indices;

for i=1:1:sv_num
		indx = sv_indices(i);
		verify_mtx(end+1,1) = T_train(indx,:);
		verify_mtx(end,2) = pred_radial_train(indx,:);
end

error_mtx = verify_mtx(:,1) - verify_mtx(:,2);
error_num = sum(error_mtx(:)!=0)

save -append -mat "./cSVM_predict_train.mat" pred_radial_train
