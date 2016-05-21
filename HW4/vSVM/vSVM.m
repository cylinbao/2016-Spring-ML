clear;

addpath '/home/myislin/Spring_2016/ML/libsvm/matlab';
load '/home/myislin/Spring_2016/ML/HW4/Training_data_hw4.mat';
load '/home/myislin/Spring_2016/ML/HW4/Test_data1_hw4.mat';

linear_m = svmtrain(T_train, X_train, '-s 1 -t 0 -c 1 -n 0.5 -b 1');
poly2_m = svmtrain(T_train, X_train, '-s 1 -t 1 -d 2 -g 0.0012755 -r 0 -c 1 -n 0.5 -b 1');
poly3_m = svmtrain(T_train, X_train, '-s 1 -t 1 -d 3 -g 0.0012755 -r 0 -c 1 -n 0.5 -b 1');
poly4_m = svmtrain(T_train, X_train, '-s 1 -t 1 -d 4 -g 0.0012755 -r 0 -c 1 -n 0.5 -b 1');
radial_m = svmtrain(T_train, X_train, '-s 1 -t 2 -g 0.0012755 -c 1 -n 0.5 -b 1');

[pred_linear, acc_linear, est_linear] = svmpredict(T_test, X_test, linear_m, '-b 1');
[pred_poly2, acc_poly2, est_poly2] = svmpredict(T_test, X_test, poly2_m, '-b 1');
[pred_poly3, acc_poly3, est_poly3] = svmpredict(T_test, X_test, poly3_m, '-b 1');
[pred_poly4, acc_poly4, est_poly4] = svmpredict(T_test, X_test, poly4_m, '-b 1');
[pred_radial, acc_radial, est_radial] = svmpredict(T_test, X_test, radial_m, '-b 1');

save -append -mat "./vSVM_trainedModels.mat" linear_m poly2_m poly3_m poly4_m radial_m
save -append -mat "./vSVM_predict.mat" acc_linear acc_poly2 acc_poly3 acc_poly4 acc_radial
