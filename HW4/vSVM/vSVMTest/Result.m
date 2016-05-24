clear;

predict = load('./vSVM_predict_n1.mat');
result_mtx(1,1) = predict.acc_linear(1);
result_mtx(2,1) = predict.acc_poly2(1);
result_mtx(3,1) = predict.acc_poly3(1);
result_mtx(4,1) = predict.acc_poly4(1);
result_mtx(5,1) = predict.acc_radial(1);

predict = load('./vSVM_predict_n25.mat');
result_mtx(1,2) = predict.acc_linear(1);
result_mtx(2,2) = predict.acc_poly2(1);
result_mtx(3,2) = predict.acc_poly3(1);
result_mtx(4,2) = predict.acc_poly4(1);
result_mtx(5,2) = predict.acc_radial(1);

predict = load('./vSVM_predict_n5.mat');
result_mtx(1,3) = predict.acc_linear(1);
result_mtx(2,3) = predict.acc_poly2(1);
result_mtx(3,3) = predict.acc_poly3(1);
result_mtx(4,3) = predict.acc_poly4(1);
result_mtx(5,3) = predict.acc_radial(1);

predict = load('./vSVM_predict_n75.mat');
result_mtx(1,4) = predict.acc_linear(1);
result_mtx(2,4) = predict.acc_poly2(1);
result_mtx(3,4) = predict.acc_poly3(1);
result_mtx(4,4) = predict.acc_poly4(1);
result_mtx(5,4) = predict.acc_radial(1);

predict = load('./vSVM_predict_n9');
result_mtx(1,5) = predict.acc_linear(1);
result_mtx(2,5) = predict.acc_poly2(1);
result_mtx(3,5) = predict.acc_poly3(1);
result_mtx(4,5) = predict.acc_poly4(1);
result_mtx(5,5) = predict.acc_radial(1);
