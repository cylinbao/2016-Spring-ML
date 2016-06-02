function [y] = Softmax_ForProp(x)

exp_x = exp(x);
y = bsxfun(@rdivide, exp_x, sum(exp_x,2));

end
