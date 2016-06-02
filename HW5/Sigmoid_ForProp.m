function [y] = Sigmoid_ForProp(x)

y = power((1+exp(-x)),-1);

end
