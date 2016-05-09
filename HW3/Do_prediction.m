function [y] = Do_prediction(x)

y = all(x==max(x,[],2),3);

end
