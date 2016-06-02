function [y] = Do_Prediction(x)

y = all(x==max(x,[],2),3);

end
