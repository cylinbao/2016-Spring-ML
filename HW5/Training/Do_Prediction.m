function [y] = Do_Prediction(x)

[a b] = max(x,[],2);
y = eye(size(x,1),size(x,2));
y = y(b,:);

end
