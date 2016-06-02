function [y] = labelize(x)

y = eye(size(x,1),10)
y = y(x,:)(:,1:max(x,1))

end
