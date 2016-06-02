function [y] = labelize(x, class_num)

y = eye(size(x,1),class_num);
y = y(x,:);

end
