function [y] = InnerProduct_ForProp(x,W,b)

y = bsxfun(@plus, x*W, b');

end
