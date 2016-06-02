function [e] = Energy_Function(out,t)

e = -sum(sum((t.*log(out))));

end
