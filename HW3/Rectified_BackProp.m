function [dEdx] = Rectified_BackProp(dEdy,x)

dEdx = (x > 0).*dEdy;

end
