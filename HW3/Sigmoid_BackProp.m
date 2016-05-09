function [dEdx] = Sigmoid_BackProp(dEdy,x)

dEdx = x.*(1.-x).*dEdy;

end
