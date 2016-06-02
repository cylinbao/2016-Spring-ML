function [dEdx,dEdW,dEdb] = InnerProduct_BackProp(dEdy,x,W,b)

dEdx = W*dEdy';
dEdx = dEdx';

dEdb = dEdy*b;
dEdb = sum(dEdb) / size(dEdb,1);

dEdW = x'*dEdy / size(dEdy,1);
end
