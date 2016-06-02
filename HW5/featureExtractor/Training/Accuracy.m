function [y] = Accuracy(p, t)

c_mtx = t - p;                                        
c_num = sum(all(c_mtx==0,2));                                                  
y= c_num / size(t,1) * 100;

end
