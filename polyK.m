function [K,sigma] = polyK(X,sigma,r)

K = (X * X' + r ).^sigma;


end