function [K,sigma] = tanhK(X,sigma,r)
% other wise k_ij=tanh(sigma <x_i, x_j> + r)
K = X*X';
K = tanh(sigma*K+r);

end