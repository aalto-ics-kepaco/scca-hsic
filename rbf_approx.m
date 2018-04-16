function [phi, sigma, K_nn, K_mn, D_nn] = rbf_approx(X, ind, varargin)

n = ind; % nystrom sample
D_mn = sqrt(abs(sqdist(X',X(n,:)')));

if nargin == 2 % median heuristic
    sigma = median(D_mn(:));
else
    sigma = varargin{1};
end

K_mn = exp(- (D_mn.^2) ./ (2 * sigma.^2));

D_nn = sqrt(abs(sqdist(X(n,:)', X(n,:)')));
K_nn = exp(- (D_nn.^2) ./ (2 * sigma.^2));
K_nn = K_nn + eye(size(K_nn)) * 0.001;

phi = K_mn * inv(K_nn^(1/2));
    
     
end





