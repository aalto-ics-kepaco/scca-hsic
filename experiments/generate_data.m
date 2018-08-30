function [X,Y] = generate_data(n,p,q,numx,type)

% generate simulated relations from a uniform distribution U[0,1].
%--------------------------------------------------------------------------
% input
% n         sample size (number of observations)
% p         dimension of X (number of variables)
% q         dimension of Y (number of variables)
% numx      number of related variables in the X view
% type      type of relation:   1   linear
%                               2   quadratic                               
%                               3   cubic
%                               4   exponential
%                               5   logarithmic

X = rand(n,p);
Y = rand(n,q);

xvar = zeros(size(X,1),1);
if numx >= 2
    for k = 2:numx
        xvar = xvar + X(:,k);
    end
end

switch type
    case 1
        Y(:,1) = (X(:,1) + xvar) - Y(:,2) + normrnd(0,0.05,[n,1]);
    case 2
        Y(:,1) = (X(:,1) + xvar).^2 - Y(:,2) + normrnd(0,0.05,[n,1]);
    case 3
        Y(:,1) = (X(:,1) + xvar).^3 - Y(:,2) + normrnd(0,0.05,[n,1]);
    case 4
        Y(:,1) = exp(X(:,1) + xvar) - Y(:,2) + normrnd(0,0.05,[n,1]);
    case 5
        Y(:,1) = log(X(:,1) + xvar) - Y(:,2) + normrnd(0,0.05,[n,1]);

end






end
