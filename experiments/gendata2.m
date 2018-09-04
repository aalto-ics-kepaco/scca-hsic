function [X,Y] = gendata2(n,p,q,type)


X = normrnd(0,1,[n,p]);
Y = normrnd(0,1,[n,q]);

switch type
    case 1 % linear
        Y(:,1) = X(:,1) + X(:,2) - Y(:,2) + normrnd(0,0.05,[n,1]);
        
    case 2 % sinusoidal
        Y(:,1) = 1 * sin(1 *(X(:,1) + X(:,2))) - Y(:,2) + normrnd(0,0.05,[n,1]);
    
    case 3 % hyperbolic      
        Y(:,1) = 1 ./ ( 1 * (X(:,1) + X(:,2)) ) - Y(:,2) + normrnd(0,0.05,[n,1]);               
end
