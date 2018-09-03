function [X,Y] = gendata2(n,p,q,type)

switch type
    case 1 % linear
%         X = zeros(n,p);
%         for i = 1:p
%             X(:,i) = normrnd(0,1,[n,1]);
%         end
%         
%         Y = zeros(n,q);
%         for j = 1:q
%             Y(:,j) = normrnd(0,1,[n,1]);
%         end

        X = normrnd(0,1,[n,p]);
        Y = normrnd(0,1,[n,p]);
        Y(:,1) = X(:,1) + X(:,2) - Y(:,2) + normrnd(0,0.05,[n,1]);
        
    case 2
        X = zeros(n,p);
        for i = 1:p
            X(:,i) = normrnd(0,1,[n,1]);
        end
        
        Y = zeros(n,q);
        for j = 1:q
            Y(:,j) = normrnd(0,1,[n,1]);
        end
        Y(:,1) = 1 * sin(1 *(X(:,1) + X(:,2))) - Y(:,2) + normrnd(0,0.05,[n,1]);
    
    case 3
        X = zeros(n,p);
        for i = 1:p
            X(:,i) = normrnd(0,1,[n,1]);
        end
        
        Y = zeros(n,q);
        for j = 1:q
            Y(:,j) = normrnd(0,1,[n,1]);
        end
        
        Y(:,1) = 1 ./ ( 1 * (X(:,1) + X(:,2)) ) - Y(:,2) + normrnd(0,0.05,[n,1]);
                
end
