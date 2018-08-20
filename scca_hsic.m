function [U,V,final_obj,tempobj,InterMediate] = scca_hsic(X,Y,hyperparams)


%% Input :
% X is n x dx data matrix (Assue that data matrices)
% Y is n x dy data matrix
% M is number of factor required
% normtypeX is norm used for X view 1= l1 norm (default) and 2 = l2 norm
% normtypeY is norm used for Y view 1= l1 norm  and 2 = l2 norm (default)
% Rep is number of repeats with dfferent initital point for each components
% eps convergence threshold
% type1 is type of kernel function for X view [default kernel is gaussian]
% type2 is type of kernel function for Y view
% sigma1 is parameters for kernels for X view [width or degree, if sigma1= 0 for gaussian kernel then use median, if sigma1= 0 for polynomial kernel then model use linear kernel ]
% sigma2 is parameters for kernels for Y view [width or degree, if sigma2= 0 for gaussian kernel then use median, if sigma2= 0 for polynomial kernel then model use linear kernel ]
%
%% Output:
% U and V are output CCA components
% InterMediate All intermediate results
% InterMediate(m,rep).u  contains all intermediate u for mth Canonical component and rep intial sart.
% InterMediate(m,rep).v  contains all intermediate v for mth Canonical component and rep intial sart.
% InterMEdiate(m,rep).obj contains intermediate objective values for each iteration for m th component and rep intial start.

%% Set up parameters

M = hyperparams.M;
normtypeX = hyperparams.normtypeX;
normtypeY = hyperparams.normtypeY;
Cx = hyperparams.Cx;
Cy = hyperparams.Cy;
Rep = hyperparams.Rep;
eps = hyperparams.eps;
type1 = hyperparams.type1;
sigma1 = hyperparams.sigma1;
type2 = hyperparams.type2;
sigma2 = hyperparams.sigma2;
r1 = hyperparams.r1;
r2 = hyperparams.r2;

rng(5) % fix the random number generator

if ~exist('Rep', 'var') || isempty(Rep)
    Rep = 10;
end

if ~exist('eps', 'var') || isempty(eps)
    eps = 1e-6;
end

if ~exist('normtypeX', 'var') || isempty(normtypeX)
    normtypeX = 1; % default l1 norm for X
end

if ~exist('normtypeY', 'var') || isempty(normtypeY)
    normtypeY = 2; % default l2 norm for Y
end

if ~exist('Cx', 'var') || isempty(Cx)
  Cx = 1; % default regularization constant for X
end

if ~exist('Cy', 'var') || isempty(Cy)
  Cy = 1; % default regularization constant for Y
end

if ~exist('type1', 'var') || isempty(type1)
    type1 = 'Gaussian'; % default kernel for X
    sigma1 = 0; % default kernel width for X is median
end

if ~exist('type2', 'var') || isempty(type2)
    type2 = 'Gaussian'; % default kernel width for X is median
    sigma2 = 0; % default kernel width for X is median
end

if ~exist('sigma1', 'var') || isempty(sigma1)
    switch type1
        case 'Gaussian' 
            sigma1=0;  %median
        case 'Polynomial'
            sigma1=0; % Linear kernel
        case 'Sigmoid'
            sigma1=1; 
    end
end

if ~exist('sigma2', 'var') || isempty(sigma2)
    switch type2
        case 'Gaussian' 
            sigma2=0; %median
        case 'Polynomial'
            sigma2=0; % Linear kernel
        case 'Sigmoid'
            sigma2=1; 
    end
end

if ~exist('r1', 'var') || isempty(r1)
    switch type1
        case 'Polynomial'
            r1=0; 
        case 'Sigmoid'
            r1=-1; 
    end
end

if ~exist('r2', 'var') || isempty(r2)
    switch type2
        case 'Polynomial'
            r2=0; 
        case 'Sigmoid'
            r2=-1; 
    end
end

% partition into training and validation sets
[~,indices] = partition(size(X,1), 3);
train = indices ~= 1;
test = indices == 1;
Xtrain = X(train,:); Xtest = X(test,:);
Ytrain = Y(train,:); Ytest = Y(test,:);

Xm = Xtrain;
Ym = Ytrain;
N = size(Xm,1);
dx = size(Xm,2);
dy = size(Ym,2);


maxit = 500;
if size(Xm,1) == size(Ym,1)
else
    printf('size of data matrix are not same');
end

InterMediate=[];
for m=1:M
    for rep=1:Rep
        %fprintf('Reps: #%d \n',rep);
        
        % intialization
        if normtypeX==1
         umr = projL1(rand(dx,1),Cx);
        end
        if normtypeX==2
          umr = projL2(rand(dx,1),Cx);
        end
        if normtypeY==1
            vmr = projL1(rand(dy,1),Cy);
        end
        if normtypeY==2
            vmr = projL2(rand(dy,1),Cy);
        end
        Xu = Xm * umr;
        Yv = Ym * vmr;
        switch type1
            case 'Gaussian'
                if sigma1>0
                    [Ku,au] = gaussK(Xu,'none',sigma1);
                else 
                    [Ku,au] = gaussK(Xu,'median',sigma1);
                end
            case 'Polynomial'
                if sigma1>=0
                    [Ku,au] = polyK(Xu,sigma1,r1);
                else
                    disp('For view X, the degree of polynomial kernel must be greater than or equal to 1')
                end
            case 'Sigmoid'
                %if sigma1>=0 && r1 <=0
                    [Ku,au] = tanhK(Xu,sigma1,r1);
                %else
                %    disp('For view X, sigma >=0 and r <=0 is recomended')
                %end
        end
        switch type2
            case 'Gaussian'
                if sigma2>0
                    [Kv,av] = gaussK(Yv,'none',sigma2);
                else 
                    [Kv,av] = gaussK(Yv,'median',sigma2);
                    %av
                end
            case 'Polynomial'
                if sigma2>=0
                    [Kv,av] = polyK(Yv,sigma2,r2);
                else
                    disp('For view Y, the degree of polynomial kernel must be greter than or equal to 1')
                end
                case 'Sigmoid'
                %if sigma2>=0 && r2 <=0
                    [Kv,av] = tanhK(Yv,sigma2,r2);
                %else
                %    disp('For view X, sigma >=0 and r <=0 is recomended')
                %end
        end
        cKu = centralizedK(Ku);
        cKv = centralizedK(Kv);
        diff = 999999;
        ite = 0;
        
        while diff > eps && ite < maxit
            ite = ite+1;
            %fprintf('Iter: #%d \n',ite);
            
            
            obj_old = f(Ku,cKv); % objective at current values
            %save gradvar.mat Ku cKv Xm au umr
            switch type1
                case 'Gaussian'
                    %all(isnan(Ku))
                    %all(isnan(cKv))
                   
                    gradu = gradf_gauss_SGD(Ku,cKv,Xm,au,umr); % gradient at current values
                case 'Polynomial'
                    gradu = gradf_poly_SGD(Ku,cKv,Xm,au,umr,r1);
                case 'Sigmoid'
                    gradu = gradf_tanh_SGD(Ku,cKv,Xm,au,umr);
            end
            %line search for u
            gamma = norm(gradu,2); % initial step size
            
            % line search start
            chk = 1;
            while chk == 1
                if normtypeX == 1
                    %umr
                    %gradu
                    %gamma
                    umr_new  = projL1(umr + gradu * gamma, Cx);
                end
                if normtypeX == 2
                    umr_new  = projL2(umr + gradu * gamma, Cx);
                end
                
                switch type1
                    case 'Gaussian'
                        if sigma1>0
                            [Ku_new,au_new] = gaussK(Xm*umr_new,'none',sigma1);
                        else
                            [Ku_new,au_new] = gaussK(Xm*umr_new,'median',sigma1);
                        end
                    case 'Polynomial'
                        [Ku_new,au_new] = polyK(Xm*umr_new,sigma1,r1);
                    case 'Sigmoid'
                        [Ku_new,au_new] = tanhK(Xm*umr_new,sigma1,r1);
                        

                end

                obj_new = f(Ku_new,cKv);
                
                if obj_new > obj_old + 1e-4*abs(obj_old)
                    chk = 0;
                    umr = umr_new;
                    Ku = Ku_new;
                    cKu = centralizedK(Ku);
                    au = au_new;
                    obj = obj_new;
                else
                    gamma = gamma/2;
                    if gamma <1e-6
                        chk=0;
                        
                    end
                end
            end
            obj=obj_new;
            InterMediate(m,rep).u(:,ite) = umr;
            InterMediate(m,rep).obj(2*ite-1) = obj;
            %line search end
            
            obj_old = obj;
            switch type2
                case 'Gaussian'
                        gradv = gradf_gauss_SGD(Kv,cKu,Ym,av,vmr);
                case 'Polynomial'
                        gradv = gradf_poly_SGD(Kv,cKu,Ym,av,vmr,r2);
                case 'Sigmoid'
                        gradv = gradf_tanh_SGD(Kv,cKu,Ym,av,vmr);
            end

            %line search for v
            gamma = norm(gradv,2); % initial step size
            
            % line search start   
            chk=1;
            while chk==1
                if normtypeY == 1
                    vmr_new  = projL1(vmr+ gradv*gamma,Cy);
                end
                if normtypeY == 2
                    vmr_new  = projL2(vmr+ gradv*gamma,Cy);
                end
                switch type1
                    case 'Gaussian'
                        if sigma1>0
                            [Kv_new,av_new] = gaussK(Ym*vmr_new,'none',sigma2);
                        else
                            [Kv_new,av_new] = gaussK(Ym*vmr_new,'median',sigma2);
                        end
                    case 'Polynomial'
                        [Kv_new,av_new] = polyK(Ym*vmr_new,sigma2,r2);
                    case 'Sigmoid'
                        [Kv_new,av_new] = tanhK(Ym*vmr_new,sigma2,r2);

                end
                cKv_new = centralizedK(Kv_new);
                obj_new = f(Ku,cKv_new);
                if obj_new > obj_old + 1e-4*abs(obj_old)
                    chk = 0;
                    vmr = vmr_new;
                    Kv = Kv_new;
                    cKv = cKv_new;
                    av = av_new;
                    obj = obj_new;
                else
                    gamma = gamma/2;
                    if gamma <1e-6
                        chk = 0;    
                    end
                end
            end
            obj=obj_new;
            InterMediate(m,rep).v(:,ite) = vmr;
            InterMediate(m,rep).obj(2*ite) = obj;
            
            %line search end
            %diff = abs(obj-obj_old) / (abs(obj+obj_old)/2);
            %diff = abs(obj-obj_old) / ((abs(obj)+abs(obj_old))/2);
            %disp(['iter = ',num2str(ite),', objtr = ',num2str(obj), ', diff = ', num2str(diff)])
            
            % Check here the value of test objective
            Kxtest = gaussK(Xtest * umr, 'median', []);
            Kytest = centralizedK(gaussK(Ytest * vmr, 'median', []));
            test_obj = f(Kxtest,Kytest);
            
            diff = abs(obj - obj_old) / abs(obj + obj_old);            
            
            disp(['iter = ',num2str(ite),', objtr = ',num2str(obj),...
                ', diff = ', num2str(diff), ...
                ', test = ', num2str(test_obj)])
        end
        InterMediate(m,rep).Result.u = umr;
        InterMediate(m,rep).Result.v = vmr;
        InterMediate(m,rep).Result.obj = obj;
        tempobj(rep) = obj;
    end
    
    [temp1,id] = max(tempobj);
    U(:,m) = InterMediate(m,id).Result.u;
    V(:,m) = InterMediate(m,id).Result.v;
    final_obj(m,1) = max(tempobj);
    
    % deflated data
    Xm = Xm - (U(:,m)*U(:,m)'*Xm')';
    Ym = Ym - (V(:,m)*V(:,m)'*Ym')';
    
end
end












