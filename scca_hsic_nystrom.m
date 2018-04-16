function [U,V,final_obj,tempobj,InterMediate] = scca_hsic_nystrom(X,Y,hyperparams)
%% This function will solve max u'X'Yv s.t. ||u||_1 =1 and ||v||_2=1
%% Input :
% X is n x dx data matrix (Assue that data matrices)
% Y is n x dy data matrix
% M is number of factor required
% normtypeX is norm used for X view 1= l1 norm (default) and 2 = l2 norm
% normtypeY is norm used for Y view 1= l1 norm  and 2 = l2 norm (default)
% App_para is the number of columns we considered in approximated kernel
% Rep is number of repeats with dfferent initital point for each components
% eps convergence threshold
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
App_para = hyperparams.App_para;
Cx = hyperparams.Cx;
Cy = hyperparams.Cy;
Rep = hyperparams.Rep;
eps = hyperparams.eps;
sigma1 = hyperparams.sigma1;
sigma2 = hyperparams.sigma2;
grad = hyperparams.grad;
maxit = hyperparams.maxit;

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
Nnym = ceil(App_para * N);

InterMediate = [];
for m = 1:M % for every component
    for rep = 1:Rep % rep times
        fprintf('Reps: #%d \n',rep);
        
        % initialise the u and v
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
        
        % random sampling
        ind = randperm(N, Nnym);
        %[~, ind, ~] = kmedoids(Xm', Nnym);
        %ind = find_centroids(Xm,Nnym);
        
        % compute the approximated kernel
        if sigma1 > 0
            [phiu, au] = rbf_approx(Xm * umr, ind, sigma1);
        else
            [phiu, au] = rbf_approx(Xm * umr, ind);
        end
        Ku = phiu' * phiu;
                        
        if sigma2 > 0
            [phiv, av] = rbf_approx(Ym * vmr, ind, sigma2);
        else
            [phiv, av] = rbf_approx(Ym * vmr, ind);
        end
        Kv = phiv' * phiv;
                
        
        % centre the kernels
        [phicu] = centre_nystrom_kernel(phiu);
        cKu = phicu' * phicu;
        [phicv] = centre_nystrom_kernel(phiv);
        cKv = phicv' * phicv;
        
        diff = 999999;
        ite = 0;
        obj_old = f_nystrom(phicu,phicv);
                     
        while diff > eps && ite < maxit  % stopping conditions
            ite = ite + 1;
            obj = obj_old;
            
            % GRADIENT WRT U
            switch grad
                case 'minibatch'
                    gradu = gradf_gauss_SGD(Ku ,cKv ,Xm(ind,:), au ,umr);
                case 'batch'
                    gradu = gradf_gauss(Ku ,cKv ,Xm(ind,:), au ,umr);
            end
           
            % LINE SEARCH FOR U
            gamma = norm(gradu,2);
            chk = 1; 
            while chk == 1
                if normtypeX == 1
                    umr_new  = projL1(umr + gradu * gamma, Cx);
                end
                if normtypeX == 2
                    umr_new  = projL2(umr + gradu * gamma, Cx);
                end
                
                if sigma1 > 0
                    [phiu_new, au_new] = rbf_approx(Xm * umr_new, ind, sigma1);
                else
                    [phiu_new, au_new] = rbf_approx(Xm * umr_new, ind);
                end
                Ku_new = phiu_new' * phiu_new;
               
                phicu_new = centre_nystrom_kernel(phiu_new);
                cKu_new = phicu_new' * phicu_new;
                obj_new = f_nystrom(phicu_new, phicv);
                
                if obj_new > obj_old + 1e-4 * abs(obj_old)
                    chk = 0;
                    umr = umr_new;
                    Ku = Ku_new;
                    au = au_new;
                    obj = obj_new;
                    cKu = cKu_new;
                    phicu = phicu_new;
                else
                    gamma = gamma/2;
                    if gamma < 1e-6
                        chk = 0;
                    end
                end
            end
            
            obj = obj_new;
            InterMediate(m,rep).u(:,ite) = umr;
            InterMediate(m,rep).obj(2*ite-1) = obj;           
            % LINE SEARCH END
           
            obj_old = obj;
            % GRADIENT WRT V
            switch grad
                case 'minibatch'
                    gradv = gradf_gauss_SGD(Kv,cKu,Ym(ind,:),av,vmr);
                case 'batch'
                    gradv = gradf_gauss(Kv,cKu,Ym(ind,:),av,vmr);
            end
            
            % LINE SEARCH FOR V
            gamma = norm(gradv, 2);
            chk = 1;
            while chk == 1
                if normtypeY == 1
                    vmr_new  = projL1(vmr + gradv * gamma, Cx);
                end
                if normtypeY == 2
                    vmr_new  = projL2(vmr + gradv * gamma,Cy);
                end
                
                if sigma2 > 0
                    [phiv_new, av_new] = rbf_approx(Ym * vmr_new, ind, sigma2);
                else
                    [phiv_new, av_new] = rbf_approx(Ym * vmr_new, ind);
                end
                Kv_new = phiv_new' * phiv_new;
                                       
                [phicv_new] = centre_nystrom_kernel(phiv_new);
                cKv_new = phicv_new' * phicv_new;
                obj_new = f_nystrom(phicu,phicv_new);
                
                if obj_new > obj_old + 1e-4 * abs(obj_old)
                    chk = 0;
                    vmr = vmr_new;
                    Kv = Kv_new;
                    cKv = cKv_new;
                    av = av_new;
                    phicv = phicv_new;
                    obj = obj_new;
                else
                    gamma = gamma/2;
                    if gamma < 1e-6
                        chk = 0;
                    end
                end
            end
            obj = obj_new;
            InterMediate(m,rep).v(:,ite) = vmr;
            InterMediate(m,rep).obj(2*ite) = obj;
            % LINE SEARCH END
            
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
    
    [~,id] = max(tempobj);
    U(:,m) = InterMediate(m,id).Result.u;
    V(:,m) = InterMediate(m,id).Result.v;
    final_obj(m,1) = max(tempobj);
    
    % deflated data
    Xm = Xm - (U(:,m)*U(:,m)'*Xm')';
    Ym = Ym - (V(:,m)*V(:,m)'*Ym')';
    
end
end












        