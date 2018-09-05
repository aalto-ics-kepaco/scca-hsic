%% Experiment B: Finding Non-Linear Relations

%--------------------------------------------------------------------------
% Uurtio, V., Bhadra, S., Rousu, J. 
% Sparse Non-Linear CCA through Hilbert-Schmidt Independence Criterion. 
% IEEE International Conference on Data Mining (ICDM 2018)
%--------------------------------------------------------------------------

clear
% HYPERPARAMETERS
% SCCA-HSIC 
hyperparams.M = 1;
hyperparams.normtypeX = 1;
hyperparams.normtypeY = 1;
hyperparams.Rep = 15;
hyperparams.eps = 1e-7;
hyperparams.sigma1 = [];
hyperparams.sigma2 = [];
hyperparams.maxit = 500;
hyperparams.flag = 2;

% CCA-HSIC
hyperparams2.M = 1;
hyperparams2.normtypeX = 2;
hyperparams2.normtypeY = 2;
hyperparams2.Rep = 15;
hyperparams2.eps = 1e-7;
hyperparams2.sigma1 = [];
hyperparams2.sigma2 = [];
hyperparams2.maxit = 500;
hyperparams2.flag = 2;

% data dimensions
p = 20;
q = 20;
n = 300;

methodnr = 2;
func = 1:3;

result(methodnr).cx = [];
result(methodnr).cy = [];
result(methodnr).u = [];
result(methodnr).v = [];
result(methodnr).xscore = [];
result(methodnr).yscore = [];
result(methodnr).hsic_test = [];

%%
for ff = 1:length(func)  
    rng('shuffle')
    
    [X,Y] = gendata2(n,p,q,func(ff));
    Xn = X; Yn = Y;
    [~,indices] = partition(size(X,1), 3);
    train = indices ~= 1; test = indices == 1;
    
    Xtrain = Xn(train,:); Xtest = Xn(test,:);
    Ytrain = Yn(train,:); Ytest = Yn(test,:);    
    Xground(:,ff) = X(test,1) + X(test,2); 
    Yground(:,ff) = Y(test,1) + Y(test,2);     
    Kxground = rbf_kernel(Xground(:,ff));
    Kyground = centre_kernel(rbf_kernel(Yground(:,ff)));
    hsic_ground(ff) = f(Kxground,Kyground);
    
    % SCCA-HSIC
    c1 = 0.5:0.5:3; c2 = 0.5:0.5:3;
    [c1_1,c2_1,HSIC] = tune_hypers(Xtrain,Ytrain,'scca-hsic',3,c1,c2);
    hyperparams.Cx = c1_1; hyperparams.Cy = c2_1;
    
    [U1,V1,final_obj,tempobj,InterMediate] = scca_hsic(Xtrain,Ytrain,hyperparams);
    
    Kxtest = rbf_kernel(Xtest * U1);
    Kytest = centre_kernel(rbf_kernel(Ytest * V1));
    
    result(1).cx(ff) = c1_1;
    result(1).cy(ff) = c2_1;
    result(1).u(:,ff) = U1;
    result(1).v(:,ff) = V1;
    result(1).xscore(:,ff) = Xtest * U1;
    result(1).yscore(:,ff) = Ytest * V1;
    result(1).hsic_test(ff) = f(Kxtest,Kytest);
    
    disp(['SCCA-HSIC ' num2str(f(Kxtest,Kytest),2) ])
         
    % CCA-HSIC
    c1 = 0.5:0.5:3; c2 = 0.5:0.5:3;
    [c1_3,c2_3] = tune_hypers(Xtrain,Ytrain,'cca-hsic',3,c1,c2);
    hyperparams2.Cx = c1_3; hyperparams2.Cy = c2_3;
    
    [U3,V3] = scca_hsic(Xtrain,Ytrain,hyperparams2);
    
    Kxtest = rbf_kernel(Xtest * U3);
    Kytest = centre_kernel(rbf_kernel(Ytest * V3));
    
    result(2).cx(ff) = c1_3;
    result(2).cy(ff) = c2_3;
    result(2).u(:,ff) = U3;
    result(2).v(:,ff) = V3;
    result(2).xscore(:,ff) = Xtest * U3;
    result(2).yscore(:,ff) = Ytest * V3;
    result(2).hsic_test(ff) = f(Kxtest,Kytest);
    
    disp(['CCA-HSIC ' num2str(f(Kxtest,Kytest),2)])
    
    
end


%%





