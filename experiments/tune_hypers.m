function [c1_opt,c2_opt,hsic_final] = tune_hypers(X,Y,method,repeats,a,b)

folds = 2;
fold_hsic = zeros(size(a,2),size(b,2),folds);
rep_hsic = zeros(size(a,2),size(b,2),repeats);

for rep = 1:repeats % repetitions of cv
    disp(['REP ' num2str(rep)])    
    X = zscore(X); Y = zscore(Y); % standardise
    [~,indices] = partition(size(X,1), folds); % partition
    rng('shuffle')
    
    for aa = 1:size(a,2)
        for bb = 1:size(b,2)
            
            for fold = 1:folds % cv
                train = indices ~= fold;
                test = indices == fold;
                Xtrain = X(train,:); Xtest = X(test,:);
                Ytrain = Y(train,:); Ytest = Y(test,:);
                
                switch method
                    case 'scca-hsic'
                        hyperparams.M = 1;
                        hyperparams.normtypeX = 1; 
                        hyperparams.normtypeY = 1;
                        hyperparams.Cx = a(aa); 
                        hyperparams.Cy = b(bb);
                        hyperparams.Rep = 8; 
                        hyperparams.eps = 1e-7;
                        hyperparams.sigma1 = [];
                        hyperparams.sigma2 = [];
                        hyperparams.maxit = 500;
                        hyperparams.flag = 0;
                        
                        [u,v] = scca_hsic(Xtrain,Ytrain,hyperparams);
                        
                        Kxtest = rbf_kernel(Xtest * u);
                        Kytest = centre_kernel(rbf_kernel(Ytest * v));
                        
                    case 'scca-hsic-nystrom'
                        hyperparams.M = 1; 
                        hyperparams.proportion = 0.15;
                        hyperparams.normtypeX = 1; 
                        hyperparams.normtypeY = 1;
                        hyperparams.Cx = a(aa); 
                        hyperparams.Cy = b(bb);
                        hyperparams.Rep = 5; 
                        hyperparams.eps = 1e-7;
                        hyperparams.sigma1 = [];
                        hyperparams.sigma2 = [];
                        hyperparams.maxit = 80;
                        hyperparams.flag = 0;
                        
                        [u,v] = scca_hsic_nystrom(Xtrain,Ytrain,hyperparams);
                        
                        Kxtest = rbf_kernel(Xtest * u);
                        Kytest = centre_kernel(rbf_kernel(Ytest * v));        
                end
                fold_hsic(aa,bb,fold) = f(Kxtest,Kytest);
            end
        end
    end    
    rep_hsic(:,:,rep) = mean(fold_hsic,3);   
end

hsic_final = mean(rep_hsic,3);
[~, ind] = max(hsic_final(:));
[row, col] = ind2sub(size(hsic_final),ind);
c1_opt = a(row); c2_opt = b(col);

