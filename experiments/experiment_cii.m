%% Experiment C(II): Increasing the Number of Noise Variables
%
%--------------------------------------------------------------------------
% Uurtio, V., Bhadra, S., Rousu, J. 
% Sparse Non-Linear CCA through Hilbert-Schmidt Independence Criterion. 
% IEEE International Conference on Data Mining (ICDM 2018)
%--------------------------------------------------------------------------

clear

hyperparams.M = 1; % number of components
hyperparams.Rep = 15; % number of random initializations
hyperparams.eps = 1e-7; % convergence limit
hyperparams.sigma1 = []; % std of rbf kernel by median trick
hyperparams.sigma2 = []; % std of rbf kernel by median trick
hyperparams.maxit = 500; % maximum number of iterations
hyperparams.flag = 2; % show only the converged result after each rep

% data dimensions
p = 10:10:40;
q = 10:10:40;
n = 300;

% setting
indeps = 3;
func = 4:5;
repss = 1;
methods = {'scca-hsic','cca-hsic'};

% preallocate
result(length(func),length(methods)).hsic_train = [];
result(length(func),length(methods)).u = [];
result(length(func),length(methods)).v = [];
result(length(func),length(methods)).hsic_test = [];
result(length(func),length(methods)).f1 = [];
result(length(func),length(methods)).ground = [];


for ff = 1:length(func)
    for ll = 1:length(p)
        for mm = 1:length(methods)
            
            % ground truth
            correct_v = zeros(q(ll),1); correct_v([1,2]) = 1;
            correct_u = zeros(p(ll),1); correct_u(1:3) = 1;
            
            % generate data
            rng('shuffle')
            [X,Y] = generate_data(n,p(ll),q(ll),3,func(ff));
            
            % tune hyperparameters for a random sample from this dataset
            rsamp = randsample(size(X,1), round(0.4 * size(X,1)));
            c1 = 0.5:0.5:2.5; c2 = 0.5:0.5:2.5;
            [c1_1,c2_1] = tune_hypers(X(rsamp,:),Y(rsamp,:),methods{mm},5,c1,c2);
            
            
            for rep = 1:repss
                % standardise and partition
                Xn = zscore(X); Yn = zscore(Y);
                [~,indices] = partition(size(X,1), 3);
                train = indices ~= 1; test = indices == 1;
                Xtrain = Xn(train,:); Xtest = Xn(test,:);
                Ytrain = Yn(train,:); Ytest = Yn(test,:);
                
                % compute ground truth
                Xground = X(test,1) + X(test,2) + X(test,3);
                Yground = Y(test,1) + Y(test,2);
                Kxground = rbf_kernel(Xground);
                Kyground = centre_kernel(rbf_kernel(Yground));
                result(ff,mm).ground(ll,rep) = f(Kxground,Kyground);
                
                hyperparams.Cx = c1_1; hyperparams.Cy = c2_1;
                if mm == 1
                    hyperparams.normtypeX = 1; % norm constraint on u
                    hyperparams.normtypeY = 1; % norm constraint on v
                elseif mm == 2
                    hyperparams.normtypeX = 2; % norm constraint on u
                    hyperparams.normtypeY = 2; % norm constraint on v
                end
                [u,v,hsic_train] = scca_hsic(Xtrain,Ytrain,hyperparams);
                Kxtest = rbf_kernel(Xtest * u);
                Kytest = centre_kernel(rbf_kernel(Ytest * v));
                
                result(ff,mm).hsic_train(ll,rep) = hsic_train;
                result(ff,mm).u{ll,rep} = u;
                result(ff,mm).v{ll,rep} = v;
                result(ff,mm).hsic_test(ll,rep) = f(Kxtest,Kytest);
                f1_u = f1_score(u,correct_u); f1_v = f1_score(v,correct_v);
                result(ff,mm).f1(ll,rep) = mean([f1_u f1_v]);
            end
        end
    end
end

% averages over the repetitions
for i = 1:length(func)
    for j = 1:length(methods)
        F1_mean(i,:,j) = mean(result(i,j).f1,2);
        HSIC_mean(i,:,j) = mean(result(i,j).hsic_test,2);
        ground_mean(i,:,j) = mean(result(i,j).ground,2);
    end
end









