%% Experiment C(III): Scalability
%

clear

% SCCA-HSIC_Nys
hyperparams.M = 1; % number of components/relations
hyperparams.normtypeX = 1; % the norm on X view
hyperparams.normtypeY = 1; % the norm on Y view
hyperparams.Cx = 1; % the value of the norm on X view
hyperparams.Cy = 1; % the value of the norm on Y view
hyperparams.Rep = 7; % number of restarts of the algorithm
hyperparams.eps = 1e-8; % stopping criterion
hyperparams.sigma1 = []; % std of Gaussian kernel on X view
hyperparams.sigma2 = []; % std of Gaussian kernel on Y view
hyperparams.maxit = 60; % maximum number of iterations
hyperparams.flag = 1;

% data dimensions
p = 20; % number of variables in view X
q = 20; % number of variables in view Y
n = [100,500,1000,5000,10000]; % sample size
props = [1, 1, 0.5, 0.1, 0.05, 0.01];

% test setting
indeps = 3; % number of independent variables in view X
func = 4; % exponential relation
repss = 1;

% preallocate
result(length(func),1).hsic_train = [];
result(length(func),1).u = [];
result(length(func),1).v = [];
result(length(func),1).hsic_test = [];
result(length(func),1).f1 = [];
result(length(func),1).time = [];

correct_v = zeros(q,1); correct_v([1,2]) = 1;

for ff = 1:length(func)
    for ll = 1:length(n)
        
        hyperparams.proportion = props(ll); % percentage of inducing variables for Nyström
        
        % ground truth
        correct_u = zeros(q,1); correct_u(1:indeps) = 1;
        
        % generate data
        rng('shuffle')
        [X,Y] = generate_data(n(ll),p,q,indeps,func);
        
        % tune hyperparameters for a random sample from this dataset
        rsamp = randsample(size(X,1), round(0.3 * size(X,1)));
        c1 = 0.5:0.5:2.5; c2 = 0.5:0.5:2.5;
        [c1_1,c2_1] = tune_hypers(X(rsamp,:),Y(rsamp,:),'scca-hsic-nystrom',3,c1,c2);
                       
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
            hsic_ground(ff,ll,rep) = f(Kxground,Kyground);
            
            % run SCCA-HSIC at the optimal hyperparameters
            hyperparams.Cx = c1_1; hyperparams.Cy = c2_1;
            tic;
            [u1,v1,hsic_train] = scca_hsic(Xtrain,Ytrain,hyperparams);
            result(ff,1).time = toc;
            
            % test hsic
            Kxtest = rbf_kernel(Xtest * u1);
            Kytest = centre_kernel(rbf_kernel(Ytest * v1));
            
            result(ff,1).hsic_train(ll,rep) = hsic_train;
            result(ff,1).u(:,ll,rep) = u1;
            result(ff,1).v(:,ll,rep) = v1;
            result(ff,1).hsic_test(ll,rep) = f(Kxtest,Kytest);
            
            f1_u = f1_score(u1,correct_u); f1_v = f1_score(v1,correct_v);
            result(ff,1).f1(ll,rep) = mean([f1_u f1_v]);
            
        end
    end
end

% averages over the repetitions
for i = 1:length(func)
    F1_mean(i,:) = mean(result(i,1).f1,2);
    HSIC_mean(i,:) = mean(result(i,1).hsic_test,2);
end








