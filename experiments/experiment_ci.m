%% Experiment C(I): Increasing the Number of Related Variables

%--------------------------------------------------------------------------
% Uurtio, V., Bhadra, S., Rousu, J. 
% Sparse Non-Linear CCA through Hilbert-Schmidt Independence Criterion. 
% IEEE International Conference on Data Mining (ICDM 2018)
%--------------------------------------------------------------------------

clear

% Hyperparameters for SCCA-HSIC
hyperparams.M = 1; % number of components
hyperparams.Rep = 15; % number of random initializations
hyperparams.eps = 1e-7; % convergence limit
hyperparams.sigma1 = []; % std of rbf kernel by median trick
hyperparams.sigma2 = []; % std of rbf kernel by median trick
hyperparams.maxit = 500; % maximum number of iterations
hyperparams.flag = 2; % show the converged result

% data dimensions
p = 20; % number of variables in view X
q = 20; % number of variables in view Y
n = 300; % sample size

% test setting
indeps = 1:4; % number of independent variables in view X
func = 4:5; % test all five relations
repss = 3;
methods = {'scca-hsic','cca-hsic'};

% preallocate
result(length(func),length(methods)).hsic_train = [];
result(length(func),length(methods)).u = [];
result(length(func),length(methods)).v = [];
result(length(func),length(methods)).hsic_test = [];
result(length(func),length(methods)).f1 = [];
result(length(func),length(methods)).ground = [];

correct_v = zeros(q,1); correct_v([1,2]) = 1;

for ff = 1:length(func)
    for ll = 1:length(indeps)
        for mm = 1:length(methods)
            
            % ground truth
            correct_u = zeros(q,1); correct_u(1:indeps(ll)) = 1;
            
            % generate data
            rng('shuffle')
            [X,Y] = generate_data(n,p,q,indeps(ll),func(ff));
            
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
                
                xvar = zeros(size(Xtest,1),1);
                if indeps(ll) >= 2
                    for k = 2:indeps(ll)
                        xvar = xvar + X(test,k);
                    end
                end
                
                Xground = X(test,1) + xvar;
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
                result(ff,mm).u(:,ll,rep) = u;
                result(ff,mm).v(:,ll,rep) = v;
                result(ff,mm).hsic_test(ll,rep) = f(Kxtest,Kytest);
                f1_u = f1_score(u,correct_u); f1_v = f1_score(v,correct_v);
                result(ff,mm).f1(ll,rep) = mean([f1_u f1_v]);
            end
            
        end
    end
end

%% averages over the repetitions
for i = 1:length(func)
    for j = 1:length(methods)
        F1_mean(i,:,j) = mean(result(i,j).f1,2);
        HSIC_mean(i,:,j) = mean(result(i,j).hsic_test,2);
        ground_mean(i,:,j) = mean(result(i,j).ground,2);
    end
end

%% visualise

marks = 's:';
figure
subplot(121)
hold on
h1 = plot(mean(mean(hsic_ground,3)),'k--');
h2 = errorbar(mean(HSIC_mean),std(HSIC_mean),marks,'MarkerSize',20,'MarkerEdgeColor','auto','MarkerFaceColor','none','linewidth',2);
set(gca,'xtick',1:4,'xticklabel',[3,4,5,6],'fontweight','bold','fontsize',16)
xlabel('Related Variables')
ylabel('Test HSIC')
box on
axis square
ylim([0 0.1])

set(findobj(gca,'type','line'),'linew',2)
set(gca,'linew',2)

subplot(122)
hold on
errorbar(mean(F1_mean),std(F1_mean),marks,'MarkerSize',15,'MarkerEdgeColor','auto','MarkerFaceColor','none',...
    'linewidth',2)
ylim([0 1])
set(gca,'xtick',1:4,'xticklabel',[3,4,5,6],'fontweight','bold','fontsize',16)
xlabel('Related Variables')
ylabel('F1')
box on
set(findobj(gca,'type','line'),'linew',2)
set(gca,'linew',2)
axis square








