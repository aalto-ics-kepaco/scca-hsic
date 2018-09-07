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
indeps = 3; % number of related variables in view X
func = 1:5; % test all five relations
repss = 10; % number of repetitions
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
            c1 = 0.5:0.5:3; c2 = 0.5:0.5:3;
            [c1_1,c2_1] = tune_hypers(X(rsamp,:),Y(rsamp,:),methods{mm},3,c1,c2);
            
            
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

%% averages over the repetitions and functions
for jj = 1:length(methods)
    F1_mean(:,jj) = mean([result(1:length(func),jj).f1],2);
    F1_std(:,jj) = std([result(1:length(func),jj).f1],0,2);
    HSIC_mean(:,jj) = mean([result(1:length(func),jj).hsic_test],2);
    HSIC_std(:,jj) = std([result(1:length(func),jj).hsic_test],0,2);
    ground_mean(:,jj) = mean([result(1:length(func),jj).ground],2);
end

%% visualise
marks = 's:';
figure
subplot(121)
hold on
h1 = plot(mean(ground_mean,2),'k--');
h2 = errorbar(HSIC_mean,HSIC_std,marks,'MarkerSize',20,'MarkerEdgeColor','auto','MarkerFaceColor','none','linewidth',2);
set(gca,'xtick',1:4,'xticklabel',[3,4,5,6],'fontweight','bold','fontsize',16)
xlabel('Noise Variables')
ylabel('Test HSIC')
box on
axis square
ylim([0 0.1])

set(findobj(gca,'type','line'),'linew',2)
set(gca,'linew',2)

[l,b] = legend({'Ground Truth','SCCA-HSIC','CCA-HSIC'},...
   'location','eastoutside',...
   'orientation','horizontal','fontsize',17,'fontname','times');
set(findobj(b,'-property','MarkerSize'),'MarkerSize',25)
hl = findobj(b,'type','line');
set(hl,'LineWidth',2);
legend boxoff
set(l,'Position', [0.27 0.75 0.45 0.2], 'Units', 'normalized');

subplot(122)
hold on
errorbar(F1_mean,F1_std,marks,'MarkerSize',15,'MarkerEdgeColor','auto','MarkerFaceColor','none',...
    'linewidth',2)
ylim([0 1])
set(gca,'xtick',1:4,'xticklabel',[3,4,5,6],'fontweight','bold','fontsize',16)
xlabel('Noise Variables')
ylabel('F1')
box on
set(findobj(gca,'type','line'),'linew',2)
set(gca,'linew',2)
axis square










