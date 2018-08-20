% Tutorial script to perform Nyström approximated SCCA-HSIC
clear

% generate data
n = 10000; % sample size
p = 20; % dimensionality of X
q = 20; % dimensionality of Y
numx = 2; % number of related variables from X view (2 in Y by default)
type = 4; % type of relation
[X,Y] = generate_data(n,p,q,numx,type);


%% standardise and partition into training and test sets

% standardise
X = zscore(X); Y = zscore(Y);

% partition into training and test sets
[~,indices] = partition(size(X,1), 3);
train = indices ~= 1;
test = indices == 1;
Xtrain = X(train,:); Xtest = X(test,:);
Ytrain = Y(train,:); Ytest = Y(test,:);

%% set the hyperparameters

hyperparams.M = 1; % number of components/relations
hyperparams.normtypeX = 1; % the norm on X view
hyperparams.normtypeY = 1; % the norm on Y view
hyperparams.App_para = 0.01; % percentage of inducing variables for Nyström
hyperparams.Cx = 1; % the value of the norm on X view
hyperparams.Cy = 1; % the value of the norm on Y view
hyperparams.Rep = 3; % number of restarts of the algorithm
hyperparams.eps = 1e-6; % stopping criterion
hyperparams.type1 = 'Gaussian';
hyperparams.type2 = 'Gaussian';
hyperparams.sigma1 = []; % std of Gaussian kernel on X view
hyperparams.sigma2 = []; % std of Gaussian kernel on Y view
hyperparams.r1 = []; % std of Gaussian kernel on X view
hyperparams.r2 = []; % std of Gaussian kernel on Y view
hyperparams.grad = 'minibatch'; % type of gradient
hyperparams.maxit = 30; % maximum number of iterations

%% train the scca-hsic model

[u,v,hsic_train,tempobj,InterMediate] = scca_hsic_nystrom(Xtrain,Ytrain,hyperparams);

%%
[u,v,hsic_train,tempobj,InterMediate] = scca_hsic(Xtrain,Ytrain,hyperparams);

%% test the scca-hsic model

Kxtest = gaussK(Xtest * u, 'median', []);
Kytest = centralizedK(gaussK(Ytest * v, 'median', []));
hsic_test = f(Kxtest,Kytest);

%% Example visualisations
%% Check the convergence
figure
hold on
for i = 1:hyperparams.Rep
    plot(InterMediate(i).obj,'o:','markersize',13)
end
xlabel('iterations')
ylabel('objective (hsic)')
box on
set(gca,'LineWidth',2)
set(gca,'fontsize',14,'fontweight','bold')
legend('Start 1','Start 2','Start 3','location','best')

%% Plot the transformations/projections/scores
figure
plot(Xtest * u, Ytest * v,'o')



