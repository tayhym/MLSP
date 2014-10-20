% % test adaboost_train
 
%% finding threshold
Xtrain = [-0.5; 0.1; 0.2; 0.3];
Ytrain = [0;1;1;1];
 
[alpha_t,best_stumps] = adaboost_train(Xtrain,Ytrain);
assert(best_stumps{1}.thres == -0.2);
assert(best_stumps{1}.direction == 1);
assert(best_stumps{1}.column==1);


%% test adaboost_train neg direction 
Xtrain = [-0.5; 0.1; 0.2; 0.3];
Ytrain = [1;1;1;-1];
 
[alpha_t,best_stumps] = adaboost_train(Xtrain,Ytrain);
assert(best_stumps{1}.thres == 0.25); 
assert(best_stumps{1}.direction == -1);


%% test adaboost_train neg direction 
Xtrain = [ 0.1; 0.2; 0.3;-0.5];
Ytrain = [-1;1;1;1];


[alpha_t,best_stumps] = adaboost_train(Xtrain,Ytrain);
assert(best_stumps{1}.thres - 0.15<1e-10);
assert(best_stumps{1}.direction == 1);

Xtrain = [0.3; 0.1; 0.2; -0.5];
Ytrain = [-1;1;1;1];
[alpha_t,best_stumps] = adaboost_train(Xtrain,Ytrain);
assert(best_stumps{1}.thres == 0.25);
assert(best_stumps{1}.direction == -1);

%% test update of D_t
Xtrain = [0.3 -0.6;
          0.5 -0.5;
          0.7 -0.1;
          0.6 -0.4;
          0.2 0.4;
          -0.8 -0.1;
          0.4 -0.9;
          0.2 0.5];
Ytrain = [1;1;1;1;-1;-1;-1;-1];

[alpha_t, best_stumps] = adaboost_train(Xtrain, Ytrain);
assert(best_stumps{1}.thres==0.45 || best_stumps{1}.thres==0.25);
assert(best_stumps{1}.direction==1);
assert(best_stumps{1}.err == (1/8));
assert(alpha_t(1)-0.97 <0.1);

assert(best_stumps{2}.thres==0.15 ||best_stumps{2}.thres==0.45);
assert(best_stumps{1}.column==1);
assert(best_stumps{2}.column==1);

%% 

[ypred] = adaboost_test(best_stumps,alpha_t,Xtrain);
 
 
 