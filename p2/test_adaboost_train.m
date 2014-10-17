% % test adaboost_train
% Xtrain = [-0.5; 0.1; 0.2; 0.3];
% Ytrain = [0;1;1;1];
%  
% [alpha_t,best_stumps] = adaboost_train(Xtrain,Ytrain);
% assert(best_stumps{1}.thres == -0.2);
% assert(best_stumps{1}.direction == 1);
% 
% 
% % test adaboost_train neg direction 
% Xtrain = [-0.5; 0.1; 0.2; 0.3];
% Ytrain = [1;1;1;0];
% 
% [alpha_t,best_stumps] = adaboost_train(Xtrain,Ytrain);
% assert(best_stumps{1}.thres == 0.25);
% assert(best_stumps{1}.direction == -1);


% test adaboost_train neg direction 
Xtrain = [ 0.1; 0.2; 0.3;-0.5];
Ytrain = [0;1;1;1];

[alpha_t,best_stumps] = adaboost_train(Xtrain,Ytrain);
assert(best_stumps{1}.thres == 0.25);
assert(best_stumps{1}.direction == -1);