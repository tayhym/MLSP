clear all; close all;
%---------boosting based face detector---------%
[eigenfaces,weights_F, weights_NF] = getEigenfaces();   % column vectors of eigenfaces
         

Xtrain = [weights_F';weights_NF'];
Ytrain = [1*ones(size(weights_F',1),1);-1*ones(size(weights_NF',1),1)];
%%
    
[alpha_t,models] = adaboost_train(Xtrain, Ytrain);
%%
[ypred] = adaboost_test(models,alpha_t,Xtrain);

accuracy = mean(ypred==Ytrain);