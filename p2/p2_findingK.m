% finding optimal K value
clear all; close all;

results = zeros(2,8); % 1st row is train acc, 2nd row is test acc
for K=2:16
%---------boosting based face detector---------%
[eigenfaces,weights_F, weights_NF] = getEigenfaces(K);   % column vectors of eigenfaces
         

Xtrain = [weights_F';weights_NF'];
Ytrain = [1*ones(size(weights_F',1),1);-1*ones(size(weights_NF',1),1)];
%%
    
[alpha_t,best_stumps] = adaboost_train(Xtrain, Ytrain);
%%
[ypred] = adaboost_test(best_stumps,alpha_t,Xtrain);

accuracy = mean(ypred==Ytrain);
results(1,K) = accuracy;

%% combined testData for faces and non-faces 

[Weights_F_test,Weights_NF_test] = getTestData(eigenfaces);


Xtest = [Weights_F_test';Weights_NF_test'];
Ytest = [1*ones(size(Weights_F_test',1),1);-1*ones(size(Weights_NF_test',1),1)];
[ypred_test] = adaboost_test(best_stumps, alpha_t, Xtest);

test_accuracy = mean(Ytest==ypred_test);
results(2,K/2) = test_accuracy;
end 
