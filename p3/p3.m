clear all; close all;
%---------boosting based face detector---------%
K=12;
[eigenfaces,weights_F, weights_NF] = getEigenfacesIncError(K);   % column vectors of eigenfaces
         

Xtrain = [weights_F';weights_NF'];
Ytrain = [1*ones(size(weights_F',1),1);-1*ones(size(weights_NF',1),1)];
%%
        
[alpha_t,best_stumps] = adaboost_train(Xtrain, Ytrain);
%%
[ypred] = adaboost_test(best_stumps,alpha_t,Xtrain);

accuracy = mean(ypred==Ytrain);

%% combined testData for faces and non-faces 

[Weights_F_test,Weights_NF_test] = getTestDataIncError(eigenfaces);


Xtest = [Weights_F_test';Weights_NF_test'];
Ytest = [1*ones(size(Weights_F_test',1),1);-1*ones(size(Weights_NF_test',1),1)];
[ypred_test] = adaboost_test(best_stumps, alpha_t, Xtest);

test_accuracy = mean(Ytest==ypred_test);

%% test data for faces and non-faces separately
X_test_faces = Weights_F_test';
Y_test_faces = ones(size(Weights_F_test',1),1);

X_test_nonFaces = Weights_NF_test';
Y_test_nonFaces = -1*ones(size(Weights_NF_test',1),1);

y_pred_faces = adaboost_test(best_stumps,alpha_t,X_test_faces);
acc_test_faces = mean(y_pred_faces==Y_test_faces);
y_pred_nonFaces = adaboost_test(best_stumps,alpha_t,X_test_nonFaces);
acc_test_nonFaces = mean(y_pred_nonFaces==Y_test_nonFaces);





