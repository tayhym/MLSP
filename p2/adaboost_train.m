% weights_F: column vectors of weights, 1 per face image
% weights_NF: column vectors of weights, 1 per non-face image
% alpha_t: weights for respective models
% models: cell array containing structs for each model
function [alpha_t, models] = adaboost_train(Xtrain, Ytrain)
    fprintf('training adaboost...');
    N = size(Xtrain,1);
    T = 500; 
    
    alpha_t = zeros(T,1);
    best_stumps = cell(T,1);
    
    D_t = ones(N,1)*1/N;
    for t=1:T
        fprintf('%d/%d\n',t,T);
        best_stump = stump_train(Xtrain,Ytrain,D_t);
        best_stumps{t} = best_stump;        
        err = best_stump.err;
        alpha_t(t) = 0.5*log((1-err_t)/err_t);
        D_t = D_t*exp(-alpha_t(t)*Ytrain.*Ypred);
        D_t = D_t/sum(D_t(:));
    end 
    fprintf('...done\n');
        
  
    