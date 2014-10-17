% weights_F: column vectors of weights, 1 per face image
% weights_NF: column vectors of weights, 1 per non-face image
% alpha_t: weights for respective models
% models: cell array containing structs for each model
function [alpha_t, best_stumps] = adaboost_train(Xtrain, Ytrain)
    fprintf('training adaboost...');
    N = size(Xtrain,1);
    T = 20; 
    
    alpha_t = zeros(T,1);
    best_stumps = cell(T,1);
    
    D_t = ones(N,1)*1/N;
    for t=1:T
        fprintf('%d/%d\n',t,T);
        [best_stump,ypred] = stump_train(Xtrain,Ytrain,D_t);
        best_stumps{t} = best_stump;        
        err_t = best_stump.err;
        err_t = max(1e-30, err_t); % avoid NaNs by div by 0
        alpha_t(t) = 0.5*log((1-err_t)/err_t);
        D_multiplier = exp(-alpha_t(t)*Ytrain.*ypred);
        D_t = D_t.*exp(-alpha_t(t)*Ytrain.*ypred);
        D_t = D_t/sum(D_t(:));
    end 
    fprintf('...done\n');
        
    
    