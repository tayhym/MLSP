% ypred      : m by 1 output classes using weighted models 
% best_stumps: cell array structs containing info for the best stump 
%            : for the t-th weighted iteration 
% alpha_t    : T by 1 array of the weights per model
function [ypred] = adaboost_test(best_stumps, alpha_t, Xtest)
    m = size(Xtest,1);
    T = size(alpha_t,1); 
    
    ypred = ones(m,1);
   
    fprintf('Generating predictions from weighted stumps...\n');
    for t=1:T 
        fprintf('%d/%d\n',t,T);
        ypred = ypred + alpha_t(t)*stump_test(best_stumps{t},Xtest);
    end 
    ypred = sign(ypred);
%     ypred(ypred<0) = -1;
end 

    