% ypred: 1 by m output classes using weighted models 
% models: cell array 1 by T of structs for the models
% alpha_t: 1 by T array of the weights per model
function [ypred] = adaboost_test(models, alpha_t, Xtest)
    m = size(Xtest,1);
    T = size(alpha_t,1); 
    
    ypred = ones(m,1);
    pred = zeros(m,1);
    
%     for i=1:m
%         pred = 0;
%         for j=1:T
%             pred = pred + (alpha_t(j).*nb_test(models{j},Xtest(i,:)));
%         end
%         if (pred<0)
%             ypred(i) = -1;
%         end
%     end
fprintf('Generating predictions from model...\n');
    for t=1:T 
        fprintf('%d/%d\n',t,T);
        pred = pred + alpha_t(t)*nb_test(models{t},Xtest);
    end 
    ypred(pred<0) = -1;
end 

    