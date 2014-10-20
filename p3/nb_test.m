% ypred: column vector of prediction labels -1 or 1
function [ypred] = nb_test(model, Xtest)
    one_prior = log(model.priors(1));
    two_prior = log(model.priors(2));
    one_conditionals = log(model.one_conditionals);
    two_conditionals = log(model.two_conditionals);
    
    m = size(Xtest,1); % num examples
    ypred = ones(m,1);
    for i=1:m
        pred_one = sum(Xtest(i,:).*one_conditionals) + one_prior;
        pred_two = sum(Xtest(i,:).*two_conditionals) + two_prior;
        
        if (pred_one<pred_two)
            ypred(i) = -1;
        end
    end
end

            