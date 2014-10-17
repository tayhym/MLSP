% stump_attr : stump attributes
% Xtest      : m by n data to make a prediction on 
function [ypred] = stump_test(stump_attr,Xtest)
    decision_column = Xtest(:,stump_attr.column);
    if (stump_attr.direction == 1)
        ypred = decision_column >= stump_attr.thres;
    else 
        ypred = decision_column <= stump_attr.thres;
    end 
end

