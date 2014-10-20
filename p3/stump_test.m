% stump_attr : stump attributes
% Xtest      : m by n data to make a prediction on 
function [ypred] = stump_test(stump_attr,Xtest)
    decision_column = Xtest(:,stump_attr.column);
    if (stump_attr.direction == 1)
        ypred = double(decision_column >= stump_attr.thres);
        ypred(ypred==0) = -1;
    else 
        ypred = double(decision_column <= stump_attr.thres);
        ypred(ypred==0) = -1;
    end 
end

