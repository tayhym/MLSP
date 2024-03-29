% best_stump: struct with best stump to classify data
%           : best_stump.column - stump column number
%           : best_stump.thres  - threshold
%           : best_stump.direction - +ve 1 for >= == face
%           :                        -ve 1 for <= == face
% D_t       : m by 1 weights for each instance (how heavy to count in error)
% ypred     : m by 1 predictions used to update the weight for this stump (based on accuracy)
function [best_stump, ypred] = stump_train(Xtrain,Ytrain,D_t)
    
    m = size(Xtrain,1);
    K = size(Xtrain,2);
    best_stump = struct('column',-1,'thres',-1,'direction',-1,'err',-1);
    
    best_column = -1;
    best_thres = -1;
    best_direc = -1;
    best_err = -1;
    
    for k=1:K
        stump = Xtrain(:,k);
        stump_unsorted = stump;
        [stump,idx] = sort(stump);
        stump_ytrain = Ytrain(idx);
        D_t_corres = D_t(idx);
        
        for thres_idx=1:m-1
            threshold = (stump(thres_idx)+stump(thres_idx+1))/2;
            pred_geq = double(stump>=threshold);
            pred_geq(pred_geq==0) = -1;
            err_geq = sum(D_t_corres(pred_geq~=stump_ytrain));
            assert(numel(err_geq)==1);
            
            pred_leq = double(stump<=threshold);
            pred_leq(pred_leq==0) = -1;
            err_leq = sum(D_t_corres(pred_leq~=stump_ytrain));
            assert(numel(err_leq)==1);
                     
            if (err_geq<=err_leq)
                stump_candidate_err = err_geq;
                stump_candidate_dir = 1;
                stump_candidate_pred = double(stump_unsorted>=threshold);
                stump_candidate_pred(stump_candidate_pred==0) = -1;
            else 
                stump_candidate_err = err_leq;
                stump_candidate_dir = -1;
                stump_candidate_pred = double(stump_unsorted<=threshold);
                stump_candidate_pred(stump_candidate_pred==0) = -1;
            end 
            
            if ((best_column == -1) ||              ...
               (best_err > stump_candidate_err))
                best_column = k;
                best_thres = threshold;
                best_direc = stump_candidate_dir;
                best_err = stump_candidate_err;
                ypred = double(stump_candidate_pred);
            end         
        end
    end    
    best_stump.column = best_column;
    best_stump.thres = best_thres;
    best_stump.direction = best_direc;
    best_stump.err = best_err;
    end
    
        
            
            
            
            
    
    
