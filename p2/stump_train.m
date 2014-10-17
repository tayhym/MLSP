% best_stump: struct with best stump to classify data
%           : best_stump.column - stump column number
%           : best_stump.thres  - threshold
%           : best_stump.direction - +ve 1 for >= == face
%           :                        -ve 1 for <= == face
% D_t       : m by 1 weights for each instance (how heavy to count in error)
function [best_stump] = stump_train(Xtrain,Ytrain,D_t)
    
    m = size(Xtrain,1);
    K = size(Xtrain,2);
    best_stump = struct('column',-1,'thres',-1,'direction',-1,'err',-1);
    
    best_column = -1;
    best_thres = -1;
    best_direc = -1;
    best_err = -1;
    
    for k=1:K
        stump = Xtrain(:,k);
        [stump,idx] = sort(stump);
        stump_ytrain = Ytrain(idx);
        
        for thres_idx=1:m
            threshold = stump(thres_idx);
            pred_geq = (stump>=threshold);
            pred_geq(pred_geq==0) = -1;
            err_geq = sum(D_t(pred_geq~=stump_ytrain));
            assert(numel(err_geq)==1);
            
            pred_leq = (stump<=threshold);
            pred_leq(pred_leq==0) = -1;
            err_leq = sum(D_t(pred_leq~=stump_ytrain));
            assert(numel(err_leq)==1);
                     
            if (err_geq<=err_leq)
                stump_candidate_err = err_geq;
                stump_candidate_dir = 1;
            else 
                stump_candidate_err = err_leq;
                stump_candidate_dir = -1;
            end 
            
            if ((best_column == -1) ||              ...
               (best_err > stump_candidate_err))
                best_column = k;
                best_thres = threshold;
                best_dir = stump_candidate_dir;
                best_err = stump_candidate_err;
            end         
        end
    end 
    
   best_stump = struct('column',-1,'thres',-1,'direction',-1,'err',-1);

    best_stump.column = best_column;
    best_stump.thres = best_thres;
    best_stump.dir = best_dir;
    best_stump.err = best_err;
    end
    
        
            
            
            
            
    
    
