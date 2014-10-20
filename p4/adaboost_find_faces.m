% eigenfaces: 361 x K column vectors of the eigenfaces learnt from adaboost
%            test data
% best_stumps: decision thresholds and direction and column for each weak
%            classifier
% alpha_t:  weights to use in final classifier
function [det_mtx] = adaboost_find_faces(best_stumps,alpha_t,eigenfaces,gimage)
    N = 19;
    M = 19;
        
    X = size(gimage,1);
    Y = size(gimage,2);
    
    
    % eigenfaces of detector trained using 19x19 images
    % to detect at other scales, the original image can be scaled such that
    % faces aproximately 19x19
    patch_sums = computeAllPatchSums(gimage,[19 19]);

    patch_scores = zeros(size(gimage));
    patch_means = patch_sums./(N*M);
    fprintf('Generating predictions from weighted stumps...\n');
    delta = 2;
    tic
    for i=1:1:X-N
        for j=1:1:Y-M
%             fprintf('Querying patch %d/%d, %d/%d...\n',i,X-N,j,Y-M);
            p_im = gimage(i:i+N-1,j:j+M-1);
            p_im = p_im - patch_means(i,j);
            weights = getWeightsIncError(eigenfaces,p_im);
            xtest_patch = weights';
            face_present = adaboost_test(best_stumps,alpha_t,xtest_patch,0);

%             if (face_present==-1)
%                 patch_scores(i,j) = 0;
%             else 
%                 patch_scores(i,j) = face_present;
%             end 
            patch_scores(i,j) = face_present;
        end 
    end
    toc
%     face_present_idx = face_present<=.5*max(max(face_present));
%     patch_scores(face_present_idx) = 0;
    det_mtx = patch_scores;
end 

    