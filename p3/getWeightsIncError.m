% column vector of eigenfaces
% weights_F: column vector of eigenface weights for each face image, 
%          : and a column of normalized error in representation
% weights_NF: column vector of eigenface weights for each non-face image,
%           : including normalied error
% K: number of eigenface vectors
function [weights_F] = getWeightsIncError(eigenfaces,p_im)
    
    weights_F = eigenfaces'*p_im;
    diff_error = eigenfaces*weights_F - p_im;
    N = size(diff_error,1);
    diff_feature = zeros(1,size(diff_error,2));
    for i=1:size(diff_error,2)
        diff_feature(i) = (1/N)*norm(diff_error(:,i))^2;
        assert(numel(norm(diff_error(:,i)))==1);
        assert(numel(diff_error(:,i))==N);
    end 
    weights_F(end+1,:) = diff_feature;
    
end
