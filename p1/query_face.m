% check if face is present, by dot product between patch_image 
% and normalized eigen_face. 
function [face_score] = query_face(p_im, eigenface,patch_mean,patch_means,sumE)
    %sumE=sum(sum(eigenface(:)));
    
    % method1: normalized dot product
    p_im = p_im - patch_mean;
    face_score = p_im(:)'*eigenface(:);
    
    
    % method2: covolution
%     nrows = size(eigenface,1);
%     ncols = size(eigenface,2);
%     tmpimg = conv2(p_im,rot90(eigenface,2));
%     convolved_image = tmpimg(nrows:end, ncols:end);
%     face_score = convolved_image - ...
%     sumE*patch_means(1:size(convolved_image,1),1:size(convolved_image,2));
%     face_score = sum(sum(face_score));
end 
