% check if face is present, by dot product between patch_image 
% and normalized eigen_face. 
function [face_score] = query_face(p_im, eigenface)
    %sumE=sum(sum(eigenface(:)));
    
    % method1: dot product
    face_score = p_im(:)'*eigenface(:);
    
    % method2: covolution
%     tmpimg = conv2(p_im,rot90(eigenface,2));
%            convolved_image = tmpimg(nrows:end, ncols:end);
%            face_score = convolved_image - ...
%            sumE*patch_means(1:size(convolved_image,1),1:size(convolved_image,2));

end 
