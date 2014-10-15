% takes gray image and queries for a face in patch of the image
% gimage: grayscale image 
% eigenface: normalized face to scan against
% patch_scores: size(gimage) matrix of patch_scores of patch cornered at
%               each i, j point
% patch_means: matrix of means of a patch cornered at (i,j)
function [patch_scores] = sliding_window2(gimage, eigenface)
    N = size(eigenface,1);
    M = size(eigenface,2);

    X = size(gimage,1);
    Y = size(gimage,2);
    
    patch_sums = computeAllPatchSums(gimage,size(eigenface));

    patch_scores = zeros(size(gimage));
    patch_means = patch_sums./(N*M);
    sumE = sum(sum(eigenface(:)));
    for i=1:X-N
        for j=1:Y-M
            p_im = gimage(i:i+N-1,j:j+M-1);
            patch_scores(i,j) = query_face(p_im, eigenface,patch_means(i,j),patch_means,sumE);
        end 
    end
    