% takes gray image, scales it to given scale and 
% queries for a face in patch of the image
% gimage: grayscale image 
% scale: scale to scan at
% eigenface: normalized face to scan against
% patch_scores: size(gimage) matrix of patch_scores of patch cornered at
%               each i, j point
% patch_means: matrix of means of a patch cornered at (i,j)
function [best_scores, best_locations] = sliding_window(gimage, eigenface, patch_means)
    N = size(eigenface,1);
    M = size(eigenface,2);
    X = size(gimage,1);
    Y = size(gimage,2);
    
    patch_scores = zeros(size(gimage));
    
    % display best scores
    num_candidates = 10; % assume cap of 20 faces
    best_scores = zeros(1,num_candidates);
    best_locations = zeros(2, num_candidates); 

    debug = 0;
    
    for i=1:X-N
        for j=1:Y-M
            p_im = gimage(i:i+N-1,j:j+M-1);
            p_m = patch_means(i,j);
            p_im = p_im - p_m;
            patch_scores(i,j) = query_face(p_im, eigenface);
            [best_scores, best_locations] = ...
            add(best_scores,best_locations,patch_scores(i,j),i,j);     
        end 
    end
   %%
   % display the best locations
    if debug
        for i=1:length(best_scores)
            x = best_locations(1,i);
            y = best_locations(2,i);
            patch = gimage(x:x+N-1,y:y+M-1);
            figure; imshow(patch, [min(patch(:)),max(patch(:))]);
        end
    end