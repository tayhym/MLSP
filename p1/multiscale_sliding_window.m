% performs sliding_window on an image rescaled at various scales
% and returns matrices of best scores and best locations
% gimage: grayscale image
% eigenface: normalized eigenface
% best_scores: 5 by num_best_scores_candidates mtx of best scores 
%              (1 row per scale)
% best_locations: 10 by num_best_scores_candidates mtx of best locations 
%                 (2 rows per scale)
% num_candidates: number of faces to shortlist
function [best_scores, best_locations] = multiscale_sliding_window(gimage,eigenface,num_candidates,threshold)
    
    num_scales = 5;
    [N_o, M_o] = size(gimage); % original image size
    % scaled images
    s_im1 = imresize(gimage,[N_o*0.5,M_o*0.5]);   % scaled to 0.5x
    s_im2 = imresize(gimage,[N_o*0.75,M_o*0.75]);   % 0.75x 
    s_im3 = gimage;                     % 1.0x
    s_im4 = imresize(gimage,[N_o*1.5,M_o*1.5]);   % 1.5x 
    s_im5 = imresize(gimage,[N_o*2.0,M_o*2.0]); % 2.0x
    
    s_im_all = cell(1,5); % 1 by 5 cell array
    s_im_all{1} = s_im1;
    s_im_all{2} = s_im2;
    s_im_all{3} = s_im3;
    s_im_all{4} = s_im4;
    s_im_all{5} = s_im5;
    scales = [0.5, 0.75, 1.0, 1.5,2.0]; % scales input image was scaled to
    
    % best scores at each of 5 scales
    bs_1 = zeros(1,num_candidates);
    bs_2 = zeros(1,num_candidates);
    bs_3 = zeros(1,num_candidates);
    bs_4 = zeros(1,num_candidates);
    bs_5 = zeros(1,num_candidates);
    % best locations at each scale
    bl_1 = zeros(2,num_candidates);
    bl_2 = zeros(2,num_candidates);
    bl_3 = zeros(2,num_candidates);
    bl_4 = zeros(2,num_candidates);
    bl_5 = zeros(2,num_candidates);
    
    % overall best scores and corresponding locations
    best_scores = [bs_1;bs_2;bs_3;bs_4;bs_5];
    best_locations = [bl_1;bl_2;bl_3;bl_4;bl_5];
    
    % compute best scores at each location and threshold them
    % for multiple-scales: compute new patch means
    for i=1:num_scales
        bl_x = 2*(i-1)+1; % x and y indices location for best_locations 
        bl_y = 2*i; 
        
        scaled_gimage = s_im_all{i};
        patch_means = computePatchMeans(s_im_all{i},eigenface);
        scale = scales(i);  % current image scale (used to compute correspondance)
        [best_scores(i,:), best_locations(bl_x:bl_y,:)] = ...
            sliding_window(scaled_gimage,eigenface, patch_means,num_candidates);
        [best_scores(i,:)] = ...
            thresholdFaces(best_scores(i,:),threshold);
        displayFaces(best_scores(i,:),best_locations(bl_x:bl_y,:),eigenface,scaled_gimage);
        
        
    end
    
         
         
    
    
    
    
    
    
    