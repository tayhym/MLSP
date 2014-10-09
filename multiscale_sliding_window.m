% performs sliding_window on an image rescaled at various scales
% and returns matrix of locations where faces detected
% gimage: grayscale image
% eigenface: normalized eigenface
function [peak_locations] = multiscale_sliding_window(gimage, eigenface)
    
    % scaled images
    s_im1 = imresize(gimage,[32,32]);   % scaled to 0.5x
    s_im2 = imresize(gimage,[48,48]);   % 0.75x 
    s_im3 = gimage;                     % 1.0x
    s_im4 = imresize(gimage,[96,96]);   % 1.5x 
    s_im5 = imresize(gimage,[128,128]); % 2.0x
    
    % patch_scores at each of 5 scales
    ps_1 = zeros(X,Y);
    ps_2 = zeros(X,Y);
    ps_3 = zeros(X,Y);
    ps_4 = zeros(X,Y);
    ps_5 = zeros(X,Y);
    
    