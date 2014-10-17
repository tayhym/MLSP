% p1_v2 updated-modular version of p1

% get eigenface
eigenface = getEigenface();


% test on group pictures
% scan test images for eigenfaces
%%
i_gimg=3;
    colorimg = imread(strcat('group_photos/',groupimages(i_gimg).name));
    gimage = squeeze(mean(colorimg,3)); % mean along r,g,b channels
    figure; imagesc(gimage); 
%%
    % get patch-means
    integralI = cumsum(cumsum(gimage,1),2); % integral image
    patch_sums = zeros(size(gimage));      % sum of nrows by ncols patch                                      
    for i=1:size(gimage,1)-nrows+1          % cornered at pixel i,j 
        for j=1:size(gimage,2)-ncols+1
            a1 = integralI(i,j);
            a2 = integralI(i+nrows-1,j);
            a3 = integralI(i,j+ncols-1);
            a4 = integralI(i+nrows-1,j+ncols-1);
            patch_sums(i,j) = a4-a2-a3+a1;
        end 
    end 
    patch_means = patch_sums./(nrows*ncols); 
    %%
    num_candidates = 20; % assume cap of 20 faces
%     [bs, bl] = sliding_window(gimage,eigenface,patch_means, num_candidates);
    
%     [bs] = thresholdFaces(bs,840); 
%     displayFaces(bs,bl,eigenface,gimage);
    
    %%
    threshold = 840;
    [bs_all, bl_all] =  multiscale_sliding_window(            ...
                        gimage,eigenface,num_candidates,threshold);
                    
                    
    % some form of voting procedure? that counts the number of votes that 
    % a pixel has, and then votes for a face at that location. final
    % location is the mean of all pixel locations that voted for the face
    % to be present there.
    
    