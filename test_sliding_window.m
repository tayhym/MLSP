clear all; close all;
% test sliding window
eigenface = getEigenface();
groupimages = dir('group_photos');
i_gimg = 3;
colorimg = imread(strcat('group_photos/',groupimages(i_gimg).name));
gimage = squeeze(mean(colorimg,3)); % mean along r,g,b channels
figure; imagesc(gimage);

nrows = size(eigenface,1);
ncols = size(eigenface,2);
% preprocess test image to normalize mean and variance 
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


[bs, bl] = sliding_window(gimage,eigenface,patch_means);

    