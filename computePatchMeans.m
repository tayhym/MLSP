% given grayscale image, compute patch means by integral image
% eigenface used to compute patch size
function [patch_means] = computePatchMeans(gimage, eigenface)
    % get patch-means
    integralI = cumsum(cumsum(gimage,1),2); % integral image
    patch_sums = zeros(size(gimage));       % sum of nrows by ncols patch 
    [nrows, ncols] = size(eigenface);
    
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
end 

