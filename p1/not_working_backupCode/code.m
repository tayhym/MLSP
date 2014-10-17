%---------------some code from part1 - normalizing patches -------%
% mean normalize and standard-dev normalize patch img
%             figure;
%             subplot(1,2,1); imagesc(patch_img); title('before normalization');
            patch_img = patch_img./(mean(patch_img(:)));
            patch_img = patch_img./norm(patch_img(:));
%             subplot(1,2,2); imagesc(patch_img); title('after mean std-dev norm');
            % compute patch-score: the dot product of eigenface with patch
            tmpimg = conv2(patch_img,rot90(eigenface,2)); % flip lr, flip up 
                                                       % (so conv becomes 
                                                       % dot product near 
                                                       % the end)
            figure; imagesc(tmpimg);                                           
            dot_gimage_patch = tmpimg(nrows:end, ncols:end);
            patch_score(i,j) = sum(sum(dot_gimage_patch));
%             sumE = sum(eigenface(:));
%             patch_score = dot_gimage_patch - sumE*patch_means(1:size(dot_gimage_patch,1),1:size(dot_gimage_patch,2));
%             figure; 
%             subplot(1,2,1); imagesc(patch_img);title(strcat('score for patch',num2str((patch_score(i,j)))));
%             subplot(i,2,2); imagesc(eigenface);

%-----------code that uses mean normalization-----------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------problem1:face_detector----------------%
%reads in training images, extracts eigenfaces,
%detects faces in given images----------------%

corpus = dir('lfw1000/');
l = length(corpus);  % read from 3rd file (first 2 files are . and ..)
nimages = l-2;
[nrows, ncols] = size(imread(strcat('lfw1000/',corpus(3).name)));

% matrix of all unrolled images (vectorized)
images = zeros((nrows*ncols),nimages);
% read in images from corpus
for i=1:nimages
    name = corpus(i+2).name;
    % double is for computation 
    % uint8 is 0-255
    image = double(histeq(imread(strcat('lfw1000/',name))));
    % mean normalization and variance normalization to remove lighting
    % effects 
%     image = image - mean(image(:));
%     image = image./norm(image(:));
    images(:,i) = image(:);
end

%% compute eigenvectors, sqrt eigenvalues == sv for sym, square, positive def
% matrices
[U,S,V]=svd(images,0);

eigenface=reshape(U(:,1),nrows,ncols);
figure; imagesc(eigenface); 
figure; imshow(eigenface,[min(eigenface(:)) max(eigenface(:))]);

%% scan test images for eigenface

% read and convert to grayscale 
groupimages = dir('group_photos');
ngimages = length(groupimages) -2; % minus '.' '..' files

for i_gimg=3
    colorimg = imread(strcat('group_photos/',groupimages(i_gimg).name));
    gimage = squeeze(mean(colorimg,3)); % mean along r,g,b channels
 
    % preprocess test image to normalize mean and variance 
    integralI = cumsum(cumsum(gimage,1),2); % integral image
    patch_means = zeros(size(gimage));      % mean of nrows by ncols patch                                      
    for i=1:size(gimage,1)-nrows+1          % cornered at pixel i,j 
        for j=1:size(gimage,2)-ncols+1
            a1 = integralI(i,j);
            a2 = integralI(i+nrows-1,j);
            a3 = integralI(i,j+ncols-1);
            a4 = integralI(i+nrows-1,j+ncols-1);
            patch_means(i,j) = a4-a2-a3+a1;
        end 
    end
    patch_means = patch_means./(nrows*ncols);   
    
    patch_score = zeros(size(gimage));
    % scan along each patch finding match against eigenface
    % using hopsize of 4 pixels
    for i=1:nrows:size(gimage,1)-nrows+1
        for j=1:ncols:size(gimage,2)-ncols+1
            patch_img = gimage(i:i+nrows-1,j:j+ncols-1);
            
            % mean normalize and standard-dev normalize patch img
%             figure;
%             subplot(1,2,1); imagesc(patch_img); title('before normalization');
            patch_img = patch_img./(mean(patch_img(:)));
            patch_img = patch_img./norm(patch_img(:));
%             subplot(1,2,2); imagesc(patch_img); title('after mean std-dev norm');
            % compute patch-score: the dot product of eigenface with patch
            tmpimg = conv2(patch_img,rot90(eigenface,2)); % flip lr, flip up 
                                                       % (so conv becomes 
                                                       % dot product near 
                                                       % the end)
            figure; imagesc(tmpimg);                                           
            dot_gimage_patch = tmpimg(nrows:end, ncols:end);
            patch_score(i,j) = sum(sum(dot_gimage_patch));
%             sumE = sum(eigenface(:));
%             patch_score = dot_gimage_patch - sumE*patch_means(1:size(dot_gimage_patch,1),1:size(dot_gimage_patch,2));
%             figure; 
%             subplot(1,2,1); imagesc(patch_img);title(strcat('score for patch',num2str((patch_score(i,j)))));
%             subplot(i,2,2); imagesc(eigenface);
        end 
    end 
end
%% display superimposed picture
figure; imagesc(gimage);
[maxVal, idx] = max(patch_score);
figure; imagesc(idx);    
%% 
figure; imshow(patch_score, [min(patch_score(:)), max(patch_score(:))]);



%-----------------code that finds peaks--------------------%
tmp = patch_score; 
[sorted_values, sort_idx] = sort(tmp, 'descend');
% assume a maximum of 10 faces, then thresholds the faces that are 
% below a threshold
numtargets = 10;
maxIdx = sort_idx(1:numtargets);
for i=1:numtargets-1
    % check that the locations of faces are not close
    [I, J] = ind2sub(maxIdx(i:i+1), size(patch_score));
    large_location = [I(1) J(1)];
    next_location = [I(2) J(2)];
    while (norm(next_location-large_location)<sqrt(nrows.^2+ncols.^2)/2) 
        next_location
        large_location
      
        % find next bounding box
        tmp(sub2ind(size(patch_score),next_location(1),next_location(2))) ...
        = 0;
        [new_sorted_values, new_sort_idx] = sort(tmp,'descend');
        newmaxIdx = new_sort_idx(1:numtargets);
        next_location = ind2sub(newmaxIdx(i+1),size(patch_score));
        next_location
    end 
    next_location
    maxIdx(i+1)
    size(patch_score)
    next_location(1)
    next_location(2)
    sub2ind(size(patch_score),next_location(1), next_location(2))
    maxIdx(i+1) = sub2ind(size(patch_score),next_location(1), next_location(2)); % update new next location
    break;
end    