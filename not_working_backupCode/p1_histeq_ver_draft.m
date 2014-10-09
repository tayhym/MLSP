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
eigenface = eigenface/norm(eigenface(:)); % normalize eigenface
figure; imagesc(eigenface); 
figure; imshow(eigenface,[min(eigenface(:)) max(eigenface(:))]);

%% scan test images for eigenface

% read and convert to grayscale 
groupimages = dir('group_photos');
ngimages = length(groupimages) -2; % minus '.' '..' files

for i_gimg=3
    colorimg = double(imread(strcat('group_photos/',groupimages(i_gimg).name)));
    gimage = squeeze(mean(colorimg,3)); % mean along r,g,b channels
    
    % integral image - preprocess test image to normalize mean and variance 
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
    for i=1:size(gimage,1)-nrows+1
        for j=1:size(gimage,2)-ncols+1
            patch_img = gimage(i:i+nrows-1,j:j+ncols-1);
            % histogram equalize path image 
            patch_img = double(histeq(uint8(patch_img)));
            patch_score(i,j) = eigenface(:)'*patch_img(:)/norm(patch_img(:));
        end 
    end 
end
%% process patch_scores
% iterate through matrix, every element of matrix that is close to a 
% neighboring value is assigned the neighbor's score
tol = 0.01;

patch_score = round(patch_score*100)/100;
for i=1:size(patch_score,1)
    for j=1:size(patch_score,2)
        current_val = round(patch_score(i,j)*100)/100;
        % create neighborhood of pixel locations
        bound = 10;
       
        min_x = max(i,min(i,mod(i-bound,size(patch_score,1))));
        max_x = max(i,mod(i+bound,size(patch_score,1)));
        min_y = max(j,min(i,mod(j-bound,size(patch_score,2))));
        max_y = max(i,mod(j+bound,size(patch_score,2)));
        
        for ii=min_x:max_x
            for jj=min_y:max_y
                if (abs(patch_score(ii,jj)-current_val)<tol) 
                    patch_score(ii,jj) = current_val;
                end
            end
        end
                
    end 
end

%%
% find peaks in matrix: find maximum, then check if maximum values are 
% near the maximum already found - if so then remove and rescan.
tmp = patch_score; 
[sorted_values, sort_idx] = sort(tmp, 'descend');
% assume a maximum of 10 faces, then thresholds the faces that are 
% below a threshold
numtargets = 10;
maxIdx = sort_idx(1:numtargets);
for i=1:numtargets-1
    % check that the locations of faces are not close
    [I, J] = ind2sub(size(patch_score),maxIdx(i:i+1));
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

%% display faces

figure; imagesc(gimage);

%% 
figure; imshow(patch_score, [min(patch_score(:)), max(patch_score(:))]);
