%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------problem1:face_detector----------------%
%reads in training images, extracts eigenfaces,
%detects faces in given images----------------%

%% read corpus of faces
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
    image = double(imread(strcat('lfw1000/',name)));
    images(:,i) = image(:);
end
    % mean normalization and variance normalization to remove lighting
    % effects
    m_images = mean(images(:));
    images = images - m_images;
    sd_images = norm(images(:));
    images = images./sd_images;

%% compute eigenvectors, sqrt eigenvalues == sv for sym, square, positive def
% matrices
[U,S,V]=svds(images,1);

eigenface=reshape(U(:,1),nrows,ncols);
figure; imagesc(eigenface); 
figure; imshow(eigenface,[min(eigenface(:)) max(eigenface(:))]);

%% scan test images for eigenface

% read and convert to grayscale 
groupimages = dir('group_photos');
ngimages = length(groupimages) -2; % minus '.' '..' files

i_gimg=3;
    colorimg = imread(strcat('group_photos/',groupimages(i_gimg).name));
    gimage = squeeze(mean(colorimg,3)); % mean along r,g,b channels
    figure; imagesc(gimage);
%%    
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
%%
% find 20 best fit face locations
    num_candidates = 10; % assume cap of 20 faces
    best_scores = zeros(1,num_candidates);
    best_locations = zeros(2, num_candidates); 
    
    N = size(eigenface,1);
    M = size(eigenface,2);
    X = size(gimage,1);
    Y = size(gimage,2);
    % scan along each patch finding match against eigenface
    for i=1:X-N
        for j=1:Y-M
            p_im = gimage(i:i+N-1,j:j+M-1);         
            p_m = patch_means(i,j);
            p_im = p_im - p_m;
%             p_std = norm(p_im(:));
%             p_im = p_im/p_std;
            p_s = abs(p_im(:)'*eigenface(:)); 
           
%           tmpimg = conv2(p_im,rot90(eigenface,2));
%           convolved_image = tmpimg(nrows:end, ncols:end);
%           patch_score = convolved_image - sumE*patch_means(1:size(dot_gimage_patch,1),1:size(dot_gimage_patch,2));
            [best_scores, best_locations] = ...
            add(best_scores,best_locations,p_s,i,j);     
        end 
    end
    %%
    for i=1:size(best_locations,2)
        patch = gimage(best_locations(1,i):best_locations(1,i)+N-1,best_locations(2,i):best_locations(2,i)+M-1);
        figure; imshow(patch, [min(patch(:)), max(patch(:))]);
    end
    figure; imshow(gimage,[min(gimage(:)),max(gimage(:))]);
    