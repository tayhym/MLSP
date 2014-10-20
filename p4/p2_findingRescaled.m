% compute train accuracy and the stump attributes at different 
% eigenface scales
clear all; close all;

results = zeros(1,5); % 1st row is train acc, 2nd row is test acc

scales = [0.5,0.75,1.0,1.5,2.0];
all_scales_best_stumps = cell(1,5); 
all_scales_stumps_alpha_t = cell(1,5);

for scale_idx=1:5
    scale = scales(scale_idx);
    
 %% read corpus of faces
    corpus = dir('BoostingData/train/face');
    l = length(corpus);  % read from 3rd file (first 2 files are . and ..)
    nimages = l-2;
    [nrows, ncols] = size(imresize(imread(strcat('BoostingData/train/face/',corpus(3).name)),scale));
    
    % matrix of all unrolled images (vectorized)
    images = zeros((nrows*ncols),nimages);
    % read in images from corpus
    for i=1:nimages
        name = corpus(i+2).name;
        % double is for computation 
        % uint8 is 0-255
        image = double(imresize(imread(strcat('BoostingData/train/face/',name)),scale));
        image = image - mean(image(:));
        image = image/norm(image(:));
        images(:,i) = image(:);
    end
 %% read corpus of non-faces
    corpus_NF = dir('BoostingData/train/non-face');
    l = length(corpus_NF);  % read from 3rd file (first 2 files are . and ..)
    nimages_NF = l-2;
    [nrows, ncols] = size(imresize(imread(strcat('BoostingData/train/non-face/',corpus_NF(3).name)),scale));
    
    % matrix of all unrolled images (vectorized)
    images_NF = zeros((nrows*ncols),nimages_NF);
    % read in images from corpus
    for i=1:nimages_NF
        name = corpus_NF(i+2).name;
        % double is for computation 
        % uint8 is 0-255
        image_NF = double(imresize(imread(strcat('BoostingData/train/non-face/',name)),scale));
        image_NF = image_NF - mean(image_NF(:));
        if (norm(image_NF(:))==0)
            images_NF(:,i) = 0;
            continue;
        end 
        
        image_NF = image_NF/norm(image_NF(:));
        images_NF(:,i) = image_NF(:);
    end
    
    K=12;    
%---------boosting based face detector---------%
% [eigenfaces,weights_F, weights_NF] = getEigenfaces(K);   % column vectors of eigenfaces
         
    [U,S,V]=svds(images,K);
%     eigenface=reshape(U(:,10),nrows,ncols);
%     figure; imshow(eigenface,[min(eigenface(:)) max(eigenface(:))]); title('Eigenface');
%     eigenface = eigenface - mean(eigenface(:));
%     eigenface = eigenface./norm(eigenface(:));
    eigenfaces = U;
    weights_F = eigenfaces'*images;
    
    weights_NF = eigenfaces'*images_NF;    

Xtrain = [weights_F';weights_NF'];
Ytrain = [1*ones(size(weights_F',1),1);-1*ones(size(weights_NF',1),1)];
%%
    
[alpha_t,best_stumps] = adaboost_train(Xtrain, Ytrain);
all_scales_best_stumps{scale_idx} = best_stumps;
all_scales_stumps_alpha_t{scale_idx} = alpha_t;
%%
[ypred] = adaboost_test(best_stumps,alpha_t,Xtrain);

accuracy = mean(ypred==Ytrain);
results(1,scale_idx) = accuracy;

end

