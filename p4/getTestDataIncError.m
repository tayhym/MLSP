% eigenfaces : column vectors of eigenfaces to project test images to 
% Xtest: column vectors of weights
% Ytest: column vector of predictions
function [weights_F,weights_NF] = getTestDataIncError(eigenfaces)
%% read corpus of faces
    corpus = dir('BoostingData/test/face');
    l = length(corpus);  % read from 3rd file (first 2 files are . and ..)
    nimages = l-2;
    [nrows, ncols] = size(imread(strcat('BoostingData/test/face/',corpus(3).name)));
    
    % matrix of all unrolled images (vectorized)
    images = zeros((nrows*ncols),nimages);
    % read in images from corpus
    for i=1:nimages
        name = corpus(i+2).name;
        % double is for computation 
        % uint8 is 0-255
        image = double(imread(strcat('BoostingData/test/face/',name)));
        image = image - mean(image(:));
        image = image/norm(image(:));
        images(:,i) = image(:);
    end
    
    weights_F = eigenfaces'*images;
    diff_error = eigenfaces*weights_F - images;
    N = size(diff_error,1);
    diff_feature = zeros(1,size(diff_error,2));
    for i=1:size(diff_error,2)
        diff_feature(i) = (1/N)*norm(diff_error(:,i))^2;
        assert(numel(norm(diff_error(:,i)))==1);
        assert(numel(diff_error(:,i))==N);
    end 
    weights_F(end+1,:) = diff_feature;
    
    %% read corpus of non-faces
    corpus_NF = dir('BoostingData/test/non-face');
    l = length(corpus_NF);  % read from 3rd file (first 2 files are . and .. last file is folder)
    nimages_NF = l-3;
    [nrows, ncols] = size(imread(strcat('BoostingData/test/non-face/',corpus_NF(3).name)));
    
    % matrix of all unrolled images (vectorized)
    images_NF = zeros((nrows*ncols),nimages_NF);
    % read in images from corpus
    for i=1:nimages_NF
        fprintf('reading %d/%d image\n',i,nimages_NF);
        name = corpus_NF(i+2).name;
        % double is for computation 
        % uint8 is 0-255
        image_NF = double(imread(strcat('BoostingData/test/non-face/',name)));
        image_NF = image_NF - mean(image_NF(:));
        if (norm(image_NF(:))==0)
            images_NF(:,i) = 0;
            continue;
        end 
        
        image_NF = image_NF/norm(image_NF(:));
        images_NF(:,i) = image_NF(:);
    end
    
    weights_NF = eigenfaces'*images_NF;
    diff_error_NF = eigenfaces*weights_NF - images_NF;
    N = size(diff_error_NF,1);
    diff_feature_NF = zeros(1,size(diff_error_NF,2));
    for i=1:size(diff_error_NF,2)
        diff_feature_NF(i) = (1/N)*norm(diff_error_NF(:,i))^2;
        assert(numel(norm(diff_error_NF(:,i)))==1);
        assert(numel(diff_error_NF(:,i))==N);
    end 
    weights_NF(end+1,:) = diff_feature_NF;
    
end
