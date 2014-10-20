% finding optimal K value
clear all; close all;

results = zeros(2,8); % 1st row is train acc, 2nd row is test acc

 %% read corpus of faces
    corpus = dir('BoostingData/train/face');
    l = length(corpus);  % read from 3rd file (first 2 files are . and ..)
    nimages = l-2;
    [nrows, ncols] = size(imread(strcat('BoostingData/train/face/',corpus(3).name)));
    
    % matrix of all unrolled images (vectorized)
    images = zeros((nrows*ncols),nimages);
    % read in images from corpus
    for i=1:nimages
        name = corpus(i+2).name;
        % double is for computation 
        % uint8 is 0-255
        image = double(imread(strcat('BoostingData/train/face/',name)));
        image = image - mean(image(:));
        image = image/norm(image(:));
        images(:,i) = image(:);
    end
 %% read corpus of non-faces
    corpus_NF = dir('BoostingData/train/non-face');
    l = length(corpus_NF);  % read from 3rd file (first 2 files are . and ..)
    nimages_NF = l-2;
    [nrows, ncols] = size(imread(strcat('BoostingData/train/non-face/',corpus_NF(3).name)));
    
    % matrix of all unrolled images (vectorized)
    images_NF = zeros((nrows*ncols),nimages_NF);
    % read in images from corpus
    for i=1:nimages_NF
        name = corpus_NF(i+2).name;
        % double is for computation 
        % uint8 is 0-255
        image_NF = double(imread(strcat('BoostingData/train/non-face/',name)));
        image_NF = image_NF - mean(image_NF(:));
        if (norm(image_NF(:))==0)
            images_NF(:,i) = 0;
            continue;
        end 
        
        image_NF = image_NF/norm(image_NF(:));
        images_NF(:,i) = image_NF(:);
    end
    
%% read corpus of faces
    corpus = dir('BoostingData/test/face');
    l = length(corpus);  % read from 3rd file (first 2 files are . and ..)
    nimages = l-2;
    [nrows, ncols] = size(imread(strcat('BoostingData/test/face/',corpus(3).name)));
    
    % matrix of all unrolled images (vectorized)
    test_images = zeros((nrows*ncols),nimages);
    % read in images from corpus
    for i=1:nimages
        name = corpus(i+2).name;
        % double is for computation 
        % uint8 is 0-255
        image = double(imread(strcat('BoostingData/test/face/',name)));
        image = image - mean(image(:));
        image = image/norm(image(:));
        test_images(:,i) = image(:);
    end
    
    
    %% read corpus of non-faces
    corpus_NF = dir('BoostingData/test/non-face');
    l = length(corpus_NF);  % read from 3rd file (first 2 files are . and .. last file is folder)
    nimages_NF = l-3;
    [nrows, ncols] = size(imread(strcat('BoostingData/test/non-face/',corpus_NF(3).name)));
    
    % matrix of all unrolled images (vectorized)
    test_images_NF = zeros((nrows*ncols),nimages_NF);
    % read in images from corpus
    for i=1:nimages_NF
        fprintf('reading %d/%d image\n',i,nimages_NF);
        name = corpus_NF(i+2).name;
        % double is for computation 
        % uint8 is 0-255
        image_NF = double(imread(strcat('BoostingData/test/non-face/',name)));
        image_NF = image_NF - mean(image_NF(:));
        if (norm(image_NF(:))==0)
            test_images_NF(:,i) = 0;
            continue;
        end 
        
        image_NF = image_NF/norm(image_NF(:));
        test_images_NF(:,i) = image_NF(:);
    end
    
        


for K=2:2:16
    
    
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
    test_weights_F = eigenfaces'*test_images;
    test_weights_NF = eigenfaces'*test_images_NF;


Xtrain = [weights_F';weights_NF'];
Ytrain = [1*ones(size(weights_F',1),1);-1*ones(size(weights_NF',1),1)];
%%
    
[alpha_t,best_stumps] = adaboost_train(Xtrain, Ytrain);
%%
[ypred] = adaboost_test(best_stumps,alpha_t,Xtrain);

accuracy = mean(ypred==Ytrain);
results(1,K/2) = accuracy;

%% combined testData for faces and non-faces 

Weights_F_test = test_weights_F;
Weights_NF_test = test_weights_NF;

Xtest = [Weights_F_test';Weights_NF_test'];
Ytest = [1*ones(size(Weights_F_test',1),1);-1*ones(size(Weights_NF_test',1),1)];
[ypred_test] = adaboost_test(best_stumps, alpha_t, Xtest);

test_accuracy = mean(Ytest==ypred_test);
results(2,K/2) = test_accuracy;
end 
