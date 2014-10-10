% returns a mean and variance normalized eigenface
% must be in right directory
function [eigenface] = getEigenface()
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
    %% compute eigenvectors, sqrt eigenvalues == sv for sym, square, positive def
    % matrices
    [U,S,V]=svds(images,1);

    eigenface=reshape(U(:),nrows,ncols);
    figure; imshow(eigenface,[min(eigenface(:)) max(eigenface(:))]); title('before normalization');
    eigenface = eigenface - mean(eigenface(:));
    eigenface = eigenface./norm(eigenface(:));
    figure; imshow(eigenface,[min(eigenface(:)) max(eigenface(:))]); title('after normalization');
end

