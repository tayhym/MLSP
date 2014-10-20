load p3accuracies.mat
%% 
% p1_v3 
% changelog: max method
%%%%%%%%%%%
% options (variable) 
normalizeTrainingFaces = 1;     % getEigenFace()
normalizeEigenface = 0;         % getEigenFace()
meanSubtractPatches = 1;        % sliding_window2()
varianceNormalizePatches = 0;   % sliding_window2()
normalizedDotProduct = 0;       % queryFace()
dotProduct = 1;                 % queryFace()
%%%%%%%%%%%

% test for eigenface on group pictures
i_gimg=3;
% i_gimg=4;
% i_gimg=5;

groupimages = dir('group_photos');
colorimg = imread(strcat('group_photos/',groupimages(i_gimg).name));
gimage = squeeze(mean(colorimg,3)); % mean along r,g,b channels
figure(1); imagesc(gimage); 

%% 
detection_matrix = adaboost_find_faces(best_stumps,alpha_t,gimage);
    
figure(8); imagesc(detection_matrix); colorbar;

%% 
% compute on different scales 
N = size(gimage,1);M=size(gimage,2);
s_im1 = imresize(gimage,[N*0.5,M*0.5]);   % scaled to 0.5x
s_im2 = imresize(gimage,[N*0.75,M*0.75]);   % 0.75x 
s_im3 = gimage;                     % 1.0x
s_im4 = imresize(gimage,[N*1.5,M*1.5]);   % 1.5x 
s_im5 = imresize(gimage,[N*2.0,M*2.0]); % 2.0x 

