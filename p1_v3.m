% p1_v3 
% changelog: max method
clear all; close all;
%%%%%%%%%%%
% options (variable) 
normalizeTrainingFaces = 1;     % getEigenFace()
normalizeEigenface = 0;         % getEigenFace()
meanSubtractPatches = 1;        % sliding_window2()
varianceNormalizePatches = 0;   % sliding_window2()
normalizedDotProduct = 0;       % queryFace()
dotProduct = 1;                 % queryFace()
%%%%%%%%%%%

% Pipeline1: Get eigenface and images
% get eigenface
eigenface = getEigenface();

% test for eigenface on group pictures
i_gimg=3;
groupimages = dir('group_photos');
colorimg = imread(strcat('group_photos/',groupimages(i_gimg).name));
gimage = squeeze(mean(colorimg,3)); % mean along r,g,b channels
figure(1); imagesc(gimage); 

%% 
patch_scores = sliding_window2(gimage, eigenface);
    
figure(8); imagesc(patch_scores); colorbar;

%% 
% compute on different scales 
N = size(gimage,1);M=size(gimage,2);
s_im1 = imresize(gimage,[N*0.5,M*0.5]);   % scaled to 0.5x
s_im2 = imresize(gimage,[N*0.75,M*0.75]);   % 0.75x 
s_im3 = gimage;                     % 1.0x
s_im4 = imresize(gimage,[N*1.5,M*1.5]);   % 1.5x 
s_im5 = imresize(gimage,[N*2.0,M*2.0]); % 2.0x 

patch_scores1 = sliding_window2(s_im1, eigenface);
figure(9); imagesc(patch_scores1); colorbar;
patch_scores2 = sliding_window2(s_im2, eigenface);
figure(10); imagesc(patch_scores2); colorbar;
patch_scores3 = sliding_window2(s_im3, eigenface);
figure(11); imagesc(patch_scores3); colorbar;
patch_scores4 = sliding_window2(s_im4, eigenface);
figure(12); imagesc(patch_scores4); colorbar;
patch_scores5 = sliding_window2(s_im5, eigenface);
figure(13); imagesc(patch_scores5); colorbar;
    
%%
f_p = patch_scores1 > (7/10)*max(patch_scores1(:));
figure(14); imagesc(f_p);
f_p2 = patch_scores2 > (7/10)*max(patch_scores2(:));
figure(15); imagesc(f_p2);
f_p3 = patch_scores3 > (7/10)*max(patch_scores3(:));
figure(16); imagesc(f_p3);
f_p4 = patch_scores4 > (6/10)*max(patch_scores4(:));
figure(17); imagesc(f_p4);
f_p5 = patch_scores5 > (7/10)*max(patch_scores5(:));
figure(18); imagesc(f_p5);


%% 
stats1 = regionprops(f_p,'area','centroid','boundingbox');
centroids = reshape([stats1.Centroid],[numel([stats1.Centroid])/2,2]);
area = reshape([stats1.Area],[numel([stats1.Area]),1]);
% face_locations = stats1(:).Centroid > mean(stats1(:).Centroid);
thres = (1/3)*median(area);
idx = area>thres;
faces_loc = round(centroids(idx,:));
faces_loc = faces_loc/0.5; % scale back to original space

stats2 = regionprops(f_p2,'area','centroid','boundingbox');
stats3 = regionprops(f_p3,'area','centroid','boundingbox');
stats4 = regionprops(f_p4,'area','centroid','boundingbox');
stats5 = regionprops(f_p5,'area','centroid','boundingbox');




