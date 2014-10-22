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
% i_gimg=4;
% i_gimg=5;
% i_gimg=6;

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
rect_attr = size(eigenface);
% shapeInserter = vision.shapeInserter;
stats1 = regionprops(f_p,'area','centroid','boundingbox');
scale1 = 0.5;
face_loc1 = getFaceLocations(stats1,scale1)
num_faces = size(face_loc1,1);
rect_loc1 = [face_loc1,repmat(rect_attr,[num_faces,1])]; 
gimage_disp1 = gimage;
figure(24); 
imagesc(gimage_disp1);
hold on;
for i=1:size(face_loc1,1)
%     gimage_disp1 = step(shapeInserter, gimage_disp1,[face_loc,repmat(rect_attr,[num_faces,1])]);
    rect = rectangle('position', rect_loc1(i,:));
end

%%
stats2 = regionprops(f_p2,'area','centroid','boundingbox');
scale2 = 0.75;
face_loc2 = getFaceLocations(stats2,scale2);
num_faces = size(face_loc2,1);
rect_loc2 = [face_loc2,repmat(rect_attr,[num_faces,1])]; 

for i=1:size(face_loc2,1)
%     gimage_disp1 = step(shapeInserter, gimage_disp1,[face_loc,repmat(rect_attr,[num_faces,1])]);
    rect = rectangle('position', rect_loc2(i,:));
end

stats3 = regionprops(f_p3,'area','centroid','boundingbox');
scale3 = 1.0;
face_loc3 = getFaceLocations(stats3,scale3);
num_faces = size(face_loc3,1);
rect_loc3 = [face_loc3,repmat(rect_attr,[num_faces,1])]; 

for i=1:size(face_loc3,1)
    rect = rectangle('position', rect_loc3(i,:));
end

stats4 = regionprops(f_p4,'area','centroid','boundingbox');
scale4 = 1.5;
face_loc4 = getFaceLocations(stats4,scale4);
num_faces = size(face_loc4,1);
rect_loc4 = [face_loc4,repmat(rect_attr,[num_faces,1])]; 

for i=1:size(face_loc4,1)
    rect = rectangle('position', rect_loc4(i,:));
end

stats5 = regionprops(f_p5,'area','centroid','boundingbox');
scale5 = 2.0;
face_loc5 = getFaceLocations(stats5,scale5);
num_faces = size(face_loc5,1);
rect_loc5 = [face_loc5,repmat(rect_attr,[num_faces,1])]; 

for i=1:size(face_loc5,1)
    rect = rectangle('position', rect_loc5(i,:));
end
hold off;
%%

all_rects = [face_loc1;face_loc2;face_loc3;face_loc4;face_loc5];
face_loc_combined = combine_faces(all_rects);
num_faces = size(face_loc_combined,1);
rect_loc_combined = [face_loc_combined,repmat(rect_attr,[num_faces,1])]; 
figure(25); 
imagesc(gimage);
hold on;
for i=1:size(face_loc_combined,1)
    rect = rectangle('position', rect_loc_combined(i,:));
end
hold off;

