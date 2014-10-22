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
i_gimg=4;
i_gimg=5;
% i_gimg=6;
groupimages = dir('group_photos');
colorimg = imread(strcat('group_photos/',groupimages(i_gimg).name));
gimage = squeeze(mean(colorimg,3)); % mean along r,g,b channels
figure(1); imagesc(gimage); 

%% 
colorimg = imread(strcat('group_photos/',groupimages(i_gimg).name));
gimage = squeeze(mean(colorimg,3)); % mean along r,g,b channels
colorimg = imresize(imread(strcat('group_photos/',groupimages(i_gimg).name)),size(gimage));
% colorimg = imread(strcat('group_photos/',groupimages(i_gimg).name));

gimage = squeeze(mean(colorimg,3)); % mean along r,g,b channels
figure; imagesc(gimage);

%% 
% compute on different scales by rescaling the images to fit the new eigenface 
% dimension
N = size(gimage,1);M=size(gimage,2);
rescaled_dimensions1 = [round(N*0.5*(19/64)),round(M*0.5*(19/64))];
s_im1 = imresize(gimage,rescaled_dimensions1);   % scaled to 0.5x
patch_scores1 = adaboost_find_faces(best_stumps,alpha_t,eigenfaces,s_im1);

rescaled_dimensions2 = [round(N*0.75*19/64), round(M*0.5*19/64)];
s_im2 = imresize(gimage,rescaled_dimensions2);   % 0.75x 
patch_scores2 = adaboost_find_faces(best_stumps,alpha_t,eigenfaces,s_im2);

rescaled_dimensions3 = [round(N*19/64), round(M*19/64)];
s_im3 = imresize(gimage, rescaled_dimensions3);                     % 1.0x
patch_scores3 = adaboost_find_faces(best_stumps,alpha_t,eigenfaces,s_im3);

rescaled_dimension4 = [round(N*1.5*19/64), round(M*1.5*19/64)];
s_im4 = imresize(gimage,rescaled_dimension4);   % 1.5x

% patch_scores4 = adaboost_find_faces(best_stumps,alpha_t,eigenfaces,s_im4);

rescaled_dimension5 = [round(N*2*19/64), round(M*2*19/64)];
s_im5 = imresize(gimage,rescaled_dimension5); % 2.0x 
% patch_scores5 = adaboost_find_faces(best_stumps,alpha_t,eigenfaces,s_im5);

%%
a1 = patch_scores1>=0.8*max(max(patch_scores1));
a2= patch_scores2>=0.8*max(max(patch_scores2));
a3 = patch_scores3>=0.8*max(max(patch_scores3));
% a4 = patch_scores4>=0.8*max(max(patch_scores4));
% a5 = patch_scores5>=0.8*max(max(patch_scores5));

figure(5); imagesc(a1);
figure(6);imagesc(a2);
figure(7); imagesc(a3);
figure(8); imagesc(a4);
figure(9); imagesc(a5);
%% 
% shapeInserter = vision.shapeInserter;
stats1 = regionprops(a1,'area','centroid','boundingbox');
scale1 = 0.5*19/64;
rect_attr = [64 64]; % size of eigenface

face_loc1 = getFaceLocations(stats1,scale1)
num_faces = size(face_loc1,1);
rect_loc1 = [face_loc1,repmat(rect_attr,[num_faces,1])]; 
gimage_disp1 = gimage;
figure(10); 
imagesc(gimage_disp1);
hold on;
for i=1:size(face_loc1,1)
%     gimage_disp1 = step(shapeInserter, gimage_disp1,[face_loc,repmat(rect_attr,[num_faces,1])]);
    rect = rectangle('position', rect_loc1(i,:));
end

%%
stats2 = regionprops(a2,'area','centroid','boundingbox');
scale2 = 0.75*19/64;

face_loc2 = getFaceLocations(stats2,scale2);
num_faces = size(face_loc2,1);
rect_loc2 = [face_loc2,repmat(rect_attr,[num_faces,1])]; 

for i=1:size(face_loc2,1)
%     gimage_disp1 = step(shapeInserter, gimage_disp1,[face_loc,repmat(rect_attr,[num_faces,1])]);
    rect = rectangle('position', rect_loc2(i,:));
end

stats3 = regionprops(a3,'area','centroid','boundingbox');
scale3 = 1.0*19/64;

face_loc3 = getFaceLocations(stats3,scale3);
num_faces = size(face_loc3,1);
rect_loc3 = [face_loc3,repmat(rect_attr,[num_faces,1])]; 

for i=1:size(face_loc3,1)
    rect = rectangle('position', rect_loc3(i,:));
end


stats4 = regionprops(a4,'area','centroid','boundingbox');
scale4 = 1.5*19/64;
 
% face_loc4 = getFaceLocations(stats4,scale4);
% num_faces = size(face_loc4,1);
% rect_loc4 = [face_loc4,repmat(rect_attr,[num_faces,1])]; 
%  
% for i=1:size(face_loc4,1)
%      rect = rectangle('position', rect_loc4(i,:));
% end
 
stats5 = regionprops(a5,'area','centroid','boundingbox');
scale5 = 2.0*19/64;
 
% face_loc5 = getFaceLocations(stats5,scale5);
% num_faces = size(face_loc5,1);
% rect_loc5 = [face_loc5,repmat(rect_attr,[num_faces,1])]; 
%  
% for i=1:size(face_loc5,1)
%     rect = rectangle('position', rect_loc5(i,:));
% end

%%
all_rects = [face_loc1;face_loc2;face_loc3];
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


