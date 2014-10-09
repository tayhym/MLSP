% test query_face 
% obtain test samples from image
groupimages = dir('group_photos');
i_gimg = 3;
colorimg = imread(strcat('group_photos/',groupimages(i_gimg).name));
gimage = squeeze(mean(colorimg,3)); % mean along r,g,b channels
figure; imagesc(gimage);

N = size(eigenface,1); % N is row-wise
M = size(eigenface,2); % M is col-wise 
face1 = gimage(106:106+N-1,97:97+M-1);

figure; imagesc(face1);
face_d = gimage(138:138+N-1,96:96+M-1);
figure; imagesc(face_d);


% 2 things: 1) dot product simlarity, not absolute
%           2) need multiple scales
% scores at each of the faces: 
score_face1 = query_face(face1, eigenface)
score_face_d = query_face(face_d, eigenface)

assert(score_face1>score_face_d);
