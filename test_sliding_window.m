% test sliding window
groupimages = dir('group_photos');
i_gimg = 3;
colorimg = imread(strcat('group_photos/',groupimages(i_gimg).name));
gimage = squeeze(mean(colorimg,3)); % mean along r,g,b channels
figure; imagesc(gimage);
[bs bl] = sliding_window(gimage,eigenface);