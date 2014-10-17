% get the best face locations given 
% stats: returned by regionprops - 
%        a structure with fields for the area 
%        of each connected region and the centroid locations for each
%        connected region
function [faces_loc] = getFaceLocations(stats1, scale)
    [stats1.Centroid]
    tmp = reshape([stats1.Centroid],[2,numel([stats1.Area])]);
    centroids = [tmp(1,:)',tmp(2,:)'];
    area = reshape([stats1.Area],[numel([stats1.Area]),1]);
    % face_locations = stats1(:).Centroid > mean(stats1(:).Centroid);
    thres = (1/3)*median(area);
    idx = area>thres;
    faces_loc = centroids(idx,:);
    faces_loc = round(faces_loc/scale); % scale back to original space
end
