% displays the faces noted in best_locations, if score is not 
% negative or 0
function [] = displayFaces(best_scores, best_locations, eigenface,gimage)
    N = size(eigenface,1);
    M = size(eigenface,2);
    for i=1:length(best_scores)
        best_score = best_scores(i);
        if (best_score>0)
            x = best_locations(1,i);
            y = best_locations(2,i);
            patch = gimage(x:x+N-1,y:y+M-1);
            figure; imshow(patch, [min(patch(:)),max(patch(:))]);
        end 
    end
end
