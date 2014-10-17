% sets score to -1 if score is below threshold
function [thresholded_bs] = thresholdFaces(bs,threshold)
    thresholded_bs = bs;
    for i=1:length(bs)
        if (bs(i) <threshold)
            thresholded_bs(i) = -1;
        end
    end
end

        
