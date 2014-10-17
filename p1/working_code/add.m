% Add if it is far away in x,y space from all points
% (currently in nominees for faces) and if it has 
% a dot product that is higher than the minimum of the
% points (looks more like eigenface)

function [out_scores, out_locations] = add(scores, locations,incoming_score, in_i, in_j)
    % create copy 
    out_scores = scores;
    out_locations = locations;
    
    far_away = 1;
    dist = 10; % distance to consider far
    for i=1:size(out_locations,2)
        x_loc = out_locations(1,i);
        y_loc = out_locations(2,i);
        if (norm([x_loc,y_loc]-[in_i,in_j])<dist)
            far_away = 0;
        end 
        % but if the flag was set due to 0s in empty matrix
        % add back
        if ((scores(i)==0) && (x_loc==0) && (y_loc==0))
            far_away = 1;
        end 
    end
    if (far_away) 
        % find closest score to replace
        min_s = min(scores);
        if (incoming_score>min_s)
            [~, idx] = find(scores==min_s,1);
            out_scores(idx) = incoming_score;
            out_locations(1,idx) = in_i;
            out_locations(2,idx) = in_j;
        end 
    end   
end
