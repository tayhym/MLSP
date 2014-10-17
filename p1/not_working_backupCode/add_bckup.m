% only add incoming point IF it is larger than at least 1 current value in matrix
% AND if it is far away in space from that point.
function [out_scores, out_locations] = add_bckup(scores, locations,incoming_score, in_i, in_j)
    % create copy 
    out_scores = scores;
    out_locations = locations;
    
    % find closest score lower than incoming_score
    min_s = min(scores);
    if (incoming_score<=min_s)
        return;
    end
    
    
    % find closest score
    [closest_score idx] = min(abs(scores-incoming_score));
    
    % empty matrix case
    if (closest_score==incoming_score)
        out_scores(idx) = incoming_score;
        out_locations(1,idx) = in_i;
        out_locations(2,idx) = in_j;
    end
     
    % find nearest x point, (and possibly y)
    [val idx] =min(abs(locations(1,:)-in_i));
    y_loc =locations(2,idx);
    
    if (norm(y_loc-in_j)>10) 
        % points far apart
        out_locations(1,idx) = in_i;
        out_locations(2,idx) = in_j;
        out_scores(idx) = incoming_score;
    else
        % points close by in y-direction: check x
        x_loc = locations(1,idx);
        if (norm(x_loc -in_i)>10)
            % far apart in x-direction
            out_locations(1,idx) = in_i;
            out_locations(2,idx) = in_j;
            out_scores(idx) = incoming_score;
        else 
            % close in x and y direction: add average
            out_locations(1,idx) = round((in_i+x_loc)/2);
            out_locations(2,idx) = round((in_j+y_loc)/2);
            
        end 
        
        
            
        
    
    
    % find attributes for score to be replaced
    replacement_score_i = find(scores==replacement_score,1);
    if (isempty(replacement_score_i))
        
    x_rs = locations(1,replacement_score_i);
    y_rs = locations(2,replacement_score_i);
    
    % find attributes for score that is incoming
    incoming_score_i = find(scores==incoming_score,1);
    x_in = locations(1,incoming_score_i);
    y_in = locations(2,incoming_score_i);
    
    loc_close = abs(norm([x_rs, y_rs]-[x_in, y_in])<10);
    
    score_close = abs(incoming_score-replacement_score)<10;
    
    if (loc_close && score_close) 
        % merge locations, and scores
        scores(replacement_score_i) = (replacement_score+incoming_score)/2;
        locations(