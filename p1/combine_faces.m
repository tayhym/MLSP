function [face_loc_combined] = combine_faces(all_rects)
    num_rects = size(all_rects,1);
    face_loc_combined = [];
    for i=1:num_rects
        rect = all_rects(i,:);
        [proximal, closest] = search_faces(rect,face_loc_combined);
        if (~proximal)
            face_loc_combined(end+1,:) = rect; 
        else 
            % replace proximal by ave of two overlaping rects
            [~,row_idx] = ismember(closest,face_loc_combined,'rows');
            face_loc_combined(row_idx,:) = (rect + closest)/2; 
            
            tmp = removerows(face_loc_combined,row_idx);
            [new_proximal,new_closest] = search_faces( ... 
            face_loc_combined(row_idx,:),tmp);
            while (new_proximal)
                [~,idx_new] = ismember( ...  
                                   new_closest,tmp,'rows');
                tmp(idx_new,:) =  ...
                        (face_loc_combined(row_idx,:)+new_closest)/2;
                face_loc_combined = tmp;
                
                tmp = removerows(tmp,idx_new);
            
                [new_proximal,new_closest] = search_faces( ... 
                face_loc_combined(idx_new,:),tmp);
            end 
            
        end 
    end 
end 

            

function [proximal,closest_rect] = search_faces(rect, all_rects)
    min_dist = -1;
    closest_rect = zeros(1,2);
    proximal = 0;
    for ii=1:size(all_rects,1)
        dist = 64; % approximate size of eigenface
        if (norm(rect- all_rects(ii,:))<=dist)
            proximal =  1;
        end
        
        if ((min_dist == -1) || (norm(rect-all_rects(ii,:))<=min_dist))
            min_dist = norm(rect-all_rects(ii,:));
            closest_rect = all_rects(ii,:);
        end
        
    end 
end 

            
        
    
        