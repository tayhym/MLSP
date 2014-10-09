%--test add function--%

%normal add case
scores = [0 10 20 30];
loc  = [0   10 20 30;
        10  20 30 40];
    

 [s, l] = add(scores, loc, 40, 40,20);
 
 new_score = [40 10 20 30];
 new_loc = [40 10 20 30;
            20 20 30 40];
 assert(sum(sum(s==new_score))==sum(sum(ones(size(new_score)))));
 assert(sum(sum(l==new_loc))==sum(sum(ones(size(new_loc)))));
 
 %close in x and y direction
 scores = [0 10 20 30];
 loc  = [0   10 20 30;
        10  20 30 40];
 [s, l] = add(scores, loc, 40, 25,30);
 new_score = [0 10 20 30];
 new_loc  = [0   10 20 30;
            10  20 30 40];
 assert(sum(sum(s==new_score))==sum(sum(ones(size(new_score)))));
 assert(sum(sum(l==new_loc))==sum(sum(ones(size(new_loc)))));
 
 %smaller than all scores
 scores = [15 10 20 30];
 loc  = [0   10 20 30;
        10  20 30 40];
 [s, l] = add(scores, loc, 9, 100,100);
 new_score = [15 10 20 30];
 new_loc  = [0   10 20 30;
            10  20 30 40];
 assert(sum(sum(s==new_score))==sum(sum(ones(size(new_score)))));
 assert(sum(sum(l==new_loc))==sum(sum(ones(size(new_loc)))));
 
 %negative score
 scores = [0 10 20 30];
 loc  = [0   10 20 30;
        10  20 30 40];
 [s, l] = add(scores, loc, -50, 100,30);
 new_score = [0 10 20 30];
 new_loc  = [0   10 20 30;
            10  20 30 40];
 assert(sum(sum(s==new_score))==sum(sum(ones(size(new_score)))));
 assert(sum(sum(l==new_loc))==sum(sum(ones(size(new_loc)))));
 
%correct replacement
 scores = [100 15 200 30];
 loc  = [0   10 20 30;
        10  20 30 40];
 [s, l] = add(scores, loc, 16, 100,30);
 new_score = [100 16 200 30];
 new_loc  = [0   100 20 30;
            10  30 30 40];
 assert(sum(sum(s==new_score))==sum(sum(ones(size(new_score)))));
 assert(sum(sum(l==new_loc))==sum(sum(ones(size(new_loc)))));
 