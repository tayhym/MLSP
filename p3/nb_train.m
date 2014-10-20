% weighted naive bayes training model
% weights: 
% weighted_err: use weights to determine the weighted error of model
function [model] = nb_train(Xtrain, Ytrain, weights)
    M = size(Xtrain,1); % counts (prior to weighting)
    N = size(Xtrain,2); % num features
    
    assert(sum(abs(weights)==weights)==size(weights,1));
    weighted_egs = Ytrain.*weights;  % assumes weights all positive
    
    class_one_eg = weighted_egs(Ytrain==1);
    class_two_eg = weighted_egs(Ytrain==-1);
    weighted_counts = (Xtrain.*repmat(weights,[1,10]));
    class_one_weighted = weighted_counts(Ytrain==1,:);
    class_two_weighted = weighted_counts(Ytrain==-1,:);
    class_one_counts_total = sum(sum(class_one_weighted));
    class_two_counts_total = sum(sum(class_two_weighted));
    
    class_one_examples = sum(abs(class_one_eg));
    class_two_examples = sum(abs(class_two_eg));
    total_examples = (class_one_examples+class_two_examples);
    
    classOne_prior = class_one_examples/total_examples;
    classTwo_prior = class_two_examples/total_examples;
    
    priors = [classOne_prior, classTwo_prior];
    
    alpha = 1; % smoothing parameter
    d = N; % num_outcomes
    class_one_conditionals =  (sum(class_one_weighted,1)+alpha)./(class_one_counts_total+(alpha*d));
    class_two_conditionals =  (sum(class_two_weighted,1)+alpha)./(class_two_counts_total+(alpha*d));
    
    
%     class_one_conditionals = sum(class_one_weighted,1)/class_one_examples;
%     class_two_conditionals = sum(class_two_weighted,1)/class_two_examples;
    
    model = struct('priors',priors,  ...
                   'one_conditionals',class_one_conditionals, ...
                   'two_conditionals',class_two_conditionals);

    
end 

    
    
    
    
    
    