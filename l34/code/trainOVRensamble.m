function ovrsp = trainOVRensamble(tset, tlab, htrain)
% Trains a set of linear classifiers (one versus rest class)
% on the training set using trainSelect function
% tset - training set samples
% tlab - labels of the samples in the training set
% htrain - handle to proper function computing separating plane
% ovosp - one versus one class linear classifiers matrix
%   the first column contains positive class label
%   the second column contains negative class label
%   columns (3:end) contain separating plane coefficients

  labels = unique(tlab);
  
  % pair containing class_lab:0 to indicate one vs rest classifiers
  pairs = zeros(rows(labels), 2);
  pairs(: , 1) = labels;
  ovrsp = zeros(rows(labels), 2 + 1 + columns(tset));
  
  for i=1:rows(pairs)
	% store labels in the first two columns
    ovrsp(i, 1:2) = pairs(i, :);
	
	% select positive samples as exaples of one class and negative as the rest of training set 
    posSamples = tset(tlab == pairs(i,1), :);
    negSamples = tset(tlab != pairs(i,1), :);
	
	% train 5 classifiers and select the best one
    [sp fp fn] = trainSelect(posSamples, negSamples, 5, htrain);
	    
  % store the separating plane coefficients (this is our classifier)
	% in ovo matrix
    ovrsp(i, 3:end) = sp;
  end
