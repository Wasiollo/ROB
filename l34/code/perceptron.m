function [sepplane fp fn] = perceptron(pclass, nclass)
% Computes separating plane (linear classifier) using
% perceptron method.
% pclass - 'positive' class (one row contains one sample)
% nclass - 'negative' class (one row contains one sample)
% Output:
% sepplane - row vector of separating plane coefficients
% fp - false positive count (i.e. number of misclassified samples of pclass)
% fn - false negative count (i.e. number of misclassified samples of nclass)

  sepplane = rand(1, columns(pclass) + 1) - 0.5;
  tset = [ ones(rows(pclass), 1) pclass; -ones(rows(nclass), 1) -nclass];
  nPos = rows(pclass); % number of positive samples
  nNeg = rows(nclass); % number of negative samples

  i = 1;
  do 
	%%% YOUR CODE GOES HERE %%%
	%% You should:
	%% 1. Check which samples are misclassified (boolean column vector)
	%% 2. Compute separating plane correction 
	%%		This is sum of misclassfied samples coordinate times learning rate 
	%% 3. Modify solution (i.e. sepplane)

	%% 4. Optionally you can include additional conditions to the stop criterion
	%%		200 iterations can take a while and probably in most cases is unnecessary

  misclassifiedIndexes = (sepplane*tset') < 0;
  if (sum(misclassifiedIndexes) == 0)
    break;
  endif;
  sumOfMisscorrections = sum(tset(misclassifiedIndexes, :), 1);
  
  correctionFactor = 1.0/sqrt(i);
  sepplane += correctionFactor*sumOfMisscorrections;

	++i;
  until i > 200;

  %%% YOUR CODE GOES HERE %%%
  %% You should:
  %% 1. Compute the numbers of false positives and false negatives
  misclassifiedIndexes  = (sepplane*tset') < 0;
  fp = sum(misclassifiedIndexes(1:nPos))/nPos;
  fn = sum(misclassifiedIndexes(nPos+1:nPos+nNeg))/nNeg;
  
  
