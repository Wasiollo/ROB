function [clab, classifiers_clab] = unamvoting(tset, clsmx)
% Simple unanimity voting function 
% 	tset - matrix containing test data; one row represents one sample
% 	clsmx - voting committee matrix
%   	the first column contains positive class label
%   	the second column contains negative class label
%   	columns (3:end) contain separating plane coefficients
% Output:
%	clab - classification result 

	% class processing
	labels = unique(clsmx(:, [1 2]));
	reject = max(labels) + 1;

	% cast votes of classifiers
	[votes, classifiers_votes] = voting(tset, clsmx);

	maxvotes = rows(labels) - 1; % unanimity voting in one vs. one scheme

	[mv clab] = max(votes, [], 2);

	% reject decision 
	clab(mv ~= maxvotes) = reject;

	classifiers_clab = {};

	for i = 1:columns(classifiers_votes)
		classifiers_clab{1, i} = classifiers_votes{1 ,i};
		[mv, classifiers_clab{2, i}] = max(classifiers_votes{2, i}, [], 2);
	endfor
