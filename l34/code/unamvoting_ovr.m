function [clab, classifiers_clab] = unamvoting_ovr(tset, clsmx)
% Simple unanimity voting function 
% 	tset - matrix containing test data; one row represents one sample
% 	clsmx - voting committee matrix
%   	the first column contains positive class label
%   	the second column contains 0 as rest classes label
%   	columns (3:end) contain separating plane coefficients
% Output:
%	clab - classification result 

	% class processing
	labels = unique(clsmx(:, [1 2]));
	reject = max(labels) + 1;

	% cast votes of classifiers
	[votes, classifiers_votes] = voting(tset, clsmx);

	unanvote_for_rest = rows(labels) - 2; 
	[mv clab] = max(votes(:, 2:end), [], 2);

	% reject decision 
    reject_idx = (votes(:, 1) != unanvote_for_rest) | (votes(:, 1) == rows(labels) - 1);
    clab(reject_idx) = reject;
    classifiers_clab = {};

	for i = 1:columns(classifiers_votes)
		classifiers_clab{1, i} = classifiers_votes{1 ,i};
		[mv, classifiers_clab{2, i}] = max(classifiers_votes{2, i}(:, 2:end), [], 2);
		reject_idx = classifiers_votes{2, i}(:, 1) ==  1;
    classifiers_clab{2, i}(reject_idx) = reject;
	endfor