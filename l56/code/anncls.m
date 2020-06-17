function lab = anncls(tset, layersWeights)
% simple ANN classifier
% tset - data to be classified (every row represents a sample) 
% hidlw - hidden layer weight matrix
% outlw - output layer weight matrix

% layersWeights = [hidlw outlw]

% lab - classification result (index of output layer neuron with highest value)
% ATTENTION: we assume that constant value IS NOT INCLUDED in tset rows

	layersNumber = numel(layersWeights) + 1;
	setsNumber = rows(tset);

	act{1} = tset;
	for i=2:layersNumber
		response{i} = [act{i-1} ones(setsNumber, 1)] * layersWeights{i-1};
		act{i} = actf(response{i});
	endfor

	[~, lab] = max(act{layersNumber}, [], 2);
end