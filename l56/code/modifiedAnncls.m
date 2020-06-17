function lab = modifiedAnncls(tset, layersWeights)
  
	layersNumber = numel(layersWeights) + 1;
	setsNumber = rows(tset);

	act{1} = tset;
	for i=2:layersNumber
		response{i} = [act{i-1} ones(setsNumber, 1)] * layersWeights{i-1};
		act{i} = modifiedActf(response{i});
	endfor

	[~, lab] = max(act{layersNumber}, [], 2);
end