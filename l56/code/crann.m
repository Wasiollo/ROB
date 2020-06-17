function layersWeights = crann(layersParams)
% generates hidden and output ANN weight matrices

	layersNumber = numel(layersParams);

	for i = 1:layersNumber - 1
		layersWeights{i} = (rand(layersParams(i) + 1, layersParams(i + 1)) - 0.5) / sqrt(layersParams(i) + 1);
	endfor
