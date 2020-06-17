function [outputNetwork terr] = modifiedBackprop(tset, tslb, layersWeights, lr)
% derivative of sigmoid activation function
% tset - training set (every row represents a sample)
% tslb - column vector of labels 

% lr - learning rate

	layersNumber = numel(layersWeights) + 1;
	labelsNumber = columns(layersWeights{layersNumber - 1});

	setsNumber = rows(tset);

	act{1} = tset;

	for i = 2:layersNumber
		response{i} = [act{i-1} ones(setsNumber, 1)] * layersWeights{i - 1};
		act{i} = modifiedActf(response{i});
	endfor

	for i = 2:layersNumber
		layersGradient{i - 1} = zeros(size(layersWeights{i - 1}));
	endfor

	prefferedOutput = zeros(setsNumber, labelsNumber);
	for i = 1:setsNumber
		prefferedOutput(i, tslb(i)) = 1;
	endfor

	d{layersNumber} = prefferedOutput - response{layersNumber};

	for i = layersNumber - 1 : -1 : 1
		d{i} = (d{i+1} * layersWeights{i}') .* [modifiedActdf(act{i}) ones(setsNumber, 1)];
		d{i} = d{i}(:, 1:end - 1);
		D{i} = d{i + 1}' * [act{i} ones(setsNumber, 1)];
		layersGradient{i} = lr * D{i}';
	endfor

	terr = 0.5 * sum((prefferedOutput - act{layersNumber})(:).^2) / setsNumber;

	for i = 2:layersNumber
		outputNetwork{i - 1} = layersWeights{i - 1} + layersGradient{i - 1};
	endfor

end