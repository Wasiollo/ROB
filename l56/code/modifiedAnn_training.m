[tvec tlab tstv tstl] = readSets(); 
[unique(tlab)'; unique(tstl)']
tlab += 1;
tstl += 1;

[unique(tlab)'; sum(tlab == unique(tlab)')]
[unique(tstl)'; sum(tstl == unique(tstl)')]


noHiddenNeurons = [100 100];
noEpochs = 50;
learningRate = 0.001;

rand()

setsNumber = rows(tvec);
indices = randperm(setsNumber);
tvec = tvec(indices, :);
tlab = tlab(indices, :);

meanTvec = mean(tvec);
stdTvec = std(tvec);
stdTvec(stdTvec == 0) = 1;

tvec = bsxfun(@minus, tvec, meanTvec);
tvec = bsxfun(@rdivide, tvec, stdTvec);

meanTstv = mean(tstv);
stdTstv = std(tstv);
stdTstv(stdTstv == 0) = 1;

tstv = bsxfun(@minus, tstv, meanTstv);
tstv = bsxfun(@rdivide, tstv, stdTstv);

% loading state of (pseudo)random number generator
load rndstate.txt 
rand("state", rndstate);

layerWeights = crann([columns(tvec) noHiddenNeurons 10]);
trainError = zeros(1, noEpochs);
testError = zeros(1, noEpochs);
resultReport = [];
for epoch=1:noEpochs
	tic();
  
  terr = 0;
  for i = 1:rows(tvec)
    [layerWeights singleTerr] = modifiedBackprop(tvec(i, :), tlab(i, :), layerWeights, learningRate);
    terr += singleTerr;
  endfor
  terr /= rows(tvec);

	clsRes = modifiedAnncls(tvec, layerWeights);
	cfmxTrain = confMx(tlab, clsRes);
	errcf = compErrors(cfmxTrain);
	trainError(epoch) = errcf(2);

	clsRes = modifiedAnncls(tstv, layerWeights);
	cfmxTest = confMx(tstl, clsRes);
	errcf2 = compErrors(cfmxTest);
	testError(epoch) = errcf2(2);
	epochTime = toc();
	disp([epoch epochTime trainError(epoch) testError(epoch)])
	resultReport = [resultReport; epoch epochTime trainError(epoch) testError(epoch)];
	fflush(stdout);
end

save modified_rep.txt resultReport 
save modified_cfmx_test.txt cfmxTest 
save modified_cfmx_train.txt cfmxTrain 

plot(1:50, trainError, 'b', 1:50, testError, 'r')
xlabel('epoch');
ylabel('error');
title ("Training and testing error during backprop");
legend ("train error", "test error");
