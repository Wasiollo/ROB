[tvec tlab tstv tstl] = readSets(); 
[unique(tlab)'; unique(tstl)']
tlab += 1;
tstl += 1;

[unique(tlab)'; sum(tlab == unique(tlab)')]
[unique(tstl)'; sum(tstl == unique(tstl)')]


noHiddenNeurons = [100 100];
noEpochs = 50;
learningRate = 0.01;

rand()

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
    [layerWeights singleTerr] = backprop(tvec(i, :), tlab(i, :), layerWeights, learningRate);
    terr += singleTerr;
  endfor
  terr /= rows(tvec);

	clsRes = anncls(tvec, layerWeights);
	cfmxTrain = confMx(tlab, clsRes);
	errcf = compErrors(cfmxTrain);
	trainError(epoch) = errcf(2);

	clsRes = anncls(tstv, layerWeights);
	cfmxTest = confMx(tstl, clsRes);
	errcf2 = compErrors(cfmxTest);
	testError(epoch) = errcf2(2);
  
	epochTime = toc();
	disp([epoch epochTime trainError(epoch) testError(epoch)])
	resultReport = [resultReport; epoch epochTime trainError(epoch) testError(epoch)];

	fflush(stdout);
end

save rep.txt resultReport
save cfmxTrain.txt cfmxTrain 
save cfmxTest.txt cfmxTest 

plot(1:50, trainError, 'b', 1:50, testError, 'r')
xlabel('epoch');
ylabel('error [%]');
title ("Training and testing error during backprop");
legend ("train error", "test error");
set(findall(gcf,'-property','FontSize'),'FontSize',12);

