function cfmx = confMx(truelab, declab)
% Computes confusion matrix cfmx given
% 	truelab - column vector of ground-truth labels (1..maxlabel)
% 	declab - column vector of classifiers decisions
% Output:
% 	cfmx - confusion matrix:
%		rows - are for ground truth
%		columns - are for classfier output

  labels = unique(truelab);
  cfmx = zeros(rows(labels), rows(labels));
  for i = 1:rows(truelab)
    cfmx(truelab(i), declab(i)) += 1;
  end
end
