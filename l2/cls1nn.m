function lab = cls1nn(ts, x)
  % 1-nn classifier using ts as training set
  % ts - training set (first column contains labels)
  % x - sample to be classified ( no label column here)
  % lab - x's neares neighbour label
  distsq = sumsq(ts(:, 2:end) - x, 2);
  [mv mi] = min(distsq);
  lab = ts(mi, 1);
  
endfunction
