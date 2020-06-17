function ercf = jackknife(ts)
% leave-one-out test of cls1nn classifier
% ts - training set (first column contains labels)
% ercf - error coefficient of cls1nn on ts
  res = zeros(rows(ts), 1);
  
  for i=1:rows(ts)
    res(i) = cls1nn(ts(1:end != i,:), ts(i, 2:end));
  endfor
  
  ercf = mean(res != ts(:, 1));
  
endfunction
