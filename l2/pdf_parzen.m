function pdf = pdf_parzen(pts, para)
% Aproksymuje wartoœæ gêstoœci prawdopodobieñstwa z wykorzystaniem okna Parzena
% pts zawiera punkty, dla których liczy siê f-cjê gêstoœci (punkt = wiersz)
% para - struktura zawieraj¹ca parametry:
%	para.samples - tablica komórek zawieraj¹ca próbki z poszczególnych klas
%	para.parzenw - szerokoœæ okna Parzena
% pdf - macierz gêstoœci prawdopodobieñstwa
%	liczba wierszy = liczba próbek w pts
%	liczba kolumn = liczba klas

	pdf = rand(rows(pts), rows(para.samples));
	
	% przy liczeniu gêstoœci warto zastanowiæ siê
	% nad kolejnoœci¹ obliczeñ (pêtli)
  for j=1:rows(para.samples)
    j % log
    hn = para.parzenw / sqrt(rows(para.samples{j}));
    for i=1:rows(pts)
      normparts = zeros(rows(para.samples{j}), 1);
      for k=1:rows(para.samples{j})
        normparts(k, 1) = prod(normpdf(para.samples{j}(k, :), pts(i, :), hn));
      endfor
      pdf(i, j) = mean(normparts);
    endfor
  endfor
end
