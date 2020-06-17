function pdf = pdf_parzen(pts, para)
% Aproksymuje warto�� g�sto�ci prawdopodobie�stwa z wykorzystaniem okna Parzena
% pts zawiera punkty, dla kt�rych liczy si� f-cj� g�sto�ci (punkt = wiersz)
% para - struktura zawieraj�ca parametry:
%	para.samples - tablica kom�rek zawieraj�ca pr�bki z poszczeg�lnych klas
%	para.parzenw - szeroko�� okna Parzena
% pdf - macierz g�sto�ci prawdopodobie�stwa
%	liczba wierszy = liczba pr�bek w pts
%	liczba kolumn = liczba klas

	pdf = rand(rows(pts), rows(para.samples));
	
	% przy liczeniu g�sto�ci warto zastanowi� si�
	% nad kolejno�ci� oblicze� (p�tli)
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
