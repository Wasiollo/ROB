function para = para_multi(ts)
% Liczy parametry dla funkcji pdf_multi
% ts zbi�r ucz�cy (pr�bka = wiersz; w pierwszej kolumnie etykiety)
% para - struktura zawieraj�ca parametry:
%	para.labels - etykiety klas
%	para.mu - warto�ci �rednie cech (wiersz na klas�)
%	para.sig - macierze kowariancji cech (warstwa na klas�)

	labels = unique(ts(:,1));
	para.labels = labels;
	para.mu = zeros(rows(labels), columns(ts)-1);
	para.sig = zeros(columns(ts)-1, columns(ts)-1, rows(labels));

	% tu trzeba wype�ni� warto�ci �rednie i macierze kowariancji dla klas
	% macierz kowariancji liczy funkcja cov
  
    for i = 1:rows(labels);
    for j = 1:columns(ts)-1;
      para.mu(i, j) = mean(ts(ts(:, 1) == labels(i), j+1));
    endfor
    para.sig(:, :, i) = cov(ts(ts(:, 1) == labels(i), 2:end));
  endfor

end