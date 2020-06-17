function decv = bayescls(samples, pdfunc, pdparams, apriori)
% Klasyfikator Bayesa 
% samples - macierz próbek do klasyfikacji (próbka = wiersz)
% pdfunc - uchwyt (handle) do funkcji licz¹cej pdf 
% pdparams - parametry dla funkcji licz¹cej pdf
% 	pdparams.labels - etykiety klas
% apriori - wektor prawdopodobieñstw apriori (wierszowy)
% decv - kolumnowy wektor etykiet (wynik klasyfikacji)

	pdfs = pdfunc(samples, pdparams);
	% a priori uwzglêdniamy tylko, jeœli podano parametr
	if nargin >= 4
		% podobnej konstrukcji u¿yliœmy na pierwszych zajêciach:
		% 	pdfs .*= repmat(apriori, rows(pdfs), 1);
		% bardziej efektywne jest u¿ycie funkcji bsxfun:
		pdfs = bsxfun(@times, pdfs, apriori);
	end
	
	% nie interesuje nas konkretna wartoœæ gêstoœci...
	[~, mi] = max(pdfs, [], 2);
	
	% translacja numer klasy -> etykieta
	decv = pdparams.labels(mi);
end