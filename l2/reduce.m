function rds = reduce(ds, parts)
% Funkcja redukcji liczby pr�bek poszczeg�lnych klas w zbiorze ds
% ds - zbi�r danych do redukcji; pierwsza kolumna zawiera etykiet�
% parts - wierszowy wektor wsp�czynnik�w redukcji dla poszczeg�lnych klas

	labels = unique(ds(:,1));
	if rows(labels) ~= columns(parts)
		error("Liczba klas nie zgadza sie z liczba wsp. redukcji.");
	end

	if max(parts) > 1 || min(parts) < 0
		error("Niewlasciwe wspolczynniki redukcji.");
	end
		
	% zdecydowanie wypadaloby uzyc randperm do mieszania probek w klasach
	% ta implementacja jest daleka od doskonalosci
	rds = [];
	for i=1:rows(labels);
		class_entries = ds(ds(:, 1) == labels(i, 1), :);
		keep_num = round(parts(i) * rows(class_entries));
		indexes = randperm(rows(class_entries), keep_num);
		rds = [rds; class_entries(indexes, :)];
	endfor
end