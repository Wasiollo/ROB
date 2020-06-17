pkg load statistics

% AD. 1
[train test] = load_cardsuits_data;

size(train)
size(test)
labels = unique(train(:,1))
unique(test(:,1))
[labels'; sum(train(:,1) == labels')]

% AD. 2
[mean(train); median(train)]

hist(train(:,1))

plot2features(train, 2, 3)

[mv midx] = max(train)
midx = 186
train(midx-1:midx+1, :)

size(train)
train(midx, :) = [];
size(train)

plot2features(train, 2, 3)

[mv midx] = min(train)
midx = 641
train(midx-1:midx+1, :)

size(train)
train(midx, :) = [];
size(train)

plot2features(train, 2, 3)

% AD. 3
first_idx = 3;
second_idx = 4;

plot2features(train, first_idx, second_idx)

train = train(:, [1 first_idx second_idx]);
test = test(:, [1 first_idx second_idx]);

pdfindep_para = para_indep(train);
pdfmulti_para = para_multi(train);
pdfparzen_para = para_parzen(train, 0.001); 

base_ercf = zeros(1,3);
base_ercf(1) = mean(bayescls(test(:,2:end), @pdf_indep, pdfindep_para) != test(:,1));
base_ercf(2) = mean(bayescls(test(:,2:end), @pdf_multi, pdfmulti_para) != test(:,1));
base_ercf(3) = mean(bayescls(test(:,2:end), @pdf_parzen, pdfparzen_para) != test(:,1));
base_ercf

% AD. 4

parts = [0.1 0.25 0.5];
rep_cnt = 5;

ad4_result = {};
for ad4_i=1:columns(parts);
    ad4_i % log
    ercf = zeros(3, rep_cnt);
    for ad4_j=1:rep_cnt;
        ad4_j % log
        red_train = reduce(train, repmat(parts(ad4_i), 1, 8));
        pdfindep_para = para_indep(red_train);
        pdfmulti_para = para_multi(red_train);
        pdfparzen_para = para_parzen(red_train, 0.001); 
        ercf(1, ad4_j) = mean(bayescls(test(:,2:end), @pdf_indep, pdfindep_para) != test(:,1));
        ercf(2, ad4_j) = mean(bayescls(test(:,2:end), @pdf_multi, pdfmulti_para) != test(:,1));
        ercf(3, ad4_j) = mean(bayescls(test(:,2:end), @pdf_parzen, pdfparzen_para) != test(:,1));
    endfor
    ad4_result{1, ad4_i} = parts(ad4_i);
    ad4_result{2, ad4_i}(1, :) = [min(ercf(1, :)), max(ercf(1, :)), mean(ercf(1, :)), std(ercf(1, :))];
    ad4_result{2, ad4_i}(2, :) = [min(ercf(2, :)), max(ercf(2, :)), mean(ercf(2, :)), std(ercf(2, :))];
    ad4_result{2, ad4_i}(3, :) = [min(ercf(3, :)), max(ercf(3, :)), mean(ercf(3, :)), std(ercf(3, :))];
endfor

ad4_result

% AD. 5

parzen_widths = [0.0001, 0.0005, 0.001, 0.005, 0.01];
parzen_res = zeros(1, columns(parzen_widths));

for ad5_i = 1:columns(parzen_widths)
  ad5_i & log
  pdfparzen_para = para_parzen(train, parzen_widths(ad5_i));
  parzen_res(ad5_i) = mean(bayescls(test(:,2:end), @pdf_parzen, pdfparzen_para) != test(:,1));
endfor  

[parzen_widths; parzen_res]
semilogx(parzen_widths, parzen_res)

% AD. 6

apriori = [0.165 0.085 0.085 0.165 0.165 0.085 0.085 0.165];
parts = [1.0 0.5 0.5 1.0 1.0 0.5 0.5 1.0];
rep_cnt = 5;

ad6_ercfs = zeros(rep_cnt, 3);
ad6_cfmxs = {}

pdfindep_para = para_indep(train);
pdfmulti_para = para_multi(train);
pdfparzen_para = para_parzen(train, 0.001); 

for ad6_i = 1:rep_cnt
  ad6_i
  
  reduced_test = reduce(test, parts);
 
  classif_res = zeros(rows(reduced_test), 3);
  classif_res(:, 1) = bayescls(reduced_test(:,2:end), @pdf_indep, pdfindep_para, apriori);
  classif_res(:, 2) = bayescls(reduced_test(:,2:end), @pdf_multi, pdfmulti_para, apriori);
  classif_res(:, 3) = bayescls(reduced_test(:,2:end), @pdf_parzen, pdfparzen_para, apriori);
  
  ad6_ercfs(ad6_i, 1) = mean(reduced_test(:,1) != classif_res(:, 1));
  ad6_ercfs(ad6_i, 2) = mean(reduced_test(:,1) != classif_res(:, 2));
  ad6_ercfs(ad6_i, 3) = mean(reduced_test(:,1) != classif_res(:, 3));
  ad6_cfmxs{1, ad6_i} = confMx(reduced_test(:,1), classif_res(:, 1));
  ad6_cfmxs{2, ad6_i} = confMx(reduced_test(:,1), classif_res(:, 2));
  ad6_cfmxs{3, ad6_i} = confMx(reduced_test(:,1), classif_res(:, 3));
  
endfor

ad6_ercfs
ad6_cfmxs

ad6_bayescls_full_result = zeros(rows(test), 3);
ad6_bayescls_full_result(:, 1) = bayescls(test(:,2:end), @pdf_indep, pdfindep_para);
ad6_bayescls_full_result(:, 2) = bayescls(test(:,2:end), @pdf_multi, pdfmulti_para);
ad6_bayescls_full_result(:, 3) = bayescls(test(:,2:end), @pdf_parzen, pdfparzen_para);

ad6_cfmxs_full = {};
ad6_cfmxs_full{1} = confMx(test(:, 1), ad6_bayescls_full_result(:, 1));
ad6_cfmxs_full{2} = confMx(test(:, 1), ad6_bayescls_full_result(:, 2));
ad6_cfmxs_full{3} = confMx(test(:, 1), ad6_bayescls_full_result(:, 3));

ad6_bayescls_full_result
ad6_cfmxs_full

% AD 7

std(train(:,2:end))

per_class = zeros(8,5);
for ad7_i = 1:8;
  per_class(ad7_i, 1) = ad7_i;
  per_class(ad7_i, 2:3) = mean(train(train(:, 1) == ad7_i, 2:end));
  per_class(ad7_i, 4:5) = std(train(train(:, 1) == ad7_i, 2:end));
endfor

per_class

for ad7_j = 1:rows(test)
  results_1nn(ad7_j, 1) = cls1nn(train, test(ad7_j, 2:end));
endfor

ercf_1nn = mean(results_1nn != test(:, 1));
confmx_1nn = confMx(test(:, 1), results_1nn(:, 1));

ercf_1nn
confmx_1nn
