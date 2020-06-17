load iris2.txt ;
iris2(:, 1) = 2 ;
load iris3.txt ;
iris3(:,1) =3 ;
iris = [iris2; iris3];
jackknife(iris)