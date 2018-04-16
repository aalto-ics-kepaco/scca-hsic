function g = gradf_gauss(K1,cK2,X,a,u)

N = size(K1,1);
g = 0 * u;
temp = 0 * (u * u');
for i = 1:N
    for j = 1:N
        temp = temp + K1(i,j) * cK2(i,j) * (X(i,:)-X(j,:))' * (X(i,:)-X(j,:));
    end
end
g = -(2 * a * u' * temp)';


end