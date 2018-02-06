function g = gradf_poly_SGD(K1,cK2,X,a,u,r)


N = size(K1,1);
g = 0*u;
temp = 0*(u*u');
[~,id1]=sort(rand(1,N));
[~,id2]=sort(rand(1,N));
N=floor(N/10);

for i = id1(1:1:N)
    for j = id2(1:1:N)
        temp = temp + K1(i,j)/(r+(X(i,:)*u)*(X(j,:)*u)) * cK2(i,j) * (X(i,:)'*X(j,:));
    end
end
g =2*a*temp*u;


end