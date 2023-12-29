function Z=pkr(A,B);
%effectue le produit de khatri-rao entre deux matrices A et B et retourne le résultat
%dans Z 

%[I,F]=size(A);                               
%J=size(B,1);
F=size(A,2);
if F~=size(B,2)
    error('Matrices A and B must have the same number of columns');
end

Z=zeros(size(A,1)*size(B,1),F);
for f=1:F
%     Af=(repmat(A(:,f),[1,J])).';
%     Z(:,f)=Af(:).*repmat(B(:,f),[I,1]);
    Z(:,f)=kron(A(:,f),B(:,f));
end