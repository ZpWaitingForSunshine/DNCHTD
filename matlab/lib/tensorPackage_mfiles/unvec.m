% Implementation of the unvec(x) operator
% obs: For an MN x 1 column vector, the unvec(x) operator is
% the inverse of the vec(x) one and returns an M x N  matrix.
function A= unvec(a,M,N);

if size(a,1)==1
    a=a.';
end
A=reshape(a,M,N);