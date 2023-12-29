function Ym = matricize2( Y )

Y = tensor(Y);
Ym = cell( ndims(Y), 1 );
for d = 1:ndims(Y)
    temp  = tenmat(Y,d);
    Ym{d} = temp.data;
end
if ndims(Y)==3
    Ym{4}=  Y(:)';
end
