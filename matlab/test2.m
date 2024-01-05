A = rand(40, 40);
A = A' + A;
X = rand(40,30);
B = rand(30, 20);

C = A * X * B;

X1 = (((A' * C * B') / (B * B') )' / (A' * A)')';

inv(A) * C -  X * B

X2 = inv(A) * C / B