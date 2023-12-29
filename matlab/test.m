
R1 = 20
R2 = 6
R3 = 100
RB1 = 30
U1 = rand(25, R1);
U2 = rand(8, R2);
U3 = rand(191, R3);

U4 = rand(4, R3);
B1 = rand(RB1, R1 * R2);
B2 = rand(RB1 * R3, 1);

B1TENSOR = tensor(reshape(B1, R1, R2, RB1));

left = ttm(B1TENSOR, {U1, U2}, [1 2]);

left = tenmat(left, 3);
left = double(left)';

B2Matrix = reshape(B2, RB1, R3, 1);

right1 = left * B2Matrix;

A = rand(20, 30 ,40);
U1 = rand(20, 5);
U2 = rand(30, 6);
U3 = rand(40, 7);
R = ttm(tensor(A), {U1, U2, U3}, [1 2 3]);
R = double(tenmat(R, 3));

R1 = U3 * tenmat(tensor(A), 3) * kron(U1', U2');

R- R1




















% C = rand(100, 400, 300);
% t = double(tenmat(C, [1]));
% D = reshape(t, 100, 400, 300);