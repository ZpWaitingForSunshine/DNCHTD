% A = rand(10, 10);
% B = rand(10, 4);
% C = rand(4, 5);
% 
% rest = vec(A * B * C);
% 
% rest2 = kron(C', A) * vec(B);

A = rand(10, 10);% 输入方阵A
B = rand(10, 4);% 输入矩阵B
C =  rand(4, 5);% 输入矩阵C

% 计算 A\C(:)
D = A\C(:);

% 求解线性方程组 B x + I = D
I = eye(size(B));
x = linsolve(B, D - I);

% 将解 x 转换回矩阵形式
X = reshape(x, size(B));