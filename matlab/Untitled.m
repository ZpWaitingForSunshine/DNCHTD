% A = rand(10, 10);
% B = rand(10, 4);
% C = rand(4, 5);
% 
% rest = vec(A * B * C);
% 
% rest2 = kron(C', A) * vec(B);

A = rand(10, 10);% ���뷽��A
B = rand(10, 4);% �������B
C =  rand(4, 5);% �������C

% ���� A\C(:)
D = A\C(:);

% ������Է����� B x + I = D
I = eye(size(B));
x = linsolve(B, D - I);

% ���� x ת���ؾ�����ʽ
X = reshape(x, size(B));