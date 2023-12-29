function X = tlrr(Y,D,lambda,rho,tau,tol,maxiter)
%TSC Tensor low rank representation
%   X = TSC(Y,D,lambda,rho,tau,tol,maxiter) solves
%                 min_X |Y-D*X|_F^2+lambda|X|_*
%   using the inexact augmented Lagrangian method (IALM).
%
%   Inputs:
%     Y         - Observed tensor data (n1-by-n2-by-n3).
%     D         - Dictionary (n1-by-K-by-n3).
%     lambda    - Sparsity parameter.
%     rho       - Initial penalty parameter (default 1e-3).
%     tau       - Penalty update parameter (default 1.2).
%     tol       - Tolerance (default 1e-6).
%     maxiter   - Maximum number of iterations (default 1000).
%
%   Output:
%     X         - Tensor low rank code (K-by-n2-by-n3).
%

if size(Y,1)~=size(D,1) || size(Y,3)~=size(D,3)
    error('Tensor dimensions must agree.')
end
if nargin<4
    rho = 1e-3;
end
if nargin<5
    tau = 1.5;
end
if nargin<6
    tol = 1e-3;
end
if nargin<7
    maxiter = 30;
end

K = size(D,2);
[~,n2,n3] = size(Y);

% OY=Y;
% OD=D;
% Y=fft(Y,[],3);
% D=fft(D,[],3);

% Initialization.
X = zeros(K,n2,n3);
DD = zeros(K,K,n3);
DY = zeros(K,n2,n3);
for i = 1:n3
    % Precompute 2D'D and 2D'Y to speed things up.
    DD(:,:,i) = 2*D(:,:,i)'*D(:,:,i);
    DY(:,:,i) = 2*D(:,:,i)'*Y(:,:,i);
end
Z = zeros(K,n2,n3);
Q = zeros(K,n2,n3);

% Compute the sparse coefficient tensor using (15)-(17).
for iter = 1:maxiter
    %% Update X.
    for i = 1:n3
        X(:,:,i) = (DD(:,:,i)+rho*eye(K))\(DY(:,:,i)+rho*Z(:,:,i)-Q(:,:,i));
        
    end

    OX=ifft(X,[],3);
    OQ=ifft(Q,[],3);
    %% Update Z.
   OT=ifft(X+Q/rho,[],3);
%     OZ=prox(OT,lambda/rho);
     OZ= proxF_tSVD_1(OT,lambda/rho,[]);
%     [OZ]  =     proxF_l1(OT,lambda/rho)    ;
%        OZ= proxF_tube_12(OT,lambda/rho);
    Z=fft(OZ,[],3);
%     Z = prox(X+Q/rho,lambda/rho);

    %% Update Q.
    R = X-Z;
    Q = Q+rho*R;
    rho = tau*rho;

    %% Check for convergence.
    if norm(R(:))<tol
        return
    end
end
disp('Maximum iterations exceeded.')
