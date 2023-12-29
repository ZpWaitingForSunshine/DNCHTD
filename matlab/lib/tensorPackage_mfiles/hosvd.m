function [S,W]=hosvd(X,W)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This function is part of the:                                    %% 
%%                                                                  %%
%% Tensor package v 1.0 (february 2011),                            %%
%% Algorithms for the CP decomposition of multy-way arrays          %%
%% X. Luciani, A.L.F de Almeida and P. Comon                        %%
%%                                                                  %%
%%                                                                  %%
%%   v 1.0 features:                                                %%
%%                                                                  %%
%%   - Works whatever the tensor order                              %%
%%   - Four optimization algorithms (see below)                     %%
%%   - Enhanced Line Search procedure (ELS)                         %%
%%   - Positivity constraint                                        %%
%%   - High Order Singular Value Decomposition                      %%
%%   - Deal with real or complex tensors                            %%
%%                                                                  %%   
%%   v 1.0 limitations:                                             %%
%%                                                                  %%
%%   - Only ALS works with complex tensor                           %%
%%   - It seems that ELS doesn't work with tensors of order 5 and 7 %%
%%   - This ELS code should not work fine with complex tensors      %%
%%   - Positivity constraint only works with ALS                    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% [S,W]=hosvd(X) computes the High-Order Singular Value Decomposition
% (HOSVD) of an N-th order tensor X of dimensions I_1 x I_2 x ... x I_N.
%
% The outputs parameters: 
% S: is an N-th array of dimensions R_1 x R_2 x ... x R_N, 
% where R_1, R_2, ..., R_N are the mode-1, mode-2, ..., mode-N ranks of the 
% unfolded matrix representations X1, X2, ..., X3 of the tensor X, respectively.
% W: is a cell-array of N elements, the n-th element is the mode-n
% eigenmatrix. 
%
% X=hosvd(S,W) rebuilds the original tensor X from S and W{1},...,W{N}
%

c=nargin;

switch c
    case 1 % Decomposition
        % dimensions
        I= size(X);
        % order
        N=length(I);
        % hosvd
        R=zeros(1,N);
        S=X;
        for n=1:N
            % n-th mode unfolded matrix
            Sd= reshape(S,I(1),prod(I(2:end)));
            % SVD of Sd
            [U,D,V]=svd(Sd);
            % mode-n rank
            R(n)=rank(D);
            %     [b,s]=seuil(diag(D));
            %     R(n)=length(b);
            %   R(n)=rg;
            % mode-n eigenmatrix and transformed tensor
            W{n}=U(:,1:R(n));
            Sd= W{n}'*Sd;
            S= reshape(Sd,[R(n) I(2:end)]);
            % circularly permute dimensions of tensor S
            S=shiftdim(S,1);
            I=size(S);
        end
    case 2 % Reconstruction
        S=X;
        N=length(W);
        I=size(S);
        for n=1:N
            %SS{n}=reshape(S,I(1),prod(I(2:end)));
            S=shiftdim(S,1);
            I=size(S);
        end
        I1=size(S);
        X1=S;
        for n=1:N
            XX= reshape(X1,I1(1),prod(I1(2:end)));
            XX=W{n}*XX;
            X1= reshape(XX,[size(W{n},1) I1(2:end)]);
            X1=shiftdim(X1,1);
            I1=size(X1);
        end
        S=X1;
end

