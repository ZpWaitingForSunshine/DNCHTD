function [L_new,res,d_new,Hess,grad,delta_new]=loop_nlcg(L,T,d,Hess,grad,delta_old)

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

%Conjugate gradient optimization with Enhanced Line Search

% jmax=10;
% eps1=0;%1e-20;
F= size(L{1},2);
I=size(T); % tensor dimensions
D=length(I); % tensor order (number of dimensions)
% reshape data tensor
Td=reshape(T,prod(I),1);
Td2=reshape(T,[I(1) prod(I(2:end))]);
N=size(Td2,2);
% old error and cost function value
p_old= zeros(sum(I)*F,1);
x_old= ones(1,F);
k=1;
for i=1:D
    p_old(k:k-1+I(i)*F)=vec(L{i}.');
    x_old=pkr(L{i},x_old);
    k=k+I(i)*F;
end
r_old=-grad;

% fd=d'*d;
% alpha=-grad'*d/(d'*Hess*d);
% p=p_old+alpha*d;
% k=1;
% for i=1:D
%     L_new{i}=unvec(p(k:k-1+I(i)*F),F,I(i)).';
%     k=k+I(i)*F;
% end
% j=1;
% tol=(alpha.^2)*fd;
% while j<=jmax && tol>eps1
%     [Hess,Jac]=construct_Hessian(L_new,T);
%     grad=construct_gradient(L_new,T);
%     alpha=-grad'*d/(d'*Hess*d);
%     p=p+alpha*d;
%     k=1;
%     for i=1:D
%         L_new{i}=unvec(p(k:k-1+I(i)*F),F,I(i)).';
%         k=k+I(i)*F;
%     end
%     tol=(alpha.^2)*fd;
%     j=j+1;
% end
% [Hess,Jac]=construct_Hessian(L_new,T);
% grad=construct_gradient(L_new,T);

k=1;
for i=1:D
    K{i,1}=unvec(d(k:k-1+I(i)*F),F,I(i)).';
    k=k+I(i)*F;
    K{i,2}=L{i};
end
alpha=els(Td2,K,N,F);
p=p_old+alpha*d;
k=1;
for i=1:D
    L_new{i}=unvec(p(k:k-1+I(i)*F),F,I(i)).';
    k=k+I(i)*F;
end
%[Hess,Jac]=construct_Hessian(L_new,T);
grad=construct_gradient(L_new,T);

x_new=ones(1,F);
for i=1:D
    x_new=pkr(L_new{i},x_new);
end
e_new= x_new*ones(F,1) - Td;
%f_new= e_new'*e_new;
res=e_new'*e_new/sum((Td(:)).^2);
r=-grad;
delta_new=r'*r;
delta1=r'*r_old;
%beta=delta_new/delta_old;
beta=(delta_new-delta1)/delta_old;
d_new=r+beta*d;

function g=construct_gradient(L,T)
I=size(T);
D=length(I);
nbs=size(L{1},2);
d=1:D;
g=zeros(sum(I)*nbs,1);
k=1;
for i=1:D   
    d=circshift(d,[0,-1]);
    TT=permute(T,d);
    Td=reshape(TT,prod(I),1);    
    Z1=ones(nbs);
    Z2=ones(1,nbs);
    for j=1:D-1
        Z1=hdp(L{d(j)}'*L{d(j)},Z1);    
        Z2=pkr(L{d(j)},Z2);        
    end
    g(k:k-1+I(i)*nbs)= kron(eye(I(i)),Z1)*vec(L{d(end)}.')...
    - kron(eye(I(i)),Z2)'*Td;
    k=k + I(i)*nbs;
end


