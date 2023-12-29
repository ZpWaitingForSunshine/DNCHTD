function [L,res,lda_new]=loop_grad(L,T,lda_old)


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

% gradient optimization

nbs= size(L{1},2);
I=size(T); % tensor dimensions
D=length(I); % tensor order (number of dimensions)
% reshape data tensor
Td=reshape(T,prod(I),1);
% old error and cost function value
p_old= zeros(sum(I)*nbs,1);
x_old= ones(1,nbs);
k=1;
for i=1:D
    p_old(k:k-1+I(i)*nbs)=vec(L{i}.');
    x_old=pkr(L{i},x_old);
    k=k+I(i)*nbs;
end
e_old= x_old*ones(nbs,1) - Td;
f_old= e_old'*e_old/(2);
f_new=2*f_old;
lda_new=lda_old;
grad=construct_gradient(L,T);

while f_new > f_old
    p_new= p_old -lda_new*grad;
    % rebuild factor matrices
    k=1;
    for i=1:D
        L_new{i}=unvec(p_new(k:k-1+I(i)*nbs),nbs,I(i)).';
        k=k+I(i)*nbs;
    end
    % new error and cost function value
    x_new=ones(1,nbs);
    for i=1:D
        x_new=pkr(L_new{i},x_new);
    end
    e_new= x_new*ones(nbs,1) - Td;
    f_new= e_new'*e_new/(2);
    % adjust the step size
    val_ref=.005; %(cela force l'augmentation du pas lorsque f_new < f_old) %.005;
    if f_new>= f_old
        lda_new=lda_old/2;
        lda_old=lda_new; % ajout� (car il faut garder lda_old pour le coup d'apr�s!)
    elseif (f_old - f_new)/f_old < val_ref
        lda_new=1.1*lda_old;
        lda_old=lda_new;
        L=L_new;
    else
        L=L_new;
    end
end
res=e_new'*e_new/sum((Td(:)).^2);

% gradient vector
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

