function [L_newv,res,lda_new,nu,Hess,grad,Jac,flag,gain]=loop_lm(L,T,lda_old,nu,Hess,grad,Jac,eps2,eps1,flag)

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

% Levenberg-Marquardt optimization 

L_newv=L;
F= size(L{1},2);
I=size(T); % tensor dimensions
D=length(I); % tensor order (number of dimensions)
% reshape data tensor
Td=reshape(T,prod(I),1);
% old error and cost function value
p_old= zeros(sum(I)*F,1);
x_old= ones(1,F);
k=1;
for i=1:D
    p_old(k:k-1+I(i)*F)=vec(L{i}.');
    x_old=pkr(L{i},x_old);
    k=k+I(i)*F;
end
e_old= x_old*ones(F,1) - Td;
f_old= e_old'*e_old/(2);
% step
dp= -inv(Hess + lda_old*eye(sum(I)*F))*grad;
if norm(dp)<eps2*(norm(p_old)+eps2);
    flag=1;
    %f_new=f_old;
    lda_new=lda_old;
    e_new=e_old;
    gain=0;
else 
    % update p-vector
    p_new= p_old + dp;
    % rebuild factor matrices
    k=1;
    for i=1:D
        L_new{i}=unvec(p_new(k:k-1+I(i)*F),F,I(i)).';
        k=k+I(i)*F;
    end
    % new error and cost function value
    x_new=ones(1,F);
    for i=1:D
        x_new=pkr(L_new{i},x_new);
    end
    e_new= x_new*ones(F,1) - Td;
    f_new= e_new'*e_new/(2);
    Jp=(Jac*dp)';
    f_new_approx = f_old + Jp*e_old + Jp*Jp'/2;    
    % calculate gain ratio
    gain= ((f_old - f_new)/(f_old - f_new_approx));    
    % update lambda (regularization factor)
    
    %%% Strategy 1
%     if gain >.75
%         lda_new=lda_old/3;
%         L_newv=L_new;
%         [Hess,Jac]=construct_Hessian(L_new,T);
%         grad=construct_gradient(L_new,T);
%         flag=(max(grad)<eps1);
%     elseif gain <.25
%         lda_new=lda_old*2;
%         L_newv=L;
%    
%     else
%         lda_new=lda_old;
%         L_newv=L;
% %         [Hess,Jac]=construct_Hessian(L_new,T);
% %         grad=construct_gradient(L_new,T);
%     end

    %%% Strategy 2
    if gain >0
        lda_new=lda_old*max(1/3,1-(2*gain-1)^3);
        nu=2;
        L_newv=L_new;
        [Hess,Jac]=construct_hessian(L_new,T);
        grad=construct_gradient(L_new,T);
        flag=(max(grad)<eps1);
    else
        lda_new=lda_old*nu;
        nu=2*nu;
%       f_new=f_old;
       e_new=e_old;
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


% Hessian and Jacobian matrices
function [Hes,Jac]=construct_hessian(L,T)
I=size(T);
D=length(I);
nbs=size(L{1},2);
%TT=T;
d=1:D;
M=prod(I);
% initialize Jacobian and Hessian matrices
Jac= zeros(prod(I),sum(I)*nbs);
%Hes= zeros(nbs,nbs);
k=1;
for i=1:D
    d=circshift(d,[0,-1]);       
    Z=ones(1,nbs);
    for j=1:D-1 
        Z=pkr(L{d(j)},Z);
    end   
    index=reshape(permute(reshape(1:M,I),d),M,1);
    for p=1:length(index)
        index_rows(index(p))=p;
    end    
    ZZ= kron(eye(I(i)),Z);  
    Jac(:,k:k-1+I(i)*nbs)= ZZ(index_rows,:);    
    k=k + I(i)*nbs;
end
Hes=Jac'*Jac;

