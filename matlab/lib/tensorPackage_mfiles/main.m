function [Lf,res,r,err_vec]=main(T,R,Algo,niter,crit,ini,elsper,ishosvd,alpha0,col,Lv);

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
%Main program:

% Performs the CP decomposition of tensor T for a given rank R.

% The string Algo specifies the optimization algorithm to be used: 
% Set algo to 'als' (Alternating Least Squares),
%             'grad', (Gradient descent)
%             'lm' (Levenberg-Marquardt) or
%             'nlcg' (Conjugate gradient with optimal step)

% niter is the maximal number of itération

%crit is the value of the stopping criterion (exit at the iteration k if
%|e(k)-e(k-1)|/e(k-1)<crit where e is the reconstruction error)


% ini=0: Random initialization of the loading matrices 
% ini=1: Initialize with the loading matrices saved in Fac.mat. Each run of
% main.m saves initial loading matrices in Fac.mat

% elsper is the ELS cycle, 0 means no ELS, (unusefull with 'nlcg') 

% ishosvd = 1 performs the HO-SVD of the tensor T before its decomposition

% alpha0 specifies the number of iterations affected by the non negativity 
% constraint. 0 means no constraint othewise, one could choose alpha0 << niter

% col is a string that specifies the line type of the error plot

% If they are known, actual loading matrices can be given in the cell Lv and
% used in order to solve permutation and scaling ambiguity. 


% Outputs:

% Lf is a cell array which stores the estimated loading matrices

% res is the relative mean squared reconstruction error term (objective
% function to be minimized)

% If Lv is given then vector r contains the Nomalized Mean Squared Error,
% computed between estimated and actual loading matrices

% vector err_vec stores the res value at each iteration 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Positivity constraint
if alpha0~=0
    alpha=[cos(2*pi*([1:1:alpha0]-1)/4*(alpha0-1)) zeros(1,100000)];
end


% Compute the HO-SVD if wanted:

if ishosvd==1
    TT=T;
    [T,W]=hosvd(T);
end


% Tensor dimensions
I=size(T)
N=prod(I(2:end));
% Tensor order
D=length(I);


% Loadings intitialisation:

% Load initial matrices saved in Fac.mat (if ini=1) or choose the best out
% of 1000 random starting matrices
if ini==1
    load Fac
    Td=reshape(T,prod(I),1);
    Z0= ones(1,R);
    for d=1:D
        Z0=pkr(Le{d},Z0);
    end
    Td0=Z0*ones(R,1);
    res=abs(Td(:)-Td0(:));
    err_old=res'*res/sum(abs(Td(:)).^2);
else
    Td=reshape(T,prod(I),1);
    err_old=1e10;
    L=cell(1,D);
    nn=norm(Td(:)).^(1/D);
    for i=1:1000
        for d=1:D;
            L{d}=nn*((rand(I(d),R)));
        end
        Z0= ones(1,R);
        for d=1:D
            Z0=pkr(L{d},Z0);
        end
        Td0=Z0*ones(R,1);
        res=abs(Td(:)-Td0(:));
        err_old1=res'*res/sum(abs(Td(:)).^2);
        if err_old1<err_old
            Le=L;
            err_old=err_old1;
        end
    end
    save Fac Le
end


% Initialize variables used in the optimization loop:

if strcmp(Algo,'lm') || strcmp(Algo,'nlcg') %only useful with LM
    grad=construct_gradient(Le,T);
    [Hess,Jac]=construct_hessian(Le,T);
    eps1=1e-3;
    eps2=1e-12;
    flag=(max(grad)<eps1);
    mu= 2;
    nu=2;
    dir=-grad;
    delta=dir'*dir;
end
r=[];
lda=0.01;
it=0;
Lit=cell(2,D);
Td=reshape(T,[I(1) prod(I(2:end))]);
K=cell(D,2);
err_vec=zeros(niter+1,1);
err_new=err_old/2;
err_rel= abs(err_new-err_old)/err_old;
err_rel_vec=zeros(niter,1);


% Optimization loop for estimating the loading matrices:

while   it<niter && (err_rel>crit || err_rel==0) %&& mu> 1e-8  && lda> 1e-8 &&flag==0

    it=it+1 ;
    err_old=err_new;
    if alpha0~=0
        alph=alpha(it);
    else
        alph=1;
    end

    switch Algo
        case 'als'
            Lit(1,:)=Le;
            [Le,err_new]=loop_als(Le,T,alph);
            Lit(2,:)=Le;
            if mod(it,elsper)==0 %&& it>10
                for d=1:D
                    K{d,1}=Lit{2,d}-Lit{1,d};
                    K{d,2}=Lit{1,d};
                end
                [p,err_new]=els(Td,K,N,R);
                for d=1:D
                    Le{d}=Lit{1,d}+p*K{d,1};
                end
            end
        case 'lm'
            if mod(it,elsper)==0
                [Le,err_new,mu,nu,Hess,grad,Jac,flag]=loop_lm_opt(Le,T,mu,nu,Hess,grad,Jac,eps2,eps1,flag);
            else
                [Le,err_new,mu,nu,Hess,grad,Jac,flag]=loop_lm(Le,T,mu,nu,Hess,grad,Jac,eps2,eps1,flag);
            end
        case 'nlcg'
                [Le,err_new,dir,Hess,grad,delta]=loop_nlcg(Le,T,dir,Hess,grad,delta);
        case 'grad'
            if mod(it,elsper)==0
                grad=construct_gradient(Le,T);
                k=1;
                for i=1:D
                    K{i,1}=-unvec(grad(k:k-1+I(i)*R),R,I(i)).';
                    k=k+I(i)*R;
                    K{i,2}=Le{i};
                end
                p=els(Td,K,N,R);
                [Le,err_new]=loop_grad_opt(Le,T,p,grad);
                lda=p;
            else
                [Le,err_new,lda]=loop_grad(Le,T,lda);
            end
    end

    err_rel= abs(err_new-err_old)/err_old;
    err_rel_vec(it)= err_rel;
    err_vec(it)=err_new;
end


% Estimation results:

if ishosvd==1
    I=size(TT);
    Lf=cell(1,D);
    for i=1:D
        Lf{i}=W{i}*Le{i};
    end
    TTe=construct_tensor(Lf);
    TTd=reshape(TT,prod(I),1);
    TTed=reshape(TTe,prod(I),1);
    res=abs(TTd(:)-TTed(:));
    res=res'*res/sum(abs(TTd(:)).^2);
else
    Lf=Le;
    res=err_new;
end


%solve permutation and scaling ambiguity:

if nargin==nargin('main');
    Lf=permscal_def(Lf,Lv);
    for i=1:D
        r(i)=norm(Lf{i}-Lv{i})./norm(Lv{i});
    end
end


% Disp estimation errors:

res
r

%fit plot:

semilogy(err_vec(1:it),col)
hold on


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

%Remove permutation and scaling ambiguity
function Ln=permscal_def(L,Lv);
N=length(L);
F=size(L{1},2);
X=[];
Xv=[];
for n=1:N
    A=L{n};
    Av=Lv{n};
    t=size(A,1);
    for i=1:F
        An(1:2:2*t-1,i)=real(A(:,i)/A(1,i));
        An(2:2:2*t,i)=imag(A(:,i)/A(1,i));
        Avn(1:2:2*t-1,i)=real(Av(:,i)/Av(1,i));
        Avn(2:2:2*t,i)=imag(Av(:,i)/Av(1,i));
    end
    X=[X;An];
    Xv=[Xv;Avn];
end
P=perms(1:F);
for i=1:size(P,1)
    r(i)=norm(Xv-X(:,P(i,:)));
end
[m,pos]=min(r);
p=P(pos,:);
Ln=L;
for n=1:N
    A=L{n}(:,p);
    Av=Lv{n};
    complexGF=any(imag(Av(:)));
    t=size(A,1);
    teta=zeros(size(A,1),1);
    for i=1:F
        for k=1:size(A,1)
            teta(k)=angle(mean(A(k,i)*Av(:,i)./A(:,i)));
        end
        Ln{n}(:,i)=(norm(abs(Av(:,i)))*abs(A(:,i))/norm(abs(A(:,i)))).*exp(j*teta);
    end
    if complexGF==0
        Ln{n}=real(Ln{n});
    end
end
