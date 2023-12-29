function [T,TT,Lf]=create_tensor(nb_T,alea,dim_T,complexT,rang_T,snr,L);

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


%Creates a tensor or/and a set of tensor 

%nb_T: number of tensor to be created
%alea: put 1 for a tensor of random values
%dim_T stores the dimensions of the tensor(s)
%rang_T: tensor(s) rank (only if alea=0)
%snr: gaussian additive noise level (signal to noise ratio), nbr=inf means
%no noise
%ComplexT: 0 for real tensor(s) or one for complex
%L is a cell array containing the matrices to be used to create the tensor(s)
%if L is not given then random matrices are used
%If nb_T=1, T is a tensor and TT is an empty matrix;
%If nb_T>1, T is a cell array containing a set of tensors and TT is a
%tensor of order T_order+1 containing the cells of T.
%Lf: cell array which contains the matrices used to create the tensor(s)




ordre_T=length(dim_T(1,:)); %tensor(s) order
T=[];
TT=[];
Lf=[];
switch alea
    case 1 %tensor(s) created with random values
        if nb_T==1
            T=randn(dim_T)+2+randn(dim_T)*i*complexT;
            Lf=[];
        else
            dimg=[nb_T dim_T];
            ch=[];
            TT=randn(dimg)+randn(dimg)*i*complexT;
            for l=1:ordre_T
                ch=[ch ',:'];
            end
            for l=1:nb_T
                eval(['T{l}=squeeze(TT(l' ch '));' ])
            end
        end

    case 0 %tensor(s) created with given or random matrices
        if nb_T==1
            if nargin==7 %given matrices 
                Lf=L;
            else %random matrices
                    
                for d=1:ordre_T
                    Lf{d}=rand(dim_T(d),rang_T)+complexT*rand(dim_T(d),rang_T)*i;
                end
            end
            T=construct_tensor(Lf);
            if snr~=inf
                Px=T(:)'*T(:)/numel(T);
                B=((randn(size(T))+i*complexT*randn(size(T))))/sqrt(complexT+1);
                B=sqrt(Px*10^(-snr/10))*B;
                Pb=B(:)'*B(:)/numel(B);
                snrv=10*log10(Px/Pb);
                T=T+B;
            end
        else
            dimg=[nb_T dim_T]
            if nargin==7
                Lf=L;
            else
                for d=1:ordre_T+1
                    Lf{d}=randn(dimg(d),rang_T)+i*complexT*randn(dimg(d),rang_T);
                end
            end
            TT=construct_tensor(Lf);
            if snr~=inf
                Px=TT(:)'*TT(:)/numel(TT);
                B=((randn(size(TT))+i*complexT*randn(size(TT))))/sqrt(complexT+1);
                B=sqrt(Px*10^(-snr/10))*B;
                Pb=B(:)'*B(:)/numel(B);
                snrv=10*log10(Px/Pb);
                TT=TT+B;
            end
            ch=[];
            for l=1:ordre_T
                ch=[ch ',:'];
            end
            for l=1:nb_T
                eval(['T{l}=squeeze(TT(l' ch '));' ])
            end
        end
end






