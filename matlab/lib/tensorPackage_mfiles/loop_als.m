  function [Ln,res]=loop_als(L,T,alph);
  
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

% ALS optimization 
  
  
%effectue l'�tape n de l'estimation des matrices (Ln{1}, Ln{2}... Ln{D}) � partir
%des r�sultats de l'�tape n-1 (L{1}, L{2}... L{D}).
%alph permet la contrainte de positivit� (mettre 1 pour une estimation libre)

I=size(T); % tensor dimensions
D=length(I); % tensor order (number of dimensions)
Ln=L;
TT=T;
d2=[1 D:-1:2];%ordre des matrices dans le calcule des produits de khatri rao (premier passage)
ind=[1:D];
%On effectue l'estimation de chaque matrice (de 1 � D)
for d1=1:D
    I=size(TT);
    %produit de khatri rao entre les diff�rentes matrices que l'on a pas �
    %estimer � ce passage dans la boucle
    Z=Ln{d2(2)};
    for d=d2(3:end)
        Z=pkr(Z,Ln{d});
    end
    %On d�plie le tenseur dans le sens adequat (correspondant � la matrice
    %� estimer)
    Td=reshape(TT,[I(1) prod(I(2:end))]);
    %estimation de la matrice
    a=pinv(Z)*Td.';
    Ln{d1}=a.';
    %contrainte de positivit�
    A0=Ln{d1};
    A0(A0<0)=0;
    Ln{d1}=alph*Ln{d1}+(1-alph)*A0;
    %permutation circulaire du des dimensions du tenseur (mise � jour du
    %sens de d�pliement du tenseur)
    ind=circshift(ind,[0 -1]);
    TT=permute(T,ind);
    %permutation du jeu d'indice pour le produit de khatri rao
    %(mise � jour de l'ordre des matrices dans le calcule des produits de khatri rao lors du prochain passage)
    d2=circshift(d2,[0,1]); 
end
%calcul du rapport entre l'�nergie de l'erreur de reconstruction et l'�nergie du tenseur
I=size(TT);
Td=reshape(TT,[I(1) prod(I(2:end))]);
Z=Ln{d2(2)};
for d=d2(3:end)
    Z=pkr(Z,Ln{d});
end
TTe=Ln{1}*Z.';
res=abs(Td(:)-TTe(:));
res=res'*res/sum(abs(Td(:)).^2);
