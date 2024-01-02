function [Out ]= Super_NCTCP_4_order(Ob,KK, Y,rate,s,PN,R)
%code for NCTCP(nonlocal coupled tensor CP decomposition for HSI MSI fusion )
% no ground truth

% input: 
%     ob锛?observed LR-HSI
%     KK锛?rank number for CP decomposition
%     Y锛?observed  MSI
%     rate:  enlarge rate
%     s:    blurring kernel 
%     PN锛?number of patches in a cluster
%     R:    spectral response function
% output:
%     Out锛歠used  HR-HSI
 
% solve min||Xn-HSX||_F+lambda*||Y-A*B'||_F
% st. X=ABC
%% important :  
%when fusing real data, using gauss3filterBack and gauss3filterForw for blurring, 
% using upsamp_HS and downsamp_HS for upsampling or down sampling,
% otherwise, using gauss3filter for blurring, and interp23tapGeneral and  inMatrix(1:rate:end,1:rate:end,:);
% for upsampling and downsampling( or both imresize 'bicubic')

tic
%preprocess
% max_HS=1;
max_HS = max(max(max(Ob)));
Ob = Ob./max_HS;
Y = Y./max_HS;
% Ob = my_upsample(Ob,rate);

% parameter set
tol=1e-3;
mu=1e-4;
lambda= 100;%100
maxIter=10;
minIter=3;

clu_method='graph';% kmeans, graph, over
nn = size(Ob);
par.patsize=10;
par.Pstep=2;

%initialize
Z = Ob;
M1=zeros(size(Z));
% M2=zeros(size(Y,3),size(Ob,3));

Ob= gauss3filter(Ob,s);
% Ob= gauss3filterBack(Ob,s);
TY=cat(3,Y,Y(:,:,end));
Npatch2    =  Im2Patch3D(TY,par);
Npatch2=Npatch2(:,1:end-1,:);

unfoldPatch = reshape(Npatch2, size(Npatch2,1)*size(Npatch2,2),size(Npatch2,3));

% L=50;

if strcmp(clu_method,'kmeans')
    %% kmeans cluster
    L=floor(size(unfoldPatch,2)/100);
    [k1 k2]= kmeans(unfoldPatch,L);
    
elseif strcmp(clu_method,'graph')
    %% sort graph cluster
 
    Y=unfoldPatch;
    %     randStartIx = round( rand * size(Y,2));
    randStartIx = 1;
    ixPic = reshape( 1 : ((nn(1)-par.patsize)/par.Pstep+1)*((nn(2)-par.patsize)/par.Pstep+1) , [ ((nn(1)-par.patsize)/par.Pstep+1) , ((nn(2)-par.patsize)/par.Pstep+1) ] );
    tempIx = ixPic(:);
    [ cur_rows , cur_cols ] = ind2sub( [ ((nn(1)-par.patsize)/par.Pstep+1) , ((nn(2)-par.patsize)/par.Pstep+1)] , tempIx );
    ixMat = [ cur_rows' ; cur_cols' ];
    x_restMat = [ Y ; ones( 1 , size(Y,2) ) ; -1/2 * sum( Y.^2 ) ];
    x_1Mat = [ Y ; -1/2 * sum(Y.^2 ) ; ones( 1 ,size(Y,2) ) ];
    
    B = 81;
    eps1 = 10^0.01;
    randVec = rand( size(Y,2) , 1 );
    disp('111');
    sortedIx = double( mxCalcPermRand( x_1Mat( : ) , x_restMat( : ) , ixMat( : ) , size(Y,1) + 2 , size(Y,2) , ((nn(1)-par.patsize)/par.Pstep+1),((nn(2)-par.patsize)/par.Pstep+1) , randStartIx , B , randVec , eps1 , 1 ) );
    disp('2222');
    %     sortedIx= randperm(size(Y,2));
    k1=zeros(size(Y,2),1);
    Num=PN;
    L= ceil (size(Y,2)/Num);
    for m=1:L-1
        k1(sortedIx((m-1)*Num+1:m*Num))=m;
    end
    
    k1(sortedIx(((L-1)*Num+1):end))=L;    
else
    par.step=5;
    par.patnum=50;
    [Sel_arr]       =  nonLocal_arr(nn, par); % PreCompute the all the patch index in the searching window
    L               =   length(Sel_arr);
    patchXpatch     = sum(unfoldPatch.*unfoldPatch,1);
    distenMat       = repmat(patchXpatch(Sel_arr),size(unfoldPatch,2),1)+repmat(patchXpatch',1,L)-2*(unfoldPatch')*unfoldPatch(:,Sel_arr);
    [~,index]       = sort(distenMat);
    index           = index(1:par.patnum,:);
    
end


% initialize
for i=1:L
    if strcmp(clu_method, 'over')
        ind{i}= index(:,i);
    else
        ind{i}=find(k1==i);
    end
    %     ind{i}=sortedIx((Num*(i-1)+1):Num*i);
    %     in_k(i)=min(ceil(length(ind{i})/5),KK);
    in_k(i)=KK;
    k=in_k(i);
    HH{i}.H{1} =  rand( par.patsize, k );
    HH{i}.H{1} = HH{i}.H{1} / diag( sqrt( sum( HH{i}.H{1}.^2 ) ) );
    
    HH{i}.H{2} =  rand( par.patsize, k );
    HH{i}.H{2} = HH{i}.H{2} / diag( sqrt( sum( HH{i}.H{2}.^2 ) ) );
   
    HH{i}.H{3} =  rand( size(Ob,3), k );
    HH{i}.H{3} = HH{i}.H{3} / diag( sqrt( sum( HH{i}.H{3}.^2 ) ) );
   
    HH{i}.H{4} =  rand( length(ind{i}), k );
    HH{i}.H{4} = HH{i}.H{4} / diag( sqrt( sum( HH{i}.H{4}.^2 ) ) );
    
    HH{i}.D =  rand( size(Npatch2,2), k );
    HH{i}.D = HH{i}.D / diag( sqrt( sum( HH{i}.D.^2 ) ) );
    
    Ytt1{i}=Npatch2(:,:,ind{i});%25*4*300
    %Ytt1{i}=permute(Ytt1{i},[1 3 2]);%25*300*4
    Ytt1{i}=reshape(Ytt1{i},par.patsize,par.patsize,size(Ytt1{i},2),size(Ytt1{i},3));
    YSO1{i}=matricize2(Ytt1{i});
    %if length(YSO1{i})~=3
        %YSO1{i}{3}=Ytt1{i}(:);
    %end
    if length(YSO1{i})~=4
        YSO1{i}{3}=Ytt1{i}(:)';
        YSO1{i}{4}=Ytt1{i}(:)';
    end
    M2{i}=zeros( size(HH{i}.D) );
end

HT= Ob;
Z=Ob;

for i=1: maxIter

    HT_old=HT;
    Z_old=Z;  
    Curpatch        = Im2Patch3D(Z, par);
    MCurpatch        = Im2Patch3D(M1, par);
    Ncur=zeros(size(Curpatch));
    W=  ones(size(Curpatch,1),size(Curpatch,3));
    
    for id =1 : L

        tt1=Curpatch(:,:,ind{id});
        %tt1=permute(tt1,[1 3 2]);
        tt1=reshape(tt1, par.patsize,par.patsize,size(tt1,2),size(tt1,3));
        Mtt1=MCurpatch(:,:,ind{id});
        %Mtt1=permute(Mtt1,[1 3 2]);     
        Mtt1=reshape(Mtt1, par.patsize,par.patsize,size(Mtt1,2),size(Mtt1,3));
        SO1=matricize2(tt1);
 
        W1 = khatrirao( HH{id}.H{ [2 3 4] }, 'r' );
        P1 = khatrirao({HH{id}.H{2},HH{id}.D,HH{id}.H{4}},'r');
        G1=(HH{id}.H{4}'*HH{id}.H{4}).*(HH{id}.H{3}'*HH{id}.H{3}).*(HH{id}.H{2}'*HH{id}.H{2});
        PTP=(HH{id}.H{4}'*HH{id}.H{4}).*(HH{id}.D'*HH{id}.D).*(HH{id}.H{2}'*HH{id}.H{2});
        %         G1= ones(k,k);
        %         for j = [2 3]
        %             G1 = G1.* (HH{id}.H{j}'*HH{id}.H{j});
        %         end
        %                        G1=W1'*W1;
        
        t1= ( mu*G1+ 2*lambda*PTP);
        MM=matricize2(Mtt1);
%         clear tt1 Mtt1;
        %         HH{id}.H{1}=(t1\(MM{1}'*W1+2*lambda*SO2*HH{id}.H{2}+mu*SO1{1}'*W1)')';
        HH{id}.H{1}=((MM{1}*W1+2*lambda*YSO1{id}{1}*P1+mu*SO1{1}*W1))/(t1);
        %            HH{id}.H{1}=max((t1\(MM{1}'*W1+2*lambda*SO2*HH{id}.H{2}+mu*SO1{1}'*W1)')',0);
        %          HH{id}.H{1}= a*b(:,1:k);
        
        W2 = khatrirao( HH{id}.H{ [1 3 4] }, 'r' );
        P2 = khatrirao({HH{id}.H{1},HH{id}.D,HH{id}.H{4}},'r');
        G2=(HH{id}.H{4}'*HH{id}.H{4}).*(HH{id}.H{3}'*HH{id}.H{3}).*(HH{id}.H{1}'*HH{id}.H{1});
        P2TP2=(HH{id}.H{4}'*HH{id}.H{4}).*(HH{id}.D'*HH{id}.D).*(HH{id}.H{1}'*HH{id}.H{1});
        %         G2 = ones(k,k);
        %         for jj= [1 3]
        %             G2 = G2 .* (HH{id}.H{jj}'*HH{id}.H{jj});
        %         end
        %                      G2=W2'*W2;
        t2= ( mu*G2 + 2*lambda*P2TP2);
        %         HH{id}.H{2}=(t2\(MM{2}'*W2+2*lambda*SO2'*HH{id}.H{1}+mu*SO1{2}'*W2)')';
        
        HH{id}.H{2}=((MM{2}*W2+2*lambda*YSO1{id}{2}*P2+mu*SO1{2}*W2))/t2;
        %         HH{id}.H{2}=max((t2\(MM{2}'*W2+2*lambda*SO2'*HH{id}.H{1}+mu*SO1{2}'*W2)')',0);
        %         HH{id}.H{2}= c(:,1:k);
        
        W3 = khatrirao( HH{id}.H{ [1 2 4] }, 'r' );
        
        %         G3 = ones(k,k);
        %         for jn= [1 2]
        %             G3 = G3 .* (HH{id}.H{jn}'*HH{id}.H{jn});
        %         end
        leftA= R'*R;
        %         rightA=W3'*W3;
        rightA =(HH{id}.H{4}'*HH{id}.H{4}).*(HH{id}.H{2}'*HH{id}.H{2}).*(HH{id}.H{1}'*HH{id}.H{1});
        rightEqu=SO1{3}*W3+MM{3}*W3/mu+R'*HH{id}.D+R'*M2{id}/mu;
        %                  G3=W3'*W3;
        %         HH{id}.H{3} = (G3\((MM{3}'*W3/mu+SO1{3}'*W3))')';
        HH{id}.H{3} =sylvester(leftA,rightA,rightEqu);
        %          HH{id}.H{3} = max((G3\((MM{3}'*W3/mu+SO1{3}'*W3))')',0);
       
        t4=(2*lambda*(W3'*W3)+mu*eye(size(W3,2)));
        HH{id}.D = ((2*lambda*YSO1{id}{3}*W3+mu*R*HH{id}.H{3}-M2{id}))/t4;
        
        W4 = khatrirao( HH{id}.H{ [1 2 3] }, 'r' );
        P4 = khatrirao({HH{id}.H{1},HH{id}.H{2},HH{id}.D},'r');
        G4=(HH{id}.H{3}'*HH{id}.H{3}).*(HH{id}.H{2}'*HH{id}.H{2}).*(HH{id}.H{1}'*HH{id}.H{1});
        P4TP4=(HH{id}.D'*HH{id}.D).*(HH{id}.H{2}'*HH{id}.H{2}).*(HH{id}.H{1}'*HH{id}.H{1});
        %         G2 = ones(k,k);
        %         for jj= [1 3]
        %             G2 = G2 .* (HH{id}.H{jj}'*HH{id}.H{jj});
        %         end
        %                      G2=W2'*W2;
        t5= ( mu*G4 + 2*lambda*P4TP4);
        %         HH{id}.H{2}=(t2\(MM{2}'*W2+2*lambda*SO2'*HH{id}.H{1}+mu*SO1{2}'*W2)')';
        
        HH{id}.H{4}=((MM{4}*W4+2*lambda*YSO1{id}{4}*P4+mu*SO1{4}*W4))/t5;
       
        tempPatch{id}=double(ktensor( {HH{id}.H{1},HH{id}.H{2},HH{id}.H{3},HH{id}.H{4}}));
        tempPatch2{id}=reshape(tempPatch{id},size(tempPatch{id},1)*size(tempPatch{id},2),size(tempPatch{id},3),size(tempPatch{id},4));

%         Ncur(:,:,ind{id})=Ncur(:,:,ind{id})+(tempPatch);
%         if strcmp(clu_method,'over')
%             W(:,ind{id})         = W(:,ind{id})+ones(size(tempPatch,1),size(tempPatch,3));
%         end
        M2{id}=M2{id}+mu*(HH{id}.D-R*HH{id}.H{3});%对应论文里的乘子Fp
    end

    for ii=1:L
        Ncur(:,:,ind{ii})=Ncur(:,:,ind{ii})+(tempPatch2{ii});
        if strcmp(clu_method,'over')
            W(:,ind{ii})         = W(:,ind{ii})+ones(size(tempPatch2{ii},1),size(tempPatch2{ii},3));
        end
    end
    %         [HT, ~]  =  Patch2Im3D( Ncur, ones(par.patsize.^2,size(Curpatch,3)), par, nn);
    [HT, ~]  =  Patch2Im3D( Ncur, W, par, nn);

    % congtive gradient
    rr= 2*Ob+mu*HT-M1;
    Z = reshape(cgsolve2(@(x) Afun(x,rate,s,nn,mu),rr(:)),nn(1),nn(2),nn(3));
    M1 = M1 + mu*(Z- HT);
    
    mu=1.01*mu;
    
    stopCond = norm(HT(:) - HT_old(:))/norm(HT_old(:));
    stopCond2 = norm(Z(:) - Z_old(:))/norm(Z_old(:));
    fprintf('the %d iter\n', i);
    %     subplot(121)
    %     imshow(H{1}*H{2}');drawnow;
    %     subplot(122)
    imshow(Z(:,:,4),[]);drawnow;
    res1=Z-HT;
    RZ=Z*max_HS;
    %     AM=max(max(RZ(:)),max(Omsi(:)));
%     AM=max(Omsi(:));
%     psnr       =  PSNR3D(RZ*255/AM,Omsi*255/AM);
    %     res2 = Y- H{1}*H{2}';
    ReChange=norm(res1(:))/norm(Z(:));
    fprintf('%10.5f\t%10.5f\t%10.5f\t\n',ReChange,stopCond,stopCond2)               ;
    if ReChange<tol&stopCond<tol&stopCond2<tol&i>(minIter-1)
        break;
    end
    
    
end

Out = Z;
Out = Out.*max_HS;
toc
end

function re= Afun(x,rate,s,sz,mu)
x=reshape(x,sz(1),sz(2),sz(3));
ous= my_upsample(my_downsample(gauss3filter(x,s),rate),rate);
ous = gauss3filter(ous,s);

%method in Hysure
% ous= my_upsample(my_downsample(gauss3filterForw(x,s),rate),rate);
% ous = gauss3filterBack(ous,s);

re= 2*ous+mu*x;
re=re(:);
end





