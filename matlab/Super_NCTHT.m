function [Out ]= Super_NCTCP(Ob,KK, Y,rate,s,PN,R)
%code for NCTCP(nonlocal coupled tensor CP decomposition for HSI MSI fusion )
% no ground truth

% input: 
%     ob£º observed LR-HSI
%     KK£º rank number for CP decomposition
%     Y£º observed  MSI
%     rate:  enlarge rate
%     s:    blurring kernel 
%     PN£º number of patches in a cluster
%     R:    spectral response function
% output:
%     Out£ºfused  HR-HSI
 
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
Ob = my_upsample(Ob,rate);

% parameter set
tol=1e-2;
mu=1e-4;
lambda= 10;%100
maxIter=10;
minIter=3;

clu_method='graph';% kmeans, graph, over
nn = size(Ob);
par.patsize=10;
par.Pstep=1;

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
    sortedIx = double( mxCalcPermRand( x_1Mat( : ) , x_restMat( : ) , ixMat( : ) , size(Y,1) + 2 , size(Y,2) , ((nn(1)-par.patsize)/par.Pstep+1),((nn(2)-par.patsize)/par.Pstep+1) , randStartIx , B , randVec , eps1 , 1 ) );
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

R1 = 90
R2 = 20
R3 = 9
RB1 = 60

th = 0.01

HT= Ob;
Z=Ob;
Curpatch        = Im2Patch3D(Z, par);
        
 
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
    
    
    Ytt1{i}=Npatch2(:,:,ind{i});
    Ytt1{i}=permute(Ytt1{i},[1 3 2]);
    YSO1{i}=matricize(Ytt1{i});
    if length(YSO1{i})~=3
        YSO1{i}{3}=Ytt1{i}(:);
    end

    
    tt1=Curpatch(:,:,ind{i});
    tt1=permute(tt1,[1 3 2]);   
    SO1=matricize(tt1);
    
    params.Tdata = 2;            % Number of target sparse vectors
    params.dictsize = R1;      %   441;      % Number of dictionary elements (# columns of D)
    params.iternum = 20;
    params.DUC =1; 
    
  
    % U1 
%     HH{i}.H{1}  = trainD( YSO1{i}{1}', YSO1{i}{1}',[],[],params);
    HH{i}.H{1} =  rand( par.patsize*par.patsize, R1);
    HH{i}.H{1} = HH{i}.H{1} / diag( sqrt( sum( HH{i}.H{1}.^2 ) ) );
    % U2
    if i == L  & R2 > length(ind{i})
%         params.dictsize = length(ind{i});  
        
        HH{i}.H{2} =  rand( length(ind{i}), length(ind{i}));
    else
%         params.dictsize = R2
        HH{i}.H{2} =  rand( length(ind{i}), R2 );
    end
%     HH{i}.H{2}  = trainD( YSO1{i}{2}', YSO1{i}{2}',[],[],params);
    HH{i}.H{2} = HH{i}.H{2} / diag( sqrt( sum( HH{i}.H{2}.^2 ) ) );
    %U3
%     params.dictsize = R3;
%     HH{i}.H{3}  = trainD( SO1{3}', SO1{3}',[],[],params);
     
    HH{i}.H{3} =  rand( size(Ob,3), R3 );
    HH{i}.H{3} = HH{i}.H{3} / diag( sqrt( sum( HH{i}.H{3}.^2 ) ) );
    %U4
%      HH{i}.D = R *  HH{i}.H{3};
    HH{i}.D =  rand( size(Npatch2,2), R3 );
    HH{i}.D = HH{i}.D / diag( sqrt( sum( HH{i}.D.^2 ) ) );
    
    %B1

    if i == L & R2 > length(ind{i})
%         params.dictsize = RB1;
%         tempD = trainD(YSO1{i}{3},YSO1{i}{3},[],[],params);
%         tempB1 = (tempD' / kron( HH{i}.H{2},  HH{i}.H{1})')';
%         HH{i}.B1 = reshape(tempB1, R1, length(ind{i}), RB1);
        HH{i}.B1 =  rand( R1 * length(ind{i}), RB1);
        HH{i}.B1 = HH{i}.B1 / diag( sqrt( sum( HH{i}.B1.^2 ) ) );
        HH{i}.B1 = reshape(HH{i}.B1, R1, length(ind{i}), RB1);
    else
%         params.dictsize = RB1;
%         tempD = trainD(YSO1{i}{3},YSO1{i}{3},[],[],params);
%         tempB1 = (tempD' / kron( HH{i}.H{2},  HH{i}.H{1})')';
        HH{i}.B1 =  rand( R1 * R2, RB1);
        HH{i}.B1 = HH{i}.B1 / diag( sqrt( sum( HH{i}.B1.^2 ) ) );
        HH{i}.B1 = reshape( HH{i}.B1, R1, R2, RB1);
    end
    
    %B2
    HH{i}.B2 =  rand( RB1, R3);
    HH{i}.B2 = HH{i}.B2 / diag( sqrt( sum( HH{i}.B2.^2 ) ) );
    
    M2{i}=zeros( size(HH{i}.D) );
end



for i=1: maxIter

    HT_old=HT;
    Z_old=Z;  
    Curpatch        = Im2Patch3D(Z, par);
    MCurpatch        = Im2Patch3D(M1, par);
    Ncur=zeros(size(Curpatch));
    W=  ones(size(Curpatch,1),size(Curpatch,3));
    
    for id =1 : L
        disp('-----------------------------------------------------------------')
        tt1=Curpatch(:,:,ind{id});
        tt1=permute(tt1,[1 3 2]);
        Mtt1=MCurpatch(:,:,ind{id});
        Mtt1=permute(Mtt1,[1 3 2]);      
        SO1=matricize(tt1);
        
        MM=matricize(Mtt1);
        

        
        
        % update U3
        

        Core = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
        K = double(tenmat(ttm(Core, {HH{id}.H{1}, HH{id}.H{2}}, [1, 2]), 3));
        left = R' * R;
        mid = K * K';
        right = SO1{3}'* K'+ MM{3}'* K' / mu + R' * HH{id}.D + R' * M2{id} / mu;
        HH{id}.H{3} = sylvester(left, mid, right);
        Core1 = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
        RMSE(double(tenmat(ttm(Core1, {HH{id}.H{1}, HH{id}.H{2},  HH{id}.H{3}}, [1 2 3]), 3)) - SO1{3}', 0)
%         
%         
%         % update U4
%         
        Core = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
        K = double(tenmat(ttm(Core, {HH{id}.H{1}, HH{id}.H{2}}, [1, 2]), 3));
        mid = 2 * lambda * K * K' + mu * eye(size(HH{id}.H{3}, 2));
        right = 2 * lambda * YSO1{id}{3}'* K' + mu * R * HH{id}.H{3} - M2{id};
        HH{id}.D = right / mid;
        Core1 = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
        RMSE(double(tenmat(ttm(Core1, {HH{id}.H{1}, HH{id}.H{2},  HH{id}.H{3}}, [1 2 3]), 3)) - SO1{3}', 0)

        
        
                % update B2
      
        K = double(tenmat(ttm(tensor(HH{id}.B1), {HH{id}.H{1}, HH{id}.H{2}}, [1, 2]), 3))';
        W6 = HH{id}.D' * HH{id}.D;
        P6 = HH{id}.H{3}' * HH{id}.H{3};
        left = (2 * lambda + mu) * K' * K;
        mid = W6 +  P6;
        right = 2 * lambda *  K' * YSO1{id}{3} * HH{id}.D + K' * MM{3} * HH{id}.H{3} + mu * K' * SO1{3} * HH{id}.H{3};
        HH{id}.B2 = ((right / mid)' / left')';
        
        Core1 = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
        RMSE(double(tenmat(ttm(Core1, {HH{id}.H{1}, HH{id}.H{2},  HH{id}.H{3}}, [1 2 3]), 3)) - SO1{3}', 0)
    
         % update B1

        K = kron(HH{id}.H{2}, HH{id}.H{1});
        W5 = 2 * lambda * HH{id}.B2 * HH{id}.D' * (HH{id}.B2 * HH{id}.D')';
        P5 = mu * HH{id}.B2 * HH{id}.H{3}' * (HH{id}.B2 * HH{id}.H{3}')';
        left = K' * K;
        mid = W5 + P5;
        right = 2 * lambda *  K' * YSO1{id}{3} *  HH{id}.D  * HH{id}.B2' + K' * MM{3} *  HH{id}.H{3} * HH{id}.B2' + mu * K' * SO1{3} *  HH{id}.H{3} * HH{id}.B2';
        HH{id}.B1 = reshape(((right / mid)' / left')', [R1, size(HH{id}.B1, 2), RB1]);
        
        Core1 = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
        RMSE(double(tenmat(ttm(Core1, {HH{id}.H{1}, HH{id}.H{2},  HH{id}.H{3}}, [1 2 3]), 3)) - SO1{3}', 0)
        


%         


                % update U1
%         
        HH{id}.H{1} * double(tenmat(Core, 1)) * kron(HH{id}.H{3}', HH{id}.H{2}');
        Core = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
        W1 =  double(tenmat(Core, 1)) * kron(HH{id}.D', HH{id}.H{2}');
        P1 = double(tenmat(Core, 1)) * kron(HH{id}.H{3}', HH{id}.H{2}');
        left1 = 2 * lambda * (W1 * W1') + mu * P1 * P1';
        right1 =  2 * lambda * YSO1{id}{1}' * W1' + MM{1}'*P1' + mu*SO1{1}'*P1';
        HH{id}.H{1} = right1 / left1;
        Core1 = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
        RMSE(double(tenmat(ttm(Core1, {HH{id}.H{1}, HH{id}.H{2},  HH{id}.H{3}}, [1 2 3]), 3)) - SO1{3}', 0)
        
 
        % updateU2
%         
        W2 =  double(tenmat(Core, 2)) * kron(HH{id}.D', HH{id}.H{1}');
        P2 = double(tenmat(Core, 2)) * kron(HH{id}.H{3}', HH{id}.H{1}');
        left2 = 2 * lambda * (W2 * W2') + mu * P2 * P2';
        right2 =  2 * lambda * YSO1{id}{2}' * W2' + MM{2}'*P2' + mu*SO1{2}'*P2';
        HH{id}.H{2} = right2 / left2;
        
        Core1 = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
        RMSE(double(tenmat(ttm(Core1, {HH{id}.H{1}, HH{id}.H{2},  HH{id}.H{3}}, [1 2 3]), 3)) - SO1{3}', 0)
        
        

%                 
%    

        
        tempPatch{id} = recover(HH{id}.H{1}, HH{id}.H{2}, HH{id}.H{3}, HH{id}.B1, HH{id}.B2);
        
%         Ncur(:,:,ind{id})=Ncur(:,:,ind{id})+(tempPatch);
%         if strcmp(clu_method,'over')
%             W(:,ind{id})         = W(:,ind{id})+ones(size(tempPatch,1),size(tempPatch,3));
%         end
        M2{id}=M2{id}+mu*(HH{id}.D-R*HH{id}.H{3});
    end

    for ii=1:L
        Ncur(:,:,ind{ii})=Ncur(:,:,ind{ii})+(tempPatch{ii});
        if strcmp(clu_method,'over')
            W(:,ind{ii})         = W(:,ind{ii})+ones(size(tempPatch{ii},1),size(tempPatch{ii},3));
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

function patch = recover(U1, U2, U3, B1, B2)
Core = ttm(tensor(B1), { B2' }, [3]);
patch = ttm(Core, {U1, U2, U3}, [1 2 3]);
patch = permute(double(patch),[1 3 2]);   
end

% function rmse = rmse(Y)
% rmse = sqrt(mean((Y).^2));
% end

