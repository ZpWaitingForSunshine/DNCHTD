function [Out ]= Super_NCTHT_PAO_r(Ob,KK, Y,rate,s1,PN,R)
%code for NCTCP(nonlocal coupled tensor CP decomposition for HSI MSI fusion )
% no ground truth

% input: 
%     ob�� observed LR-HSI
%     KK�� rank number for CP decomposition
%     Y�� observed  MSI
%     rate:  enlarge rate
%     s:    blurring kernel 
%     PN�� number of patches in a cluster
%     R:    spectral response function
% output:
%     Out��fused  HR-HSI
 
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
lambda= 100;%100
maxIter=10;
minIter=3;

clu_method='graph';% kmeans, graph, over
nn = size(Ob);
par.patsize=5;
par.Pstep=1;

R1_th = 0.01;
R2_th = 0.0001;
R3_th = 0.01;
RB1_th = 0.001;

the = 0.01;

%initialize
Z = Ob;
M1=zeros(size(Z));
% M2=zeros(size(Y,3),size(Ob,3));

Ob= gauss3filter(Ob,s1);
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
    
%     params.Tdata = 2;            % Number of target sparse vectors
%     params.dictsize = R1;      %   441;      % Number of dictionary elements (# columns of D)
%     params.iternum = 20;
%     params.DUC =1; 
    
  
%     U1 
%     HH{i}.H{1}  = trainD( YSO1{i}{1}', YSO1{i}{1}',[],[],params);
    
%      [U1, s, v] = svd( YSO1{i}{1}');
%      s = diag(s);
%      R1 = 1: 24;
% %      R1 = find(s > R1_th);
%      HH{i}.H{1} = U1(:, R1);
%      
%     % U2
%     [U2, s, v] = svd( YSO1{i}{2}');
%      s = diag(s);
% %      R2 = find(s > R2_th);
%     if(size( U2, 2)  == 300)
%         R2 = 1: 250;
%     else
%         R2 = size( U2, 2) - 1
%     end
%     
%     HH{i}.H{2} = U2(:, R2);
%      
%     % U3
%      [U3, s, v] = svd( SO1{3}');
%      s = diag(s);
%      R3 = find(s > R3_th);
%      HH{i}.H{3} = U3(:, R3);
%      
%     %U4
%      HH{i}.D = R *  HH{i}.H{3};
%     
%     %B1
% %     [B1U, s, v] = svd( SO1{3});
%     B1U = v;
%     RB1 = find(s > RB1_th);
%     B1 = kron(HH{i}.H{2}', HH{i}.H{1}') * B1U(:, RB1);
%     HH{i}.B1 = reshape(B1, [size(HH{i}.H{1}, 2), size(HH{i}.H{2}, 2), size(RB1, 1)]);
%     
%     %B2
%     HH{i}.B2 =  rand( size(RB1, 1), size(R3, 1));
%     HH{i}.B2 = HH{i}.B2 / diag( sqrt( sum( HH{i}.B2.^2)));
  
    R1 = 24;
    HH{i}.H{1} =  rand( par.patsize*par.patsize, R1 );
    HH{i}.H{1} = HH{i}.H{1} / diag( sqrt( sum( HH{i}.H{1}.^2 ) ) );
    
    if(size( SO1{2}, 2)  == PN)
        R2 = PN - 10;
    else
        R2 = size( SO1{2}, 2) - 1;
    end

    HH{i}.H{2} =  rand( length(ind{i}), R2 );
    HH{i}.H{2} = HH{i}.H{2} / diag( sqrt( sum( HH{i}.H{2}.^2 ) ) );
    
    R3 = 30;
    HH{i}.H{3} =  rand( size(Ob,3), R3);
    HH{i}.H{3} = HH{i}.H{3} / diag( sqrt( sum( HH{i}.H{3}.^2 ) ) );
    
    HH{i}.D =  rand( size(Npatch2,2), R3 );
    HH{i}.D = HH{i}.D / diag( sqrt( sum( HH{i}.D.^2 ) ) );
    
    RB1 = 35;
    HH{i}.B2 =  rand( RB1, R3);
    HH{i}.B2 = HH{i}.B2  / diag( sqrt( sum(HH{i}.B2 .^2 ) ) );
    
    HH{i}.B1 =  rand(R1, R2 * RB1);
    HH{i}.B1 =  HH{i}.B1 / diag( sqrt( sum(  HH{i}.B1.^2 ) ) );
    HH{i}.B1 = reshape(HH{i}.B1, [R1, R2, RB1]);
    
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
        

        
        for inner = 1 : 1
            
            loss(HH{id}.H{1}, HH{id}.H{1}, HH{id}.H{2}, HH{id}.H{2}, HH{id}.H{3}, HH{id}.D, HH{id}.B1, HH{id}.B1, HH{id}.B2, HH{id}.B2, YSO1{id}{1}',  SO1{1}', MM{1}', mu, lambda,  0, R, M2{id})
            
                        
           %             update B2
            HH{id}.B2 = updateB2( HH{id}.H{1},  HH{id}.H{2},  HH{id}.H{3},  HH{id}.D, HH{id}.B1, HH{id}.B2, YSO1{id}{3}',  SO1{3}', MM{3}', mu,lambda, the);
%             loss(HH{id}.H{1}, HH{id}.H{1}, HH{id}.H{2}, HH{id}.H{2}, HH{id}.H{3}, HH{id}.D, HH{id}.B1, HH{id}.B1, HH{id}.B2, HH{id}.B2, YSO1{id}{1}',  SO1{1}', MM{1}', mu, lambda,  the, R, M2{id})
            
           
%             loss(HH{id}.H{1},  HH{id}.H{2},  HH{id}.H{3},  HH{id}.D, HH{id}.B1, HH{id}.B2, YSO1{id}{1}',  SO1{1}', mu, lambda)
                        % update B1
%             B1pre = HH{id}.B1;
            HH{id}.B1 =  updateB1( HH{id}.H{1},  HH{id}.H{2},  HH{id}.H{3},  HH{id}.D, HH{id}.B1, HH{id}.B2, Ytt1{id}, tt1, Mtt1, mu, lambda, the);   
            
            % update U2
%             U2pre =  HH{id}.H{2};
            
            HH{id}.H{2} =  updateU2( HH{id}.H{1},  HH{id}.H{2},  HH{id}.H{3},  HH{id}.D, HH{id}.B1, HH{id}.B2, YSO1{id}{2}',  SO1{2}', MM{2}', mu, lambda, the);
%             loss(HH{id}.H{1}, HH{id}.H{1}, HH{id}.H{2}, HH{id}.H{2}, HH{id}.H{3}, HH{id}.D, HH{id}.B1, HH{id}.B1, HH{id}.B2, HH{id}.B2, YSO1{id}{1}',  SO1{1}', MM{1}', mu, lambda,  the, R, M2{id})
            
            % update U1
            U1pre = HH{id}.H{1};
           
            HH{id}.H{1} =  updateU1( HH{id}.H{1},  HH{id}.H{2},  HH{id}.H{3},  HH{id}.D, HH{id}.B1, HH{id}.B2, YSO1{id}{1}',  SO1{1}', MM{1}', mu, lambda, the);
         
            % update U3
            try
                HH{id}.H{3} = updateU3( HH{id}.H{1},  HH{id}.H{2},  HH{id}.H{3},  HH{id}.D, HH{id}.B1, HH{id}.B2, YSO1{id}{3}',  SO1{3}', MM{3}', mu, R, M2{id}, the);
            catch
            end
            %             loss(HH{id}.H{1}, HH{id}.H{1}, HH{id}.H{2}, HH{id}.H{2}, HH{id}.H{3}, HH{id}.D, HH{id}.B1, HH{id}.B1, HH{id}.B2, HH{id}.B2, YSO1{id}{1}',  SO1{1}', MM{1}', mu, lambda,  the, R, M2{id})
            
            % update U4
            HH{id}.D = updateU4( HH{id}.H{1},  HH{id}.H{2},  HH{id}.H{3},  HH{id}.D, HH{id}.B1, HH{id}.B2, YSO1{id}{3}',  SO1{3}', MM{3}', mu, lambda, R, M2{id}, the);
            loss(HH{id}.H{1}, HH{id}.H{1}, HH{id}.H{2}, HH{id}.H{2}, HH{id}.H{3}, HH{id}.D, HH{id}.B1, HH{id}.B1, HH{id}.B2, HH{id}.B2, YSO1{id}{1}',  SO1{1}', MM{1}', mu, lambda,  the, R, M2{id})
            

            
            
%                         loss(HH{id}.H{1},  HH{id}.H{2},  HH{id}.H{3},  HH{id}.D, HH{id}.B1, HH{id}.B2, YSO1{id}{1}',  SO1{1}', mu, lambda)

            
   
             
%             Core = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
%             K = double(tenmat(ttm(Core, {HH{id}.H{1}, HH{id}.H{2}}, [1, 2]), 3));
%             left = R' * R;
%             mid = K * K';
%             right = SO1{3}'* K'+ MM{3}'* K' / mu + R' * HH{id}.D + R' * M2{id} / mu;
%             HH{id}.H{3} = sylvester(left, mid, right);
%             Core1 = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
%             RMSE(double(tenmat(ttm(Core1, {HH{id}.H{1}, HH{id}.H{2},  HH{id}.H{3}}, [1 2 3]), 3)) - SO1{3}', 0)
%     %         
%     %         
%     %         % update U4
%     %         
%             Core = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
%             K = double(tenmat(ttm(Core, {HH{id}.H{1}, HH{id}.H{2}}, [1, 2]), 3));
%             mid = 2 * lambda * K * K' + mu * eye(size(HH{id}.H{3}, 2));
%             right = 2 * lambda * YSO1{id}{3}'* K' + mu * R * HH{id}.H{3} - M2{id};
%             HH{id}.D = right / mid;
%             Core1 = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
%             RMSE(double(tenmat(ttm(Core1, {HH{id}.H{1}, HH{id}.H{2},  HH{id}.H{3}}, [1 2 3]), 3)) - SO1{3}', 0)
% 
% 
%             
%                     % update B2
% 
%             K = double(tenmat(ttm(tensor(HH{id}.B1), {HH{id}.H{1}, HH{id}.H{2}}, [1, 2]), 3))';
%             W6 = HH{id}.D' * HH{id}.D;
%             P6 = HH{id}.H{3}' * HH{id}.H{3};
%             left = (2 * lambda + mu) * K' * K;
%             mid = W6 +  P6;
%             right = 2 * lambda *  K' * YSO1{id}{3} * HH{id}.D + K' * MM{3} * HH{id}.H{3} + mu * K' * SO1{3} * HH{id}.H{3};
%             HH{id}.B2 = ((right / mid)' / left')';
% 
%             Core1 = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
%             RMSE(double(tenmat(ttm(Core1, {HH{id}.H{1}, HH{id}.H{2},  HH{id}.H{3}}, [1 2 3]), 3)) - SO1{3}', 0)
% 
%              % update B1
% 
%             K = kron(HH{id}.H{2}, HH{id}.H{1});
%             W5 = 2 * lambda * HH{id}.B2 * HH{id}.D' * (HH{id}.B2 * HH{id}.D')';
%             P5 = mu * HH{id}.B2 * HH{id}.H{3}' * (HH{id}.B2 * HH{id}.H{3}')';
%             left = K' * K;
%             mid = W5 + P5;
%             right = 2 * lambda *  K' * YSO1{id}{3} *  HH{id}.D  * HH{id}.B2' + K' * MM{3} *  HH{id}.H{3} * HH{id}.B2' + mu * K' * SO1{3} *  HH{id}.H{3} * HH{id}.B2';
%             HH{id}.B1 = reshape(((right / mid)' / left')', [R1, size(HH{id}.B1, 2), RB1]);
% 
%             Core1 = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
%             RMSE(double(tenmat(ttm(Core1, {HH{id}.H{1}, HH{id}.H{2},  HH{id}.H{3}}, [1 2 3]), 3)) - SO1{3}', 0)
% 
% 
% 
%     %         
% 
% 
%                     % update U1
%     %         
%             HH{id}.H{1} * double(tenmat(Core, 1)) * kron(HH{id}.H{3}', HH{id}.H{2}');
%             Core = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
%             W1 =  double(tenmat(Core, 1)) * kron(HH{id}.D', HH{id}.H{2}');
%             P1 = double(tenmat(Core, 1)) * kron(HH{id}.H{3}', HH{id}.H{2}');
%             left1 = 2 * lambda * (W1 * W1') + mu * P1 * P1';
%             right1 =  2 * lambda * YSO1{id}{1}' * W1' + MM{1}'*P1' + mu*SO1{1}'*P1';
%             HH{id}.H{1} = right1 / left1;
%             Core1 = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
%             RMSE(double(tenmat(ttm(Core1, {HH{id}.H{1}, HH{id}.H{2},  HH{id}.H{3}}, [1 2 3]), 3)) - SO1{3}', 0)
% 
% 
%             % updateU2
%     %         
%             W2 =  double(tenmat(Core, 2)) * kron(HH{id}.D', HH{id}.H{1}');
%             P2 = double(tenmat(Core, 2)) * kron(HH{id}.H{3}', HH{id}.H{1}');
%             left2 = 2 * lambda * (W2 * W2') + mu * P2 * P2';
%             right2 =  2 * lambda * YSO1{id}{2}' * W2' + MM{2}'*P2' + mu*SO1{2}'*P2';
%             HH{id}.H{2} = right2 / left2;
% 
%             Core1 = ttm(tensor( HH{id}.B1), { HH{id}.B2' }, [3]);
%             RMSE(double(tenmat(ttm(Core1, {HH{id}.H{1}, HH{id}.H{2},  HH{id}.H{3}}, [1 2 3]), 3)) - SO1{3}', 0)
        
        end

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
%     imshow(HT(:,:,4),[]);
    % congtive gradient
    rr= 2*Ob+mu*HT-M1;
    Z = reshape(cgsolve2(@(x) Afun(x,rate,s1,nn,mu),rr(:)),nn(1),nn(2),nn(3));
    
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
% Core = ttm(tensor(B1), { B2 }, [3]);
% patch = ttm(Core, {U1, U2, U3}, [1 2 3]);

patch = double(tensor(M2T(U1, U2, U3, B1, B2)));
patch = permute(double(patch),[1 3 2]);   
end

% function rmse = rmse(Y)
% rmse = sqrt(mean((Y).^2));
% end
function T = M2T(U1, U2, U3, B1, B2)
    W = size(U1, 1);
    H = size(U2, 1);
    L = size(U3, 1);
    temp = U3 * double(tenmat( ttm(tensor(B1), B2', 3), 3)) * kron(U2', U1');
%     np.dot(U3,   np.dot(tl.unfold(tl.tenalg.mode_dot(B1, B2, mode=2), mode=2), np.kron(U1.T, U2.T)))
    T = tensor(reshape(temp', [W, H, L]));
end
    
function loss(U1, U1_, U2, U2_, U3, U4, B1, B1_, B2, B2_, Y, X, M, mu, lda,  theta, R, F)
    disp("loss")
    p = 0.0001;
    a = lda * norm(double(tenmat(tensor(M2T(U1, U2, U4, B1, B2)), 1)) - Y, 'fro');
%     rmse = RMSE(double(tenmat(tensor(M2T(U1, U2, U4, B1, B2)), 1)), Y)
    b = mu / 2 * norm(double(tenmat(tensor(M2T(U1, U2, U3, B1, B2)), 1)) - X - M/mu, 'fro');
    c = p * norm(B1(:), 1) + p * norm(B2, 1);
    d = theta * norm(U1 - U1_) + theta * norm(U2 - U2_) + theta * norm(B1(:) - B1_(:)) + theta * norm(B2 - B2_);
    e = mu / 2 * norm(U4 - R * U3, 'fro') + sum(sum(F .* (U4 - R * U3)));
    a + b + e
end


% function loss2(U1, U2, U3, U4, B1, B2, Y, X, mu, lda, M)
%     disp("loss2")
%     a = double(tenmat(tensor(M2T(U1, U2, U4, B1, B2)), 1));
%     b = double(tenmat(tensor(M2T(U1, U2, U3, B1, B2)), 1));
%     c = M * (b -X)
%     lda * norm(a - Y) + mu * norm(b - X)/ 2 + 
% end


function U1 = updateU1(U1, U2, U3, U4, B1, B2, Y, X, M, mu, lda, theta)
    core = ttm(tensor(B1), B2', 3);
    A = double(tenmat(ttm(core, {U2, U4}, [2, 3]), 1));
    B = double(tenmat(ttm(core, {U2, U3}, [2, 3]), 1));
    C = theta;
%     Ymode 1
    right = 2 * lda * Y * A' + mu * (X + M / mu) * B' + C * U1;
    left = 2 * lda * A * A' + mu * B * B' + C * eye(size(A, 1));
%     U1 = CG(left', right')';
    U1 = right / left;
end



function U2 = updateU2(U1, U2, U3, U4, B1, B2, Y, X, M, mu, lda, theta)
    core = ttm(tensor(B1), B2', 3);
    A = double(tenmat(ttm(core, {U1, U4}, [1, 3]), 2));
    B = double(tenmat(ttm(core, {U1, U3}, [1, 3]), 2));
    C = theta;
%     Ymode 1
    right = 2 * lda * Y * A' + mu * (X + M / mu) * B' + C * U2;
    left = 2 * lda * A * A' + mu * B * B' + C * eye(size(A, 1));
%     U2 = CG(left', right')';
    U2 =  right / left;
end


function U3 = updateU3(U1, U2, U3, U4, B1, B2, Y, X, M, mu, R, F, theta)
    core = ttm(tensor(B1), B2', 3);
    A = double(tenmat(ttm(core, {U1, U2}, [1, 2]), 3));
    right1 = R' * R * mu;
    right2 = mu * A * A' + 2 * theta * eye(size(A, 1));
    left = mu * (X + M / mu) * A' + mu * R' * (U4 + F / mu) + 2 * theta * U3;
    U3 = sylvester(right1, right2, left);
end


function U4 = updateU4(U1, U2, U3, U4, B1, B2, Y, X, M, mu, lda, R, F, theta)
    core = ttm(tensor(B1), B2', 3);
    A = double(tenmat(ttm(core, {U1, U2}, [1, 2]), 3));
    left = 2 * lda * A * A' + (mu + 2 * theta) * eye(size(A, 1));
    right = 2 * lda * Y * A' + mu * (R * U3 - F) + 2 * theta * U4;
%     U4 = CG(left', right')';
    U4 =  right / left;
end

function B1 = updateB1(U1, U2, U3, U4, B1, B2, Y, X, M, mu, lda, theta)
    mu_ = 0.01;
    C2 = B1;
    C1 = C2;
    V1 = zeros(size(C1));
    disp('B1')
    for loop = 0:5
%         losss(U3 * B2', U2, U1, X + M / mu, C, C2, V2,  mu / 2, mu_);
        C2 = updateCore(U3 * B2', U2, U1, X + M / mu, C2, V1, mu / 2, mu_);
        C1 = updateCore(U4 * B2', U2, U1, Y, C1, V1, lda, mu_);
        V1 = V1 - (C1 - C2);
    end
    B1 = C1;
end
% function B1 = updateB1(U1, U2, U3, U4, B1, B2, Y, X, M, mu, lda, theta)
%     mu_ = 0.01;
%     beta_ = theta;
%     lamda_ = 0.0001;
% 
%     C = B1;
%     Cpre = C;
%     C1 = C;
%     C2 = C;
%     V1 = zeros(size(C));
%     V2 = zeros(size(C));
%     disp('B1')
% %     lda * norm(M2T(U1, U2, U4, B1, B2) - Y)
% %     lda * norm(double(tenmat(tensor(M2T(U1, U2, U4, B1, B2)), 1)) - double( tenmat(tensor(Y), 1)))
%     for loop = 0:5
% %         losss(U3 * B2', U2, U1, X + M / mu, C, C2, V2,  mu / 2, mu_);
%         C2 = updateCore(U3 * B2', U2, U1, X + M / mu, C2, V2, mu / 2, mu_);
%         
% %         losss(U3 * B2', U2, U1, X + M / mu, C, C2, V2,  mu / 2, mu_);
%         
%         
% %         losss(U4 * B2', U2, U1, Y, C, C1, V1, lda, mu_);
% %         lda * norm(M2T(U1, U2, U4, C1, B2) - Y)
%         C1 = updateCore(U4 * B2', U2, U1, Y, C1, V1, lda, mu_);
% %         losss(U4 * B2', U2, U1, Y, C, C1, V1, lda, mu_);
% %         lda * norm(M2T(U1, U2, U4, C1, B2) - Y)
%         
% 
% 
% %         C1 = C1(:);
% %         C2 = C2(:);
% 
%         a = (mu_ * (C1 + V1 + C2 + V2) + beta_ * Cpre) / (2 * mu_ + beta_);
%         b = lamda_ / (4 * mu_ + 2 * beta_);
%         C = sign(a) .* max(abs(a) - b, 0);
%           
% %         dd = losss(U3 * B2', U2, U1, X + M / mu, C2, C, V2,  mu / 2, mu_)  +   losss(U4 * B2', U2, U1, Y, C1, C, V1, lda, mu_)
%         V1 = V1 - (C - C1);
%         V2 = V2 - (C - C2);
%         
%         
% %         lda * norm(M2T(U1, U2, U4, C, B2) - Y)
%     end
%     B1 = C;
% 
% % ������admm�ķ�������ϡ��
% 
% % ��������ϡ��ķ���
% % temp = kron(U2, U1);
% % A = kron(U3 * B2', temp);
% % C = kron(U4 * B2', temp);
% % right = lda * A' * A + mu / 2 * C' * C + theta * eye(size(A, 2));
% % left = lda * A' * Y(:) + mu / 2 * C' * (X{:} + M{:} / mu) + theta * B1(:);
% % B1 = right / left;
% end

function losss(U3, U2, U1, Y, C1, C, V1, lda, mu)
    a =  (Y - ttm(tensor(C), {U1, U2, U3}, [1 2 3]));
    b = C1 - C  + V1;
    lda * norm( a(:)) 
end


function C2 = updateCore(S, H1, W1, HSI, C, V2, lda, mu)

    C = C(:);
    V2 = V2(:);
    
%     losss(S, H1, W1, HSI, C, C, V2, lda, mu)
    
    STS = S' * S;
    [SS, SU] = eig(STS);
    SU = diag(SU);
    H1TH1 = H1' * H1;
    [H1S, H1U] = eig(H1TH1);
    H1U = diag(H1U);
     W1TW1 = W1' *  W1;
    [W1S, W1U] = eig(W1TW1);
    W1U = diag(W1U);

    temp1 = kron(SU, H1U);
    temp1 = kron(temp1, W1U);
    mid = lda * temp1 + mu * ones(size(temp1));
    mid = 1 ./ mid;
    
    
%      size(HSI)
%     size(W1)
%     size(H1)
%     size(S)   
    D2Y = double(ttm(tensor(HSI), {W1', H1', S'}, [1 2 3]));
    


    D2Y_ = D2Y(:);
    right = reshape(lda * D2Y_ + mu * C - mu * V2, size(D2Y));
    
    temp1 = double(ttm(tensor(right), {W1S', H1S', SS'}, [1 2 3]));
    temp1 = temp1(:);
    temp1 = temp1 .* mid;
    left = reshape(temp1, [size(W1S, 1), size(H1S, 1), size(SS, 1)]);
    C2 = double(ttm(tensor(left), {W1S, H1S, SS}, [1 2 3]));
    
    C2 = C2(:);
    C2 = reshape(C2, [size(W1, 2), size(H1, 2), size(S, 2)]);
    
end


function B2 = updateB2(U1, U2, U3, U4, B1, B2, Y, X, M, mu, lda, theta)
    K = double(tenmat(ttm(tensor(B1), {U1, U2}, [1, 2]), 3));
    A = K * K';
    B = 2 * lda * U4' * U4;
    C = mu * U3' * U3;
    D =  2 * lda * K * Y' * U4 + mu * K * (M / mu +  X)' * U3 + 2 * theta * B2;
    B2 = reshape( linsolve( (kron(B' + C', A) + theta * eye(size(B, 1) * size(A, 1))), vec(D)), size(B2));
%     B2 = vec(C) / (kron(B', A) + theta);
    
%     B2 = reshape( linsolve( (kron(B' + D', A) + theta), vec(C)), size(B2));
   
%     B2 = (eye(size(A)) - (B + D) * inv(A)) * inv(A) * C;

    
%     B2 = Syl(A, B2, B, theta, D, C);
    
    
%     W6 = 2 * lda * U4' * U4;
%     P6 = mu * U3' * U3;
%     mid = K * K';
%     left =  (2 * lda + mu) * (W6 +  P6) + 2 * theta * ones(size(W6, 1));
%     right = 2 * lda *  U4' * Y * K' + mu * U3' * (M / mu +  X) * K' + 2 * theta * B2;
% %     B2 = ((right / mid)' / left');
%     B2 = (inv(left) * right / mid)';
    

end
% 
% function B2 = updateB2(U1, U2, U3, U4, B1, B2, Y, X, M, mu, lda, theta)
%     mu_ = 100;
%     beta_ = theta;
%     lamda_ = 0.0001;
%     C = B2';
%     Cpre = C;
%     C1 = C;
%     C2 = C;
%     V1 = C;
%     V2 = C;
% %     core = tl.unfold(tl.tenalg.multi_mode_dot(B1, [U1, U2], [0, 1]), mode=2)
%     
%     core = double(tenmat(ttm(tensor(B1), {U1, U2}, [1 2]), 3));
% 
%     for loop = 1:20 
%         C1 = updateCore2(U4, core, Y, C1, V1, lda, mu_);
%         C2 = updateCore2(U3, core,  X + M / mu, C2, V2, mu / 2, mu_);
%         
%         a = (mu_ * (C1 + V1 + C2 + V2) + beta_ * Cpre) / (2 * mu_ + beta_);
%         b = lamda_ / (4 * mu_ + 2 * beta_);
%         C = sign(a) .* max(abs(a) - b, 0);
% 
%         V1 = V1 - (C - C1);
%         V2 = V2 - (C - C2);
%     end
%     
%     B2 = C';
% end

function C = updateCore2(U4, core, Y, C1, V1, lda, mu_)
    AAT = core * core';
    [AS, AU] = eig(AAT);
    AU = diag(AU);

    U4TU4 = U4' * U4;
    [U4S, U4U] = eig(U4TU4);
    U4U = diag(U4U);
    
    
    temp1 = kron(AU, U4U);

    mid = lda * temp1 + mu_ * ones(size(temp1));
    mid = 1 ./ mid;

    right = lda * (U4' * Y * core') + mu_ * (C1 - V1);
    right = right(:);

    temp2 = U4S' * reshape(right, [size(U4S, 1), size(AS, 1)]) * AS;
    temp2 = temp2(:);

    mid2 = temp2 .* mid;

    temp3 = U4S * reshape(mid2, [size(U4S, 2), size(AS, 2)]) * AS';
    temp3 = temp3(:);

     C = reshape(temp3, [size(U4, 2), size(core, 1)]);
     C = C;
end


% AXB+X+AXD=C
% function X = Syl(A, X, B, theta, D, C)
%     
% end
