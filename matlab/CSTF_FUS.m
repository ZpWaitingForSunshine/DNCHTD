function HR_HSI= CSTF_FUS(HSI,MSI,T,BW,BH,downsampling_scale,par,s0)
mu=0.001;
lambda=par.lambda;

%%  simulate LR-HSI
Y_h_bar=hyperConvert2D(HSI);
HSI1=Unfold(HSI,size(HSI),1);
HSI2=Unfold(HSI,size(HSI),2);
HSI3=Unfold(HSI,size(HSI),3);

%%  simulate HR-MSI
MSI1=Unfold(MSI,size(MSI),1);
MSI2=Unfold(MSI,size(MSI),2);
MSI3=Unfold(MSI,size(MSI),3);

%% inilization D1 D2 D3 C
[m n]=size(MSI1);
D=MSI1;
params.Tdata = 2;            % Number of target sparse vectors
params.dictsize =par.W;      %   441;      % Number of dictionary elements (# columns of D)
params.iternum = 100;
params.DUC =1; 
      
D1 = trainD(D,MSI1,[],[],params);
params.dictsize =par.H;
D2 = trainD(MSI2,MSI2,[],[],params);
D3=vca(Y_h_bar,par.S);


D_1=ifft(fft(D1).*repmat(BW,[1 par.W]));
D_1=D_1(s0:downsampling_scale:end,:);
  
D_2=ifft(fft(D2).*repmat(BH,[1 par.H]));
D_2=D_2(s0:downsampling_scale:end,:);
D_3=T*D3;
  
D11{1}=D_1;
D11{2}=D_2;
D11{3}=D3;
D22{1}=D1;
D22{2}=D2;
D22{3}=D_3;
C=zeros(size(D1,2),size(D2,2),size(D3,2));
C   =  sparse_tucker2( D11,D22, HSI,MSI, lambda,C,mu );
       
HR_HSI=ttm(tensor(C),{D1,D2,D3},[1 2 3]);
HR_HSI = double(HR_HSI);


