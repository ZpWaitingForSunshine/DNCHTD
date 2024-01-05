function [] = demo_batch(RR)

    % load data
    % clear
    % close all
    addpath('my tensor SVD');
    addpath(genpath('lib'));
    % addpath(genpath('data'));
    % addpath(genpath('tSVD'));
    addpath(genpath('mxPerm'));
    % addpath(genpath('ktsvd-master'));
    addpath(genpath('Quality_Indices'));
    % addpath(genpath('method'));
    addpath(genpath('.'));
    
    mband=4;
    
    % A = rand(10, 2);
    % B = rand(11, 3);
    % C = tensor(rand(2, 3, 4));
    % D = double(tenmat(C, 3));
    % E = reshape(D', [2 3 4])
    % 
    % D =  double(tenmat(ttm(C, {A, B}, [1 2]), 3))'; 
    % 
    % E = kron(B, A) * double(tenmat(C, [1 2]));
    
    
    I_REF=imread('original_rosis.tif');
    % I_REF=imread('dc5.tif');
    
    I_REF=double(I_REF);
    % % 
    load LANDSAT.mat
    R=R(1:mband,1:size(I_REF,3));
    % 
%         I_REF=imread('dc.tif');
%         I_REF=double(I_REF(861:1280, 1:300,:));
%     %     I_REF=double(I_REF(861:910, 1:50,:));
%         load LANDSAT.mat
%         R=R(1:mband,1:size(I_REF,3));
    
    %     load Sandiego.mat
    %     Sandiego(:,:,[1:6 33:35 97 104:110 153:166 221:224 ]) = [];
    %     Sandiego(:,:,[94 95 96]) = [];
    %     I_REF= Sandiego(1:200,1:200,:);clear Sandiego
    %     load LANDSAT.mat
    %     R=R(1:mband,:);
    %     R(:,[1:6 33:35 97 104:110 153:166 221:224 ])=[];
    %     R(:,[94 95 96]) = [];
    
    %% data generation
    
    R=R./repmat(sum(R,2),1,size(R,2));
    I_temp= reshape(I_REF,size(I_REF,1)*size(I_REF,2),size(I_REF,3));
    I_ms=R*I_temp';
    I_MSI=reshape(I_ms',size(I_REF,1),size(I_REF,2),mband);
    ratio = 5;
    % overlap = 1:41; % commun bands (or spectral domain) between I_PAN and I_HS
    size_kernel=[9 9];
    sig = (1/(2*(2.7725887)/ratio^2))^0.5;
    %sig =2;
    s=fspecial('gaussian',size_kernel,sig);
    
    I_HSn = gauss3filter(I_REF,s);
    I_HS=my_downsample(I_HSn,ratio);
    
    % pavia = pavia_3_order;
    % % pavia = pavia_3_order.*max(max(max(I_HS)));
    % imshow(pavia);
    
    tic
    %% running test
    % proposed method
    tic
    disp("patchsize")
    disp(RR)
    
    [I_CTD] = Super_NCTHT_PAO_r(I_HS,RR,I_MSI,ratio,s,300,R);
    % [I_CTD] = Super_NCTCP(I_HS,180,I_MSI,ratio,s,32,R);
    time_CTD=toc
    QI_CTD = QualityIndices(I_CTD,I_REF,ratio); % measure cc, sam0, rmse, ergas
    AM=max(I_REF(:));
    psnr_CTD    =  PSNR3D(I_CTD*255/AM,I_REF*255/AM);
    disp(psnr_CTD);
    disp("--------------------------------------------------------------------------------")

    toc
end

