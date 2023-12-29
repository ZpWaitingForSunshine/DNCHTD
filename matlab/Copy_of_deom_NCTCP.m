% load data
clear
close all
addpath('my tensor SVD');
addpath(genpath('lib'));
% addpath(genpath('data'));
% addpath(genpath('tSVD'));
addpath(genpath('mxPerm'));
% addpath(genpath('ktsvd-master'));
addpath(genpath('Quality_Indices'));
% addpath(genpath('method'));
addpath(genpath('.'));
% addpath('ImprovedDL')
% addpath('functions', 'sisal')

mband=4;


% I_REF=imread('original_rosis.tif');
% % I_REF=imread('dc5.tif');
% 
%  I_REF=double(I_REF);
% % 
%  load LANDSAT.mat
%  R=R(1:mband,1:size(I_REF,3));
% 
    I_REF=imread('dc.tif');
    I_REF=double(I_REF(861:910,1:50,:));
%     I_REF=double(I_REF(861:1280,1:300,:));
    load LANDSAT.mat
    R=R(1:mband,1:size(I_REF,3));

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

tic

%% init by cstf 
[M,N,L] = size(I_REF);

downsampling_scale=5;

s0=downsampling_scale/2;
% BW=ones(5,1)/5;
% BW1=psf2otf(BW,[M 1]);
% BH=ones(5,1)/5;
% BH1=psf2otf(BH,[N 1]);
% 
% 
% par.W=M -5; par.H=N-5;  par.S=50; par.lambda=1e-3;

% I_HS = CSTF_FUS(I_HS,I_MSI,R,BW1,BH1,downsampling_scale,par,s0);
% 
% save I_HS.mat I_HS;

load I_HS.mat;

% 
% AM=max(I_REF(:));
% psnr_CTD1 =  PSNR3D(I_HS*255/AM,I_REF*255/AM);

tic

%% running test
% proposed method

[I_CTD] = Super_NCTHT_PAO_r(I_HS,120,I_MSI,ratio,s,25,R);
% [I_CTD] = Super_NCTCP(I_HS,180,I_MSI,ratio,s,32,R);
time_CTD=toc;
QI_CTD = QualityIndices(I_CTD,I_REF,ratio); % measure cc, sam0, rmse, ergas
AM=max(I_REF(:));
psnr_CTD    =  PSNR3D(I_CTD*255/AM,I_REF*255/AM);
disp(psnr_CTD);
toc