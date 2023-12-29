% Perform three-dimensional Gaussian smoothing in the frequency domain.
% This is achieved by replacing the spatial domain convolution with 
% Fourier coefficient multiplication.
% R = gauss3filter(I);
% R = gauss3filter(I, sigma);
% R = gauss3filter(I, sigma, pixelspacing);
% In a spatial domain representation, R = convn(I, f(x,y,z));
% The Gaussian kernel f(x,y,z) is different depending on the function 
% inputs, see the description below. No image padding is provided, pay
% attention to the Fourier wrap-around artifacts.
%
% Anisotropic smoothing is partly supported, anisotropic voxel size is 
% fully supported. Suband_1.5 frequency oversampling is employed to reduce 
% numerical erros when sigma is less than the voxel length. Please refer to 
% following paper for the Subband_x frequency oversampling technique:
%   Max W. K. Law and Albert C. S. Chung, "Efficient Implementation for Spherical Flux Computation and Its Application to Vascular Segmentation",
%   IEEE Transactions on Image Processing, 2009, Volume 18(3), 596�V612
%
%
%   R = gauss3filter(I);
%   Smooth the image using isotropic smoothing with sigma = 1 voxel-length,
%       f(x,y,z) = (2*pi)^(-3/2) * exp(-(x.^2/2 - y.^2/2 - z.^2/2));
%
%   R = gauss3filter(I, sigma);
%   If sigma is a scalar, it smooths the image using isotropic smoothing with
%   sigma voxel-length,
%       f(x,y,z) = (2*pi)^(-3/2)/(sigma^3) * exp(-(x.^2/sigma^2/2 - y.^2/sigma^2/2 - z.^2/sigma^2/2));
%   If sigma is a 3D vector, i.e. sigma = [sigma_x sigma_y sigma_z], it 
%   smooths the image using anisotropic smoothing (oriented anisotropic 
%   Gaussian is not supported),
%       f(x,y,z) = (2*pi)^(-3/2)/sigma(1)/sigma(2)/sigma(3) * exp(-(x.^2/sigma(1)^2/2 - y.^2/sigma(2)^2/2 - z.^2/sigma(3)^2/2));
%
%   R = gauss3filter(I, sigma, pixelspacing);
%   If sigma is a scalar, smooth the image using isotropic smoothing with
%   sigma physical-length. pixelspacing is a 3D vector. It defines the size 
%   of a voxel in physical-length,
%       f(x,y,z) = (2*pi)^(-3/2)/(sigma^3) * exp(-((x*pixelspacing(1)).^2/sigma^2/2 - (y*pixelspacing(2)).^2/sigma^2/2 - (z*pixelspacing(3)).^2/sigma^2/2));
%   If sigma is a 3D vector, sigma = [sigma_x sigma_y sigma_z],
%       f(x,y,z) = (2*pi)^(-3/2)/sigma(1)/sigma(2)/sigma(3) * exp(-((x*pixelspacing(1)).^2/sigma(1)^2/2 - (y*pixelspacing(2)).^2/sigma(2)^2/2 - (z*pixelspacing(3)).^2/sigma(3)^2/2));
%
% Remarks
%   The outputs of gauss3filter(I), gauss3filter(I, 1) and
%   gauss3filter(I, 1, [1 1 1]) are identical.
% 
%   To enable GPU computation (Matlab 2012a or later, CUDA 1.3 GPU are required), use
%   R = gauss3filter(gpuArray(I), sigma, pixelspacing).
%
%   The Gaussian kernel in the frequency domain is
%   exp(-2*pi*pi* (u.^2 *sigma1 + v.^2 *sigma2 + w.^2 * sigma3));
%
%   Please kindly cite the following paper if you use this program, or any code 
%   extended from this program.
%       Max W. K. Law and Albert C. S. Chung, "Efficient Implementation for Spherical Flux Computation and Its Application to Vascular Segmentation��,
%       IEEE Transactions on Image Processing, 2009, Volume 18(3), 596�V612
%
% Author: Max W.K. Law
% Email:  max.w.k.law@gmail.com
% Page:   http://www.cse.ust.hk/~maxlawwk/



function result=gauss3filterBack(image, sigma, pixelspacing)



% using method in HySure
FBm = fft2(sigma);
FBC = conj(FBm);
% FBC=FBm;
% Blur Ym
Ym = im2mat(image);
Ymb = ConvC(Ym, FBC, size(image,1));
% Masked version of Ym (to use when estimating the spatial blur)
result = mat2im(Ymb, size(image,1));

    
end

