addpath('my tensor SVD');
addpath(genpath('lib'));
% addpath(genpath('data'));
% addpath(genpath('tSVD'));
addpath(genpath('mxPerm'));
% addpath(genpath('ktsvd-master'));
addpath(genpath('Quality_Indices'));
% addpath(genpath('method'));
addpath(genpath('.'));

for i = 1:100
   [P,U0,out]= cp_als(tensor(double(dc5)),4)
end