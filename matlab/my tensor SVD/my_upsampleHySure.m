 function outMatrix = my_upsampleHySure(inMatrix, rate)
% 

outMatrix = upsamp_HS(inMatrix, rate, size(inMatrix,1)*rate, size(inMatrix,2)*rate, 0);