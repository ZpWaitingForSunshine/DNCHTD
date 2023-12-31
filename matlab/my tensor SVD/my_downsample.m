 function [outMatrix, inMatrix] = my_downsample(inMatrix, rate)

% inSize = size(inMatrix);
% dim = nb_dims(inMatrix);
% 
% if (dim==1)
%     if inSize(1)==1
%         inMatrix = inMatrix';
%         inSize = size(inMatrix);
%     end
%     lengthOut = ceil(max(inSize)/rate);
%     diffSize = rate*lengthOut - inSize(1);
%     outMatrix = zeros(lengthOut,1);
%     inMatrix = [inMatrix;inMatrix(end)*ones(diffSize,1)];
%     outSize = size(outMatrix);
%     for i = 1:rate
%         outMatrix = outMatrix+inMatrix(i:rate:end);
%     end
%     outMatrix = outMatrix/rate;
% elseif (dim==2)
%     if inSize(1)==1 || inSize(2)==1
%         error('Logical Error');
%     end
%     lengthOut = ceil(inSize/rate);
%     diffSize = rate*lengthOut - inSize;
%     outMatrix = zeros(lengthOut);
%     if diffSize(1)~=0
%         inMatrix = [inMatrix; repmat(inMatrix(end,:),[diffSize(1),1] )];
%     end
%     if diffSize(2)~=0
%         inMatrix = [inMatrix repmat(inMatrix(:,end), [1, diffSize(2)] )];
%     end
%     for i = 1:rate
%         for j = 1:rate
%             outMatrix = outMatrix + inMatrix(i:rate:end,j:rate:end);
%         end
%     end
%     outMatrix = outMatrix/(rate*rate);
% elseif (dim==3)
%     if inSize(1)==1 || inSize(2)==1 || inSize(3)==1
%         error('Logical Error');
%     end
%     lengthOut = ceil(inSize/rate);
%     diffSize = rate*lengthOut - inSize;
%     outMatrix = zeros(lengthOut);
%     if diffSize(1)~=0
%         inMatrix = [inMatrix; repmat(inMatrix(end,:,:),[diffSize(1),1] )];
%     end
%     if diffSize(2)~=0
%         inMatrix = [inMatrix repmat(inMatrix(:,end,:), [1, diffSize(2)] )];
%     end
%     if diffSize(3)~=0
%         inMatrix(:,:,end+1:end+diffSize(3)) = repmat(inMatrix(:,:,end), [1,1,diffSize(3)]);
%     end
%     for i = 1:rate
%         for j = 1:rate
%             for k = 1:rate
%                 outMatrix = outMatrix + inMatrix(i:rate:end,j:rate:end,k:rate:end);
%             end
%         end
%     end
%     outMatrix = outMatrix/(rate^3);
% else
%     error('TODO');
% end

% outMatrix = imresize(inMatrix,1/rate,'bicubic');
 outMatrix = inMatrix(1:rate:end,1:rate:end,:);
% outMatrix = downsamp_HS(inMatrix, rate, 0);


% outMatrix = zeros (size(inMatrix,1)/rate,size(inMatrix,2)/rate,size(inMatrix,3));
% for i=1:rate
%     for j=1:rate
%         temp= inMatrix(i:rate:end, j:rate:end,:);
%         outMatrix= outMatrix + temp;
%     end
% end
% outMatrix= outMatrix/rate/rate;


% [M ,N, B]=size(inMatrix);
% G=create_G([1 M/rate 1 N/rate], rate);
% Y_h_bar=hyperConvert2d(inMatrix)*G;
% outMatrix=hyperConvert3d(Y_h_bar,M/rate, N/rate );


