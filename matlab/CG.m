
% A = rand(100, 100);
% A = A * A';
% x1 = rand(100, 2);
% 
% b = A * x1;
% x = rand(size(x1));
% x = cg(A, b);
% 
% A * x - b


function [X_] = CG(A, B)
% 
    X_ = zeros(size(B));
    nn = size(B);
    for i =1:nn(2)
        X_(:, i) = conjgrad(A, B(:, i), X_(:, i));
    end
end


function [x] = conjgrad(A, b, x)
    r = b - A * x;
    p = r;
    rsold = r' * r;

    for i = 1:30
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
            break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end