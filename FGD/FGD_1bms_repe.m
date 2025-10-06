function [error,errl,itererr] = FGD_1bms_repe(n,r,m,eta,gradient,reb,numiter,seed,repe)
% gradient == '1bms': use 1-bit sensing gradient; otherwise, use single
% index model gradient
errlfgd = zeros(repe,numiter+1);
rng('default');
for i = 1:repe
    rng(seed);
    seed = seed + 1;
    %% data generation
    X =  randn(n, r)*randn(r,n);
    X =  X / norm(X,"fro");
    A_mat = randn(m, n^2);
    y = A_mat * X(:);
    y = sign(y);  
    
    %% initialization
    Z0 = reshape(y'*A_mat/m,[n,n]);
    [U,S,V] = svd(Z0);
    Z0 = U(:,1:r) * S(1:r,1:r) * (V(:,1:r))';
    X0 = Z0/norm(Z0,"fro");

    %% Loop
    errlfgd(i,:) = FGD_1bms(A_mat,y,X0,X,eta,n,r,gradient,reb,numiter);       
end

%% one-bit matrix sensing result
errl = errlfgd(:,numiter+1);
error = mean(errl);
itererr = mean(errlfgd);
 

