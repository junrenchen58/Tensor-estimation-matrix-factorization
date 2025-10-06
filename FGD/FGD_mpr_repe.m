function [error,errl,itererr] = FGD_mpr_repe(n,r,m,sigma,eta,ini,reb,numiter,seed,repe)
% ini: initialization parameter  
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
    y = abs(y);  
    y = y + sigma*randn(m,1);
    
    %% initialization
    SS = randn(n,n);
    SS = SS/norm(SS,'fro');
    Z0 = ini* X+(1-ini)*SS;
    [U,S,V] = svd(Z0);
    Z0 = U(:,1:r) * S(1:r,1:r) * (V(:,1:r))';
    X0 = Z0/norm(Z0,"fro");

    %% Loop
    errlfgd(i,:) = FGD_mpr(A_mat,y,X0,X,eta,n,r,reb,numiter);       
end

%% matrix phase retrieval result
errl = errlfgd(:,numiter+1);
error = mean(errl);
itererr = mean(errlfgd);
 

