function [error,errl] = tsim_truncation_repe(n,r,m,mu,numiter,seed,repe)
%% Truncation Nonlinearity f(a) = sign(a)*min(|a|,1); mu = 0.683;
errlrgd = zeros(repe,numiter+1);
rng('default');
for i = 1:repe
    rng(seed);
    seed = seed + 1;
    p1 = n; p2 = n; p3 = n;
    r1 = r; r2 = r; r3 = r;
    %% data generation
    S = tensor(randn(r1, r2, r3));
    S = S/norm(S); % ensure parameter norm 1
    E1 = randn(p1, r1);
    E2 = randn(p2, r2);
    E3 = randn(p3, r3);
    [U1,~,~] = svds(E1,r1);
    [U2,~,~] = svds(E2,r2);
    [U3,~,~] = svds(E3,r3);
    A = tensor(randn(p1, p2, p3,m));
    U = {U1, U2, U3};
    X = ttm(S, U, [1:3]);
    A_mat = tenmat(A, 4);
    A_mat = A_mat.data;
    y = A_mat * X(:);
    y = sign(y).* min(abs(y),1); %% the nonlinearity
    
    %% initialization
    [Xt,Ut] = tini_nonli(A,y,r,n);

    %% RGD  
    [errlrgd(i,:),~] = RGD_tsims(A_mat,y,Xt, Ut,X, U, p1, p2, p3,r1,r2,r3, numiter,mu);       
     
end
errl = errlrgd(:,numiter+1);
error = mean(errl);
 

