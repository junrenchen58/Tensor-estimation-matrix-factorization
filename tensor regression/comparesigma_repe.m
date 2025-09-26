function [errlrgd,timergd,errlrgn,timergn] = comparesigma_repe(n,r,m,sigma,numiter,seed,repe)
errlrgd = zeros(repe,numiter+1);
errlrgn = zeros(repe,numiter+1);
timergd = zeros(1,repe); 
timergn = zeros(1,repe);
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
    eps = sigma * randn(m,1);
    A_mat = tenmat(A, 4);
    A_mat = A_mat.data;
    y = A_mat * X(:) + eps;
    
    %% initialization
    W = reshape(tensor(y' * A_mat),[p1, p2, p3])/m;
    init_result = hosvd(W,norm(W),'ranks',[r1 ,r2 ,r3],'sequential',false,'verbosity',0);
    Xt = ttm(init_result.core, init_result.u,[1:3]);
    Ut = init_result.u;

    %% RGD with fixed eta = 1
    [errlrgd(i,:),timergd(i)] = RGD_scalar_tensor_regression(A_mat,y,Xt, Ut,X, U, p1, p2, p3,r1,r2,r3, numiter, 1 );
     
    %% RGN 
    retra_type = 'hosvd';
    [errlrgn(i,:),timergn(i)] = RGN_scalar_tensor_regression(A,y,Xt, Ut,X, U, p1, p2, p3,r1,r2,r3, numiter, retra_type );
     
end
errlrgd = mean(errlrgd);
timergd = mean(timergd);
 
errlrgn = mean(errlrgn); 
timergn = mean(timergn);

