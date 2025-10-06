% Inputs: A_mat: m-by-n1n2 matrix, each row is a vectorization of the covariate matrix; 
% y: observation; X0: initialization of X; 
% X: true parameter of interest; n : n* n matrix; r: matrix rank
% iter_max: the maximum number of iteration;  
% reb == 1: use rebalancing step 
% FOR THE LOCAL CONTRACTION OF FGD for Matrix Phase Retrieval
function errl = FGD_mpr(A_mat,y,X0,X,eta,n,r,reb,iter_max)
Xt = X0;
[U,S,V]=svd(Xt);
Ut = U(:,1:r) * sqrt(S(1:r,1:r));
Vt = V(:,1:r) * sqrt(S(1:r,1:r));
Xt_err = norm(Xt - X,"fro")/norm(X,"fro");
errl = [Xt_err];
m = size(A_mat,1);
for iter = 1:iter_max    
    % compute original gradient
    haty = A_mat * Xt(:);
    coeff = sign(haty);
    haty = abs(haty);
    H = reshape((coeff.*(haty - y))' * A_mat/m, [n,n]);
    % FGD
    Ut = Ut - eta *(H* Vt);
    Vt = Vt - eta *(H'* Ut);
    % rebalancing
    if reb == 1
        [Ut,Vt] = rebalance(Ut,Vt);
    end
    % normalization
    Xt = Ut * Vt';
    %Ut = Ut / sqrt(norm(Xt,"fro"));
    %Vt = Vt / sqrt(norm(Xt,"fro"));
    %Xt = Xt/norm(Xt,"fro");  
    Xt_err_new = norm(Xt - X,"fro")/norm(X,"fro");
    Xt_err = Xt_err_new;
    errl = [errl, Xt_err];
end
end