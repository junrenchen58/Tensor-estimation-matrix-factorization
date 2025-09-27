function [tenini,U] = tini_nonli(A,y,r,n)
%% assume br = (r,r,r)
p1 = n;
p2 = n;
p3 = n;
A_mat = tenmat(A, 4);
A_mat = A_mat.data;
m = size(A_mat,1);
W = reshape(tensor(y' * A_mat),[p1, p2, p3])/m; %% Average data tensor

yA_mat = y.*A_mat;

yA_mat_sum = sum(yA_mat);
yA_sum = reshape(yA_mat_sum,[p1,p2,p3]);
yA_sum = tensor(yA_sum);

U = cell(1,3);
for i = 1:3
    yA_sum_mati = tenmat(yA_sum,i);
    yA_sum_mati = yA_sum_mati.data;
    hatNi = yA_sum_mati * yA_sum_mati';
    DiaM = zeros(p1,p1); %% assume p1 = p2 = p3 = n in this step
    for j = 1:m
        yA_matj = yA_mat(j,:);
        yA_matj = reshape(yA_matj,[p1,p2,p3]);
        yA_matj = tensor(yA_matj);
        yA_matj = tenmat(yA_matj,i);
        yA_matj = yA_matj.data;
        DiaM = DiaM + yA_matj * yA_matj';
    end
    hatNi = (hatNi - DiaM)/(m*(m-1));
    [U1,~] = eigs(hatNi, r, 'la');
    U{i} = U1 * U1';
end
tenini = ttm(W, U, [1:3]);
tenini = tenini / norm(tenini); %% normalization the output
end

