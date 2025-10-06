function [U1,V1] = rebalance(U,V)
[Q1,R1] = qr(U);
[Q2,R2] = qr(V);
[A,L,B] = svd(R1 * R2');
U1 = Q1*A*sqrt(L);
V1 = Q2*B*sqrt(L);
end

