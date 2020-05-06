function [W, tr, u, s]  = solve_global_svd(cor_mat_noise, pms)

[u, s, ~] = svd(cor_mat_noise);
tr = trace(s(1:pms.target_k, 1:pms.target_k));
W = u(:,1:pms.target_k);
end