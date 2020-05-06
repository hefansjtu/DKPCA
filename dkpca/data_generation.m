function [X_noise, X, data, num, tr_gt, sigma_gt] = data_generation(pms)
thres = 5000;
beta = 2;
local_m = floor(pms.m/pms.worker_num);
    V_gt =  [orth([randn(pms.n, pms.n)])];
    U_gt = orth(randn(pms.m, pms.m));

% sigma_gt = [diag(sort([thres; thres/2; thres/4; thres/1/100*abs(ones(pms.k-3,1));zeros(pms.m - pms.k,1)],'descend')) zeros(pms.m , pms.n-pms.m )];
if pms.m <= pms.n
    sigma_gt = [diag(sort([thres*ones(pms.target_k,1); thres/beta; thres/100*abs(ones(pms.k-2,1)); zeros(pms.m - pms.k,1)],'descend')) zeros(pms.m , pms.n-pms.m )];
else
    sigma_gt = [diag(sort([thres*ones(pms.target_k,1); thres/beta; thres/100*abs(ones(pms.k-2,1)); zeros(pms.n - pms.k,1)],'descend')); zeros(pms.m-pms.n, pms.n)];
end
tr_gt = trace(sigma_gt(1:pms.target_k, 1:pms.target_k));
% U_gt =  orth(randn(pms.m, pms.k));
X = (U_gt*sqrt(sigma_gt)*V_gt);
% X_noise = zeros(size(X));    
% U_gt = cell(pms.worker_num,1);

X_noise=[];

num = cell(pms.worker_num,1);
data = cell(pms.worker_num,1);
% cor_mat = cell(pms.worker_num,1);
noise_local = cell(pms.worker_num,1);
anchor = cell(pms.worker_num+1,1);
anchor{1} = 1;


for iter = 1: pms.worker_num
    if iter~= pms.worker_num
        anchor{iter+1} = anchor{iter} + local_m;
    else
        anchor{iter+1} = pms.m+1;
    end
    noise_local{iter} = 2*rand(1)*randn(local_m, pms.n) + 0*rand(1);
    
    data{iter} = X(anchor{iter}: anchor{iter+1} - 1, :);
%     data{iter} = tmp;
    X_noise(anchor{iter}: anchor{iter+1} - 1, :) =  data{iter};
%     cor_mat{iter} = data{iter}'*data{iter}/num{iter};
end

% tr_svd = trace(u'*cor_mat_gt*u);
end