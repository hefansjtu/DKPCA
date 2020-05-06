clc; clear; close all;


alldata = load('total.mat');
X_total = table2array(alldata.table_total(:,2:end));
[pms.m,pms.n] = size(X_total);

variable_lm = [10:20:500 pms.m];
variable_J = unique(ceil(pms.m./variable_lm),'stable');
variable_lm = floor(pms.m./variable_J);
pms.target_k = 10;
pms.local_target_k = pms.target_k+5;
pms.sigma = sqrt(pms.m)/3;
pms.centralize = true;
error_kernel = zeros(length(variable_J), 1);
time_kernel = zeros(length(variable_J), 1);

kernel_gt = cal_RBF(X_total, pms.sigma);
[kernel_gt] = centralize_kernel(kernel_gt);
[W_svd, tr_kernel_svd, u, s]  = solve_global_svd(kernel_gt, pms);

for ii_j = 1:length(variable_J)
    pms.worker_num = variable_J(ii_j);
    local_m = floor(pms.m/pms.worker_num);
    anchor = cell(pms.worker_num+1,1);
    anchor{1} = 1;
    data = cell(pms.worker_num,1);
    for iter = 1: pms.worker_num
        if iter~= pms.worker_num
            anchor{iter+1} = anchor{iter} + local_m;
        else
            anchor{iter+1} = pms.m+1;
        end
        data{iter} = X_total(anchor{iter}: anchor{iter+1} - 1, :);
    end
    
    %% rbf
    
    [W_dkpca, kernel_hat, lam_hat, lam_max, run_time] = solve_dkpca(data, pms,kernel_gt, 'RBF');
    time_kernel(ii_j) = time_kernel(ii_j) + run_time;
    error_kernel(ii_j) = error_kernel(ii_j) + pms.target_k-sum(max(abs(W_dkpca'*W_svd)).^2);
end


error_kernel(error_kernel == 0) = 1e-17;


figure; hold on;
yyaxis left
plot((variable_lm), log10(error_kernel),':^','LineWidth',3);
yyaxis right
plot((variable_lm), time_kernel, '-o','LineWidth',3);
legend('-DynamicLegend');

xlabel('J')
