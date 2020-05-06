clc; clear; close all;
variable_n = [50:50:500 537];
alldata = load('total.mat');
X_total = table2array(alldata.table_total(:,2:end));
[pms.m, ~] = size(X_total);

error_kernel = zeros(length(variable_n), 1);
time_kernel = zeros(length(variable_n), 1);

pms.sigma = sqrt(pms.m)/3;
pms.worker_num = 60;
pms.target_k = 10;
pms.local_target_k = pms.target_k+5;
pms.centralize = true;
local_m = floor(pms.m/pms.worker_num);
anchor = cell(pms.worker_num+1,1);
anchor{1} = 1;
data_total = cell(pms.worker_num,1);
for iter = 1: pms.worker_num
    if iter~= pms.worker_num
        anchor{iter+1} = anchor{iter} + local_m;
    else
        anchor{iter+1} = pms.m+1;
    end
    data_total{iter} = X_total(anchor{iter}: anchor{iter+1} - 1, :);
end

for ii_n = 1:length(variable_n)
    for repeat  = 1: 50
        data = cell(pms.worker_num,1);
        X = [];
        pms.n = variable_n(ii_n);
        idx = randperm(size(X_total,2), pms.n);
        for ii_worker = 1: pms.worker_num
            data{ii_worker} = data_total{ii_worker}(:,idx);
            X = [X; data{ii_worker}];
        end
        
        
        %% rbf
        kernel_gt = cal_RBF(X, pms.sigma);
        if pms.centralize == true
            [kernel_gt] = centralize_kernel(kernel_gt);
        end
        [W_svd, tr_kernel_svd, u, s]  = solve_global_svd(kernel_gt, pms);
        [W_dkpca, kernel_hat, lam_hat, lam_max, run_time] = solve_dkpca(data, pms, kernel_gt, 'RBF');
        time_kernel(ii_n) = time_kernel(ii_n) + run_time;
        error_kernel(ii_n) = error_kernel(ii_n) + pms.target_k-sum(max(abs(W_dkpca'*W_svd)).^2);
    end
    
end

error_kernel = error_kernel./repeat;
error_kernel(error_kernel == 0) = 1e-17;

figure; hold on;
yyaxis left
plot(variable_n, log10(error_kernel),':^','LineWidth',3);
yyaxis right
plot(variable_n, time_kernel, '-o','LineWidth',3);
legend('-DynamicLegend');

xlabel('N')
