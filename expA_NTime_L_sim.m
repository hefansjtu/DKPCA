clc; clear; close all;
variable_n = [50:50:1000];
variable_m = [1000];
pms.k = 50;
pms.worker_num = 5;
pms.target_k = 1;
pms.local_target_k = pms.target_k;
pms.centralize = false;
error_linear = zeros(length(variable_n), length(variable_m));
approx_linear = zeros(length(variable_n), length(variable_m));
lam_linear = zeros(length(variable_n), length(variable_m));
delta_linear = zeros(length(variable_n), length(variable_m));
time_linear = zeros(length(variable_n), length(variable_m));
for ii_m = 1: length(variable_m)
    for repeat = 1:50
        pms.m = variable_m(ii_m);
        pms.n = max(variable_n);
        pms.sigma = sqrt(pms.m)/10;
        [X_total, ~, data_total, ~, tr_gt, sigma_gt] = data_generation(pms);
        for ii_n = 1:length(variable_n)
            data = cell(pms.worker_num,1);
            X = [];
            pms.n = variable_n(ii_n);
            for ii_worker = 1: pms.worker_num
                data{ii_worker} = data_total{ii_worker}(:,1:pms.n);
                X = [X; data{ii_worker}];
            end
            
            %% linear
            
            kernel_gt = X'*X/(pms.m);
            [W_svd, tr_linear_svd, ~,s]  = solve_global_svd(kernel_gt, pms);
            [W_dkpca, kernel_hat, lam_hat, lam_max, run_time] = solve_dkpca(data, pms, kernel_gt, 'Linear');
            time_linear(ii_n, ii_m) = time_linear(ii_n, ii_m) + run_time;
            error_linear(ii_n, ii_m) = error_linear(ii_n, ii_m) + pms.target_k-sum(max(abs(W_dkpca'*W_svd)).^2);%%
            approx_linear(ii_n,ii_m) = approx_linear(ii_n,ii_m) + norm((kernel_gt-kernel_hat),'fro');
            lam_tmp = diag(s);
            lam_linear(ii_n, ii_m) = lam_linear(ii_n, ii_m) + sum(lam_max);
            delta_linear(ii_n, ii_m) = delta_linear(ii_n, ii_m) + (lam_tmp(pms.target_k) - lam_tmp(pms.target_k+1));
            
        end
    end
end
error_linear=error_linear./repeat;
error_linear(error_linear == 0) = 1e-17;
approx_linear=approx_linear./repeat;
tmp_linear = abs(lam_linear)./delta_linear;
time_linear = time_linear./repeat;

comm_cost = cell(length(variable_m),1);
comp_cost = cell(length(variable_m),1);
for ii_m = 1: length(variable_m)
    comm_cost{ii_m} = (1+pms.n)*pms.target_k./(variable_m(ii_m)./pms.worker_num.*variable_n);
    comp_cost{ii_m} = max(variable_n, variable_m(ii_m)./pms.worker_num+pms.target_k.*pms.worker_num);
end

figure; hold on;
for ii_m = 1: length(variable_m)
    yyaxis left
    plot(variable_n, log10(error_linear(:,ii_m)),':^','DisplayName',sprintf('m=%d',variable_m(ii_m)),'LineWidth',3);
    yyaxis right
    plot(variable_n, time_linear(:,ii_m),'-o','DisplayName',sprintf('m=%d',variable_m(ii_m)),'LineWidth',3);
    legend('-DynamicLegend');
end
xlabel('N')

