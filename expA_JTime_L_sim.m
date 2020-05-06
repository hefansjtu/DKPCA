clc; clear; close all;
thres = 5000; % principal eigenvalue of sigma
pms.m = 1000;
pms.centralize = false;
variable_lm = [1:2:10 50 100 200:200:1000 ];
variable_J = unique(ceil(pms.m./variable_lm));
variable_n = [200 400 600];
error_linear = zeros(length(variable_J), length(variable_n));
approx_linear = zeros(length(variable_J), length(variable_n));
lam_linear = zeros(length(variable_J), length(variable_n));
delta_linear = zeros(length(variable_J), length(variable_n));
time_linear = zeros(length(variable_J), length(variable_n));
for ii_n = 1: length(variable_n)
    for repeat = 1:50
        pms.n = variable_n(ii_n);
        pms.k = 50; % rank of sigma 
        pms.target_k = 1;
        pms.local_target_k = pms.target_k;
        pms.sigma = sqrt(pms.m)/5;
        V_gt =  [orth([randn(pms.n, pms.n)])];
        U_gt = orth(randn(pms.m, pms.m));
        if pms.m <= pms.n
            sigma_gt = [diag(sort([thres; thres/2; thres/4; thres/1/100*abs(ones(pms.k-3,1));zeros(pms.m - pms.k,1)],'descend')) zeros(pms.m , pms.n-pms.m )];
        else
            sigma_gt = [diag(sort([thres; thres/2; thres/4; thres/1/100*abs(ones(pms.k-3,1));zeros(pms.n - pms.k,1)] ,'descend')); zeros(pms.m-pms.n, pms.n)];
        end
        X_gt = (U_gt*sqrt(sigma_gt)*V_gt);
        for ii_j = 1:length(variable_J)
            pms.worker_num = variable_J(ii_j);
            local_m = floor(pms.m/pms.worker_num);
            anchor = cell(pms.worker_num+1,1);
            anchor{1} = 1;
            data = cell(pms.worker_num,1);
            X=[];
            for iter = 1: pms.worker_num
                if iter~= pms.worker_num
                    anchor{iter+1} = anchor{iter} + local_m;
                else
                    anchor{iter+1} = pms.m+1;
                end
                data{iter} = X_gt(anchor{iter}: anchor{iter+1} - 1, :);
                X=[X; data{iter}];
            end
            %% linear
            tic
            kernel_gt = X'*X/(pms.m);            
            [W_svd, tr_linear_svd, ~,s]  = solve_global_svd(kernel_gt, pms);

            [W_dkpca, kernel_hat, lam_hat, lam_max, run_time] = solve_dkpca(data, pms, kernel_gt, 'Linear');
            time_linear(ii_j, ii_n) = time_linear(ii_j, ii_n) + run_time;
            error_linear(ii_j, ii_n) = error_linear(ii_j, ii_n) + pms.target_k-sum(max(abs(W_dkpca'*W_svd)).^2);%%
            approx_linear(ii_j,ii_n) = approx_linear(ii_j,ii_n) + norm((kernel_gt-kernel_hat),'fro');
            lam_tmp = diag(s);
            lam_linear(ii_j, ii_n) = lam_linear(ii_j, ii_n) + sum(lam_max);
            delta_linear(ii_j, ii_n) = delta_linear(ii_j, ii_n) + (lam_tmp(pms.target_k) - lam_tmp(pms.target_k+1));
            
        end
    end
end
% save workspace

variable_lm = floor(pms.m./variable_J);

error_linear=abs(error_linear)./repeat;
error_linear(error_linear == 0) = 1e-17;
approx_linear=approx_linear./repeat;
tmp_linear = abs(lam_linear)./delta_linear;
time_linear = time_linear./repeat;

comm_cost = cell(length(variable_n),1);
comp_cost = cell(length(variable_n),1);
for ii_n = 1: length(variable_n)
    comm_cost{ii_n} = (1+variable_n(ii_n))*pms.target_k./(pms.m./variable_J*variable_n(ii_n));
    comp_cost{ii_n} = max(variable_n(ii_n), pms.m./variable_J+pms.target_k.*variable_J);
end

figure; hold on;
for ii_n = 1: length(variable_n)
    yyaxis left
    plot(log10(variable_lm), log10(error_linear(:,ii_n)),':^','DisplayName',sprintf('n=%d',variable_n(ii_n)),'LineWidth',3);
    yyaxis right
    plot(log10(variable_lm), time_linear(:,ii_n),'-o','DisplayName',sprintf('n=%d',variable_n(ii_n)),'LineWidth',3);    
    legend('-DynamicLegend');
end
xlabel('J')




