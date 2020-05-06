clc; clear; close all;

pms.m = 1000;
pms.n = 100;
pms.k = 50;
pms.worker_num = 10;
pms.target_k = 1;
pms.c = 4;
pms.maxIter = 20;
pms.maxInnerIter = 12;
pms.centralize = false;
repeat_num = 50;

error_dkpca = zeros(1, repeat_num);
error_dkpca_1 = zeros(1, repeat_num);
error_dpca_all = zeros(pms.maxIter, repeat_num);
error_dpca_half = zeros(pms.maxIter, repeat_num);
error_dpca_one = zeros(pms.maxIter, repeat_num);

for repeat = 1: repeat_num
    
    [X_total, ~, data_total, ~, tr_gt, sigma_gt] = data_generation(pms);
    
    %% svd
    kernel_gt = X_total'*X_total/(pms.m);
    [u,s,~] = svd(kernel_gt);
    U_gt = X_total*u(:,1:pms.target_k);
    U_gt = U_gt/norm(U_gt);
    
    uu_gt = U_gt*U_gt';
    %% ours
    
    pms.local_target_k = pms.target_k;
    [W_dkpca, ~, ~, ~, ~] = solve_dkpca(data_total, pms, kernel_gt, 'Linear');
    U_dkpca = X_total*W_dkpca;
    U_dkpca = U_dkpca/norm(U_dkpca);
    
    error_dkpca(repeat) = norm(U_dkpca*U_dkpca'-uu_gt,'fro')^2;
    
    pms.local_target_k = pms.target_k+1;
    [W_dkpca, ~, ~, ~, ~] = solve_dkpca(data_total, pms, kernel_gt, 'Linear');
    U_dkpca = X_total*W_dkpca;
    U_dkpca = U_dkpca/norm(U_dkpca);
    
    error_dkpca_1(repeat) = norm(U_dkpca*U_dkpca'-uu_gt,'fro')^2;
    
    %% d-pca
    N_local = cell(pms.worker_num, 1);
    
    for worker_iter = 1: pms.worker_num        
        N_local{worker_iter} = 1:pms.worker_num;    
    end
     [error_dpca_all(:,repeat)]   = dpca(data_total, X_total, N_local, uu_gt, pms);

    for worker_iter = 1: pms.worker_num        
        N_local{worker_iter} = [mod((worker_iter-2),pms.worker_num) mod((worker_iter-1),pms.worker_num) worker_iter mod((worker_iter+1),pms.worker_num) mod((worker_iter+2),pms.worker_num)];%1:pms.worker_num;
        N_local{worker_iter}(find(N_local{worker_iter} == 0)) = pms.worker_num;        
    end
     [error_dpca_half(:,repeat)]   = dpca(data_total, X_total, N_local, uu_gt, pms);
     
     
    for worker_iter = 1: pms.worker_num        
        N_local{worker_iter} = [mod((worker_iter-1),pms.worker_num) worker_iter mod((worker_iter+1),pms.worker_num) ];
        N_local{worker_iter}(find(N_local{worker_iter} == 0)) = pms.worker_num;        
    end
     [error_dpca_one(:,repeat)]   = dpca(data_total, X_total, N_local, uu_gt, pms);
end

mean_err_dpca_half = mean(error_dpca_half,2);
mean_err_dpca_all = mean(error_dpca_all,2);
mean_err_dpca_one = mean(error_dpca_one,2);
mean_err_dkpca = mean(error_dkpca);
mean_err_dkpca_1 = mean(error_dkpca_1);

figure; hold on;
plot(1:pms.maxIter, log(mean_err_dkpca)*ones(1,pms.maxIter),':','LineWidth',3,'DisplayName','DKPCA');
% plot(1:pms.maxIter, log(mean_err_dkpca_1)*ones(1,pms.maxIter));
plot(1:pms.maxIter, log(mean_err_dpca_all),'LineWidth',3,'DisplayName','DPCA10');
plot(1:pms.maxIter, log(mean_err_dpca_half),'LineWidth',3,'DisplayName','DPCA5');
plot(1:pms.maxIter, log(mean_err_dpca_one),'LineWidth',3,'DisplayName','DPCA3');
legend('show')