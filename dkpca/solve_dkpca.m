function [W, kernel, lam_hat, lam_max, run_time] = solve_dkpca(data, pms, kernel_gt, kernel_type)
start_time_tol = tic;
W=[]; kernel=[];
lam_max = zeros(pms.worker_num,1);%0;
if strcmp(kernel_type, 'RBF')
    kernel = ones(pms.n, pms.n);
elseif strcmp(kernel_type, 'Linear')
    kernel = zeros(pms.n, pms.n);
else
    fprintf('error kernel type!')
    return;
end
v_local = cell(pms.worker_num,1);
lam_local = cell(pms.worker_num,1);
err = cell(pms.worker_num,1);

start_time_loc = tic;
for iter = 1: pms.worker_num
    if strcmp(kernel_type, 'RBF')
        ker_tmp = cal_RBF(data{iter}, pms.sigma);
    elseif strcmp(kernel_type, 'Linear')
        ker_tmp = (data{iter})'*data{iter};
    end
    [v_tmp, lam_tmp, ~] = svd(ker_tmp);
    lam_tmp = diag(lam_tmp);
    v_local{iter} = v_tmp(:,1:pms.local_target_k);
    lam_local{iter} = diag(lam_tmp(1:pms.local_target_k));
    lam_max(iter) = sum(lam_tmp(pms.local_target_k+1:end));
    err{iter} = norm((v_local{iter}*lam_local{iter}*(v_local{iter})') - ker_tmp);
%     if sum(lam_tmp(pms.local_target_k+1:end)) > lam_max
%         lam_max = sum(lam_tmp(pms.local_target_k+1:end)); %lam_tmp(pms.local_target_k+1); %
%     end
end
run_time_loc = toc(start_time_loc);

for iter_worker = 1: pms.worker_num
    if strcmp(kernel_type, 'RBF')
        kernel = kernel .* (v_local{iter_worker}*lam_local{iter_worker}*(v_local{iter_worker})');
    elseif strcmp(kernel_type, 'Linear')
        kernel = kernel + (v_local{iter_worker}*lam_local{iter_worker}*(v_local{iter_worker})');
    end
end
if strcmp(kernel_type, 'Linear')
    kernel = kernel/(pms.m);
end
if pms.centralize == true
    [kernel] = centralize_kernel(kernel);
end
[v_tmp, lam_tmp, ~] = svd(kernel);
W = v_tmp(:,1: pms.target_k);

lam_hat = diag(lam_tmp);

run_time_tol = toc(start_time_tol);
run_time = run_time_tol - run_time_loc + run_time_loc/pms.worker_num;
% fprintf('\t\t center running time: %0.4f s\n.', toc(start_time));

end
