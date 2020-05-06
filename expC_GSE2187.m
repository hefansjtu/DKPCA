clc; clear; close all;

alldata = load('total.mat'); %  tox+fib.mat   % tox+azo.mat
X_total = table2array(alldata.table_total(:,2:end));
type1_size = 181; %the first 181 samples are toxicants.
type2_size = size(X_total,2) - type1_size;
num_train = 200; 
idx1_size = ceil(num_train*type1_size/(type1_size+type2_size));
idx2_size = num_train - idx1_size;
Y_total = [ones(type1_size,1); -1*ones(type2_size,1)];
[pms.m, n_total] = size(X_total);
variable_sig = [50*ones(50,1)];
variable_k = [1 5 10 20 50:50:200];
error = zeros(length(variable_k),length(variable_sig));
error_ours = zeros(length(variable_k),length(variable_sig));
error_linear = zeros(length(variable_k),length(variable_sig));
pms.centralize = true;

for ii_sig = 1: length(variable_sig)

    idx_train = randperm(n_total, num_train);
    idx_test = setdiff([1:n_total], idx_train);
    test_X = X_total(:,idx_test,:);
    test_Y = Y_total(idx_test);
    X =X_total(:,idx_train);
    train_Y = Y_total(idx_train);
    
    [pms.m, pms.n] = size(X);
    
    pms.worker_num = 40;
    
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
        data{iter} = X(anchor{iter}: anchor{iter+1} - 1, :);
    end
    
    
    pms.sigma = variable_sig(ii_sig);
    kernel_ori = cal_RBF(X, pms.sigma);
    [kernel_gt] = centralize_kernel(kernel_ori);
    [U, ~,~]  = svd(kernel_gt);
    for ii_k = 1: length(variable_k)
        
        pms.target_k = variable_k(ii_k);
        pms.local_target_k = min(variable_k(ii_k)+10, num_train);
        % tic
        %% svd
        W_svd = U(:,1:pms.target_k);
        % fprintf('common time: %f\n',toc);
        [projed_train_data] = proj_kpca(W_svd, data, X, pms, kernel_ori);
        [projed_test_data] = proj_kpca(W_svd, data, test_X, pms, kernel_ori);
        
        svmModel = fitcsvm(projed_train_data', train_Y, 'BoxConstraint', 5, 'KernelFunction', 'linear');
        [test_pre,~] = predict(svmModel, projed_test_data');
        error(ii_k,ii_sig) = (length(test_Y)-sum(test_Y==test_pre))/length(test_Y);
        
        %% ours
        [W_dkpca, kernel, lam_hat, lam_max] = solve_dkpca(data, pms, kernel_gt, 'RBF');
        % fprintf('ours time: %f\n',toc);
        
        %         appro_err = pms.target_k - trace((W_dkpca'*W_svd).^2);
        [projed_train_data] = proj_kpca(W_dkpca, data, X, pms, kernel_ori);
        [projed_test_data] = proj_kpca(W_dkpca, data, test_X, pms, kernel_ori);
        
        svmModel = fitcsvm(projed_train_data', train_Y, 'BoxConstraint', 5, 'KernelFunction', 'linear');
        [test_pre,~] = predict(svmModel, projed_test_data');
        error_ours(ii_k,ii_sig) = (length(test_Y)-sum(test_Y==test_pre))/length(test_Y);
        
        %% linear
        pca_linear = (X-mean(X))'*(X-mean(X));
        [W_linear,~,~] = svd(pca_linear);
        W_linear = W_linear(:,1:pms.target_k);
        proj_vec = (X-mean(X))*W_linear;
%        proj_vec = proj_vec;
        projed_train_data = proj_vec'*(X-mean(X));
        projed_test_data = proj_vec'*(test_X-mean(test_X));
        
        svmModel = fitcsvm(projed_train_data', train_Y, 'BoxConstraint', 5, 'KernelFunction', 'linear');
        [test_pre,~] = predict(svmModel, projed_test_data');
        error_linear(ii_k,ii_sig) = (length(test_Y)-sum(test_Y==test_pre))/length(test_Y);
        
    end
end
mean_k=mean(error,2);
mean_l=mean(error_linear,2);
mean_ours=mean(error_ours,2);
std_k = std(error,0,2);
std_l = std(error_linear,0,2);
std_ours = std(error_ours,0,2);
% svmModel = fitcsvm(X', train_Y, 'BoxConstraint', 5, 'KernelFunction', 'linear');
% [test_pre,~] = predict(svmModel, test_X');
% error_best = (length(test_Y)-sum(test_Y==test_pre))/length(test_Y);