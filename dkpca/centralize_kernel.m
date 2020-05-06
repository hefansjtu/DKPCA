function [kernel_result] = centralize_kernel(kernel)
[m,n] = size(kernel);
kernel_result = kernel;
if m~=n
    return;
end
mean_mat = ones(n,n)/n;
kernel_result = kernel - kernel*mean_mat - mean_mat*kernel + mean_mat*kernel*mean_mat;
end
