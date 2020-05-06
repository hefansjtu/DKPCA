function [kernel] = cal_RBF(X, sigma)
kernel = zeros(size(X,2),size(X,2));
for iter_i = 1: size(X,2)
    for iter_j = iter_i: size(X,2)
        kernel(iter_i, iter_j) = cal_RBF_fun(X(:,iter_i), X(:,iter_j), sigma);
        kernel(iter_j, iter_i) = kernel(iter_i, iter_j);
    end
end
% kernel=kernel./length(kernel);
end

function [val] = cal_RBF_fun(x,y, sigma)
val = exp(-norm(x-y,2)^2/sigma^2);
end