function [y] = proj_kpca(alpha, data, proj_data, pms, kernel_gt)
[~, proj_n] = size(proj_data);

% calculate \|x-x_i\| distributedly
norm_x = zeros(pms.n, proj_n);
anchor = 1;
for worker_iter = 1: pms.worker_num
    tmp_proj = proj_data(anchor:anchor+size(data{worker_iter},1)-1,:);
    anchor = anchor+size(data{worker_iter},1);
    for ii = 1: proj_n
        tmp = data{worker_iter} - tmp_proj(:,ii)*ones(1,pms.n);
        norm_x(:, ii) = norm_x(:, ii) + (sum(abs(tmp).^2))';
    end
end
k = exp(-norm_x/pms.sigma^2);
y = zeros(size(alpha,2), proj_n);
[kernel_proj] = centralize_test(kernel_gt, k');
for ii = 1: size(alpha,2)
    y(ii,:) = alpha(:,ii)'*kernel_proj';
end
% figure; hold on;
% plot3(y(1,1:201),y(2,1:201), y(3,1:201),'ro')
% plot3(y(1,202:402),y(2,202:402),y(3,202:402), 'g*')
% plot3(y(1,403:end),y(2,403:end),y(3,403:end), 'k^')
end