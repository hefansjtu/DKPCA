function [kernel_result] = centralize_test(kernel, test)
[l,n] = size(test);
mean1 = ones(n,n)*1/n;
mean2 = ones(n,l)*1/n;
kernel_result = test - test*mean1 - mean2'*kernel +mean2'*kernel*mean1;
end