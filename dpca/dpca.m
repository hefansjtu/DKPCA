function [error_dpca]   = dpca(data_total, X_total, N_local, uu_gt, pms)

error_dpca = zeros(pms.maxIter, 1);

C = randn(pms.target_k, pms.m);
    C = C/norm(C);
    Y = C*X_total;
    C_local = cell(pms.worker_num, 1);
    Y_local = cell(pms.worker_num, 1);
    V_local = cell(pms.worker_num, 1);
% 
    tmp_y = cell(pms.worker_num, 1);
    
    for worker_iter = 1: pms.worker_num
        C_local{worker_iter} = zeros(pms.target_k, size(data_total{worker_iter},1));
        Y_local{worker_iter} = Y;
        V_local{worker_iter} = zeros(size(Y_local{worker_iter},1), size(Y_local{worker_iter},2)*length(N_local{worker_iter}));
    end
    
    for iter = 1: pms.maxIter
        C_old = C_local;
        error_tmp = 0;
        for worker_iter = 1: pms.worker_num
            C_local{worker_iter} = inv(Y_local{worker_iter}*Y_local{worker_iter}')*Y_local{worker_iter}*(data_total{worker_iter})';
            error_tmp = error_tmp + norm(C_local{worker_iter} - C_old{worker_iter});
        end
        
        if error_tmp < 1e-6
            break;
        end
        
        for inIter = 1: pms.maxInnerIter
            for worker_iter = 1: pms.worker_num
                tmp_y{worker_iter} = zeros(size(Y_local{worker_iter},1), size(Y_local{worker_iter},2)*length(N_local{worker_iter}));
                tmp_zeros = zeros(1, length(N_local{worker_iter}));
                for nei_iter = 1: length(N_local{worker_iter})
                    nei_idx = N_local{worker_iter}(nei_iter);
                    tmp_e = tmp_zeros;
                    tmp_e(nei_iter) = 1;
                    tmp_y{worker_iter} = tmp_y{worker_iter} + kron(Y_local{nei_idx}, tmp_e);
                end
                V_local{worker_iter} = V_local{worker_iter} + 0.5*pms.c*(kron(Y_local{worker_iter}, ones(1, length(N_local{worker_iter}))) - tmp_y{worker_iter});
            end
            
            Y_old = Y_local;
            error_tmp = 0;
            for worker_iter = 1: pms.worker_num
                tmp_v = zeros(size(V_local{worker_iter}));
                tmp_zeros = zeros(length(N_local{worker_iter}));
                for nei_iter = 1: length(N_local{worker_iter})
                    nei_idx = N_local{worker_iter}(nei_iter);
                    tmp_e = zeros(length(N_local{nei_idx}), length(N_local{worker_iter}));
                    idx_i = find(N_local{nei_idx} == worker_iter);
                    idx_j = find(N_local{worker_iter} == nei_idx);
                    tmp_e(idx_i, idx_j) = 1;
                    tmp_v = tmp_v + V_local{nei_idx}*kron(eye(pms.n), tmp_e);
                end
                tmp = 2*C_local{worker_iter}*data_total{worker_iter} - (V_local{worker_iter} - tmp_v)*kron(eye(pms.n), ones(length(N_local{worker_iter}), 1))...
                    + pms.c*((kron(Y_local{worker_iter}, ones(1, length(N_local{worker_iter}))) + tmp_y{worker_iter})*kron(eye(pms.n), ones(length(N_local{worker_iter}), 1)));
                Y_local{worker_iter} = inv(2*C_local{worker_iter}*C_local{worker_iter}' + 2*pms.c*length(N_local{worker_iter}).*eye(pms.target_k))*tmp;
                error_tmp = error_tmp + norm(Y_local{worker_iter} - Y_old{worker_iter} );
            end
            if error_tmp < 1e-5
                break;
            end
        end
        C=[];
        for worker_iter = 1: pms.worker_num
            C = [C  C_local{worker_iter}];
        end
        UU = C'*inv(C*C')*C;
        error_dpca(iter) = norm(UU-uu_gt,'fro')^2;
        
        if iter>1 && abs(error_dpca(iter) - error_dpca(iter-1)) < 1e-6
            break;
        end
       
    end
    error_dpca(iter+1:end) = error_dpca(iter)*ones(pms.maxIter - iter,1);
end