function H_optimized_reshape = Optimize(H_dynamic)
    
    % Flatten initial guess: 2x 4x4 matrices into a 32x1 vector
    H_realsense_tracker = eye(4);
    H_ground_PCA = eye(4);
    H_static0 = [H_ground_PCA(:); H_realsense_tracker(:)];

    options = optimoptions(@lsqnonlin, 'Algorithm', 'levenberg-marquardt',...
                                       'Display', 'iter', 'StepTolerance', 0.001, 'OptimalityTolerance', 0.001);
    H_optimized = lsqnonlin(@(H_static) CostFunction(H_static, H_dynamic), H_static0, [], [], options);

    % Reshape back to 4x4 matrices
    H_optimized_reshape.H_ground_PCA         = reshape(H_optimized(1:16), 4, 4);
    H_optimized_reshape.H_realsense_tracker = reshape(H_optimized(17:32), 4, 4);

    function cost = CostFunction(H_static, H_dynamic)
        H_ground_PCA        = reshape(H_static(1:16), 4, 4);
        H_realsense_tracker = reshape(H_static(17:32), 4, 4);

        n = length(H_dynamic.ARUCO_ground_realsense);
        cost = zeros(n, 1);
        for ii = 1 : n
            cost(ii) = norm( H_dynamic.HTC_tracker_PCA{ii} * H_realsense_tracker * H_dynamic.ARUCO_ground_realsense{ii} - H_ground_PCA, 'fro');
        end
    end 

end

