
% The MATLAB script acting as the aerospace regulatory validation tool. It plots the localized Von Mises stress mapping, evaluates the minimum Factor of Safety (FOS) across all elements, and confirms structural airworthiness.

function verify_aerospace_bracket()
    % VERIFY_AEROSPACE_BRACKET Evaluates bracket safety performance.
    fprintf('=== Initiating Aerospace Structural Airworthiness Audit ===\n');
    
    config_file = '../configs/aerospace_bracket_config.json';
    export_file = '../topology_export.json';
    
    if ~exist(config_file, 'file') || ~exist(export_file, 'file')
        error('Audit Terminated: Essential configuration or data payload files missing.');
    end
    
    % Read configurations and operational payloads
    config = jsondecode(fileread(config_file));
    payload = jsondecode(fileread(export_file));
    
    rho = payload.density_matrix;
    von_mises_pa = payload.von_mises_field; % Extracted from updated pipeline exporter
    
    yield_strength_mpa = config.aerospace_specifications.yield_strength_mpa;
    yield_strength_pa = yield_strength_mpa * 1e6;
    
    % Step 1: Max Stress Identification & Factor of Safety (FOS) Verification
    % Filter out background noise in void areas (where material density is zero)
    solid_mask = rho >= 0.5;
    active_stresses = von_mises_pa(solid_mask);
    
    max_stress_pa = max(active_stresses(:));
    max_stress_mpa = max_stress_pa / 1e6;
    min_fos = yield_strength_pa / max_stress_pa;
    
    fprintf('Material Selection Matrix: %s\n', config.aerospace_specifications.material);
    fprintf('Material Tensile Yield Limit: %.1f MPa\n', yield_strength_mpa);
    fprintf('Maximum Predicted Internal Stress: %.1f MPa\n', max_stress_mpa);
    fprintf('Calculated Minimum Structural Factor of Safety (FOS): %.2f\n', min_fos);
    
    %% Step 2: Render Multi-Physics Performance Dashboard
    figure('Name', 'Aerospace Bracket Compliance Dashboard', 'Color', [1 1 1]);
    
    subplot(1, 2, 1);
    imagesc(rho); colormap(gca, gray); colorbar;
    title('AI Optimized Lightweight Bracket Topology');
    xlabel('Width Elements'); ylabel('Height Elements'); axis equal tight;
    
    subplot(1, 2, 2);
    imagesc(von_mises_pa / 1e6); colormap(gca, jet); colorbar;
    title('Von Mises Equivalent Stress Profile (MPa)');
    xlabel('Width Elements'); ylabel('Height Elements'); axis equal tight;
    
    %% Step 3: Regulatory Sign-Off Gate
    if min_fos >= 1.5
        fprintf('AUDIT SIGN-OFF RESULT: APPROVED. Component clears the aerospace FOS safety threshold (>= 1.5).\n');
    else
        warning('AUDIT SIGN-OFF RESULT: REJECTED. Material yielding or structural micro-cracking risk is too high.');
    end
end

