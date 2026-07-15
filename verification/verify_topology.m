% The verification program running in MATLAB to provide structural diagnostics and print high-fidelity geometry matrices.


function verify_topology()
    % VERIFY_TOPOLOGY Post-processing check for the Python generated data.
    fprintf('=== Initializing Automated Structural Verification Pipeline ===\n');
    
    target_json = '../topology_export.json';
    if ~exist(target_json, 'file')
        error('Execution Error: The required data export file does not exist.');
    end
    
    % Read and parse telemetry metrics
    payload = jsondecode(fileread(target_json));
    rho = payload.density_matrix;
    u_x = payload.displacement_x;
    
    % Evaluate geometry configurations
    actual_volume = sum(rho(:) >= 0.5) / numel(rho);
    fprintf('Evaluated Volume Fraction Allocation: %.2f%%\n', actual_volume * 100);
    
    % Calculate spatial variations (Checkerboard noise filter check)
    [d_rho_dx, d_rho_dy] = gradient(rho);
    total_variation = sum(abs(d_rho_dx(:)) + abs(d_rho_dy(:)));
    fprintf('Calculated Geometry Total Variation Vector: %.4f\n', total_variation);
    
    % Render plots
    figure('Name', 'PITO Validation Dashboard', 'Color', [1 1 1]);
    subplot(1,2,1); imagesc(rho); colormap(gca, gray); colorbar;
    title('Optimized AI Density Profile (\rho)'); axis equal tight;
    
    subplot(1,2,2); imagesc(u_x); colormap(gca, jet); colorbar;
    title('Displacement Map Field (U_x)'); axis equal tight;
end
