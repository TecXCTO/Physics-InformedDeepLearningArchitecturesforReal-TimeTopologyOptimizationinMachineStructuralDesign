"""
Implements the multi-physics autograd core. This ensures that the generated engineering designs honor Navier-Cauchy static equilibrium laws.
"""
import torch

def compute_2d_plane_stress_loss(rho, u_x, u_y, boundary_tensor, config):
    """
    Computes custom loss residuals derived from 2D continuum elasticity equations.
    """
    batch, _, H, W = rho.shape
    device = rho.device
    
    # Extract structural configs
    E0 = config["material_properties"]["youngs_modulus_e0"]
    nu = config["material_properties"]["poissons_ratio_nu"]
    p = config["material_properties"]["simp_penalty_p"]
    vol_target = config["optimization_constraints"]["target_volume_fraction"]
    lambda_vol = config["optimization_constraints"]["volume_loss_weight_lambda"]

    # Generate analytical differentiation grids across pixels
    y_grid, x_grid = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
    x_coords = x_grid.to(device).requires_grad_(True).expand(batch, 1, H, W)
    y_coords = y_grid.to(device).requires_grad_(True).expand(batch, 1, H, W)

    # Material properties penalization channel (SIMP engine mapping)
    E_field = 1e-9 + (rho ** p) * E0
    constitutive_factor = E_field / (1 - nu**2)
    
    # 1st Order Gradients tracking: Displacements to Strains
    dux_dx = torch.autograd.grad(u_x, x_coords, torch.ones_like(u_x), create_graph=True)[0]
    dux_dy = torch.autograd.grad(u_x, y_coords, torch.ones_like(u_x), create_graph=True)[0]
    duy_dx = torch.autograd.grad(u_y, x_coords, torch.ones_like(u_y), create_graph=True)[0]
    duy_dy = torch.autograd.grad(u_y, y_coords, torch.ones_like(u_y), create_graph=True)[0]
    
    # Constitutive transformations: Strains to Stress fields
    sigma_xx = constitutive_factor * (dux_dx + nu * duy_dy)
    sigma_yy = constitutive_factor * (duy_dy + nu * dux_dx)
    sigma_xy = (E_field / (2 * (1 + nu))) * (dux_dy + duy_dx)
    
    # 2nd Order Gradients tracking: Stress Divergence
    ds_xx_dx = torch.autograd.grad(sigma_xx, x_coords, torch.ones_like(sigma_xx), create_graph=True)[0]
    ds_xy_dy = torch.autograd.grad(sigma_xy, y_coords, torch.ones_like(sigma_xy), create_graph=True)[0]
    ds_yy_dy = torch.autograd.grad(sigma_yy, y_coords, torch.ones_like(sigma_yy), create_graph=True)[0]
    ds_xy_dx = torch.autograd.grad(sigma_xy, x_coords, torch.ones_like(sigma_xy), create_graph=True)[0]
    
    # Isolate applied external loading parameters from incoming tensors
    body_force_x = boundary_tensor[:, 0:1, :, :]
    body_force_y = boundary_tensor[:, 1:2, :, :]
    
    # Navier-Cauchy Static Conservation Field Residual Evaluation
    residual_x = ds_xx_dx + ds_xy_dy + body_force_x
    residual_y = ds_yy_dy + ds_xy_dx + body_force_y
    
    loss_pde = torch.mean(residual_x**2) + torch.mean(residual_y**2)
    loss_vol = torch.mean((torch.mean(rho) - vol_target)**2)
    
    return loss_pde + (lambda_vol * loss_vol)
