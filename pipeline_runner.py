"""
The core executable module orchestrating configurations loading, training steps execution, and validation artifacts compilation.
"""
import os
import json
import torch
import torch.optim as optim
from core.model import PhysicsInformedCNN
from core.physics_engine import compute_2d_plane_stress_loss

def run_pito_pipeline():
    print("Execution Engine Initialized. Parsing Project Configurations...")
    
    # Load configuration schema
    config_path = os.path.join("configs", "default_hyperparameters.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
        
    res = config["training_parameters"]["spatial_resolution"]
    epochs = config["training_parameters"]["epochs"]
    lr = config["training_parameters"]["learning_rate"]
    
    # Initialize infrastructure models
    model = PhysicsInformedCNN(channels_in=3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create simulated engineering boundaries: Shape (1, 3, H, W)
    # Channels: 0 = Load X force, 1 = Load Y force, 2 = Fixed Boundary anchor index
    simulated_boundary_input = torch.randn(1, 3, res[0], res[1], requires_grad=True)
    
    print(f"Beginning Optimization Routine across {epochs} physical evaluation cycles...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Execute forward pass through structural evaluation dimensions
        rho, ux, uy = model(simulated_boundary_input)
        
        # Quantify mechanical residuals mathematically
        loss = compute_2d_plane_stress_loss(rho, ux, uy, simulated_boundary_input, config)
        
        # Backpropagation optimization path
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Cycle {epoch+1:02d}/{epochs} | Equilibrium Loss Matrix Gradient: {loss.item():.4e}")
            
    # Serialize outputs to coordinate verification boundaries with MATLAB layer
    export_payload = {
        "density_matrix": rho.detach().squeeze().cpu().numpy().tolist(),
        "displacement_x": ux.detach().squeeze().cpu().numpy().tolist()
    }
    
    with open("topology_export.json", "w") as export_file:
        json.dump(export_payload, export_file)
        
    print("Pipeline sequence executed successfully. Artifacts logged to topology_export.json.")

if __name__ == "__main__":
    run_pito_pipeline()
  
