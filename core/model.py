"""
Handles the deep feature-extraction pipeline, translating multi-channel spatial structural load masks into structural material layout parameters
"""
import torch
import torch.nn as nn

class PhysicsInformedCNN(nn.Module):
    """
    Deep Convolutional Encoder-Decoder mapping spatial boundary conditions 
    (Forces, Constraints) directly into field layouts for Density (Rho) 
    and Displacement Vector Components (U_x, U_y).
    """
    def __init__(self, channels_in=3):
        super(PhysicsInformedCNN, self).__init__()
        
        # Downsampling Encoder Stage
        self.encoder = nn.Sequential(
            nn.Conv2d(channels_in, 16, kernel_size=3, padding=1),
            nn.Tanh(), # Tanh ensures smooth C2 continuity for mathematical differentiation
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), 
            nn.Tanh()
        )
        
        # Upsampling Decoder Stage
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), 
            nn.Tanh(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1) 
        )

    def forward(self, boundary_tensor):
        latent = self.encoder(boundary_tensor)
        output = self.decoder(latent)
        
        # Enforce distinct mathematical envelopes for design criteria
        rho = torch.sigmoid(output[:, 0:1, :, :]) # Normalizes local material density to [0.0, 1.0]
        u_x = output[:, 1:2, :, :]
        u_y = output[:, 2:3, :, :]
        
        return rho, u_x, u_y
      
