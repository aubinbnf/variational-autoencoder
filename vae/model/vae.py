import torch
import torch.nn as nn
import torch.nn.functional as F
from vae.model import Encoder, Decoder

class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, log_var):
        epsilon = torch.randn_like(log_var)
        
        sigma = torch.exp(0.5 * log_var)

        z = mu + sigma * epsilon
        
        return z
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        
        z = self.reparameterize(mu, log_var)
        
        recon_x = self.decoder(z)
        
        return recon_x, mu, log_var