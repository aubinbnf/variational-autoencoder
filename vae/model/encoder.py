import torch
import torch.nn as nn
import torch.nn.functional as F
from vae.model import ResidualBlock

class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.res2 = ResidualBlock(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.res3 = ResidualBlock(256)
        
        self.fc = nn.Linear(16384, latent_dim * 2)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.res1(x)
        
        x = self.conv2(x)
        x = self.res2(x)
        
        x = self.conv3(x)
        x = self.res3(x)
        
        x = x.flatten(1)
        
        x = self.fc(x)
        
        mu = x[:, :self.latent_dim]
        log_var = x[:, self.latent_dim:]
        
        return mu, log_var