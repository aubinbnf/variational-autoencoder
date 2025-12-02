import torch
import torch.nn as nn
import torch.nn.functional as F
from vae.model import ResidualBlock

class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, 16384)
        
        self.res1 = ResidualBlock(256)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', 
                                     align_corners=False)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        
        self.res2 = ResidualBlock(128)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=False)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        self.res3 = ResidualBlock(64)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=False)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 8, 8)
        
        x = self.res1(x)
        x = self.upsample1(x)
        x = self.conv1(x)
        
        x = self.res2(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        
        x = self.res3(x)
        x = self.upsample3(x)
        x = self.conv3(x)
        
        return x