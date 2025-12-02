import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels)
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
    
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)
    
        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)

        out = out + residual
    
        return out