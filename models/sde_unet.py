import math
import torch
from torch import nn
import torch.nn.functional as F
from models.unet import EncBlock, DecBlock


class Diffusion(nn.Module):
    def __init__(self, channels):
        super(Diffusion, self).__init__()
        self.diffusion_blocks = nn.ModuleList([
            EncBlock(channels[i], channels[i+1]) for i in range(len(channels)-1)
        ])
    def forward(self, p):
        diffusion_terms = []
        for diffusion_block in self.diffusion_blocks:
            x, p = diffusion_block(p)
            diffusion_terms.append(x.sigmoid())
        return diffusion_terms

class DriftBlock(nn.Module):
    def __init__(self, in_c, out_c, deltat=1):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding='same'),
            nn.ReLU()),
            nn.Sequential(nn.Conv2d(out_c, out_c, kernel_size=3, padding='same'),
            nn.ReLU())
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_channels = in_c
        self.out_channels = out_c
        self.deltat = deltat
        
    def forward(self, inputs, diff_term):
        x = self.net[0](inputs)
        x_tmp = x + diff_term*math.sqrt(self.deltat)*torch.randn_like(x).to(x)
        x = self.net[1](x) + x_tmp
        p = self.pool(x)
        return x, p
    
# Drift: the actual SDE-UNet
class SDEUNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels=[64, 128, 256, 512]):
        super(SDEUNet, self).__init__()
        
        channels = [in_channels] + channels
        self.diffusion = Diffusion(channels)
        self.layer_depth = len(channels)
        self.deltat = 4./self.layer_depth

        self.encs = nn.ModuleList([
            DriftBlock(channels[i], channels[i+1], deltat=self.deltat) for i in range(len(channels)-1)
        ])
        self.decs = nn.ModuleList([
            DecBlock(channels[i], channels[i-1]) for i in range(len(channels)-1, 1, -1)
        ])

        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)
        # self.outconv = nn.Conv2d(64, out_channels + 1, kernel_size=1) # output mean and sigma
        self.sigma = 0.5

    def forward(self, p, train_diffusion=False):
        if not train_diffusion:

            diffusion_terms = self.diffusion(p) # 4 --> 64, ...
            skips = []
            for i, (enc_block, diff_term) in enumerate(zip(self.encs, diffusion_terms)):
                x, p = enc_block(p, self.sigma*diff_term)
                skips.append(x)

            for dec_block, skip in zip(self.decs, reversed(skips[:-1])):
                x = dec_block(x, skip)
            out = self.outconv(x)
            return out
        else:
            final_out = self.diffusion(p.detach())
            return final_out
