# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 09:41:35 2025

@author: franc
"""

from TorchPropagator import WFS
from PhaseDataset import Zernike

import torch.nn as nn
import torch
from mmengine import Config
import os
import torch.nn.functional as F


# class ResidualBlock(nn.Module):
#     """Residual block with two convolutions and skip connection"""
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                               stride=stride, padding=1, bias=False)

#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
#                               stride=1, padding=1, bias=False)


#         # Skip connection
#         if stride != 1 or in_channels != out_channels:
#             self.skip = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1,
#                          stride=stride, bias=False),
#             )
#         else:
#             self.skip = nn.Identity()

#     def forward(self, x):
#         identity = self.skip(x)

#         out = F.gelu(self.conv1(x))
#         out = self.conv2(out)

#         out += identity
#         out = F.gelu(out)

#         return out


class UNetWithMLP(nn.Module):
    pass
    # """U-Net with residual blocks and MLP output for 80x80 single-channel input"""
    # def __init__(self, input_channels=1, mlp_output_size=200):
    #     super(UNetWithMLP, self).__init__()

    #     # Encoder (downsampling path)
    #     self.enc1 = ResidualBlock(input_channels, 32)
    #     self.enc2 = ResidualBlock(32, 64, stride=2)  # 40x40
    #     self.enc3 = ResidualBlock(64, 128, stride=2)  # 20x20
    #     self.enc4 = ResidualBlock(128, 256, stride=2)  # 10x10

    #     # Bottleneck
    #     self.bottleneck = ResidualBlock(256, 512, stride=2)  # 5x5

    #     # Decoder (upsampling path)
    #     self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    #     self.dec4 = ResidualBlock(512, 256)  # 256 + 256 from skip connection

    #     self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    #     self.dec3 = ResidualBlock(256, 128)  # 128 + 128 from skip connection

    #     self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    #     self.dec2 = ResidualBlock(128, 64)  # 64 + 64 from skip connection

    #     self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
    #     self.dec1 = ResidualBlock(64, 32)  # 32 + 32 from skip connection

    #     # Final convolution to get feature maps
    #     self.final_conv = nn.Conv2d(32, 16, kernel_size=1)

    #     # Global Average Pooling to reduce spatial dimensions
    #     self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    #     # MLP head
    #     self.mlp = nn.Sequential(
    #         nn.Flatten(),
    #         nn.Linear(16, 128),
    #         nn.GELU(),
    #         nn.Dropout(0.5),
    #         nn.Linear(128, mlp_output_size)
    #     )

    # def forward(self, x):

    #     x = x.unsqueeze(1)
    #     # Encoder
    #     e1 = self.enc1(x)      # 80x80x32
    #     e2 = self.enc2(e1)     # 40x40x64
    #     e3 = self.enc3(e2)     # 20x20x128
    #     e4 = self.enc4(e3)     # 10x10x256

    #     # Bottleneck
    #     b = self.bottleneck(e4)  # 5x5x512

    #     # Decoder with skip connections
    #     d4 = self.upconv4(b)                    # 10x10x256
    #     d4 = torch.cat([d4, e4], dim=1)        # 10x10x512
    #     d4 = self.dec4(d4)                     # 10x10x256

    #     d3 = self.upconv3(d4)                  # 20x20x128
    #     d3 = torch.cat([d3, e3], dim=1)        # 20x20x256
    #     d3 = self.dec3(d3)                     # 20x20x128

    #     d2 = self.upconv2(d3)                  # 40x40x64
    #     d2 = torch.cat([d2, e2], dim=1)        # 40x40x128
    #     d2 = self.dec2(d2)                     # 40x40x64

    #     d1 = self.upconv1(d2)                  # 80x80x32
    #     d1 = torch.cat([d1, e1], dim=1)        # 80x80x64
    #     d1 = self.dec1(d1)                     # 80x80x32

    #     # Final processing
    #     features = self.final_conv(d1)         # 80x80x16
    #     pooled = self.global_avg_pool(features)  # 1x1x16

    #     # MLP output
    #     output = self.mlp(pooled)              # 200

    #     return output


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layers):
        super().__init__()
        layers = []
        for _ in range(conv_layers):
            layers.append(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, padding=1, bias=True
                )
            )
            layers.append(nn.GELU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGGNet(nn.Module):
    def __init__(self, wfsParams, input_channels=4):
        super().__init__()

        Nmodes = wfsParams["Nmodes"]

        self.features = nn.Sequential(
            VGGBlock(input_channels, 64, conv_layers=2),
            VGGBlock(64, 128, conv_layers=2),
            VGGBlock(128, 256, conv_layers=3),
            VGGBlock(256, 512, conv_layers=3),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.01),
            nn.Linear(512, Nmodes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Applies custom initialization to the network weights.
        Weights are drawn from a normal distribution and biases are set to a constant.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(
                module.weight, nonlinearity="relu"
            )  # Normal distribution
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


class PapyrusBlock(nn.Module):
    def __init__(self, Nmodes, initial_kernel_size=3):
        super().__init__()

        kernel_size = 3
        padding = (kernel_size - 1) // 2
        initial_padding = (initial_kernel_size - 1) // 2

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=initial_kernel_size, padding=initial_padding),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=initial_kernel_size, padding=initial_padding),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Dropout(0.2),
        )

        # self.outputlayer = nn.Linear(256, Nmodes)
        self.outputlayer = nn.Sequential(nn.Linear(1024, Nmodes))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.outputlayer(x)
        return x


class PapyrusBlock2(nn.Module):
    def __init__(self, channels_in, Nmodes, initial_kernel_size=3):
        super().__init__()

        self.enc1 = ResidualBlock(channels_in, 32, initial_kernel_size)
        self.enc3 = ResidualBlock(32, 64)
        self.enc4 = ResidualBlock(64, 128)
        self.enc5 = ResidualBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        # self.outputlayer = nn.Linear(256, Nmodes)
        self.outputlayer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, Nmodes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool(x)
        x = self.enc3(x)
        x = self.pool(x)
        x = self.enc4(x)
        x = self.pool(x)
        x = self.enc5(x)
        x = self.pool(x)
        x = self.outputlayer(x)
        return x


class Papyrus(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()

        Nmodes = wfsParams["Nmodes"]
        channels = 1

        self.papy1 = PapyrusBlock2(channels, Nmodes // 3, 11)
        self.papy2 = PapyrusBlock2(channels, Nmodes // 3, 7)
        self.papy3 = PapyrusBlock2(channels, Nmodes - 2 * (Nmodes // 3))

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.papy1(x)
        x2 = self.papy2(x)
        x3 = self.papy3(x)

        out = torch.cat([x1, x2, x3], dim=1)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.act = nn.LeakyReLU()
        # projection for channel match
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        out = self.conv1(x)
        out = self.act(out)
        return out + residual


# class Papyrus(nn.Module):
#     def __init__(self, wfsParams, atmosParams, device):
#         super().__init__()

#         self.N = 17
#         x = torch.linspace(-self.N/2, self.N/2, self.N)
#         x,y = torch.meshgrid(x,x)
#         self.pupil = ((x**2 + y**2) < ((17/2)**2 + 300/self.N)).to(device = device)

#         Nmodes = wfsParams["Nmodes"]

#         # encoder stage blocks (no stride inside blocks)

#         self.enc1 = ResidualBlock(1, 64)
#         self.enc2 = ResidualBlock(64, 128)
#         self.enc3 = ResidualBlock(128, 256)
#         self.enc4 = ResidualBlock(256, 128)
#         self.enc5 = ResidualBlock(128, 64)
#         self.enc6 = ResidualBlock(64, 1)

#         # compute concat_channels:


#         self.pool = nn.MaxPool2d(2)


#         # initialize weights (optional but helpful)
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x):
#         """
#         x: [B, 80, 80]  (grayscale)
#         returns: [B, Nmodes]
#         """
#         x = x.unsqueeze(1)  # -> [B,1,80,80]

#         f1 = self.enc1(x)   # [B,64,80,80]
#         p1 = self.pool(f1)

#         f2 = self.enc2(p1)  # [B,128,40,40]
#         p2 = self.pool(f2)

#         f3 = self.enc3(p2)  # [B,256,20,20]
#         p3 = f3

#         f4 = self.enc4(p3)  # [B,256,20,20]
#         p4 = f4

#         f5 = self.enc5(p4)  # [B,256,20,20]
#         p5 = f5

#         f6 = self.enc6(p5)  # [B,256,20,20]
#         p6 = f6

#         # pooled skip features (small spatial footprints)

#         # concatenate flattened pooled features

#         out = F.interpolate(p6, size=(self.N, self.N), mode='bilinear').squeeze()
#         out = out[:, self.pupil == 1]
#         out = out - out.mean(dim = -1, keepdim = True)
#         return out

# class Papyrus(nn.Module):
#     def __init__(self, wfsParams, atmosParams, device, dropout=0.01, start_filters = 16):
#         super().__init__()

#         Nmodes = wfsParams["Nmodes"]


#         # Encoder
#         self.enc1 = self.conv_block(1, start_filters)
#         self.enc2 = self.conv_block(start_filters, start_filters*2)
#         self.enc3 = self.conv_block(start_filters*2, start_filters*4)
#         self.enc4 = self.conv_block(start_filters*4, start_filters*8)

#         # Bottleneck
#         self.bottleneck = self.conv_block(start_filters*8, start_filters*16)

#         # Decoder
#         self.up3 = nn.ConvTranspose2d(start_filters*16, start_filters*8, kernel_size=2, stride=2)
#         self.dec3 = self.conv_block(start_filters*16, start_filters*8)

#         self.up2 = nn.ConvTranspose2d(start_filters*8, start_filters*4, kernel_size=2, stride=2)
#         self.dec2 = self.conv_block(start_filters*8, start_filters*4)

#         self.up1 = nn.ConvTranspose2d(start_filters*4, start_filters*2, kernel_size=2, stride=2)
#         self.dec1 = self.conv_block(start_filters*4, start_filters*2)

#         self.up0 = nn.ConvTranspose2d(start_filters*2, start_filters, kernel_size=2, stride=2)
#         self.dec0 = self.conv_block(start_filters*2, start_filters)

#         self.final_conv = nn.Conv2d(start_filters, 1, kernel_size=3, padding=1)

#         # Output head (convert multi-scale features to Zernike)
#         self.global_pool = nn.AdaptiveAvgPool2d((10, 10))
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(dropout),
#             nn.Linear(start_filters*10*10, 512),
#             nn.GELU(),
#             nn.Linear(512, Nmodes)
#         )

#     self._init_weights()

# def _init_weights(self):
#     for m in self.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#         elif isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.ones_(m.weight)
#             nn.init.zeros_(m.bias)

#     def conv_block(self, in_ch, out_ch):
#         return nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.GELU()
#         )

#     def forward(self, x):
#         # Encoder
#         x = x.unsqueeze(1)  # [B, 4, 80, 80]
#         e1 = self.enc1(x)            # [B, 64, 80, 80]
#         e2 = self.enc2(F.max_pool2d(e1, 2))  # [B, 128, 40, 40]
#         e3 = self.enc3(F.max_pool2d(e2, 2))  # [B, 256, 20, 20]
#         e4 = self.enc4(F.max_pool2d(e3, 2))  # [B, 512, 10, 10]

#         # Bottleneck
#         b = self.bottleneck(F.max_pool2d(e4, 2))  # [B, 1024, 5, 5]

#         # Decoder with skip connections
#         d3 = self.up3(b)               # [B, 512, 10, 10]
#         d3 = self.dec3(torch.cat([d3, e4], dim=1))

#         d2 = self.up2(d3)              # [B, 256, 20, 20]
#         d2 = self.dec2(torch.cat([d2, e3], dim=1))

#         d1 = self.up1(d2)              # [B, 128, 40, 40]
#         d1 = self.dec1(torch.cat([d1, e2], dim=1))

#         d0 = self.up0(d1)              # [B, 64, 40, 40]
#         d0 = self.dec0(torch.cat([d0, e1], dim=1))

#         # Global pooling from the last decoder feature
#         pooled = self.global_pool(d0)  # [B, 64, 1, 1]
#         out = self.fc(pooled)          # [B, Nmodes]

#         return out


class PapyrusPhase(nn.Module):
    def __init__(self, pupil, device):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
        )

        # self.outputlayer = nn.Linear(256, Nmodes)
        self.outputlayer = nn.Sequential(
            nn.Linear(512, pupil.sum().to(dtype=torch.int32))
        )

    def forward(self, x):
        x = self.encoder(x.unsqueeze(1))
        # x = self.encoder(x)
        x = self.outputlayer(x)
        return x


class DataFusion(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()

        Nmodes = wfsParams["Nmodes"]

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.01),
        )

        # self.outputlayer = nn.Linear(256, Nmodes)
        self.outputlayer = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, Nmodes)
        )

    def forward(self, x):
        x = x.type(torch.float32)
        x = self.encoder(x)
        x = self.outputlayer(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()

        Nmodes = wfsParams["Nmodes"]

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11, padding=7),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=7, padding=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=2, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.1),
        )

        self.outputlayer = nn.Sequential(
            nn.Linear(512, Nmodes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder(x[:, 0].unsqueeze(1))
        x = self.outputlayer(x)
        return x


class Papyrus1stStage(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()

        Nmodes = wfsParams["Nmodes"]
        self.binning = wfsParams["Bin_factor"]
        self.bin_step = nn.AvgPool2d(self.binning)

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=11, padding=7),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=7, padding=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=2, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.1),
        )

        self.outputlayer = nn.Sequential(
            nn.Linear(512, Nmodes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.binning > 1:
            x = self.bin_step(x)
        x = self.encoder(x)
        x = self.outputlayer(x)
        return x


class Rama(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()

        Nmodes = wfsParams["Nmodes"]

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=7, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=7, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(0.1),
        )

        self.outputlayer = nn.Sequential(
            nn.Linear(128*4*4, Nmodes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.outputlayer(x)
        return x

# class Papyrus2ndStage(nn.Module):
#     def __init__(self, wfsParams, atmosParams, device):
#         super().__init__()

#         Nmodes = wfsParams["Nmodes"]
#         channels = 2

#         self.papy1 = PapyrusBlock2(channels, Nmodes // 2, 11)
#         self.papy2 = PapyrusBlock2(channels, Nmodes - (Nmodes // 2))

#     def forward(self, x):
#         x1 = self.papy1(x)
#         x2 = self.papy2(x)

#         out = torch.cat([x1, x2], dim=1)
#         return out


class Papyrus2ndStage(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()

        Nmodes = wfsParams["Nmodes"]
        self.binning = wfsParams["Bin_factor"]
        self.bin_step = nn.AvgPool2d(self.binning)

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=11, padding=7),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=7, padding=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=2, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.1),
        )

        self.outputlayer = nn.Sequential(
            nn.Linear(512, Nmodes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.binning > 1:
            x = self.bin_step(x)
        x = self.encoder(x)
        x = self.outputlayer(x)
        return x


class ViT_PyTorch(nn.Module):
    def __init__(
        self,
        img_size=90,
        patch_size=16,
        in_channels=2,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        out_dim=50,
        dropout=0.1,
    ):
        super().__init__()

        # Patch embedding (Conv does everything)
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        num_patches = (img_size // patch_size) ** 2

        # Positional embeddings only (no CLS)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, x):
        # x: [B, C, H, W]

        x = self.patch_embed(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, D]

        x = x + self.pos_embed

        x = self.encoder(x)

        x = x.mean(dim=1)

        x = self.head(x)

        return x


class OptimizedLinearEstimator(nn.Module):
    "Learned Linear Estimator with a learned reconstruction matrix and ref intensity"

    "They are initalized using the propagator code from the starting point"

    def __init__(self, init=0, WFS=None, Nmodes=0):

        super().__init__()

        # Initialization with the  reconstruction matrix at starting point
        if init == 1:

            print("Initalization of the reconstruction matrix")
            [z, z_FullRes] = Zernike(
                WFS.pupil, Nmodes
            )
            z_FullRes = z_FullRes
            WFS.BuildReconstructionMatrix(z_FullRes, WFS.mask)
            self.WFS = WFS
            self.param = nn.Parameter(WFS.reconstructionMatrix)
            self.param_name = "Reconstruction matrix as a parameter"
        # Reconstruction matrix initalized at 0
        else:
            number_of_pixels = WFS.Npix**2
            self.param = nn.Parameter(
                torch.zeros((Nmodes, number_of_pixels), dtype=torch.float64)
            )

    def forward(self, image):

        ## (Learned) Matrix multiplication

        reduced_intensity = image

        EstimatedZernike = torch.matmul(
            reduced_intensity.flatten(start_dim=-2), self.param.T
        )

        return EstimatedZernike


class LinearEstimator(nn.Module):

    def __init__(self, WFS: WFS):

        super().__init__()
        self.WFS = WFS

    def forward(self, image):

        ## Build the reconstruction matrix for each forward (because it depends on the optimized parameters)
        # self.WFS.BuildReconstructionMatrix(self.z_FullRes, self.WFS.mask)
        if self.training:
            self.WFS.BuildReferenceIntensity()
            self.WFS.BuildReconstructionMatrix(self.z_FullRes)

        return self.WFS.GetReconstructedPhase(image)


if __name__ == "__main__":
    print("Nothing to do here")
