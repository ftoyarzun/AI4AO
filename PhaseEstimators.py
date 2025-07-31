# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 09:41:35 2025

@author: franc
"""

from TorchPropagator import WFS, Zernike

import torch.nn as nn
import torch
from mmengine import Config
import os
import torch.nn.functional as F

from line_profiler import profile

class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)

        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.gelu(self.conv1(x))
        out = self.conv2(out)
        
        out += identity
        out = F.gelu(out)
        
        return out

class UNetWithMLP(nn.Module):
    """U-Net with residual blocks and MLP output for 80x80 single-channel input"""
    def __init__(self, input_channels=1, mlp_output_size=200):
        super(UNetWithMLP, self).__init__()
        
        # Encoder (downsampling path)
        self.enc1 = ResidualBlock(input_channels, 32)
        self.enc2 = ResidualBlock(32, 64, stride=2)  # 40x40
        self.enc3 = ResidualBlock(64, 128, stride=2)  # 20x20
        self.enc4 = ResidualBlock(128, 256, stride=2)  # 10x10
        
        # Bottleneck
        self.bottleneck = ResidualBlock(256, 512, stride=2)  # 5x5
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(512, 256)  # 256 + 256 from skip connection
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(256, 128)  # 128 + 128 from skip connection
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(128, 64)  # 64 + 64 from skip connection
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(64, 32)  # 32 + 32 from skip connection
        
        # Final convolution to get feature maps
        self.final_conv = nn.Conv2d(32, 16, kernel_size=1)
        
        # Global Average Pooling to reduce spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, mlp_output_size)
        )
    
    def forward(self, x):
        
        x = x.unsqueeze(1)
        # Encoder
        e1 = self.enc1(x)      # 80x80x32
        e2 = self.enc2(e1)     # 40x40x64
        e3 = self.enc3(e2)     # 20x20x128
        e4 = self.enc4(e3)     # 10x10x256
        
        # Bottleneck
        b = self.bottleneck(e4)  # 5x5x512
        
        # Decoder with skip connections
        d4 = self.upconv4(b)                    # 10x10x256
        d4 = torch.cat([d4, e4], dim=1)        # 10x10x512
        d4 = self.dec4(d4)                     # 10x10x256
        
        d3 = self.upconv3(d4)                  # 20x20x128
        d3 = torch.cat([d3, e3], dim=1)        # 20x20x256
        d3 = self.dec3(d3)                     # 20x20x128
        
        d2 = self.upconv2(d3)                  # 40x40x64
        d2 = torch.cat([d2, e2], dim=1)        # 40x40x128
        d2 = self.dec2(d2)                     # 40x40x64
        
        d1 = self.upconv1(d2)                  # 80x80x32
        d1 = torch.cat([d1, e1], dim=1)        # 80x80x64
        d1 = self.dec1(d1)                     # 80x80x32
        
        # Final processing
        features = self.final_conv(d1)         # 80x80x16
        pooled = self.global_avg_pool(features)  # 1x1x16
        
        # MLP output
        output = self.mlp(pooled)              # 200
        
        return output


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layers):
        super().__init__()
        layers = []
        for _ in range(conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True))
            layers.append(nn.GELU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGGNet(nn.Module):
    def __init__(self, wfsParams, input_channels=4):
        super().__init__()
        
        Nzernike = wfsParams["Nzernike"]
        
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
            nn.Linear(512, Nzernike)
        )
       
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Applies custom initialization to the network weights.
        Weights are drawn from a normal distribution and biases are set to a constant.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')  # Normal distribution
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

    
class Papyrus(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()
        
        Nzernike = wfsParams["Nzernike"]

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
            nn.Dropout(0.2)  
        )

        # self.outputlayer = nn.Linear(256, Nzernike)
        self.outputlayer = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, Nzernike)
        )

    def forward(self, x):
        x = self.encoder(x.unsqueeze(1))
        # x = self.encoder(x)
        x = self.outputlayer(x)
        return x

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
            nn.Dropout(0.2)  
        )

        # self.outputlayer = nn.Linear(256, Nzernike)
        self.outputlayer = nn.Sequential(
            nn.Linear(512, pupil.sum().to(dtype = torch.int32))
        )

    def forward(self, x):
        x = self.encoder(x.unsqueeze(1))
        # x = self.encoder(x)
        x = self.outputlayer(x)
        return x
    
class DataFusion(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()
        
        Nzernike = wfsParams["Nzernike"]

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
            nn.Dropout(0.01)  
        )

        # self.outputlayer = nn.Linear(256, Nzernike)
        self.outputlayer = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, Nzernike)
        )

    def forward(self, x):
        x = x.type(torch.float32)
        x = self.encoder(x)
        x = self.outputlayer(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()
        
        
        N = wfsParams['Nres'] * 2
        self.numberOfFrames = wfsParams["FrameBufferLength"]
        Nzernike = wfsParams["Nzernike"]
        Nbatch = atmosParams["Nphases"]
        
        self.encoder = nn.Sequential(
            nn.Conv2d(self.numberOfFrames, 8, kernel_size=5, padding=2),
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
            nn.Dropout(0.01)  
        )

        # self.outputlayer = nn.Linear(256, Nzernike)
        self.outputlayer = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, Nzernike)
        )
        
        
        self.frameBuffer = torch.zeros(Nbatch, self.numberOfFrames, N, N, dtype = torch.float32, device = device)

    # def forward(self, x):
    #     if self.numberOfFrames > 1:
    #         self.frameBuffer = self.frameBuffer.detach()
    #         self.frameBuffer[:, 1:, :, :] = self.frameBuffer[:, :-1, :, :]
    #         self.frameBuffer[:, 0, :, :] *= 0
        
    #     # self.frameBuffer = self.frameBuffer.detach() * 0
        
    #     self.frameBuffer[:, 0, :, :] += x
    #     self.frameBuffer[:, 1, :, :] += x.detach()
    #     self.frameBuffer[:, 2, :, :] += x.detach()
    #     self.frameBuffer[:, 3, :, :] += x.detach()
        
        
    #     x = self.encoder(self.frameBuffer)   
    #     x = self.outputlayer(x)
    #     return x
    #     x = self.encoder(x.unsqueeze(1))   
    #     x = self.outputlayer(x)
    #     return x
        
    # def forward(self, x):
    #     x = x.detach()
    #     if self.numberOfFrames > 1:
    #         self.frameBuffer = self.frameBuffer.detach()
    #         new_frame_buffer = torch.cat((x.unsqueeze(1), self.frameBuffer[:, :-1, :, :]), dim=1)
    #         self.frameBuffer = new_frame_buffer
    #     else:
    #         self.frameBuffer = x.unsqueeze(1) # Or handle single frame case appropriately

    #     x = self.encoder(self.frameBuffer)
    #     x = self.outputlayer(x)
    #     return x
    def forward(self, x):
    #     x = x.detach()
    #     if self.numberOfFrames > 1:
    #         self.frameBuffer = self.frameBuffer.detach()
    #         new_frame_buffer = torch.cat((x.unsqueeze(1), self.frameBuffer[:, :-1, :, :]), dim=1)
    #         self.frameBuffer = new_frame_buffer
    #     else:
    #         self.frameBuffer = x.unsqueeze(1) # Or handle single frame case appropriately
    
        x = self.encoder(x.unsqueeze(1))
        x = self.outputlayer(x)
        return x



class PatchEmbedding(nn.Module):
  def __init__(self, embed_dim, img_size, patch_size, dropout, in_channels = 1):
      super().__init__()
      
      num_patches = (img_size // patch_size) ** 2
      
      self.patcher = nn.Sequential(
          # We use conv for doing the patching
          nn.Conv2d(
              in_channels=in_channels,
              out_channels=embed_dim,
              # if kernel_size = stride -> no overlap
              kernel_size=patch_size,
              stride=patch_size
          ),
          # Linear projection of Flattened Patches. We keep the batch and the channels (b,c,h,w)
          nn.Flatten(2))
      self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
      self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
      self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
      # Create a copy of the cls token for each of the elements of the BATCH
      cls_token = self.cls_token.expand(x.shape[0], -1, -1)
      # Create the patches
      x = self.patcher(x).permute(0, 2, 1)
      # Unify the position with the patches
      x = torch.cat([cls_token, x], dim=1)
      # Patch + Position Embedding
      x = self.position_embeddings + x
      x = self.dropout(x)
      return x
  
    
class ViT_PyTorch(nn.Module):
    def __init__(self, embed_dim, img_size, patch_size, dropout, num_heads, num_encoders, expansion, Nzernike):
        super().__init__()
        
        self.inst_norm = nn.InstanceNorm2d(1)
        
        self.embeddings_block = PatchEmbedding(embed_dim, img_size, patch_size, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=int(embed_dim*expansion), activation="gelu", batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=Nzernike)
        )

    def forward(self, x):
        x = x.unsqueeze(1).type(torch.float32)
        x = self.inst_norm(x)
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])  # Apply MLP on the CLS token only
        return x



class OptimizedLinearEstimator (nn.Module) :
    " Learned Linear Estimator with a learned reconstruction matrix and ref intensity"
    "They are initalized using the propagator code from the starting point"
    
    def __init__(self,init=0,WFS=None,Nzernike=0) :
        
        super().__init__()
        
        # Initialization with the  reconstruction matrix at starting point
        if init == 1 :
            
            print("Initalization of the reconstruction matrix")
            [z, z_FullRes] = Zernike(WFS.pupil.cpu(), WFS.pupil_logical, WFS.Nres, Nzernike)     
            z_FullRes = z_FullRes
            WFS.BuildReconstructionMatrix(z_FullRes, WFS.mask)
            self.WFS = WFS
            self.param = nn.Parameter(WFS.reconstructionMatrix)
            self.param_name = "Reconstruction matrix as a parameter"
        # Reconstruction matrix initalized at 0
        else :
            number_of_pixels = WFS.Npix**2
            self.param = nn.Parameter(torch.zeros((Nzernike,number_of_pixels),dtype = torch.float64))
            
            
    def forward(self, image):
        
         ## (Learned) Matrix multiplication
         
        
         reduced_intensity= image
         
         EstimatedZernike = torch.matmul(reduced_intensity.flatten(start_dim = -2), self.param.T) 
         
         return EstimatedZernike
     
class LinearEstimator (nn.Module) :
    
    def __init__(self, WFS: WFS) :
        
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