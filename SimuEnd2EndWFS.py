#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:45:31 2025

@author: ptrouve
"""

from TorchPropagator import WFS, Zernike

import torch.nn as nn
import torch
from mmengine import Config


class SimpleNet (nn.Module) :
    
    def __init__(self,Nzernike,Npix) :
        
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.GELU(),
        
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Extra conv layer
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.GELU(),
            nn.MaxPool2d(2),  # First pooling
        
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.GELU(),
        
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Extra conv layer
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.GELU(),
            nn.MaxPool2d(2),  # Second pooling
        
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.GELU(),
        
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Extra conv layer
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.GELU(),
            nn.MaxPool2d(2),  # Third pooling
        
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.GELU(),
        
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Extra conv layer
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.GELU(),
            nn.MaxPool2d(2),  # Fourth pooling
        
            nn.AdaptiveAvgPool2d((1, 1)),  
            nn.Flatten(),
            nn.Dropout(0.3)  
        )

        self.outputlayer = nn.Linear(256, Nzernike)  # 512 comes from final channel size
       
        
    def forward(self, x):
        # no complex32 in pytorch
        x = self.encoder(x[:,None,:,:].type(torch.float32))
       
        x= self.outputlayer(x)
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
    
    def __init__(self, WFS,Nzernike,device) :
        
        super().__init__()
        
        self.WFS = WFS
        self.Nzernike = Nzernike
        [z, z_FullRes] = Zernike(self.WFS.pupil.cpu(), self.WFS.pupil_logical, self.WFS.Nres, self.Nzernike)
        
        self.z_FullRes = z_FullRes.to(device)         
        
        
        self.param = nn.Parameter(torch.tensor([0.1],dtype = torch.float64).to(device))
        self.param_name = "toy example parameter not used here"
        self.WFS.BuildReconstructionMatrix(self.z_FullRes, self.WFS.mask)
       
        
    def forward(self, image):
        
         ## Build the reconstruction matrix for each forward (because it depends on the optimized parameters)
        self.WFS.BuildReconstructionMatrix(self.z_FullRes, self.WFS.mask)
        
        EstimatedZernike = torch.matmul(image.flatten(start_dim = -2), self.WFS.reconstructionMatrix.T.detach()) 
         
        return  EstimatedZernike
    
     
class WFSmodule (nn.Module) :
    
    def __init__(self, ParamsDict,device) :
        
        super().__init__()
        

        self.param = nn.Parameter(torch.tensor(ParamsDict['InitParam']).to(device))
        self.param_name = "toy example parameter vector"
        
        self.WFS = WFS(ParamsDict['Nres'],ParamsDict['sampling'],ParamsDict['D'],ParamsDict['Nphotons'], ParamsDict['RON'],ParamsDict['useNoise'],self.param,ParamsDict['MaskType'],device)
       
        
    def forward(self, phase):
        
        
        return self.WFS.Propagator(phase)     


    
class End2EndWFS (nn.Module):
    
    
    def __init__(self, ParamsDict,device):
        
        super().__init__()
     
        self.WFSmodule = WFSmodule(ParamsDict,device)
        # Choice of phase estimator (Linear or learned optimized)
        Nzernike = ParamsDict['Nzernike']
        self.maskType = ParamsDict['MaskType']
        # change here the processing for phase estimation
        #self.PhaseEstimator = LinearEstimator(self.WFSmodule.WFS, Nzernike,device)
        
        #self.PhaseEstimator = OptimizedLinearEstimator(0,self.WFSmodule.WFS, Nzernike).to(device)
        self.PhaseEstimator = SimpleNet(Nzernike,self.WFSmodule.WFS.Nres*2).to(device)
        
        if self.maskType == "Free":
            self.maskGenerator = MaskGenerator().to(device)
            
            self.N = ParamsDict['Nres'] * ParamsDict['sampling']  # Resolution
            
            # Generate a grid of frequency coordinates
            u = torch.linspace(-1, 1, self.N)  # Normalized frequency range
            v = torch.linspace(-1, 1, self.N)
            U, V = torch.meshgrid(u, v, indexing="xy")  # Create the full grid
            
            # Flatten and stack into (N^2, 2) shape
            self.uv_coords = torch.stack([U.flatten(), V.flatten()], dim=1).to(device)
            self.mask = self.maskGenerator(self.uv_coords)
       
        
        
    def forward(self, x):
        # input : x (tensor) : input phase
        # output : estimated phase
        
        # Reload the ZernikeMask according to current parameter
         if self.maskType == "Pyramid":
             
             self.WFSmodule.WFS.BuildPyramidMask()
             
         elif self.maskType == "Zernike":
             self.WFSmodule.WFS.BuildZernikeMask()
         
            
         elif self.maskType == "Free":
             self.mask = self.maskGenerator(self.uv_coords).view(self.N, self.N)
             self.WFSmodule.WFS.SetMask(self.mask)
             
             
         # self.WFSmodule.WFS.BuildReferenceIntensity()
         
         # Compute the image from the phase
        
         Image = self.WFSmodule(x)       
         
         # Estimate the phase 
         
         EstimatedPhase = self.PhaseEstimator(Image) #-self.WFSmodule.WFS.reference_intensity)
         
         return EstimatedPhase


class MaskGenerator(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),  # Input: (u, v)
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # Output: Mask value
        )

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.1)  # Normal distribution
            nn.init.zeros_(module.bias)  # Set bias to zero

    def forward(self, uv_coords):
        return self.net(uv_coords)         
         
         
if __name__ == "__main__":
    
    device = 'cuda'
    
    paramfile = 'params_exp.py'

    WFSParams = Config.fromfile(paramfile)['WFSParams']
 
    
    MyEnd2EndWFS = End2EndWFS(WFSParams,device)
    
    
    # for p in MyEnd2EndWFS.PhaseEstimator.parameters():
    #     if p.requires_grad:
    #         print(p.name, p.data)