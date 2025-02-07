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
    
    def __init__(self,Nzernike) :
        
        super().__init__()
        
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, stride=5),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(3),
           
            nn.Flatten()
        )
        
        self.outputlayer = nn.Linear(3200,Nzernike)

       
        
    def forward(self, x):
      
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
     
class WFSmodule (nn.Module) :
    
    def __init__(self, ParamsDict,device) :
        
        super().__init__()
        

        self.param = nn.Parameter(torch.tensor([1.5,0.5]).to(device))
        self.param_name = "toy example parameter vector"
        
        self.WFS = WFS(ParamsDict['Nres'],ParamsDict['sampling'],ParamsDict['D'],ParamsDict['Nphotons'], ParamsDict['RON'],ParamsDict['useNoise'],self.param,device)
       
        
    def forward(self, phase):
        
        
        return self.WFS.Propagator(phase)     

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
        
        return self.WFS.GetReconstructedPhase(image)
    
    
class End2EndWFS (nn.Module):
    
    
    def __init__(self, ParamsDict,device):
        
        super().__init__()
     
        self.WFSmodule = WFSmodule(ParamsDict,device)
        # Choice of phase estimator (Linear or learned optimized)
        Nzernike = ParamsDict['Nzernike']
        
        # change here the processing for phase estimation
        #self.PhaseEstimator = LinearEstimator(self.WFSmodule.WFS, Nzernike,device)
        
        self.PhaseEstimator = OptimizedLinearEstimator(0,self.WFSmodule.WFS, Nzernike).to(device)
        #self.PhaseEstimator = SimpleNet(Nzernike).to(device)
       
        
        
    def forward(self, x):
        # input : x (tensor) : input phase
        # output : estimated phase
        
        # Reload the ZernikeMask according to current parameter
        
         #self.WFSmodule.WFS.BuildPyramidMask()
         self.WFSmodule.WFS.BuildZernikeMask()
         self.WFSmodule.WFS.BuildReferenceIntensity()
         
         # Compute the image from the phase
        
         Image = self.WFSmodule(x)       
         
         # Estimate the phase 
         
         EstimatedPhase = self.PhaseEstimator(Image-self.WFSmodule.WFS.reference_intensity)
         
         return EstimatedPhase
         
         
         
if __name__ == "__main__":
    
    device = 'cuda'
    
    paramfile = 'params_exp.py'

    WFSParams = Config.fromfile(paramfile)['WFSParams']
 
    
    MyEnd2EndWFS = End2EndWFS(WFSParams,device)
    
    
    # for p in MyEnd2EndWFS.PhaseEstimator.parameters():
    #     if p.requires_grad:
    #         print(p.name, p.data)