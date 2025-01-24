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



    
    
class OptimizedLinearEstimator (nn.Module) :
    " Learned Linear Estimator with a learned reconstruction matrix and ref intensity"
    "They are initalized using the propagator code from the starting point"
    
    def __init__(self,init=0,WFS=None,Nzernike=0) :
        
        super().__init__()
        
        # Initialization with the  reconstruction matrix at starting point
        if init == 1 :
            
            print("Initalization of the reconsctruction matrix")
            [z, z_FullRes] = Zernike(WFS.pupil, WFS.pupil_logical, WFS.Nres, Nzernike)     
            z_FullRes = z_FullRes
            WFS.BuildReconstructionMatrix(z_FullRes, WFS.mask)
    
            self.LearnedReconstructionMatrix = nn.Parameter(WFS.reconstructionMatrix)
            
        # Reconstruction matrix initalized at 0
        else :
            self.LearnedReconstructionMatrix = nn.Parameter(torch.zeros((50,22500),dtype = torch.float64))

          
    def forward(self, image):
        
         ## (Learned) Matrix multiplication
         

         EstimatedZernike = self.LearnedReconstructionMatrix @ image.flatten()
        
         return EstimatedZernike    

class LinearEstimator (nn.Module) :
    
    def __init__(self, WFS,Nzernike) :
        
        super().__init__()
        
        self.WFS = WFS
        self.Nzernike = Nzernike
        [z, z_FullRes] = Zernike(self.WFS.pupil, self.WFS.pupil_logical, self.WFS.Nres, self.Nzernike)
        
        self.z_FullRes = z_FullRes            
        
    def forward(self, image):
        
         ## Build the reconstruction matrix for each forward (because it depends on the optimized parameters)
        self.WFS.BuildReconstructionMatrix(self.z_FullRes, self.WFS.mask)
    
        return self.WFS.GetReconstructedPhase(image)
    
    
class End2EndWFS (nn.Module):
    
    
    def __init__(self, ParamsDict,device):
        
        super().__init__()
     
        self.WFS = WFS(ParamsDict['Nres'],ParamsDict['sampling'],ParamsDict['D'],ParamsDict['Nphotons'], ParamsDict['RON'],ParamsDict['useNoise'],device)
        Nzernike =ParamsDict['Nzernike']
        
        # Choice of phase estimator (Linear or learned optimized)
        
        self.PhaseEstimator = LinearEstimator(self.WFS, Nzernike)
        #self.PhaseEstimator = OptimizedLinearEstimator(0,self.WFS, Nzernike).to(device)
       
        
        
    def forward(self, x):
        # input : x (tensor) : input phase
        # output : estimated phase
        
        # Reload the PyramidMask according to current parameter
        
         self.WFS.BuildPyramidMask()
         
         # Compute the image from the phase
        
         Image = self.WFS.Propagator(x)       
         
         # Estimate the phase 
         
         EstimatedPhase = self.PhaseEstimator(Image )
         
         return EstimatedPhase
         
         
         
if __name__ == "__main__":
    
    device = 'cpu'
    
    paramfile = 'params_exp.py'

    WFSParams = Config.fromfile(paramfile)['WFSParams']
 
    
    MyEnd2EndWFS = End2EndWFS(WFSParams,device)
    
    
    