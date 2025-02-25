#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:21:44 2025

@author: ptrouve
"""
import torch
from mmengine import Config
import Propagator as Propagator
import TorchPropagator as TorchPropagator
from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt


class PhaseDataset(Dataset):
    """Dataset of phase or zernike coefficients for given atmospheric, telescope  and sampling parameters"""
    "  Each batch of phase/zernik corresponds to randoms values of r0 and L0 in a given range"
    
    def __init__(self,D,Nres,Nzernike,L0,r0,Nphases, Nactuator,levelOfCorrection,loopFrequency, delayFrames, windSpeedVector, device ,transform=None):
            
        self.Nphases = Nphases       
        self.device=device       
        self.L0 = L0
        self.r0 = r0
        self.D = D
        self.Nactuator = Nactuator
        self.loopFrequency = loopFrequency
        self.delayFrames = delayFrames
     
  
        x = np.linspace(-Nres/2, Nres/2, Nres)                                          # Build the mesh
        [x,y] = np.meshgrid(x,x) 
                                       
        self.pupil = (x**2 + y**2) <= ((Nres+1)/2)**2
        self.pupil_logical = np.where(np.reshape(self.pupil,Nres*Nres)>0)

        #  ## Compute some example PSDs
        [self.dF, self.fx, self.fy] = TorchPropagator.GetSpatialFrequencies(D, Nres)

    # ## Compute the first Nzernike Zernike polynomials and the inverse to obtain the perfect reconstructor
        [z, z_FullRes] = Propagator.Zernike(self.pupil, self.pupil_logical, Nres, Nzernike)
        
        
        self.invZ = torch.from_numpy(np.linalg.pinv(z)).to(self.device, dtype=torch.float32)
        self.pupil = torch.from_numpy(self.pupil).to(self.device, dtype=torch.float32)
        
    def __len__(self):
        
        return self.Nphases

    def __getitem__(self, idx):
        
        
        r0 = np.power(10,random.uniform(self.r0[0],self.r0[1]))   # Fried parameter (m)
        
        L0 = random.uniform(self.L0[0],self.L0[1])
        
        levelOfCorrection = random.uniform(0, 1)

        windSpeedVector_x = random.uniform(-10, 10)
        windSpeedVector_y = random.uniform(-10, 10)
        
        windSpeedVector = [windSpeedVector_x, windSpeedVector_y]

         
        atmosphere_PSD = TorchPropagator.GetAtmospherePSD(self.fx, self.fy, self.dF, r0, L0, self.pupil, self.pupil_logical)      
        fitting_PSD = TorchPropagator.GetFittingPSD(self.fx, self.fy, self.dF, self.D, self.Nactuator, levelOfCorrection)        
        temporalErrorPSD = TorchPropagator.GetTemporalErrorPSD(self.fx, self.fy, self.dF, self.loopFrequency, self.delayFrames, windSpeedVector)
        # if atmospheric only
        #[outPhaseMap, outZe] = Propagator.GetMultiplePhaseMapAndZernike(atmosphere_PSD, self.pupil, self.pupil_logical, self.invZ, self.Nphases)  
        
        [outPhaseMap, outZe] = TorchPropagator.GetMultiplePhaseMapAndZernike(atmosphere_PSD * fitting_PSD + temporalErrorPSD * atmosphere_PSD, self.pupil, self.pupil_logical, self.invZ, self.Nphases)  
        
        
        # outPhaseMap = np.moveaxis(outPhaseMap, -1, 0)
        # outZe = np.moveaxis(outZe, -1, 0)
        
        
        # outPhaseMap = torch.from_numpy(outPhaseMap).to(self.device, dtype=torch.float32)
        # outZe = torch.from_numpy(outZe).to(self.device, dtype=torch.float32)
        
        
         
        return outPhaseMap, outZe, r0, L0
           
    
if __name__ == "__main__":
    
    
    device = 'cuda'
    
    paramfile = 'params_exp.py'

    AtmosParams = Config.fromfile(paramfile)['AtmosParams']
    WFSParams = Config.fromfile(paramfile)['WFSParams']
    LoopParams = Config.fromfile(paramfile)['LoopParams']

    dataset = PhaseDataset(WFSParams['D'],WFSParams['Nres'],WFSParams['Nzernike'],                       
                           AtmosParams['L0'],AtmosParams['r0'],AtmosParams['Nphases'],
                           WFSParams['Nactuator'],LoopParams['levelOfCorrection'],
                           LoopParams['loopFrequency'], LoopParams['delayFrames'], LoopParams['windSpeedVector'],
                           device)
    
    outPhaseMap, outZe,_,_ = dataset[0]
    
    plt.figure(1)
    plt.imshow(outPhaseMap[0,:,:].cpu().data.numpy())
    plt.colorbar()
    plt.show()
    
    