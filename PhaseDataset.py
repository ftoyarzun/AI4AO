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
import os


class PhaseDataset(Dataset):
    """Dataset of phase or zernike coefficients for given atmospheric, telescope  and sampling parameters"""
    "  Each batch of phase/zernik corresponds to randoms values of r0 and L0 in a given range"
    
    def __init__(self,D,Nres,Nzernike,L0,r0,Nphases, Nactuator,levelOfCorrection,loopFrequency, delayFrames, windSpeedVector, photonRange, RONRange, device, transform=None):
            
        self.Nphases = Nphases       
        self.device=device       
        self.L0 = L0
        self.r0 = r0
        self.D = D
        self.Nres = Nres
        self.Nactuator = Nactuator
        self.loopFrequency = loopFrequency
        self.delayFrames = delayFrames
        self.Nzernike = Nzernike
        self.levelOfCorrection = levelOfCorrection
        self.photonRange = photonRange
        self.RONRange = RONRange
        
        self.movingWavefrontGenerator = None
        
        self.testDatasetPath = "test_dataset.pth"
     
  
        x = np.linspace(-Nres/2, Nres/2, Nres)                                          # Build the mesh
        [x,y] = np.meshgrid(x,x) 
                                       
        self.pupil = (x**2 + y**2) <= ((Nres+1)/2)**2
        self.pupil_logical = np.where(np.reshape(self.pupil,Nres*Nres)>0)

        #  ## Compute some example PSDs
        [self.dF, self.fx, self.fy] = TorchPropagator.GetSpatialFrequencies(D, Nres)

    # ## Compute the first Nzernike Zernike polynomials and the inverse to obtain the perfect reconstructor
        [z, z_FullRes] = Propagator.Zernike(self.pupil, self.pupil_logical, Nres, Nzernike)
        
        self.z_FullRes = torch.from_numpy(z_FullRes)
        self.z = torch.from_numpy(z)
        
        self.invZ = torch.from_numpy(np.linalg.pinv(z)).to(self.device, dtype=torch.float32).transpose(0, 1)
        self.pupil = torch.from_numpy(self.pupil).to(self.device, dtype=torch.float32)
        
        
        if not os.path.exists(self.testDatasetPath):
            self.GenerateTestDataSet(10)
        
    def __len__(self):
        
        return self.Nphases

    def __getitem__(self, idx):
        
        device = self.pupil.device  # Ensure tensors stay on the same device
        
        # Generate batch of random parameters
        r0 = torch.empty(self.Nphases, 1, 1).uniform_(self.r0[0], self.r0[1])  # Fried parameter
        L0 = torch.empty(self.Nphases, 1, 1).uniform_(self.L0[0], self.L0[1])  # Outer scale
        levelOfCorrection = torch.empty(self.Nphases, 1, 1).uniform_(self.levelOfCorrection[0], self.levelOfCorrection[1])

        windSpeedVector_x = torch.empty(self.Nphases, 1, 1).uniform_(-10, 10)
        windSpeedVector_y = torch.empty(self.Nphases, 1, 1).uniform_(-10, 10)
     

        # Compute the PSDs in batch mode
        atmosphere_PSD = TorchPropagator.GetAtmospherePSD(self.fx, self.fy, self.dF, r0, L0, self.pupil, self.pupil_logical)  # Shape: (Nphases, H, W)
        fitting_PSD = TorchPropagator.GetFittingPSD(self.fx, self.fy, self.dF, self.D, self.Nactuator, levelOfCorrection)  # Shape: (Nphases, H, W)
        temporalErrorPSD = TorchPropagator.GetTemporalErrorPSD(self.fx, self.fy, self.dF, self.loopFrequency, self.delayFrames, windSpeedVector_x, windSpeedVector_y)  # Shape: (Nphases, H, W)
        
        
        total_PSD = atmosphere_PSD * fitting_PSD + temporalErrorPSD * atmosphere_PSD
        
        
        resolution = total_PSD.shape[-1]
        sqrt_fftshift_PSD = torch.sqrt(torch.fft.fftshift(total_PSD, dim=(-2, -1))).to(device)  # FFT shift along spatial dims
        randMap_real = torch.randn(self.Nphases, resolution, resolution, dtype=torch.float32, device=device)
        randMap_imag = torch.randn(self.Nphases, resolution, resolution, dtype=torch.float32, device=device)
        phaseMap = torch.fft.ifft2(sqrt_fftshift_PSD * (randMap_real + 1j * randMap_imag), dim=(-2, -1), norm="ortho").real
        

        phaseMap = phaseMap - torch.mean(phaseMap[:, self.pupil.bool()], dim=-1, keepdim=True).unsqueeze(-1)
        phaseMap = self.pupil * phaseMap  # Apply pupil mask

        # Compute Zernike decomposition
        Ze = torch.matmul(phaseMap[:, self.pupil.bool()], self.invZ)
        
        Nphotons = torch.pow(10, torch.empty(self.Nphases, 1, 1).uniform_(self.photonRange[0], self.photonRange[1])).to(self.device)
        RON = torch.empty(self.Nphases, 1, 1).uniform_(self.RONRange[0], self.RONRange[1]).to(self.device)
         
        return phaseMap, Ze, Nphotons, RON, r0.to(device)
           
    
    def GenerateTestDataSet(self, Ntest):
        # Generate all data
        inputs = []
        outputs = []
        photons = []
        rons = []
        r0s = []
        
        for i in range(Ntest):
            a, b, c, d, e = self.__getitem__(0)  # Get input and output
            inputs.append(a)
            outputs.append(b)
            photons.append(c)
            rons.append(d)
            r0s.append(e)
        
        # Convert lists to tensors
        inputs = torch.stack(inputs)   # Shape: (dataset_size, ...)
        outputs = torch.stack(outputs) # Shape: (dataset_size, ...)
        photons = torch.stack(photons)
        rons = torch.stack(rons)
        r0s = torch.stack(r0s)
        
        # Save to a file
        torch.save({"inputs": inputs,
                    "outputs": outputs,
                    "photons": photons,
                    "rons": rons,
                    "r0s": r0s}, self.testDatasetPath)
        
        print("Test dataset saved successfully.")



    def GetMovingWavefront(self, nLayers, t, fractionalr0 = None, windSpeedVector_x = None, windSpeedVector_y = None):
        
        if self.movingWavefrontGenerator is None:
            upSize = 3
            [self.dF_moving, self.fx_moving, self.fy_moving] = TorchPropagator.GetSpatialFrequencies(self.D * upSize, self.Nres * upSize)
            device = self.pupil.device  # Ensure tensors stay on the same device
            
            # Generate batch of random parameters
            self.r0_moving = torch.empty(self.Nphases, 1, 1).uniform_(self.r0[0], self.r0[1])  # Fried parameter
            L0 = torch.empty(self.Nphases, 1, 1).uniform_(self.L0[0], self.L0[1])  # Outer scale
            levelOfCorrection = torch.empty(self.Nphases, 1, 1).uniform_(self.levelOfCorrection[0], self.levelOfCorrection[1])

            if fractionalr0 is None:
                self.fractionalr0 = torch.empty(nLayers, self.Nphases, 1, 1).uniform_(0, 1).to(self.device)
                #self.fractionalr0 = torch.ones(nLayers, self.Nphases, 1, 1).to(self.device)
            else:
                self.fractionalr0 = fractionalr0
            
            self.fractionalr0 /= torch.sum(self.fractionalr0, dim = 0) 
                
            if windSpeedVector_x is None:
                self.windSpeedVector_x = torch.empty(nLayers, self.Nphases, 1, 1).uniform_(-10, 10)
                self.windSpeedVector_y = torch.empty(nLayers, self.Nphases, 1, 1).uniform_(-10, 10)
         
            else:
                self.windSpeedVector_x = windSpeedVector_x
                self.windSpeedVector_y = windSpeedVector_y

            # Compute the PSDs in batch mode
            atmosphere_PSD = TorchPropagator.GetAtmospherePSD(self.fx_moving, self.fy_moving, self.dF_moving, self.r0_moving, L0, self.pupil, self.pupil_logical)  # Shape: (Nphases, H, W)
            fitting_PSD = TorchPropagator.GetFittingPSD(self.fx_moving, self.fy_moving, self.dF_moving, self.D, self.Nactuator, levelOfCorrection)  # Shape: (Nphases, H, W)
            
            
            total_PSD = atmosphere_PSD * fitting_PSD
            
            
            resolution = total_PSD.shape[-1]
            sqrt_fftshift_PSD = torch.sqrt(torch.fft.fftshift(total_PSD, dim=(-2, -1))).to(device)  # FFT shift along spatial dims
            randMap_real = torch.randn(nLayers, self.Nphases, resolution, resolution, dtype=torch.float32, device=device)
            randMap_imag = torch.randn(nLayers, self.Nphases, resolution, resolution, dtype=torch.float32, device=device)
            self.movingWavefrontGenerator = sqrt_fftshift_PSD * (randMap_real + 1j * randMap_imag)
            self.translationPhase = torch.fft.fftshift(torch.exp(1j * 2 * torch.pi * t * (self.windSpeedVector_x * self.fx_moving.unsqueeze(0).unsqueeze(0) + self.windSpeedVector_y * self.fy_moving.unsqueeze(0).unsqueeze(0))), dim = (-1, -2)).to(device=device)
        
            self.Nphotons = torch.pow(10, torch.empty(self.Nphases, 1, 1).uniform_(self.photonRange[0], self.photonRange[1])).to(self.device)
            self.RON = torch.empty(self.Nphases, 1, 1).uniform_(self.RONRange[0], self.RONRange[1]).to(self.device)
        
        self.movingWavefrontGenerator *= self.translationPhase
        layeredPhase = torch.fft.fft2(self.movingWavefrontGenerator, dim=(-2, -1), norm="ortho").real
        
        croppedLayeredPhase = layeredPhase[:, :, :self.Nres, :self.Nres]
        
        phaseMap = (self.fractionalr0 * croppedLayeredPhase).sum(dim=0)

        phaseMap = phaseMap - torch.mean(phaseMap[:, self.pupil.bool()], dim=-1, keepdim=True).unsqueeze(-1)
        phaseMap = self.pupil * phaseMap  # Apply pupil mask

        # Compute Zernike decomposition
        Ze = torch.matmul(phaseMap[:, self.pupil.bool()], self.invZ)
        
        
         
        return phaseMap, Ze, self.Nphotons, self.RON, self.r0_moving.to(self.device), torch.stack((self.windSpeedVector_x,self.windSpeedVector_y)), self.fractionalr0
        
    def ResetMovingWavefront(self):
        self.movingWavefrontGenerator = None

class PermanentPhaseDataset(Dataset):
    def __init__(self, file_path="test_dataset.pth"):
        data = torch.load(file_path)
        self.inputs = data["inputs"]
        self.outputs = data["outputs"]
        self.photons = data["photons"]
        self.rons = data["rons"]
        self.r0s = data["r0s"]
        self.Nzernike = self.outputs.shape[-1]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], self.photons[idx], self.rons[idx], self.r0s[idx]

    
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
                           WFSParams['Nphotons'], WFSParams['RON'], device)
    
    outPhaseMap, outZe,_,_ = dataset[0]
    
    plt.figure(1)
    plt.imshow(outPhaseMap[0,:,:].cpu().data.numpy())
    plt.colorbar()
    plt.show()
    
    