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
import time

from line_profiler import profile


class PhaseDataset(Dataset):
    """Dataset of phase or zernike coefficients for given atmospheric, telescope  and sampling parameters"""
    "  Each batch of phase/zernik corresponds to randoms values of r0 and L0 in a given range"
    
    def __init__(self, WFSParams, AtmosParams, LoopParams, device, transform=None):
            
        
        self.D = WFSParams['D']
        self.Nres = WFSParams['Nres']
        self.Nzernike = WFSParams['Nzernike']
        self.photonRange = WFSParams['Nphotons']
        self.RONRange = WFSParams['RON']
        self.Nactuator = WFSParams['Nactuator']
                       
        self.L0Range = AtmosParams['L0']
        self.r0Range = AtmosParams['r0']
        self.Nphases = AtmosParams['Nphases']
        self.nLayers = AtmosParams["Layers"]
                               
                               
        self.levelOfCorrectionRange = LoopParams['levelOfCorrection']
        self.loopFrequency = LoopParams['loopFrequency']
        self.delayFrames = LoopParams['delayFrames']
        self.windSpeedRange = LoopParams['windSpeedVector']
     
        self.device=device       

        self.translationPhase = 1.
        self.movingCount = 0
        
        self.testDatasetPath = "test_dataset.pth"
        self.movingTestDatasetPath = "moving_test_dataset.pth"
     
  
        x = np.linspace(-self.Nres/2, self.Nres/2, self.Nres)                                          # Build the mesh
        [x,y] = np.meshgrid(x,x) 
                                       
        self.pupil = (x**2 + y**2) <= ((self.Nres+1)/2)**2
        self.pupil_logical = np.where(np.reshape(self.pupil,self.Nres*self.Nres)>0)

        #  ## Compute some example PSDs
        [self.dF, self.fx, self.fy] = TorchPropagator.GetSpatialFrequencies(self.D, self.Nres)
        self.fsqr = self.fx**2 + self.fy**2
        
        upSize = 2
        [self.dF_moving, self.fx_moving, self.fy_moving] = TorchPropagator.GetSpatialFrequencies(self.D * upSize, self.Nres * upSize, self.device)
        self.fsqr_moving = self.fx_moving**2 + self.fy_moving**2

    # ## Compute the first Nzernike Zernike polynomials and the inverse to obtain the perfect reconstructor
        [z, z_FullRes] = Propagator.Zernike(self.pupil, self.pupil_logical, self.Nres, self.Nzernike)
        
        self.z_FullRes = torch.from_numpy(z_FullRes)
        self.z = torch.from_numpy(z)
        
        #self.invZ = torch.from_numpy(np.linalg.pinv(z)).to(self.device, dtype=torch.float32).transpose(0, 1)
        self.invZ = torch.linalg.pinv(self.z_FullRes.flatten(0,1)).to(self.device, dtype=torch.float32).transpose(0, 1)
        self.pupil = torch.from_numpy(self.pupil).to(self.device, dtype=torch.float32)
        
        
        if not os.path.exists(self.testDatasetPath):
            self.GenerateTestDataSet(10)
            
        if not os.path.exists(self.movingTestDatasetPath):
            self.GenerateMovingTestDataSet()
            
            
            
        self.r0_moving = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.L0 = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.levelOfCorrection = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.Nphotons = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.RON = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.fractionalr0 = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device)
        self.windSpeedVector_x = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device)
        self.windSpeedVector_y = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device)
        
    def __len__(self):
        
        return self.Nphases

    def __getitem__(self, idx):
        
        device = self.pupil.device  # Ensure tensors stay on the same device
        
        # Generate batch of random parameters
        r0 = torch.empty(self.Nphases, 1, 1).uniform_(self.r0Range[0], self.r0Range[1])  # Fried parameter
        L0 = torch.empty(self.Nphases, 1, 1).uniform_(self.L0Range[0], self.L0Range[1])  # Outer scale
        levelOfCorrection = torch.empty(self.Nphases, 1, 1).uniform_(self.levelOfCorrectionRange[0], self.levelOfCorrectionRange[1])

        windSpeedVector_x = torch.empty(self.Nphases, 1, 1).uniform_(-10, 10)
        windSpeedVector_y = torch.empty(self.Nphases, 1, 1).uniform_(-10, 10)
     

        # Compute the PSDs in batch mode
        atmosphere_PSD = TorchPropagator.GetAtmospherePSD(self.fsqr, self.dF, r0, L0)  # Shape: (Nphases, H, W)
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
        Ze = torch.matmul(phaseMap.flatten(1,2), self.invZ)
        
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


    def GenerateMovingTestDataSet(self):
        
        self.ResetMovingWavefront()
        
        _,_, Nphotons, ron, r0, wind, fractionalr0 = self.GetMovingWavefront(3, 1/self.loopFrequency)
        
        movingWavefrontGenerator = self.movingWavefrontGenerator
        translationPhase = self.translationPhase
        windSpeedVector_x = self.windSpeedVector_x
        windSpeedVector_y = self.windSpeedVector_y
        
        torch.save(
            {"movingWavefrontGenerator": movingWavefrontGenerator,
             "translationPhase": translationPhase,
             "Nphotons": Nphotons,
             "ron": ron,
             "r0": r0,
             "windSpeedVector_x": windSpeedVector_x,
             "windSpeedVector_y": windSpeedVector_y,
             "fractionalr0": fractionalr0},
            self.movingTestDatasetPath)
        
        print("Test dataset saved successfully.")
        
        
        
        
    @profile
    def GetMovingWavefront(self, generateClosedLoop = False):

        if self.movingCount == 0:
            # Generate batch of random parameters
            self.DrawRandomParameters()    

            # Compute the PSDs in batch mode
            total_PSD = self.BuildAtmospherePSD(generateClosedLoop)
            
            resolution = total_PSD.shape[-1]
            sqrt_fftshift_PSD = torch.sqrt(torch.fft.fftshift(total_PSD, dim=(-2, -1)))  # FFT shift along spatial dims
            randMap = torch.randn(self.nLayers, self.Nphases, resolution, resolution, dtype=torch.complex32, device=self.device) 
            self.movingWavefrontGenerator = sqrt_fftshift_PSD * randMap 
       
        if self.movingCount == 1:
            phase_factor = (1j * 2 * torch.pi / self.loopFrequency * (self.windSpeedVector_x * self.fx_moving.unsqueeze(0).unsqueeze(0) + self.windSpeedVector_y * self.fy_moving.unsqueeze(0).unsqueeze(0))).to(device = self.device, dtype = torch.complex32)
            self.translationPhase = torch.fft.fftshift(torch.exp(phase_factor), dim = (-1, -2))

        self.movingWavefrontGenerator *= self.translationPhase
        
        
        phaseMap = self.CompressAtmosphere() 
        
        # Compute Zernike decomposition
        Ze = torch.matmul(phaseMap.flatten(1,2), self.invZ)
        
        self.movingCount += 1
         
        return phaseMap, Ze, self.Nphotons, self.RON, self.r0_moving, torch.stack((self.windSpeedVector_x,self.windSpeedVector_y)), self.fractionalr0
    
    
    def CompressAtmosphere(self):
        layeredPhase = torch.fft.fft2(self.movingWavefrontGenerator, dim=(-2, -1), norm="ortho").real
        croppedLayeredPhase = layeredPhase[:, :, :self.Nres, :self.Nres]
        phaseMap = (torch.sqrt(self.fractionalr0) * croppedLayeredPhase).sum(dim=0)
        phaseMap = self.pupil * phaseMap  # Apply pupil mask
        return phaseMap
        
    
    def ResetMovingWavefront(self):
        self.translationPhase = 1.
        self.movingWavefrontGenerator = None
        self.movingCount = 0
        
        
    def DrawRandomParameters(self):
        self.r0_moving = self.r0_moving.uniform_(*self.r0Range)  # Fried parameter
        self.L0 = self.L0.uniform_(*self.L0Range)  # Outer scale
        self.levelOfCorrection = self.levelOfCorrection.uniform_(*self.levelOfCorrectionRange)
        self.Nphotons = torch.pow(10, self.Nphotons.uniform_(*self.photonRange))
        self.RON = self.RON.uniform_(*self.RONRange)

        self.fractionalr0 = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device).uniform_(0, 1)
        self.fractionalr0 /= torch.sum(self.fractionalr0, dim = 0) 
            
        self.windSpeedVector_x = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device).uniform_(*self.windSpeedRange)
        self.windSpeedVector_y = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device).uniform_(*self.windSpeedRange)

     
    def BuildAtmospherePSD(self, generateClosedLoop):
        atmosphere_PSD = TorchPropagator.GetAtmospherePSD(self.fsqr_moving, self.dF_moving, self.r0_moving, self.L0)  # Shape: (Nphases, H, W)
        total_PSD = atmosphere_PSD# * fitting_PSD
        if not generateClosedLoop:
            return total_PSD
        
        total_PSD = total_PSD.repeat(self.nLayers, 1, 1, 1)
        fitting_PSD = TorchPropagator.GetFittingPSD(self.fx_moving, self.fy_moving, self.dF_moving, self.D, self.Nactuator, self.levelOfCorrection)  # Shape: (Nphases, H, W)
        temporalErrorPSD = TorchPropagator.GetTemporalErrorPSD(self.fx_moving, self.fx_moving, self.dF_moving, self.loopFrequency, self.delayFrames, self.windSpeedVector_x, self.windSpeedVector_y)  # Shape: (Nphases, H, W)
        total_PSD *= fitting_PSD
        total_PSD += temporalErrorPSD * atmosphere_PSD
        
        return total_PSD
        
        
    def LoadTestMovingWavefront(self, file_path="moving_test_dataset.pth"):
        data = torch.load(file_path)
        self.movingWavefrontGenerator = data["movingWavefrontGenerator"]
        self.translationPhase = data["translationPhase"]
        self.Nphotons = data["Nphotons"]
        self.RON = data["ron"]
        self.r0_moving = data["r0"]
        self.windSpeedVector_x = data["windSpeedVector_x"]
        self.windSpeedVector_y = data["windSpeedVector_y"]
        self.fractionalr0 = data["fractionalr0"]
        
        

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

    dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, device)
    
    # outPhaseMap, outZe,_,_,_ = dataset[0]
    
    t1 = time.perf_counter()
    dataset.ResetMovingWavefront()
    outPhaseMap, outZe,_,_,_,_,fr0 = dataset.GetMovingWavefront(generateClosedLoop = True)
    t2 = time.perf_counter()
    print(1/(t2 - t1))
    
    
    plt.figure(1)
    plt.imshow(outPhaseMap[0,:,:].cpu().data.numpy())
    plt.colorbar()
    plt.show()
    
    