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
import astropy.units as u
import poppy
from scipy.io import loadmat


class PhaseDataset(Dataset):
    """
    PhaseDataset is a custom PyTorch dataset used to generate synthetic wavefront phase maps 
    and corresponding Zernike decompositions for training neural networks in adaptive optics systems.

    It supports:
    - Static and dynamic (moving) wavefront generation.
    - Randomized atmospheric parameters such as Fried parameter (r0), outer scale (L0), and wind.
    - Configurable parameters from wavefront sensor (WFS), atmospheric, and control loop settings.
    - PSD-based generation of phase maps using Fourier domain techniques.
    - Automatic generation and caching of static and dynamic test datasets.
    - Computation of Zernike coefficients for phase map reconstruction and analysis.

    Attributes:
        D (float): Aperture diameter.
        Nres (int): Resolution of the wavefront map (number of pixels).
        Nzernike (int): Number of Zernike modes to generate.
        photonRange (tuple): Log-scale range of photon count per measurement.
        RONRange (tuple): Read-Out Noise (RON) range.
        Nactuator (int): Number of actuators for the deformable mirror.
        r0Range (tuple): Range for the Fried parameter.
        L0Range (tuple): Range for the atmospheric outer scale.
        Nphases (int): Number of phase maps to generate in each sample.
        nLayers (int): Number of atmospheric turbulence layers.
        loopFrequency (float): Adaptive optics loop frequency.
        delayFrames (int): Control loop delay in frames.
        windSpeedRange (tuple): Range of wind speed values for each layer.
        pupil (Tensor): Circular aperture mask.
        z_FullRes (Tensor): Precomputed full-resolution Zernike polynomials.
        invZ (Tensor): Pseudo-inverse of the Zernike matrix for decomposition.
        testDatasetPath (str): File path for cached static dataset.
        movingTestDatasetPath (str): File path for cached moving dataset.

    Usage:
        dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, device)
        sample = dataset[idx]  # Returns phaseMap, Zernike coefficients, photons, RON, r0

    Dependencies:
        Requires `TorchPropagator` and `Propagator` modules for PSD and Zernike generation.
    """
    def __init__(self, WFSParams, AtmosParams, LoopParams, device):
        """
        Initialize the PhaseDataset.
    
        Parameters:
            WFSParams (dict): Parameters related to the wavefront sensor.
            AtmosParams (dict): Parameters describing atmospheric conditions.
            LoopParams (dict): Parameters for the adaptive optics control loop.
            device (torch.device): Device to store tensors (CPU or GPU).
            transform (callable, optional): Transform to apply on the data.
        """    
        
        self.D = WFSParams['D']
        self.Nres = WFSParams['Nres']
        self.Nzernike = WFSParams['Nzernike']
        self.photonRange = WFSParams['Nphotons']
        self.RONRange = WFSParams['RON']
        self.Nactuator = WFSParams['Nactuator']
        self.c_obs = WFSParams['c_obs']
                       
        self.L0Range = AtmosParams['L0']
        self.r0Range = AtmosParams['r0']
        self.Nphases = AtmosParams['Nphases']
        self.nLayersRange = AtmosParams["Layers"]
        self.f_slope = AtmosParams["f_slope"]
                               
                               
        self.levelOfCorrectionRange = LoopParams['levelOfCorrection']
        self.loopFrequency = LoopParams['loopFrequency']
        self.delayFrames = LoopParams['delayFrames']
        self.windSpeedRange = LoopParams['windSpeedVector']
     
        self.device=device       

        self.translationPhase = 1.
        self.movingCount = 0
        
        self.testDatasetPath = "test_dataset.pth"
        self.movingTestDatasetPath = "moving_test_dataset.pth"
     
  
        # x = np.linspace(-self.Nres/2, self.Nres/2, self.Nres) # Build the mesh
        # [x,y] = np.meshgrid(x,x) 
        # radius = (self.Nres+1) / 2
        # self.pupil = np.bitwise_and((x**2 + y**2) <= radius**2, (x**2 + y**2) >= (self.c_obs * radius)**2)
        # IRCS+AO188 pupil parameters
        sp_offset = 1.278  # Spider offset [m]
        sp_angle = 51.75   # Spider angle [deg]
        sp_thick = 0.224   # Spider thickness [m]
        # Subaru IRCS+AO188
        ap = poppy.CircularAperture(radius=self.D/2.0*u.m)
        sec = poppy.AsymmetricSecondaryObscuration(
                secondary_radius=self.c_obs/2.0*u.m,
                support_angle=(90-sp_angle, 90+sp_angle, 270-sp_angle, 270+sp_angle),
                support_width=[sp_thick, sp_thick, sp_thick, sp_thick],
                support_offset_x=[-sp_offset/2., -sp_offset/2., sp_offset/2., sp_offset/2.])
        pupil = poppy.CompoundAnalyticOptic(opticslist=[ap,sec], name='AO188+IRCS')
        mask = pupil.sample(npix=self.Nres, grid_size=self.D, what='amplitude')
        self.pupil = mask

        self.pupilSum = self.pupil.sum()

        self.pupil_logical = np.where(self.pupil.flatten() > 0)

        #  ## Compute some example PSDs
        [self.dF, self.fx, self.fy] = TorchPropagator.GetSpatialFrequencies(self.D, self.Nres, self.device)
        self.fsqr = self.fx**2 + self.fy**2
        
        upSize = 2
        [self.dF_moving, self.fx_moving, self.fy_moving] = TorchPropagator.GetSpatialFrequencies(self.D * upSize, self.Nres * upSize, self.device)
        self.fsqr_moving = self.fx_moving**2 + self.fy_moving**2

    # ## Compute the first Nzernike Zernike polynomials and the inverse to obtain the perfect reconstructor
    
        # self.pupil = torch.from_numpy(self.pupil).to(self.device, dtype=torch.float32)
    
        if WFSParams['ModalBasis'] == "Zernike":
            [z, z_FullRes] = Propagator.Zernike(self.pupil, self.pupil_logical, self.Nres, self.Nzernike)
            
            self.z_FullRes = torch.from_numpy(z_FullRes).to(device=device, dtype = torch.float32)
            self.z = torch.from_numpy(z)
            self.pupil = torch.from_numpy(self.pupil).to(self.device, dtype=torch.float32)
        
        elif WFSParams['ModalBasis'] == "Papyrus_KL":
            M2C = torch.from_numpy(loadmat('M2C_KL_OOPAO_synthetic_IF')["M2C_KL"]).to(device = device, dtype = torch.float32)[:,:self.Nzernike]
            papyrus_dm = torch.from_numpy(np.load("papyrus_dm.npy").astype(np.float32)).to(device=device) * 1e7
            papyrus_modal_dm = (papyrus_dm @ M2C).view(80,80,-1)[1:-1, 1:-1, :]

            
            self.pupil = torch.from_numpy(self.pupil).to(self.device, dtype=torch.float32)
            self.z_FullRes = papyrus_modal_dm * self.pupil.unsqueeze(-1)
            self.z = self.z_FullRes[self.pupil.bool()]
            
        elif WFSParams['ModalBasis'] == "Papyrus_Zernike":
            Z2C = torch.from_numpy(np.load("Z2C.npy").astype(np.float32)).to(device=device).T[:,:self.Nzernike]
            papyrus_dm = torch.from_numpy(np.load("papyrus_dm.npy").astype(np.float32)).to(device=device) * 1e7
            papyrus_modal_dm = (papyrus_dm @ Z2C).view(80,80,-1)[1:-1, 1:-1, :]

            
            self.pupil = torch.from_numpy(self.pupil).to(self.device, dtype=torch.float32)
            self.z_FullRes = papyrus_modal_dm * self.pupil.unsqueeze(-1)
            self.z = self.z_FullRes[self.pupil.bool()]
            
        elif WFSParams['ModalBasis'] == "Papyrus_Zonal":
            papyrus_dm = torch.from_numpy(np.load("papyrus_dm.npy").astype(np.float32)).to(device=device) * 1e7
            papyrus_dm = papyrus_dm.view(80,80,-1)[1:-1, 1:-1, :]

            self.pupil = torch.from_numpy(self.pupil).to(self.device, dtype=torch.float32)
            self.z_FullRes = papyrus_dm * self.pupil.unsqueeze(-1)
            self.z = self.z_FullRes[self.pupil.bool()]
        
        elif WFSParams['ModalBasis'] == "Oziriis_KL":
            oziriis_modal_dm = torch.from_numpy(np.load(r'C:\Users\foyarzun\Nextcloud\PostDoc\OZIRIIS\Data\OZIRIIS_KL_90x90.npy')).to(device = device, dtype = torch.float32)[:self.Nzernike]
            oziriis_modal_dm = oziriis_modal_dm.view(-1,90,90).permute(1,2,0)

            self.pupil = torch.from_numpy(self.pupil).to(self.device, dtype=torch.float32)
            self.z_FullRes = oziriis_modal_dm * self.pupil.unsqueeze(-1)
            self.z = self.z_FullRes[self.pupil.bool()]

        elif WFSParams['ModalBasis'] == "Oziriis_Zonal":
            oziriis_zonal_dm = torch.from_numpy(np.load(r'C:\Users\foyarzun\Nextcloud\PostDoc\OZIRIIS\Data\OZIRIIS_Zonal_90x90.npy')).to(device = device, dtype = torch.float32)[:self.Nzernike]
            oziriis_zonal_dm = oziriis_zonal_dm.view(-1,90,90).permute(1,2,0)

            self.pupil = torch.from_numpy(self.pupil).to(self.device, dtype=torch.float32)
            self.z_FullRes = oziriis_zonal_dm * self.pupil.unsqueeze(-1)
            self.z = self.z_FullRes[self.pupil.bool()]

        else:
            raise ValueError(f"Unknown basis: {WFSParams['ModalBasis']}")
            
        
        #self.invZ = torch.from_numpy(np.linalg.pinv(z)).to(self.device, dtype=torch.float32).transpose(0, 1)
        self.invZ = torch.linalg.pinv(self.z_FullRes.flatten(0,1)).to(self.device, dtype=torch.float32).transpose(0, 1)
        
        
        self.r0_moving = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.L0 = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.levelOfCorrection = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.Nphotons = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.RON = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.nLayers = self.nLayersRange[0]
        # self.nLayers = np.random.randint(*self.nLayersRange)
        self.fractionalr0 = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device)
        self.windSpeedVector_x = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device)
        self.windSpeedVector_y = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device)
        
        
    def __len__(self):
        
        return self.Nphases

    def __getitem__(self, idx):
        
        device = self.fsqr.device  # Ensure tensors stay on the same device
        # Generate batch of random parameters
        r0 = torch.zeros(self.Nphases, 1, 1, device=device) + self.r0Range[0]  # Fried parameter
        L0 = torch.zeros(self.Nphases, 1, 1, device=device) + self.L0Range[0]  # Outer scale
        levelOfCorrection = torch.zeros(self.Nphases, 1, 1, device=device)
        # windSpeedVector_x = torch.empty(self.Nphases, 1, 1).uniform_(-10, 10)
        # windSpeedVector_y = torch.empty(self.Nphases, 1, 1).uniform_(-10, 10)
     

        # Compute the PSDs in batch mode
        atmosphere_PSD = TorchPropagator.GetAtmospherePSD(self.fsqr, self.dF, r0, L0)  # Shape: (Nphases, H, W)
        fitting_PSD = TorchPropagator.GetFittingPSD(self.fx, self.fy, self.dF, self.D, self.Nactuator, levelOfCorrection)  # Shape: (Nphases, H, W)
        #temporalErrorPSD = TorchPropagator.GetTemporalErrorPSD(self.fx, self.fy, self.dF, self.loopFrequency, self.delayFrames, windSpeedVector_x, windSpeedVector_y)  # Shape: (Nphases, H, W)
        
        
        total_PSD = atmosphere_PSD * fitting_PSD #+ temporalErrorPSD * atmosphere_PSD
        
        
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
        """
        Generate and save a static test dataset with given number of samples.
    
        Parameters:
            Ntest (int): Number of test samples to generate.
    
        Saves:
            A .pth file containing static wavefronts, Zernike coefficients, 
            and noise/atmospheric parameters.
        """
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
        """
        Generate and save a test dataset of temporally-evolving wavefronts
        using a synthetic moving turbulence model.
    
        Saves:
            A .pth file containing initial wavefront generator state and parameters
            needed for further time evolution.
        """
        self.ResetMovingWavefront()
        
        _,_, Nphotons, ron, r0, wind, fractionalr0 = self.GetMovingWavefront()
        
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
        
        
    @torch.no_grad() 
    def GetMovingWavefront(self, generateClosedLoop = False):
        """
        Generate a moving (time-evolving) wavefront phase map.
    
        Parameters:
            generateClosedLoop (bool): Whether to include closed-loop phase modeling.
    
        Returns:
            Tuple of:
                - phaseMap (torch.Tensor): Current phase map.
                - Ze (torch.Tensor): Corresponding Zernike coefficients.
                - Nphotons (torch.Tensor): Photon count.
                - RON (torch.Tensor): Read-out noise.
                - r0 (torch.Tensor): Fried parameter.
                - wind (torch.Tensor): Wind speed vectors for each layer.
                - fractionalr0 (torch.Tensor): Relative r0 contribution of each layer.
        """
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
            self.translationPhase = torch.fft.fftshift(torch.exp(phase_factor), dim = (-2, -1))

        self.movingWavefrontGenerator *= self.translationPhase
        
        
        phaseMap = self.CompressAtmosphere() 
        
        phaseMap = self.RemovePiston(phaseMap)
        
        # Compute Zernike decomposition
        Ze = torch.matmul(phaseMap.flatten(1,2), self.invZ)
        
        self.movingCount += 1
         
        return phaseMap, Ze, self.Nphotons, self.RON, self.r0_moving, torch.stack((self.windSpeedVector_x,self.windSpeedVector_y)), self.fractionalr0
    
    def RemovePiston(self, phaseMap):
        mask = self.pupil.unsqueeze(0)  # shape (1, H, W)
        
        masked_mean = (phaseMap * mask).sum(dim=(-2, -1), keepdim=True) / self.pupilSum
        return phaseMap - masked_mean * mask
    
    def CompressAtmosphere(self):
        """
        Compress the multilayer turbulence phase maps into a single wavefront phase.
    
        Returns:
            torch.Tensor: Resulting phase map cropped and projected onto the pupil.
        """
        layeredPhase = torch.fft.fft2(self.movingWavefrontGenerator, dim=(-2, -1), norm="ortho").real
        croppedLayeredPhase = layeredPhase[:, :, :self.Nres, :self.Nres]
        phaseMap = (torch.sqrt(self.fractionalr0) * croppedLayeredPhase).sum(dim=0)
        phaseMap = self.pupil * phaseMap  # Apply pupil mask
        return phaseMap
        
    
    def ResetMovingWavefront(self):
        """
        Reset the temporal evolution state of the moving wavefront generator.
        """
        self.translationPhase = 1.
        self.movingWavefrontGenerator = None
        self.movingCount = 0
        
    def DrawRandomParameters(self):
        """
        Draw new random parameters for generating moving wavefronts.
        Updates r0, L0, correction level, photon/RON noise, fractional layer weights,
        and wind speed vectors for each atmospheric layer.
        """
        self.nLayers = np.random.randint(self.nLayersRange[0], self.nLayersRange[1])
        
        self.r0_moving = torch.empty(self.Nphases, 1, 1, device=self.device).uniform_(*self.r0Range)
        self.L0 = torch.empty(self.Nphases, 1, 1, device=self.device).uniform_(*self.L0Range)
        self.levelOfCorrection = torch.empty(self.Nphases, 1, 1, device=self.device).uniform_(*self.levelOfCorrectionRange)
        self.Nphotons = torch.empty(self.Nphases, 1, 1, device=self.device).uniform_(*self.photonRange)
        self.Nphotons = torch.pow(10, self.Nphotons)
        self.RON = torch.empty(self.Nphases, 1, 1, device=self.device).uniform_(*self.RONRange)

        self.fractionalr0 = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device).uniform_(0., 1.)
        self.fractionalr0 /= torch.sum(self.fractionalr0, dim = 0) 
        
        self.windSpeedVector_x = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device).uniform_(*self.windSpeedRange)
        self.windSpeedVector_y = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device).uniform_(*self.windSpeedRange)
        
    def DrawRandomParameters_ultimate(self):
        """
        Ultimate branch verison of draw random parameters, since
        turbulence statistics have to be constant.
        """
        self.nLayers = self.nLayersRange[0]
        self.r0_moving = torch.zeros(self.Nphases, 1, 1, device=self.device) + self.r0Range[0]
        self.levelOfCorrection = torch.zeros(self.Nphases, 1, 1, device=self.device) + self.levelOfCorrectionRange
        self.Nphotons = torch.empty(self.Nphases, 1, 1, device=self.device).uniform_(*self.photonRange)
        self.Nphotons = torch.pow(10, self.Nphotons)
        self.RON = torch.empty(self.Nphases, 1, 1, device=self.device).uniform_(*self.RONRange)
        
        self.fractionalr0 = torch.tensor(
            [0.7316, 0.0650, 0.0193, 0.0252, 0.0574, 0.0500, 0.0515]
            ).reshape(-1, 1, 1, 1).expand(-1, self.Nphases, 1, 1)
     
        self.windSpeedVector_x = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device).uniform_(*self.windSpeedRange)
        self.windSpeedVector_y = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device).uniform_(*self.windSpeedRange)
        

    def BuildAtmospherePSD(self, generateClosedLoop):
        """
        Construct the atmospheric power spectral density (PSD) for all layers.
    
        Parameters:
            generateClosedLoop (bool): If True, include closed-loop correction modeling.
    
        Returns:
            torch.Tensor: Atmospheric PSD with optional closed-loop correction applied.
        """
        atmosphere_PSD = TorchPropagator.GetAtmospherePSD(self.fsqr_moving, self.dF_moving, self.r0_moving, self.L0, self.f_slope)  # Shape: (Nphases, H, W)
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
        """
        Load a saved moving wavefront dataset from a file and restore internal state.
    
        Parameters:
            file_path (str): Path to the .pth file containing the saved moving wavefront data.
    
        Restores:
            - movingWavefrontGenerator: Function to generate temporal wavefronts.
            - translationPhase: Precomputed translation phase map.
            - Nphotons: Photon count.
            - RON: Read-out noise.
            - r0_moving: Fried parameter for the moving turbulence model.
            - windSpeedVector_x: Wind speeds in the x-direction for each layer.
            - windSpeedVector_y: Wind speeds in the y-direction for each layer.
            - fractionalr0: Relative turbulence strength of each layer.
        """
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
        """
        Initialize a dataset for permanent (precomputed) static wavefront samples.
    
        Parameters:
            file_path (str): Path to the .pth file containing saved dataset.
    
        Loads:
            - inputs (torch.Tensor): Input wavefront sensor measurements or images.
            - outputs (torch.Tensor): Ground truth Zernike coefficients.
            - photons (torch.Tensor): Photon count for each sample.
            - rons (torch.Tensor): Read-out noise values for each sample.
            - r0s (torch.Tensor): Fried parameter for each sample.
        """
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
    
    