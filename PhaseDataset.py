#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:21:44 2025

@author: ptrouve
"""
import torch
from mmengine import Config
from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time
from scipy.io import loadmat


def Zernike(pupil, j = 100):
    """
    Creates the Zernike polynomial basis

    Args:
       pupil (torch array): Aperture of the telescope
       pupil_logical (torch array): Logical values of the pupil for vectorization
       resolution (int): pixels in the diameter
       j (int): Number of zernike modes to use
    Returns:
       out (torch array): 2D Matrix in which each column corresponds to a zernike mode
       outFullRes (torch array): 3D Matrix in which each 2D slice corresponds to a zernike mode

    """

    def zernIndex(j):
        """
        ADAPTED FROM AOTOOLS PACKAGE:https://github.com/AOtools/aotools

        Find the [n,m] list giving the radial order n and azimuthal order
        of the Zernike polynomial of Noll index j.

        Parameters:
            j (int): The Noll index for Zernike polynomials

        Returns:
            list: n, m values
        """
        n = int((-1.0 + np.sqrt(8 * (j - 1) + 1)) / 2.0)
        p = j - (n * (n + 1)) / 2.0
        k = n % 2
        m = int((p + k) / 2.0) * 2 - k

        if m != 0:
            if j % 2 == 0:
                s = 1
            else:
                s = -1
            m *= s

        return [n, m]

    def zernikeRadialFunc(n, m, r):
        """
        ADAPTED FROM AOTOOLS PACKAGE:https://github.com/AOtools/aotools
        Function to calculate the Zernike radial function

        Parameters:
            n (int): Zernike radial order
            m (int): Zernike azimuthal order
            r (ndarray): 2-d array of radii from the centre the array

        Returns:
            ndarray: The Zernike radial function
        """
        try:
            factorial = np.math.factorial
        except:
            import scipy

            factorial = scipy.special.factorial

        R = torch.zeros_like(r)
        # Can cast the below to "int", n,m are always *both* either even or odd
        for i in range(0, int((n - m) / 2) + 1):

            R += (
                r ** (n - 2 * i)
                * (((-1) ** (i)) * factorial(n - i))
                / (
                    factorial(i)
                    * factorial(int(0.5 * (n + m) - i))
                    * factorial(int(0.5 * (n - m) - i))
                )
            )
        return R
    

    pupil_logical = torch.where(pupil.view(-1) > 0)[0]
    resolution = pupil.shape[-1]

    device = pupil.device
    # pupil = pupil.cpu()
    X, Y = torch.where(pupil > 0)

    X = (X - (resolution + resolution % 2 - 1) / 2) / resolution
    Y = (Y - (resolution + resolution % 2 - 1) / 2) / resolution
    R = torch.sqrt(X**2 + Y**2)
    R = R / R.max()
    theta = torch.arctan2(Y, X)
    out = torch.zeros((int(torch.sum(pupil).item()), j), dtype=torch.float32, device = device)
    outFullRes = torch.zeros((resolution**2, j), dtype=torch.float32, device = device)

    for i in range(1, j + 1):
        n, m = zernIndex(i + 1)
        n_t = torch.tensor(n, dtype=torch.float32)

        if m == 0:
            Z = torch.sqrt(n_t + 1) * zernikeRadialFunc(n, 0, R)
        else:
            if m > 0:  # j is even
                Z = (
                    torch.sqrt(2 * (n_t + 1))
                    * zernikeRadialFunc(n, m, R)
                    * torch.cos(m * theta)
                )
            else:  # i is odd
                m = abs(m)
                Z = (
                    torch.sqrt(2 * (n_t + 1))
                    * zernikeRadialFunc(n, m, R)
                    * torch.sin(m * theta)
                )

        Z -= Z.mean()
        Z *= 1 / torch.std(Z)

        # clip
        out[:, i - 1] = Z

        outFullRes[pupil_logical, i - 1] = Z.to(outFullRes.dtype)

    outFullRes = torch.reshape(outFullRes, [resolution, resolution, j])

    return out, outFullRes


def GetSpatialFrequencies(D, resolution, device="cpu"):
    """
    Computes the spatial frequencies for a given diameter and resolution.

    Args:
        D (float): Diameter of the telescope
        resolution (int): Resolution of the telescope

    Returns:
        tuple:
            - dF (float): Frequency step size
            - fx (torch array): Spatial frequency components in the x direction
            - fy (torch array): Spatial frequency components in the y direction
    """
    dF = 1 / (D)
    fx = (
        torch.linspace(
            -resolution / 2,
            resolution / 2 - 1,
            resolution,
            dtype=torch.float32,
            device=device,
        )
        * dF
    )
    [fx, fy] = torch.meshgrid(fx, fx)
    return dF, fx, fy


# def GetAtmospherePSD(fx, fy, dF, r0, L0, pupil, pupilLogical):
def GetAtmospherePSD(fsqr, dF, r0, L0, f_slope=11.0 / 6.0):
    """
    Computes the atmospheric power spectral density (PSD) for phase aberrations based on the spatial frequencies.

    Args:
        fx (torch array): Spatial frequency components in the x direction
        fy (torch array): Spatial frequency components in the y direction
        dF (float): Frequency step size
        r0 (float): Fried parameter (m)
        L0 (float): Outer scale of turbulence (m)
        pupil (torch array): Pupil function of the system
        pupilLogical (torch array): Logical pupil mask indicating valid regions of the pupil

    Returns:
        torch array: Atmospheric power spectral density (PSD) for phase aberrations
    """
    resolution = fsqr.shape[-1]
    l0 = 1e-10  # Default value for the inner scale   ##PTP warning ?
    # fsqr = fx**2 + fy**2
    fm = 5.92 / l0 / (2 * torch.pi)
    # frecuencia de escala interna [1/m]
    f0 = 1 / L0
    # frecuencia de escala externa [1/m]
    PSD_phi = (
        0.023
        * r0 ** (-5 / 3)
        / (fsqr + f0**2) ** (f_slope)
        * dF**2
        * resolution**2
        * torch.exp(-fsqr / fm**2)
    )
    PSD_phi[..., resolution // 2, resolution // 2] = 0
    return PSD_phi


def GetFittingPSD(fx, fy, dF, D, Nactuator, levelOfCorrection=1):
    """
    Computes a fitting power spectral density (PSD) filter, including both low-pass and high-pass components.

    Args:
        fx (torch array): Spatial frequency components in the x direction
        fy (torch array): Spatial frequency components in the y direction
        dF (float): Frequency step size
        D (float): Diameter of the telescope
        Nactuator (int): Number of actuators in the diameter of the deformable mirror
        levelOfCorrection (float, optional): Correction factor for high-pass filter (default is 1)

    Returns:
        torch array: High-pass filter for the fitting PSD
    """
    fc = Nactuator / 2 / D

    low_pass_filter = (fx < fc) & (fy > -fc) & (fy < fc) & (fx > -fc)
    high_pass_filter = 1 - low_pass_filter * levelOfCorrection

    return high_pass_filter


def openLoopTransferFunction(freq, ao_freq, ki, leak, nb_frame_delay):
    """
    Return the temporal open loop transfer function for a integrator controller.
    Source: AOPERA (R. Fetick)
    Parameters
    ----------
    freq : np.array
        Array of temporal frequencies to evaluate the CLTF on.
    ao_freq : float
        The sampling temporal frequency of the AO loop.
    ki : float
        Integrator gain.
    leak : float
        Leaky integrator.
    nb_frame_delay : float
        Number of frame delay.
        Must include: RTC, pixel transfert, DM rise.
        Must not include: WFS integration, DM zero-order-hold.
    """
    z = torch.exp(2j*torch.pi*freq/ao_freq) # it is one method to pass from Tp to z
    not_zero_issue = 1 - 1e-8 # avoid issue to divide by zero at leak/z = 1
    H = ki/(1-not_zero_issue*leak/z) # controler
    H *= 1/z**(nb_frame_delay+1) # delay + WFS + zero order hold
    H *= torch.sinc(freq/ao_freq)

    return H

def closedLoopTransferFunction(*args, **kwargs):
    """
    Return the temporal closed loop transfer function.
    See the open_loop_transfer arguments.
    Source: AOPERA (R. Fetick)
    """
    return 1/(1+openLoopTransferFunction(*args, **kwargs))



def GetTemporalErrorPSD(
    fx, fy, freq, ki, leak, delayFrames, windSpeedVector_x, windSpeedVector_y
):
    """
    Computes the temporal error power spectral density (PSD) given the spatial frequencies and other parameters.

    Args:
        fx (torch array): Spatial frequency components in the x direction
        fy (torch array): Spatial frequency components in the y direction
        dF (float): Frequency step size
        freq (float): Temporal frequency of the system
        delayFrames (int): Number of frames for delay
        windSpeedVector (torch array): Wind speed vector [vx, vy]

    Returns:
        torch array: Temporal error power spectral density
    """
    fx_temporal = fx * windSpeedVector_x + 1e-7
    fy_temporal = fy * windSpeedVector_y + 1e-7

    f_temporal = fx_temporal + fy_temporal

    ETF = closedLoopTransferFunction(f_temporal, freq, ki, leak, delayFrames)
    ETF = torch.abs(ETF) ** 2

    return ETF


class PhaseDataset(Dataset):
    """
    PhaseDataset is a custom PyTorch dataset used to generate synthetic wavefront phase maps 
    and corresponding mode decompositions for training neural networks in adaptive optics systems.

    It supports:
    - Static and dynamic (moving) wavefront generation.
    - Randomized atmospheric parameters such as Fried parameter (r0), outer scale (L0), and wind.
    - Configurable parameters from wavefront sensor (WFS), atmospheric, and control loop settings.
    - PSD-based generation of phase maps using Fourier domain techniques.
    - Automatic generation and caching of static and dynamic test datasets.
    - Computation of mode coefficients for phase map reconstruction and analysis.

    Attributes:
        D (float): Aperture diameter.
        Nres (int): Resolution of the wavefront map (number of pixels).
        Nmodes (int): Number of modes to generate.
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
        z_FullRes (Tensor): Precomputed full-resolution modes.
        invZ (Tensor): Pseudo-inverse of the modes matrix for decomposition.
        testDatasetPath (str): File path for cached static dataset.
        movingTestDatasetPath (str): File path for cached moving dataset.

    Usage:
        dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, device)
        sample = dataset[idx]  # Returns phaseMap, modes coefficients, photons, RON, r0
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
        self.Nmodes = WFSParams['Nmodes']
        self.photonRange = WFSParams['Nphotons']
        self.RONRange = WFSParams['RON']
        self.Nactuator = WFSParams['Nactuator']
                       
        self.L0Range = AtmosParams['L0']
        self.r0Range = AtmosParams['r0']
        self.Nphases = AtmosParams['Nphases']
        self.nLayersRange = AtmosParams["Layers"]
        self.f_slope = AtmosParams["f_slope"]
        self.wavelength = AtmosParams["Wavelength"]
        self.useScintillation = AtmosParams["Scintillation"]
                               
                               
        self.levelOfCorrectionRange = LoopParams['levelOfCorrection']
        self.loopFrequency = LoopParams['loopFrequency']
        self.delayFrames = LoopParams['delayFrames']
        self.loopGainRange = LoopParams['loopGain']
        self.loopLeakRange = LoopParams['loopLeak']
        self.windSpeedRange = LoopParams['windSpeedVector']
     
        self.device=device       

        self.translationPhase = 1.
        self.movingCount = 0

        self.generateClosedLoop = False
        
        self.testDatasetPath = "test_dataset.pth"
        self.movingTestDatasetPath = "moving_test_dataset.pth"
     
  
        x = torch.linspace(-self.Nres/2, self.Nres/2, self.Nres, device = self.device, dtype = torch.float32)                                          # Build the mesh
        [x,y] = torch.meshgrid(x,x, indexing='ij') 
                                       
        self.pupil = (x**2 + y**2) <= ((self.Nres+1)/2)**2
        self.pupilSum = self.pupil.sum()
        self.pupil_logical = torch.where(self.pupil.view(-1,1)>0)

        #  ## Compute some example PSDs
        [self.dF, self.fx, self.fy] = GetSpatialFrequencies(self.D, self.Nres, self.device)
        self.fsqr = self.fx**2 + self.fy**2
        
        upSize = 4 if self.useScintillation else 2
        [self.dF_moving, self.fx_moving, self.fy_moving] = GetSpatialFrequencies(self.D * upSize, self.Nres * upSize, self.device)
        self.fsqr_moving = self.fx_moving**2 + self.fy_moving**2

        if self.useScintillation:
            x = torch.linspace(-self.Nres/2*upSize, self.Nres/2*upSize-1, self.Nres*upSize, device = self.device, dtype = torch.float32)                                          # Build the mesh
            [x,y] = torch.meshgrid(x,x, indexing='ij') 

            self.pupilASP = (x**2 + y**2) <= ((self.Nres*upSize+1)/2.2)**2
            self.masterPropagatorPhase = torch.fft.fftshift(torch.exp(-1j * torch.pi * self.wavelength * self.fsqr_moving), dim = (-2, -1))

    # ## Compute the first Nmodes modes and the inverse to obtain the perfect reconstructor

        if WFSParams['ModalBasis'] == "Zernike":
            [self.z, self.z_FullRes] = Zernike(self.pupil, self.Nmodes)
            
        elif WFSParams['ModalBasis'] == "Papyrus_KL":
            M2C = torch.from_numpy(loadmat(r'C:\Users\foyarzun\Nextcloud\PhD\Code\Python\WFS_CoConception\M2C_KL_OOPAO_synthetic_IF.mat')["M2C_KL"]).to(device = device, dtype = torch.float32)[:,:self.Nmodes]
            papyrus_dm = torch.from_numpy(np.load(r"C:\Users\foyarzun\Nextcloud\PhD\Code\Python\WFS_CoConception\papyrus_dm.npy").astype(np.float32)).to(device=device) * 1e7
            papyrus_modal_dm = (papyrus_dm @ M2C).view(80,80,-1)[1:-1, 1:-1, :]

            self.z_FullRes = papyrus_modal_dm * self.pupil.unsqueeze(-1)
            self.z = self.z_FullRes[self.pupil.bool()]
            
        elif WFSParams['ModalBasis'] == "Papyrus_Zernike":
            Z2C = torch.from_numpy(np.load("Z2C.npy").astype(np.float32)).to(device=device).T[:,:self.Nmodes]
            papyrus_dm = torch.from_numpy(np.load("papyrus_dm.npy").astype(np.float32)).to(device=device) * 1e7
            papyrus_modal_dm = (papyrus_dm @ Z2C).view(80,80,-1)[1:-1, 1:-1, :]

            
            self.z_FullRes = papyrus_modal_dm * self.pupil.unsqueeze(-1)
            self.z = self.z_FullRes[self.pupil.bool()]
            
        elif WFSParams['ModalBasis'] == "Papyrus_Zonal":
            # papyrus_dm = torch.from_numpy(np.load("papyrus_dm.npy").astype(np.float32)).to(device=device) * 1e7
            papyrus_dm = papyrus_dm.view(80,80,-1)[1:-1, 1:-1, :]

            self.z_FullRes = papyrus_dm * self.pupil.unsqueeze(-1)
            self.z = self.z_FullRes[self.pupil.bool()]
        
        elif WFSParams['ModalBasis'] == "Oziriis_KL":
            oziriis_modal_dm = torch.from_numpy(np.load(r'C:\Users\foyarzun\Nextcloud\PostDoc\OZIRIIS\Data\OZIRIIS_KL_90x90.npy')).to(device = device, dtype = torch.float32)[:self.Nmodes]
            oziriis_modal_dm = oziriis_modal_dm.view(-1,90,90).permute(1,2,0)

            self.z_FullRes = oziriis_modal_dm * self.pupil.unsqueeze(-1)
            self.z = self.z_FullRes[self.pupil.bool()]

        elif WFSParams['ModalBasis'] == "Oziriis_Zonal":
            oziriis_zonal_dm = torch.from_numpy(np.load(r'C:\Users\foyarzun\Nextcloud\PostDoc\OZIRIIS\Data\OZIRIIS_Zonal_90x90.npy')).to(device = device, dtype = torch.float32)[:self.Nmodes]
            oziriis_zonal_dm = oziriis_zonal_dm.view(-1,90,90).permute(1,2,0)

            self.z_FullRes = oziriis_zonal_dm * self.pupil.unsqueeze(-1)
            self.z = self.z_FullRes[self.pupil.bool()]

        else:
            raise ValueError(f"Unknown basis: {WFSParams['ModalBasis']}")
            
        
        self.invZ = torch.linalg.pinv(self.z_FullRes.flatten(0,1)).to(self.device, dtype=torch.float32).transpose(0, 1)
        
        
        self.r0_moving = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.L0 = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.levelOfCorrection = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.Nphotons = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.RON = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.nLayers = np.random.randint(*self.nLayersRange)
        self.fractionalr0 = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device)
        self.windSpeedVector_x = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device)
        self.windSpeedVector_y = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device)
        self.loopGain = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.loopLeak = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.layerHeights = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device)
        
        
        
        # if not os.path.exists(self.testDatasetPath):
        #     self.GenerateTestDataSet(10)
            
        # if not os.path.exists(self.movingTestDatasetPath):
        #     self.GenerateMovingTestDataSet()
        
    def __len__(self):
        
        return self.Nphases

    def __getitem__(self, idx):
        if idx == 0:
            self.ResetMovingWavefront()
        return self.GetMovingWavefront(idx)
           
        
    @torch.no_grad() 
    def GetMovingWavefront(self, idx):
        """
        Generate a moving (time-evolving) wavefront phase map.
    
        Parameters:
            generateClosedLoop (bool): Whether to include closed-loop phase modeling.
    
        Returns:
            Tuple of:
                - phaseMap (torch.Tensor): Current phase map.
                - Ze (torch.Tensor): Corresponding mode coefficients.
                - Nphotons (torch.Tensor): Photon count.
                - RON (torch.Tensor): Read-out noise.
                - r0 (torch.Tensor): Fried parameter.
                - wind (torch.Tensor): Wind speed vectors for each layer.
                - fractionalr0 (torch.Tensor): Relative r0 contribution of each layer.
        """
        if idx == 0:
            # Generate batch of random parameters
            self.DrawRandomParameters()    

            # Compute the PSDs in batch mode
            total_PSD, atm_PSD = self.BuildAtmospherePSD()
            
            resolution = total_PSD.shape[-1]
            sqrt_fftshift_PSD = torch.sqrt(torch.fft.fftshift(total_PSD, dim=(-2, -1)))  # FFT shift along spatial dims
            randMap = torch.randn(self.nLayers, self.Nphases, resolution, resolution, dtype=torch.complex64, device=self.device) 
            self.movingWavefrontGenerator = sqrt_fftshift_PSD * randMap 
            
            if self.useScintillation:
                sqrt_fftshift_atm_PSD = torch.sqrt(torch.fft.fftshift(atm_PSD, dim=(-2, -1)))
                self.movingScintillationWavefrontGenerator = sqrt_fftshift_atm_PSD * randMap
       
        if idx == 1:
            phase_factor = (1j * 2 * torch.pi / self.loopFrequency * (self.windSpeedVector_x * self.fx_moving.unsqueeze(0).unsqueeze(0) + self.windSpeedVector_y * self.fy_moving.unsqueeze(0).unsqueeze(0)))
            self.translationPhase = torch.fft.fftshift(torch.exp(phase_factor), dim = (-2, -1))
 
        self.layeredPhase = self.MakeLayersFromGenerator(idx, self.movingWavefrontGenerator)
        phaseMap = self.CompressAtmosphere() 

        if self.useScintillation:
            N = self.layeredPhase.shape[-1]
            if self.generateClosedLoop:
                self.scintillationLayeredPhase = self.MakeLayersFromGenerator(idx, self.movingScintillationWavefrontGenerator)
                pupilMap = self.ComputeScintillation(self.scintillationLayeredPhase).abs()[:, N//2-self.Nres//2:N//2+self.Nres//2, N//2-self.Nres//2:N//2+self.Nres//2]
            else:
                pupilMap = self.ComputeScintillation(self.layeredPhase).abs()[:, N//2-self.Nres//2:N//2+self.Nres//2, N//2-self.Nres//2:N//2+self.Nres//2]
        else:
            pupilMap = self.pupil.repeat(self.Nphases, 1, 1)
        
        # Compute mode decomposition
        Ze = torch.matmul(phaseMap.flatten(1,2), self.invZ)
        
        self.movingCount += 1
         
        return phaseMap, pupilMap, Ze, self.Nphotons, self.RON, self.r0_moving, torch.stack((self.windSpeedVector_x,self.windSpeedVector_y)), self.fractionalr0
    
    def RemovePiston(self, phaseMap):
        mask = self.pupil.unsqueeze(0)  # shape (1, H, W)
        masked_mean = (phaseMap * mask).sum(dim=(-2, -1), keepdim=True) / self.pupilSum
        return phaseMap - masked_mean * mask
    
    def MakeLayersFromGenerator(self, idx, generator):
        layeredPhase = torch.fft.fft2(generator * self.translationPhase ** idx, dim=(-2, -1), norm="ortho").real
        layeredPhase *= torch.sqrt(self.fractionalr0)
        return layeredPhase
    
    def CompressAtmosphere(self):
        """
        Compress the multilayer turbulence phase maps into a single wavefront phase.
    
        Returns:
            torch.Tensor: Resulting phase map cropped and projected onto the pupil.
        """
        N = self.layeredPhase.shape[-1]
        croppedLayeredPhase = self.layeredPhase[:, :, N//2-self.Nres//2:N//2+self.Nres//2, N//2-self.Nres//2:N//2+self.Nres//2]
        phaseMap = croppedLayeredPhase.sum(dim=0)
        phaseMap = self.pupil * phaseMap  # Apply pupil mask
        phaseMap = self.RemovePiston(phaseMap)
        return phaseMap
    
    def ComputeScintillation(self, layeredPhase):

        scintillationSupport = self.pupilASP.repeat(self.Nphases, 1, 1).to(dtype=torch.complex64)

        for i in range(self.nLayers):
            dist = self.layerHeights[i] - self.layerHeights[i + 1] if i < self.nLayers-1 else self.layerHeights[0]
            scintillationSupport = scintillationSupport * torch.exp(1j * layeredPhase[i])
            scintillationSupport = self.ASP(scintillationSupport, dist)
        
        scintillationSupport = scintillationSupport#[:, N//2-self.Nres//2:N//2+self.Nres//2, N//2-self.Nres//2:N//2+self.Nres//2]
        return scintillationSupport
    
    def ResetMovingWavefront(self):
        """
        Reset the temporal evolution state of the moving wavefront generator.
        """
        self.translationPhase = 1.
        self.movingCount = 0
        
    def DrawRandomParameters(self):
        """
        Draw new random parameters for generating moving wavefronts.
        Updates r0, L0, correction level, photon/RON noise, fractional layer weights,
        and wind speed vectors for each atmospheric layer.
        """
        self.nLayers = np.random.randint(*self.nLayersRange)
        self.r0_moving = torch.empty(self.Nphases, 1, 1, device=self.device).uniform_(*self.r0Range)
        self.L0 = torch.empty(self.Nphases, 1, 1, device=self.device).uniform_(*self.L0Range)

        self.levelOfCorrection = torch.empty(self.Nphases, 1, 1, device=self.device).uniform_(*self.levelOfCorrectionRange)
        self.loopGain = torch.empty(self.Nphases, 1, 1, device=self.device).uniform_(*self.loopGainRange)
        self.loopLeak = torch.empty(self.Nphases, 1, 1, device=self.device).uniform_(*self.loopLeakRange)

        self.Nphotons = torch.empty(self.Nphases, 1, 1, device=self.device).uniform_(*self.photonRange)
        self.Nphotons = torch.pow(10, self.Nphotons)
        self.RON = torch.empty(self.Nphases, 1, 1, device=self.device).uniform_(*self.RONRange)

        self.fractionalr0 = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device).uniform_(0., 1.)
        random_to_sort = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device).uniform_(0., 0.5)
        _,index_sorted = torch.sort(self.fractionalr0 + random_to_sort, dim = 0) 
        self.fractionalr0 = torch.gather(self.fractionalr0, dim=0, index=index_sorted)
        self.fractionalr0 /= torch.sum(self.fractionalr0, dim = 0)

        self.layerHeights = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device).exponential_(lambd = 0.001) * 2e0
        self.layerHeights,_ = torch.sort(self.layerHeights, dim = 0, descending = True)
        
        self.windSpeed = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device).uniform_(*self.windSpeedRange)
        self.windSpeedVector_x = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device).uniform_(*[-1,1])
        self.windSpeedVector_y = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device).uniform_(*[-1,1])

        currentIntegratedWindSpeed = torch.sum(self.fractionalr0 *
                                            torch.sqrt(self.windSpeedVector_x ** 2 +
                                                        self.windSpeedVector_y ** 2) ** (5/3), dim = 0) ** (3/5)

        normalization = self.windSpeed / currentIntegratedWindSpeed
        self.windSpeedVector_x = self.windSpeedVector_x * normalization
        self.windSpeedVector_y = self.windSpeedVector_y * normalization
        

     
    def BuildAtmospherePSD(self):
        """
        Construct the atmospheric power spectral density (PSD) for all layers.
    
        Parameters:
            generateClosedLoop (bool): If True, include closed-loop correction modeling.
    
        Returns:
            torch.Tensor: Atmospheric PSD with optional closed-loop correction applied.
        """
        atmosphere_PSD = GetAtmospherePSD(self.fsqr_moving, 
                                                          self.dF_moving, 
                                                          self.r0_moving, 
                                                          self.L0, 
                                                          self.f_slope)  # Shape: (Nphases, H, W)
        
        total_PSD = atmosphere_PSD# * fitting_PSD
        total_PSD = total_PSD.repeat(self.nLayers, 1, 1, 1)
        if not self.generateClosedLoop:
            return total_PSD, total_PSD
        
        fitting_PSD = GetFittingPSD(self.fx_moving,
                                                    self.fy_moving, 
                                                    self.dF_moving, 
                                                    self.D, 
                                                    self.Nactuator, 
                                                    self.levelOfCorrection)  # Shape: (Nphases, H, W)
        
        temporalErrorPSD = GetTemporalErrorPSD(self.fx_moving,
                                                                self.fy_moving,
                                                                self.loopFrequency,
                                                                self.loopGain,
                                                                self.loopLeak, 
                                                                self.delayFrames,
                                                                self.windSpeedVector_x,
                                                                self.windSpeedVector_y)  # Shape: (Nphases, H, W)
        
        total_PSD *= fitting_PSD
        total_PSD += temporalErrorPSD * (1. - fitting_PSD) * atmosphere_PSD
        
        return total_PSD, atmosphere_PSD
  

    def ASP(self, input_field, distance):
        
        field_freq = torch.fft.fft2(torch.fft.ifftshift(input_field, dim = (-2,-1)), dim = (-2,-1))
        field_filtered = field_freq * self.masterPropagatorPhase ** distance
        field_out = torch.fft.fftshift(torch.fft.ifft2(field_filtered, dim = (-2,-1)), dim = (-2,-1))

        return field_out
        
  