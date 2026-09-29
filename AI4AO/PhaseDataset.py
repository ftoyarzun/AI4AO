#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:21:44 2025

@author: foyarzun
"""
import torch # type: ignore[import]
from torch.utils.data import Dataset # type: ignore[import]
import numpy as np
import math

from .Utils import MakePupil
# The PSD and transfer-function helpers live in PSD.py (shared with PSFDataset);
# they are re-exported here so existing `from AI4AO.PhaseDataset import ...` keep working.
from .PSD import (  # noqa: F401
    GetSpatialFrequencies,
    GetAtmospherePSD,
    GetFittingPSD,
    openLoopTransferFunction,
    closedLoopTransferFunction,
    GetTemporalErrorPSD,
    CenterCrop,
    DrawAOParameters,
)



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

        factorial = math.factorial

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
    out = torch.zeros((j, int(torch.sum(pupil).item())), dtype=torch.float32, device = device)
    outFullRes = torch.zeros((j, resolution**2), dtype=torch.float32, device = device)

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
        out[i - 1] = Z

        outFullRes[i - 1, pupil_logical] = Z.to(outFullRes.dtype)

    outFullRes = torch.reshape(outFullRes, [j, resolution, resolution])

    return out, outFullRes


class PhaseDataset(Dataset):
    """
    PhaseDataset is a custom PyTorch dataset used to generate synthetic wavefront phase maps 
    and corresponding mode decompositions for training neural networks in adaptive optics systems.

    It supports:
    - Static and dynamic (moving) wavefront generation.
    - Randomized atmospheric parameters such as Fried parameter (r0), outer scale (L0), and wind.
    - Configurable parameters from wavefront sensor (WFS), atmospheric, and control loop settings.
    - PSD-based generation of phase maps using Fourier domain techniques.

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
        testDatasetPath (str): File path for cached static dataset.
        movingTestDatasetPath (str): File path for cached moving dataset.

    Usage:
        dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, device)
        sample = dataset[idx]  # Returns opdMap (meters), modes coefficients, photons, RON, r0
    """
    def __init__(self, WFSParams, AtmosParams, LoopParams, DMParams, device):
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
        self.photonRange = WFSParams['Nphotons']
        self.RONRange = WFSParams['RON']
        self.wavelength = WFSParams["Wavelength"]
        self.wavenumber = 2 * torch.pi / self.wavelength

        # Wavelength at which r0 is conventionally quoted (standard AO literature value).
        # Used to convert the Kolmogorov phase PSD (rad^2, defined at this wavelength)
        # into an OPD PSD (m^2), which is wavelength-independent.
        self.referenceWavelength = 500e-9

        self.L0Range = AtmosParams['L0']
        self.r0Range = AtmosParams['r0']
        self.Nphases = AtmosParams['Nphases']
        self.nLayersRange = AtmosParams["Layers"]
        self.f_slope = AtmosParams["f_slope"]
        self.useScintillation = AtmosParams["Scintillation"]
                                           
        self.levelOfCorrectionRange = LoopParams['levelOfCorrection']
        self.loopFrequency = LoopParams['loopFrequency']
        self.delayFrames = LoopParams['delayFrames']
        self.loopGainRange = LoopParams['loopGain']
        self.loopLeakRange = LoopParams['loopLeak']
        self.windSpeedRange = LoopParams['windSpeedVector']

        self.Nactuator = DMParams['Nactuator']
     
        self.device=device       

        self.translationPhase = 1.
        self.movingCount = 0

        self.height_exp_dist_lambda = 0.0003

        self.generateClosedLoop = False
        
        self.testDatasetPath = "test_dataset.pth"
        self.movingTestDatasetPath = "moving_test_dataset.pth"
     
  
        self.pupil = MakePupil(self.Nres, self.device)
        self.pupilSum = self.pupil.sum()
        self.pupil_logical = torch.where(self.pupil.view(-1,1)>0)

        #  ## Compute some example PSDs
        [self.dF, self.fx, self.fy] = GetSpatialFrequencies(self.D, self.Nres, self.device)
        self.fsqr = self.fx**2 + self.fy**2
        
        upSize = 4 if self.useScintillation else 2
        [self.dF_moving, self.fx_moving, self.fy_moving] = GetSpatialFrequencies(self.D * upSize, self.Nres * upSize, self.device)
        self.fsqr_moving = self.fx_moving**2 + self.fy_moving**2

        if self.useScintillation:
            self.pupilASP = MakePupil(self.Nres * upSize, self.device, Rpx=(self.Nres * upSize + 1) / 2.2)
            self.masterPropagatorPhase = torch.fft.fftshift(torch.exp(-1j * torch.pi * self.wavelength * self.fsqr_moving), dim = (-2, -1))

        
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
                - opdMap (torch.Tensor): Current OPD map (meters).
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
            randMap = torch.randn(self.nLayers, self.Nphases, resolution, resolution, dtype=torch.complex64, device=self.device) * math.sqrt(2)
            self.movingWavefrontGenerator = sqrt_fftshift_PSD * randMap 
            
            if self.useScintillation:
                sqrt_fftshift_atm_PSD = torch.sqrt(torch.fft.fftshift(atm_PSD, dim=(-2, -1)))
                self.movingScintillationWavefrontGenerator = sqrt_fftshift_atm_PSD * randMap
       
        if idx == 1:
            phase_factor = (1j * 2 * torch.pi / self.loopFrequency * (self.windSpeedVector_x * self.fx_moving.unsqueeze(0).unsqueeze(0) + self.windSpeedVector_y * self.fy_moving.unsqueeze(0).unsqueeze(0)))
            self.translationPhase = torch.fft.fftshift(torch.exp(phase_factor), dim = (-2, -1))
 
        self.layeredOPD = self.MakeLayersFromGenerator(idx, self.movingWavefrontGenerator)
        opdMap = self.CompressAtmosphere()

        if self.useScintillation:
            if self.generateClosedLoop:
                self.scintillationLayeredOPD = self.MakeLayersFromGenerator(idx, self.movingScintillationWavefrontGenerator)
                pupilMap = CenterCrop(self.ComputeScintillation(self.scintillationLayeredOPD).abs(), self.Nres)
            else:
                pupilMap = CenterCrop(self.ComputeScintillation(self.layeredOPD).abs(), self.Nres)
        else:
            pupilMap = self.pupil.repeat(self.Nphases, 1, 1)

        # Compute mode decomposition
        # Ze = torch.matmul(opdMap.flatten(1,2), self.invZ)

        self.movingCount += 1

        return {
                "opd": opdMap,
                "pupil": pupilMap,
                "nphotons": self.Nphotons,
                "ron": self.RON,
                "r0": self.r0_moving.squeeze(),
                "L0": self.L0.reshape(-1),
                "wind": torch.stack((self.windSpeedVector_x, self.windSpeedVector_y)).squeeze(),
                "fractional_r0": self.fractionalr0.squeeze(),
                "loop_gain": self.loopGain.reshape(-1, 1),
                "loop_leak": self.loopLeak.reshape(-1, 1),
                "level_of_correction": self.levelOfCorrection.reshape(-1, 1),
                }
    
    def RemovePiston(self, opdMap):
        mask = self.pupil.unsqueeze(0)  # shape (1, H, W)
        masked_mean = (opdMap * mask).sum(dim=(-2, -1), keepdim=True) / self.pupilSum
        return opdMap - masked_mean * mask

    def MakeLayersFromGenerator(self, idx, generator):
        layeredOPD = torch.fft.fft2(generator * self.translationPhase ** idx, dim=(-2, -1), norm="ortho").real
        layeredOPD *= torch.sqrt(self.fractionalr0)
        return layeredOPD

    def CompressAtmosphere(self):
        """
        Compress the multilayer turbulence OPD maps into a single wavefront OPD.

        Returns:
            torch.Tensor: Resulting OPD map (meters) cropped and projected onto the pupil.
        """
        croppedLayeredOPD = CenterCrop(self.layeredOPD, self.Nres)
        opdMap = croppedLayeredOPD.sum(dim=0)
        opdMap = self.pupil * opdMap  # Apply pupil mask
        opdMap = self.RemovePiston(opdMap)
        return opdMap

    def ComputeScintillation(self, layeredOPD):

        scintillationSupport = self.pupilASP.repeat(self.Nphases, 1, 1).to(dtype=torch.complex64)

        for i in range(self.nLayers):
            dist = self.layerHeights[i] - self.layerHeights[i + 1] if i < self.nLayers-1 else self.layerHeights[-1]
            # Convert this layer's OPD (meters) to phase (radians) at the sensing wavelength
            # before it can act as a complex-field phasor.
            scintillationSupport = scintillationSupport * torch.exp(1j * self.wavenumber * layeredOPD[i])
            scintillationSupport = self.ASP(scintillationSupport, dist)
        
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
        params = DrawAOParameters(
            self.Nphases, self.nLayersRange, self.r0Range, self.L0Range,
            self.levelOfCorrectionRange, self.loopGainRange, self.loopLeakRange,
            self.photonRange, self.RONRange, self.windSpeedRange,
            self.height_exp_dist_lambda, self.device,
        )
        self.aoParameters = params  # canonical shapes, see DrawAOParameters

        def column(t):
            # (B,) -> (B, 1, 1) and (L, B) -> (L, B, 1, 1), broadcasting against (..., H, W)
            return t.reshape(*t.shape, 1, 1)

        self.nLayers = params["fractional_r0"].shape[0]
        self.r0_moving = column(params["r0"])
        self.L0 = column(params["L0"])
        self.levelOfCorrection = column(params["level_of_correction"])
        self.loopGain = column(params["loop_gain"])
        self.loopLeak = column(params["loop_leak"])
        self.Nphotons = column(params["nphotons"])
        self.RON = column(params["ron"])
        self.fractionalr0 = column(params["fractional_r0"])
        self.layerHeights = column(params["layer_heights"])
        self.windSpeed = column(params["wind_speed"])
        self.windSpeedVector_x = column(params["wind"][0])
        self.windSpeedVector_y = column(params["wind"][1])

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
                                                          self.f_slope)  # Shape: (Nphases, H, W), rad^2 units at self.referenceWavelength
        # Turbulence-induced OPD is wavelength-independent; rescale the rad^2 phase PSD
        # (implicitly quoted at self.referenceWavelength) into an OPD^2 (m^2) PSD.
        atmosphere_PSD = atmosphere_PSD * (self.referenceWavelength / (2 * torch.pi)) ** 2

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
        
        field_freq = torch.fft.fft2(torch.fft.fftshift(input_field, dim = (-2,-1)), dim = (-2,-1))
        field_filtered = field_freq * self.masterPropagatorPhase ** distance
        field_out = torch.fft.ifftshift(torch.fft.ifft2(field_filtered, dim = (-2,-1)), dim = (-2,-1))

        return field_out
        
  