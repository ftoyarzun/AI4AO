# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:02:43 2024

@author: pauline : conversion of propagator to torch
"""

import torch
import pylab as plt
import aotools as ao
import math
import numpy as np
np.math = math
import torch.nn as nn
import random

from torch.fft import fft2, fftshift, ifft2, ifftshift

from Constants import double_transmision_masks


from line_profiler import profile

def Zernike(pupil, pupil_logical, resolution, j):
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
     
     
     Warning : requires the librairy aotools which works only on cpu'
     
    """
    pupil = pupil.cpu()
    X, Y = torch.where(pupil > 0)
                        
    X = ( X-(resolution + resolution%2-1)/2 ) / resolution
    Y = ( Y-(resolution + resolution%2-1)/2 ) / resolution
    R = torch.sqrt(X**2 + Y**2)
    R = R/R.max()
    theta = torch.arctan2(Y, X)
    out = torch.zeros([torch.sum(pupil),j],dtype =torch.float32)
    outFullRes = torch.zeros([resolution**2, j],dtype =torch.float32)
    

    for i in range(1, j+1):
        n, m = ao.zernike.zernIndex(i+1)
        n_t = torch.tensor(n, dtype=torch.float32)
       
        
        if m == 0:
            Z = torch.sqrt(n_t+1) * ao.zernike.zernikeRadialFunc(n, 0, R)
        else:
            if m > 0: # j is even
                Z = torch.sqrt(2*(n_t+1)) * ao.zernike.zernikeRadialFunc(n, m, R) * torch.cos(m * theta)
            else:   #i is odd
                m = abs(m)
                Z = torch.sqrt(2*(n_t+1)) * ao.zernike.zernikeRadialFunc(n, m, R) * torch.sin(m * theta)
        
        Z -= Z.mean()
        Z *= (1/torch.std(Z))

        # clip
        out[:, i-1] = Z
        
       
       
        outFullRes[pupil_logical[0], i-1] = Z.to(outFullRes.dtype)
        
    outFullRes = torch.reshape( outFullRes, [resolution, resolution, j] )
    return out, outFullRes

def ZernikeFullView(resolution, modes):
    x = torch.linspace(-1, 1, resolution)
    X, Y = torch.meshgrid(x,x, indexing = "xy")


    R = torch.sqrt(X**2 + Y**2)
    pupil = R < 1

    theta = torch.arctan2(Y, X)
    outFullRes = torch.zeros([len(modes), resolution, resolution],dtype = torch.float32)
    
    index = 0

    for i in modes:
        n, m = ao.zernike.zernIndex(i+1)
        n_t = torch.tensor(n, dtype=torch.float32)
       
        
        if m == 0:
            Z = torch.sqrt(n_t+1) * ao.zernike.zernikeRadialFunc(n, 0, R)
        else:
            if m > 0: # j is even
                Z = torch.sqrt(2*(n_t+1)) * ao.zernike.zernikeRadialFunc(n, m, R) * torch.cos(m * theta)
            else:   #i is odd
                m = abs(m)
                Z = torch.sqrt(2*(n_t+1)) * ao.zernike.zernikeRadialFunc(n, m, R) * torch.sin(m * theta)
        
        Z *= pupil
        Z -= Z[pupil].mean()
        Z *= (1/torch.std(Z[pupil]))

 
       
        outFullRes[index, :, :] = Z.to(outFullRes.dtype)
        index += 1
        

    return outFullRes


def GetSpatialFrequencies(D, resolution, device = "cpu"):
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
    fx = torch.linspace(-resolution/2, resolution/2-1, resolution,dtype=torch.float32, device = device) * dF
    [fx,fy] = torch.meshgrid(fx,fx)
    return dF, fx, fy


# def GetAtmospherePSD(fx, fy, dF, r0, L0, pupil, pupilLogical):
def GetAtmospherePSD(fsqr, dF, r0, L0):
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
    l0 = 1e-10 # Default value for the inner scale   ##PTP warning ?
    # fsqr = fx**2 + fy**2
    fm = 5.92/l0/(2*torch.pi); 		# frecuencia de escala interna [1/m]
    f0 = 1/L0; 			          # frecuencia de escala externa [1/m]
    PSD_phi = 0.023*r0**(-5/3) / (fsqr + f0**2)**(11/6) * dF**2 * resolution**2 * torch.exp(-fsqr/fm**2);
    PSD_phi[..., resolution//2,resolution//2] = 0;
    return PSD_phi

def GetFittingPSD(fx, fy, dF, D, Nactuator, levelOfCorrection = 1):
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
    fc = Nactuator/2/D   
    
    low_pass_filter = (fx < fc) & (fy > -fc) & (fy < fc) & (fx > -fc)
    high_pass_filter = 1 - low_pass_filter * levelOfCorrection
    
    return high_pass_filter
    
def transferFunc(Nd,ki,Tp):
    """
    Computes the closed-loop transfer function for an adaptive optics system.

    Args:
        Nd (float): The total system delay in frames, representing the delay in the loop.
        ki (float): The integrator gain of the adaptive optics control system.
        Tp (float): The Laplace variable.

    Returns:
        f_cl (float): The closed-loop transfer function of the adaptive optics system.
    """
    Hdm = (1-torch.exp(-Tp))/Tp;
    Hwfs = (1-torch.exp(-Tp))/Tp;
    Hdelay = torch.exp(-Nd*Tp);
    Hcorr = ki/ (1 - torch.exp(-Tp))
    f_ol = Hdm*Hwfs*Hdelay*Hcorr;
    f_cl = 1/(1+f_ol);
    n_cl = f_cl*Hdm*Hcorr*Hdelay;
    return f_cl

def GetTemporalErrorPSD(fx, fy, dF, freq, delayFrames, windSpeedVector_x, windSpeedVector_y):
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
    
    f_temporal = torch.sqrt(fx_temporal**2 + fy_temporal**2)
    
    T_delay = delayFrames / freq
    T_integration = 1 / freq
    
    
    return 1 - 2*torch.cos(2 * torch.pi * T_delay * f_temporal) * torch.sinc(T_integration * f_temporal) + torch.sinc(T_integration * f_temporal)**2


def GetMultiplePhaseMapAndZernike(PSD, pupil, pupilLogical, CM, Nphases):
    """
     Creates multiple atmospheric phase screens considering the according PSD and computes the zernike modes corresponding to them

     Args:
        PSD (torch array): Power spectral density to be used
        pupil (torch array): Aperture of the telescope
        invZ (torch array): Inverse of the zernike 2D matrix to compute the true coefficients
        Nphases (int): Number of phases to simulate
     Returns:
        phaseMap (torch tensor): Output phase map dim (Nphasex1xresolutionxresolution)
        
        Ze (torch tensor): True zernike coefficients of the phaseMap dim (Nphasex1xCM.shape[0])
     """
    resolution = PSD.shape[0]
     
    sqrt_fftshift_PSD = torch.sqrt(torch.fft.fftshift(PSD)).to(pupil.device)
    
    randMap = torch.randn(Nphases,resolution,resolution,dtype=torch.float32).to(pupil.device)
   
    
    phaseMap = torch.fft.ifft2(sqrt_fftshift_PSD * torch.fft.fft2(randMap))
    phaseMap = phaseMap.real
    
   
    phaseMap = pupil * phaseMap
   
    Ze = torch.matmul(phaseMap[:, pupil.bool()], CM.transpose(0, 1))
    
 
   
    return phaseMap, Ze


def PoissonNoise(x):
    '''From M. Dufraisse PhD : differentiable Poisson Noise Model using Gaussian approx for each pixel and reparametrization tricks'''
    
    return x + torch.sqrt(torch.clamp(x, min=1e-9)) * torch.randn(x.shape, device=x.device, dtype=x.dtype)



class WFS:
    def __init__(self, ParamsDict, device):
        """
        The wavefront sensor object is in charge of the propagation and reconstruction of the phase aberrations.

        Parameters
        ----------
        resolution : int
            Number of pixels in the diameter of the telescope.
        sampling : int
            Zero padding factor to be used in the fourier transforms.
        diameter : float
            diameter of the telescope.
        Nphotons : int
            number of photons in a single integration of the detector.
        RON : int
            read-out noise in units of electrons per frame per pixel.

        Returns
        -------
        None.

        """

        self.Nres = ParamsDict['Nres']
        self.sampling = ParamsDict['sampling']
        self.Npix = int(self.Nres * self.sampling)
        self.crop_size = self.Npix#2 * self.Nres
        self.D = ParamsDict['D']
        self.useNoise = ParamsDict['useNoise']
        self.device = device
        self.reference_intensity = None
        self.modulation = 0
        self.maskType = ParamsDict['MaskType']
        self.param = ParamsDict['InitParam']
        self.beamSplitProportionForWFSDetector = ParamsDict['beamSplitProportionForWFSDetector']
        
        self.Nphotons = 1e7
        self.RON = 2
        
        x = torch.linspace(-self.Nres/2, self.Nres/2, self.Nres, dtype=torch.float32).to(device)
        [self.x,self.y] = torch.meshgrid(x,x)
        
        x_mask = torch.linspace(-self.Npix/2, self.Npix/2-1, self.Npix, dtype=torch.float32).to(device)
        [self.x_mask,self.y_mask] = torch.meshgrid(x_mask,x_mask)
        
        self.rho_mask = torch.sqrt(self.x_mask ** 2 + self.y_mask ** 2)
        self.abs_x_mask = torch.abs(self.x_mask)
        self.abs_y_mask = torch.abs(self.y_mask)        
                                        
        self.pupil = (self.x ** 2 + self.y ** 2) <= ((self.Nres+1)/2)**2
        self.pupil_logical =  torch.where(self.pupil.reshape(self.Nres * self.Nres) > 0)
        
        
        
        if self.maskType == "Pyramid":
            
            self.BuildPyramidMask()
            
        elif self.maskType == "Zernike":
            self.BuildZernikeMask()
        
           
        elif self.maskType == "Free":
            pass
   
    
    def Propagator(self, phase):
        """
         Simulates the propagation considering a input phase aberration and a phase mask

         Args:
            phase (torch tensor): Input phase aberration dim (NphasesxNresxNres)
         Returns:
            torch tensor: Sensor measurement NphasesxNresxNres
            
         """

        
        pad = int(self.Nres * (self.sampling - 1))//2            

        uin = self.pupil[None, :, :] * torch.exp(1j * phase) / torch.sqrt(self.pupil.sum())
        uin_padded = torch.nn.functional.pad(uin,(pad,pad,pad,pad))       # Pad the pupil 
        
        
      
        ufocal = torch.fft.fft2(torch.fft.fftshift(uin_padded,[-2,-1]))                           # Propagation of the field to the focal plane
        
        # if self.modulation != 0:
        #     nSteps = round(6.28*self.modulation / 4) * 4
        #     frame_no_noise = torch.abs(torch.zeros_like(ufocal))
            
        #     for i in range(nSteps):
        #         modulation_phase = 2 * torch.pi * i / nSteps
        #         modulation_amplitude_in_pixels = self.modulation * self.sampling
        #         pyr_mask_step = torch.pi/4 * (torch.abs(self.x_mask - modulation_amplitude_in_pixels * np.cos(modulation_phase)) + torch.abs(self.y_mask - modulation_amplitude_in_pixels * np.sin(modulation_phase)))
        #         self.SetMask(phaseMask = pyr_mask_step)
        #         upupil_step = torch.fft.fft2(ufocal * torch.fft.fftshift(self.mask))
        #         frame_no_noise += torch.abs(torch.fft.fftshift(upupil_step,[-2,-1]))**2 / nSteps
     
        if self.modulation != 0:
            nSteps = round(6.28*self.modulation / 4) * 4
            frame_no_noise = torch.abs(torch.zeros_like(ufocal))
            self.psf_no_noise = torch.abs(torch.zeros_like(ufocal))
            
            for i in range(nSteps):
                modulation_phase = 2 * torch.pi * i / nSteps
                tip_tilt_mirror_phase = torch.exp(1j * 2 * torch.pi * self.modulation / self.Npix * self.sampling * (self.x_mask * np.cos(modulation_phase) + self.y_mask * np.sin(modulation_phase)))
                ufocal_step = torch.fft.fft2(torch.fft.fftshift(uin_padded * tip_tilt_mirror_phase,[-2,-1]))
                upupil_step = torch.fft.fft2(ufocal_step * torch.fft.fftshift(self.mask))
                frame_no_noise += torch.abs(torch.fft.fftshift(upupil_step,[-2,-1]))**2 / nSteps
                self.psf_no_noise += torch.abs(torch.fft.fftshift(ufocal_step,[-2,-1]))**2 / nSteps
        
        elif self.maskType in double_transmision_masks:
            upupil = torch.fft.fft2(ufocal.unsqueeze(1) * torch.fft.fftshift(self.mask))                        # Multiplication to the phase mask and propagation to the detector
            frame_no_noise = torch.abs(torch.fft.fftshift(upupil,[-2,-1]))**2
            frame_no_noise = self.crop_center(frame_no_noise, self.Nres)
            top_left     = frame_no_noise[:, 0]  # [10, 70, 70]
            top_right    = frame_no_noise[:, 1]
            bottom_left  = frame_no_noise[:, 2]
            bottom_right = frame_no_noise[:, 3]
            
            # Concatenate horizontally and vertically to make 140x140
            top_row    = torch.cat([top_left, top_right], dim=2)    # [10, 70, 140]
            bottom_row = torch.cat([bottom_left, bottom_right], dim=2)  # [10, 70, 140]
            
            # Now concatenate vertically
            frame_no_noise = torch.cat([top_row, bottom_row], dim=1)         # [10, 140, 140]
            
        
        else:
            upupil = torch.fft.fft2(ufocal * torch.fft.fftshift(self.mask))                        # Multiplication to the phase mask and propagation to the detector
            frame_no_noise = torch.abs(torch.fft.fftshift(upupil,[-2,-1]))**2                    # Return the noisy image, normalized the the number of counts

        focal_plane_image = torch.abs(torch.fft.fftshift(ufocal,[-2,-1]))**2
        focal_plane_image /= focal_plane_image.sum(dim=(-2,-1), keepdim=True)
     
        if not self.useNoise:
            normalization_factor_wfs = frame_no_noise.sum(dim=(-2,-1), keepdim=True)
            self.ufocal = focal_plane_image
            return self.crop_center(frame_no_noise / normalization_factor_wfs, self.crop_size)


        
        frame_with_noise = PoissonNoise(frame_no_noise * self.Nphotons * self.beamSplitProportionForWFSDetector) + self.RON * torch.randn_like(frame_no_noise)
        normalization_factor_wfs = frame_with_noise.sum(dim=(-2,-1), keepdim=True)
        
        
        if self.beamSplitProportionForWFSDetector != 1.:
            focal_plane_image_with_noise = PoissonNoise(focal_plane_image * self.Nphotons * (1.-self.beamSplitProportionForWFSDetector) ) + 4 * torch.randn_like(focal_plane_image)
            
            focal_plane_image_with_noise /= focal_plane_image_with_noise.sum(dim=(-2,-1), keepdim=True)
            self.ufocal = focal_plane_image_with_noise
        else:
            self.ufocal = focal_plane_image
        
        
        
        return self.crop_center(frame_with_noise / normalization_factor_wfs, self.crop_size)
    
    
    def Propagator2(self, phase):
        """
         Simulates the propagation considering a input phase aberration and a phase mask

         Args:
            phase (torch tensor): Input phase aberration dim (NphasesxNresxNres)
         Returns:
            torch tensor: Sensor measurement NphasesxNresxNres
            
         """

        
        pad = int(self.Nres * (self.sampling - 1))//2            

        uin = self.pupil.unsqueeze(0) * torch.exp(1j * phase) / torch.sqrt(self.pupil.sum())
        uin_padded = torch.nn.functional.pad(uin,(pad,pad,pad,pad))       # Pad the pupil 
        
        uin_padded = uin_padded.unsqueeze(1)
        
      
        ufocal = fft2(fftshift(uin_padded,[-2,-1]))                           # Propagation of the field to the focal plane
        
        # if self.modulation != 0:
        #     nSteps = round(6.28*self.modulation / 4) * 4
        #     frame_no_noise = torch.abs(torch.zeros_like(ufocal))
            
        #     for i in range(nSteps):
        #         modulation_phase = 2 * torch.pi * i / nSteps
        #         modulation_amplitude_in_pixels = self.modulation * self.sampling
        #         pyr_mask_step = torch.pi/4 * (torch.abs(self.x_mask - modulation_amplitude_in_pixels * np.cos(modulation_phase)) + torch.abs(self.y_mask - modulation_amplitude_in_pixels * np.sin(modulation_phase)))
        #         self.SetMask(phaseMask = pyr_mask_step)
        #         upupil_step = torch.fft.fft2(ufocal * torch.fft.fftshift(self.mask))
        #         frame_no_noise += torch.abs(torch.fft.fftshift(upupil_step,[-2,-1]))**2 / nSteps
     
        if self.modulation != 0:
            nSteps = round(6.28*self.modulation / 4) * 4
            frame_no_noise = torch.abs(torch.zeros_like(ufocal))
            self.psf_no_noise = torch.abs(torch.zeros_like(ufocal))
            
            for i in range(nSteps):
                modulation_phase = 2 * torch.pi * i / nSteps
                tip_tilt_mirror_phase = torch.exp(1j * 2 * torch.pi * self.modulation / self.Npix * self.sampling * (self.x_mask * np.cos(modulation_phase) + self.y_mask * np.sin(modulation_phase)))
                ufocal_step = fft2(fftshift(uin_padded * tip_tilt_mirror_phase,[-2,-1]))
                upupil_step = ifftshift(ifft2(ufocal_step * torch.fft.fftshift(self.mask)))
                frame_no_noise += torch.abs(fftshift(upupil_step,[-2,-1]))**2 / nSteps

        
        elif self.maskType in double_transmision_masks:
            upupil = ifft2(ufocal * fftshift(self.mask))                        # Multiplication to the phase mask and propagation to the detector
            frame_no_noise = torch.abs(fftshift(upupil,[-2,-1]))**2
            frame_no_noise = self.crop_center(frame_no_noise, self.Nres)
            top_left     = frame_no_noise[:, 0]  # [10, 70, 70]
            top_right    = frame_no_noise[:, 1]
            bottom_left  = frame_no_noise[:, 2]
            bottom_right = frame_no_noise[:, 3]
            
            # Concatenate horizontally and vertically to make 140x140
            top_row    = torch.cat([top_left, top_right], dim=2)    # [10, 70, 140]
            bottom_row = torch.cat([bottom_left, bottom_right], dim=2)  # [10, 70, 140]
            
            # Now concatenate vertically
            frame_no_noise = torch.cat([top_row, bottom_row], dim=1)         # [10, 140, 140]
            
        
        else:
            upupil = ifft2(ufocal * fftshift(self.mask))                        # Multiplication to the phase mask and propagation to the detector
            frame_no_noise = torch.abs(fftshift(upupil,[-2,-1]))**2                    # Return the noisy image, normalized the the number of counts

        focal_plane_image = torch.abs(fftshift(ufocal,[-2,-1]))**2
        focal_plane_image /= focal_plane_image.sum(dim=(-2,-1), keepdim=True)
     
        if not self.useNoise:
            normalization_factor_wfs = frame_no_noise.sum(dim=(-2,-1), keepdim=True)
            self.ufocal = focal_plane_image
            return self.crop_center(frame_no_noise / normalization_factor_wfs, self.crop_size)


        
        frame_with_noise = PoissonNoise(frame_no_noise * self.Nphotons * self.beamSplitProportionForWFSDetector) + self.RON * torch.randn_like(frame_no_noise)
        normalization_factor_wfs = frame_with_noise.sum(dim=(-2,-1), keepdim=True)
        
        
        if self.beamSplitProportionForWFSDetector != 1.:
            focal_plane_image_with_noise = PoissonNoise(focal_plane_image * self.Nphotons * (1.-self.beamSplitProportionForWFSDetector) ) + 4 * torch.randn_like(focal_plane_image)
            
            focal_plane_image_with_noise /= focal_plane_image_with_noise.sum(dim=(-2,-1), keepdim=True)
            self.ufocal = focal_plane_image_with_noise
        else:
            self.ufocal = focal_plane_image
        
        
        
        return self.crop_center(frame_with_noise / normalization_factor_wfs, self.crop_size)
    
    
    def SetPhotonsAndRON(self, Nphotons, RON):
        self.Nphotons = Nphotons
        self.RON = RON
    
    
    def crop_center(self, img, crop_size):
        """
        Crops the central 2*Nres pixels from an image.
    
        Parameters:
            img (torch.Tensor): Input image tensor of shape (B, C, Npix, Npix)
            Nres (int): Resolution parameter
            sampling (int): Sampling factor
        
        Returns:
            torch.Tensor: Cropped image of shape (B, C, 2*Nres, 2*Nres)
        """

        center = img.shape[-1] // 2  # Center index
        
        # Compute cropping boundaries
        start = center - (crop_size // 2)
        end = center + (crop_size // 2)
    
        # Crop the image
        return img[..., start:end, start:end]
    
    def GetPSF(self, phase):
        """
        Computes the Point Spread Function (PSF) for a given phase aberration.
    
        Args:
            phase (complex torch tensor): Input phase aberration
        Returns:
            torch tensor: Point Spread Function (PSF) in the focal plane
        """
        pad = int(self.Nres * (self.sampling - 1))//2              

        uin = self.pupil[None, :, :] * torch.exp(1j * phase) / torch.sqrt(self.pupil.sum())
        uin_padded = torch.nn.functional.pad(uin,(pad,pad,pad,pad))       # Pad the pupil 
        
        
      
        ufocal = torch.fft.fft2(torch.fft.fftshift(uin_padded,[-2,-1]))               # Pad the pupil 
        return torch.abs(torch.fft.fftshift(ufocal))**2     # Propagation of the field to the focal plane

    
    
    def SetMask(self,phaseMask = None, transmisionMask = None):
        """
        Sets the phase mask by converting the input mask to a complex exponential and normalizing it.
    
        Args:
            phaseMask (torch tensor): Input phase mask (real-valued)
            transmisionMask (torch tensor): Input transmision mask (real-valued)
        Returns:
            None
        """
        if phaseMask is not None:
            self.mask = torch.ones_like(phaseMask, dtype=torch.cfloat)  / self.Npix ** 2

        elif transmisionMask is not None:  
            self.mask = torch.ones_like(transmisionMask)  / self.Npix ** 2
        
        
        if phaseMask is not None:
            self.mask *= torch.exp(1j * phaseMask)

        if transmisionMask is not None:  
            self.mask *= transmisionMask

        
        
        
    def BuildZernikeMask(self):
       """
       Builds a Zernike mask and sets it using the SetMask function.
   
       Args:
           dot_diameter (float): Diameter of the dot in units of lambda/d
           dot_depth (float): Depth of the dot in radians
       Returns:
           None
       """

       
       
       diameter_in_pixels = self.param[0] * self.sampling
       
       # this line is not differentiable I use a tanh function to model the mask
       
       #zernike_mask = self.param[1] * (rho < diameter_in_pixels / 2.)
       slope = self.param[2]
      
       ring_mask = torch.tanh(slope*( diameter_in_pixels/2. -self.rho_mask))/2
       annular = ring_mask+0.5

       
       zernike_mask = self.param[1] * annular
       
       
       
       self.SetMask(zernike_mask)
        
    

    def BuildPyramidMask(self):
        """
        Builds a pyramid phase mask and sets it using the SetMask function.
    
        Args:
            None
        Returns:
            None
        """

        pyramid_mask = (self.abs_x_mask * self.param[0] + self.abs_y_mask * self.param[1])
        
        self.SetMask(pyramid_mask)
    
    
    
    def BuildReferenceIntensity(self):
        """
        Builds the reference intensity by propagating a zero-phase aberration.
    
        Args:
            None
        Returns:
            None
        """
        tempUseNoise = self.useNoise
        self.useNoise = False
        self.reference_intensity = self.Propagator(torch.zeros((1, self.Nres, self.Nres), dtype=torch.float32, device = self.device))
        self.reference_intensity= self.reference_intensity.squeeze() 
        self.useNoise = tempUseNoise
    
    def BuildReconstructionMatrix(self, modes, batch_size=30, phaseOffset = 0):
        """
        Builds the reconstruction matrix by computing the signals for each mode using finite differences.
    
        Args:
            modes (torch tensor): Modes (3D array with shape (Npix, Npix, Nmodes)) representing different phase aberrations
            mask (torch tensor): Phase mask used in the propagation (not directly used in this function)
        Returns:
            None
        """
        tempUseNoise = self.useNoise
        self.useNoise = False
        delta = 1e-5
        
        Nmodes = modes.shape[0]
        iMat_parts = []
        
        for i in range(0, Nmodes, batch_size):
            modes_batch = modes[i:i + batch_size]  # (Npix^2, batch_size)
            
            # reshape to (1, Npix, Npix, batch_size) if needed by Propagator
            push = self.Propagator(modes_batch * delta + phaseOffset)
            pull = self.Propagator(-modes_batch * delta + phaseOffset)
            
            signal = (push - pull) / (2. * delta)
            signal_flat = signal.flatten(start_dim=-2)  # shape: (batch_size, Npix^2)
    
            iMat_parts.append(signal_flat)

        self.iMat = torch.cat(iMat_parts, dim=0)  # shape: (Nmodes, Npix^2)
            
        self.useNoise = tempUseNoise
        self.reconstructionMatrix = torch.linalg.pinv(self.iMat)
        
        
    # def BuildReconstructionMatrix(self, modes):
    #     """
    #     Builds the reconstruction matrix by computing the signals for each mode using finite differences.
    
    #     Args:
    #         modes (torch tensor): Modes (3D array with shape (Npix, Npix, Nmodes)) representing different phase aberrations
    #         mask (torch tensor): Phase mask used in the propagation (not directly used in this function)
    #     Returns:
    #         None
    #     """
    #     tempUseNoise = self.useNoise
    #     self.useNoise = False
    #     delta = 1e-5
        
        
        
    #     push = self.Propagator(modes * delta)
    #     pull = self.Propagator(-modes * delta)
        
    #     signal = (push - pull) / 2. / delta
        
    #     self.iMat = signal.flatten(start_dim = -2)
            
        
    #     self.useNoise = tempUseNoise
    #     self.reconstructionMatrix = torch.linalg.pinv(self.iMat)
        
    def GetReconstructedPhase(self, intensity):
        """
        Reconstructs the phase aberration from the intensity measurement by applying the reconstruction matrix.
    
        Args:
            intensity (torch tensor): Measured intensity (with noise, if applicable)
        Returns:
            torch tensor: Reconstructed phase aberration
        """
            
        reduced_intensity = intensity - self.reference_intensity
        
        # print(f"self.reference_intensity.shape = {self.reference_intensity.shape}")
        # print(f"intensity.shape = {intensity.shape}")
  
        temp = torch.matmul(reduced_intensity.flatten(start_dim = -2), self.reconstructionMatrix)  
       
        return temp

#%% Set general parameter and build classes
if __name__ == "__main__":
    
    plt.close('all')
    ## WFS parameters
    Nres = 50                                                                       # Number of pixels in the aperture of the telescope
    sampling = 4                                                                   # Zero-padding factor (2 is Shannon)
    Npix = Nres * sampling                                                          # Total number of pixels
    D = 1                                                                           # Telescope diameter (m)
    Nphotons = 1e7                                                                  # Number of photons in measurement    
    RON = 0                                                                         # Read-out noise in photons per pixel per frame
    Nzernike = 50                                                                   # Number of Zernike modes to reconstruct
    Nactuator = 10                                                                  # Number of actuators across the diameter of the DM    
    useNoise  = False                                                                # Add Noise or not
                                                                
    maskType = 'Pyramid'                                                            # Type of mask Pyramidal or Zernike                                                             
    ## Atmosphere parameters
    r0 = 0.15                                                                       # Fried parameter (m)
    l0 = 1e-10                                                                      # Inner scale (m)
    L0 = 20                                                                         # Outter scale (m)
    Nphases = 30                                                                # Number of independent phase screens to simulate
    
    
    ## Loop parameters
    loopFrequency = 1000
    delayFrames = 1
    windSpeedVector = torch.tensor([5,10])
    
    ## device ('cuda' or 'cpu')
    
    device  = 'cuda'
    
    ## intialization of the parameter vector 
    param = torch.tensor([0.78,0.78],dtype=torch.float32)
    ## Generate the wfs object
    wfs = WFS(Nres, sampling, D, Nphotons, RON,useNoise,param,maskType,device)
    wfs.modulation = 5
    wfs.BuildReferenceIntensity()
    ## By default this generates the mask for the pyramid waferont sensor. Use the method wfs.SetMask(mask) to change to the desired mask
    
    
    ## Compute the first Nzernike Zernike polynomials and the inverse to obtain the perfect reconstructor
    #Warning : requires the librairy aotools which works only on cpu'
    
    
    [z, z_FullRes] = Zernike(wfs.pupil.cpu(), wfs.pupil_logical, wfs.Nres, Nzernike)
  
   
    invZ = torch.linalg.pinv(z).to(device)
   
    z_FullRes =z_FullRes.to(device)
    # ## Build the reconstruction matrix
    wfs.BuildReconstructionMatrix(z_FullRes, wfs.mask)
    
    
    # ## Compute some example PSDs
    [dF, fx, fy] = GetSpatialFrequencies(D, Nres)
    atmosphere_PSD = GetAtmospherePSD(fx, fy, dF, r0, L0, wfs.pupil, wfs.pupil_logical)
    fitting_PSD = GetFittingPSD(fx, fy, dF, D, Nactuator, 1)
    #temporalErrorPSD = GetTemporalErrorPSD(fx, fy, dF, loopFrequency, delayFrames, windSpeedVector)  
    
    
    # #%% This section is just to test how the images look like, what would be the perfect estimation and what is the phase estimation using a linear reconstructor 
  
    [outPhaseMap_test, outZe_test] = GetMultiplePhaseMapAndZernike(atmosphere_PSD, wfs.pupil, wfs.pupil_logical, invZ, Nphases)  
   
    test_frame = wfs.Propagator(outPhaseMap_test)
    
    
    ## Set initial data for figures
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Full atmospheric turbulence', fontsize=16)
    img = axes[0].imshow(test_frame[0,:,:].cpu().detach().numpy())
    line1, = axes[1].plot(outZe_test[0,:].cpu())
    line2, = axes[1].plot(outZe_test[0,:].cpu())
    axes[1].legend(['Ground truth', 'Linear reconstructor'])
    
    ## run through the small dataset to observe the system in action
    test_frame = wfs.Propagator(outPhaseMap_test)
    estimated_phase = wfs.GetReconstructedPhase(test_frame)
    
    for ii in  range(Nphases):
      
        plt.figure(1)
        plt.subplot(1,2,1)
        img.set_data(test_frame[ii,:,:].cpu().detach().numpy())

        plt.subplot(1,2,2)
        line1.set_ydata(outZe_test[ii,:].cpu().detach().numpy())
        line2.set_ydata(estimated_phase.squeeze()[ii,:].cpu().detach().numpy())
        plt.xlabel('Zernike mode index')
        plt.ylabel('Zernike mode Amplitude')
        plt.pause(0.1)
        plt.show()
        
    [outPhaseMap_test, outZe_test] = GetMultiplePhaseMapAndZernike(atmosphere_PSD * fitting_PSD, wfs.pupil, wfs.pupil_logical, invZ, Nphases)
    fig.suptitle('Residual turbulence after the AO loop', fontsize=16)
    
    test_frame = wfs.Propagator(outPhaseMap_test)
    estimated_phase = wfs.GetReconstructedPhase(test_frame)
    
    for ii in  range(Nphases):
        
        plt.figure(1)
        plt.subplot(1,2,1)
        img.set_data(test_frame[ii,:,:].cpu().detach().numpy())
        plt.subplot(1,2,2)
        line1.set_ydata(outZe_test[ii,:].cpu())
        line2.set_ydata(estimated_phase[ii,:].cpu().detach().numpy())
        plt.xlabel('Zernike mode index')
        plt.ylabel('Zernike mode Amplitude')
        plt.pause(0.1)
        plt.show()
      
    # #%% Generate datasets
    
    # Ndataset_each = 500
    # ## Generate full turbulence data
    # ## We have to simulate as many different atmospheric conditions as possible. To do this, we have to change r0 and L0
    # ## r0 could have values like torch.logspace(-2,-0.5,5)
    # ## L0 could have values like torch.linspace(20,40)
    # ## Some examples...
    # #atmosphere_PSD = GetAtmospherePSD(fx, fy, dF, 0.01, 30, wfs.pupil, wfs.pupil_logical)
    # #atmosphere_PSD = GetAtmospherePSD(fx, fy, dF, 0.05, 40, wfs.pupil, wfs.pupil_logical)
    # #atmosphere_PSD = GetAtmospherePSD(fx, fy, dF, 0.1, 20, wfs.pupil, wfs.pupil_logical)
    # #atmosphere_PSD = GetAtmospherePSD(fx, fy, dF, 0.2, 30, wfs.pupil, wfs.pupil_logical)
    
    # [outPhaseMap_fullTurbulence, outZe_fullTurbulence] = GetMultiplePhaseMapAndZernike(atmosphere_PSD, wfs.pupil, wfs.pupil_logical, invZ, Ndataset_each) 
    # ## Generate perfect correction data
    # ## We have to test different levels of correction
    # ## Some examples...
    # ## fitting_PSD = GetFittingPSD(fx, fy, dF, D, Nactuator, 0)
    # ## fitting_PSD = GetFittingPSD(fx, fy, dF, D, Nactuator, 0.5)
    # ## fitting_PSD = GetFittingPSD(fx, fy, dF, D, Nactuator, 1)
    # [outPhaseMap_correction, outZe_correction] = GetMultiplePhaseMapAndZernike(atmosphere_PSD*fitting_PSD, wfs.pupil, wfs.pupil_logical, invZ, Ndataset_each) 


   # Check of the phase mask compared to the numpy version
 
    # plt.figure()
    # plt.imshow(wfs.mask.real.cpu().data.numpy())
    # plt.show()

   



