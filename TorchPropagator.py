# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:02:43 2024

@author: franc
"""

import torch
import pylab as plt
import aotools as ao
import numpy as np
import torch.nn as nn
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
     
    X, Y = torch.where(pupil > 0)
                        
    X = ( X-(resolution + resolution%2-1)/2 ) / resolution
    Y = ( Y-(resolution + resolution%2-1)/2 ) / resolution
    R = torch.sqrt(X**2 + Y**2)
    R = R/R.max()
    theta = torch.arctan2(Y, X)
    out = torch.zeros([torch.sum(pupil),j])
    outFullRes = torch.zeros([resolution**2, j])
    

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
       
        outFullRes[pupil_logical[0], i-1] = Z.float()
        
    outFullRes = torch.reshape( outFullRes, [resolution, resolution, j] )
    return out, outFullRes


def GetSpatialFrequencies(D, resolution):
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
    fx = torch.linspace(-resolution/2, resolution/2-1, resolution) * dF
    [fx,fy] = torch.meshgrid(fx,fx)
    return dF, fx, fy


def GetAtmospherePSD(fx, fy, dF, r0, L0, pupil, pupilLogical):
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
    resolution = fx.shape[0]
    l0 = 1e-10 # Default value for the inner scale   ##PTP warning ?
    f = torch.sqrt(fx**2 + fy**2)
    fm = 5.92/l0/(2*torch.pi); 		# frecuencia de escala interna [1/m]
    f0 = 1/L0; 			          # frecuencia de escala externa [1/m]
    PSD_phi = 0.023*r0**(-5/3) * torch.exp(-(f/fm)**2) / (f**2 + f0**2)**(11/6) * dF**2 * resolution**2;
    PSD_phi[resolution//2,resolution//2] = 0;
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

def GetTemporalErrorPSD(fx, fy, dF, freq, delayFrames, windSpeedVector):
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
    fx_temporal = fx * windSpeedVector[0] + 1e-7
    fy_temporal = fy * windSpeedVector[1] + 1e-7
    
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
    
    randMap = torch.randn(Nphases,resolution,resolution).to(pupil.device)
   
    
    phaseMap = torch.fft.ifft2(sqrt_fftshift_PSD * torch.fft.fft2(randMap))
    phaseMap = phaseMap.real
    
   
    phaseMap = pupil * phaseMap
   
    Ze = torch.matmul(phaseMap[:,pupil],CM.transpose(0,1))
   
    return phaseMap, Ze


def PoissonNoise(x):
    '''From M. Dufraisse PhD : differentiable Poisson Noise Model using Gaussian approx for each pixel and reparametrization tricks'''
    
    return x + torch.sqrt(torch.clamp(x, min=1e-9)) * torch.randn(x.shape, device=x.device, dtype=x.dtype)



class WFS:
    def __init__(self, resolution, sampling, diameter, Nphotons, RON, useNoise,param,device):
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
        self.Nres = resolution
        self.sampling = sampling
        self.Npix = resolution * sampling
        self.D = diameter
        self.Nphotons = Nphotons
        self.RON = RON
        self.useNoise = useNoise  ### changed to a setting parameter
        self.device = device
        x = torch.linspace(-self.Nres/2, self.Nres/2, self.Nres, dtype=torch.float64).to(device)
                                         # Build the mesh
        [x,y] = torch.meshgrid(x,x)                                        
        self.pupil = (x**2 + y**2) <= ((self.Nres+1)/2)**2
        self.pupil_logical =  torch.where(self.pupil.reshape(self.Nres * self.Nres) > 0)
        
        self.param = param

        self.BuildPyramidMask()
        
    def Propagator(self, phase):
        """
         Simulates the propagation considering a input phase aberration and a phase mask

         Args:
            phase (torch tensor): Input phase aberration dim (NphasesxNresxNres)
         Returns:
            torch tensor: Sensor measurement NphasesxNresxNres
            
         """
         
         # TODO : modify it to make it work with a batch of phase

        
        pad = self.Nres * (self.sampling - 1)//2
        
        uin = self.pupil[None, :, :] * torch.exp(1j * phase).to(self.device)
        uin_padded = torch.nn.functional.pad(uin,(pad,pad,pad,pad))       # Pad the pupil 
       
       
        ufocal = torch.fft.fft2(torch.fft.fftshift(uin_padded,[1,2]))                           # Propagation of the field to the focal plane
          
        upupil = torch.fft.fft2(ufocal * torch.fft.fftshift(self.mask[None, :, :] ))                        # Multiplication to the phase mask and propagation to the detector
        frame_no_noise = torch.abs(torch.fft.fftshift(upupil,[1,2]))**2                    # Return the noisy image, normalized the the number of counts

     
        if not self.useNoise:
            normalization_factor = frame_no_noise.sum(1).sum(1)
            return frame_no_noise / normalization_factor[:,None,None]
        
        normalization_factor = frame_no_noise.sum(1).sum(1)
        
        frame_no_noise = frame_no_noise / normalization_factor[:,None,None] * self.Nphotons    # Set the number of photons in the frame
        frame_with_noise = PoissonNoise(frame_no_noise) + self.RON * torch.randn(*frame_no_noise.shape).to(frame_no_noise.device)
        normalization_factor = frame_with_noise.sum(1).sum(1)
        
        return frame_with_noise / normalization_factor[:,None,None]
    
    def GetPSF(self, phase):
        """
        Computes the Point Spread Function (PSF) for a given phase aberration.
    
        Args:
            phase (complex torch tensor): Input phase aberration
        Returns:
            torch tensor: Point Spread Function (PSF) in the focal plane
        """
        uin = self.pupil[None, :, :] * torch.exp(1j * phase)
        pad = self.Nres * (self.sampling - 1)//2
        uin_padded = torch.nn.functional(uin, (pad,pad,pad,pad))                # Pad the pupil 
        return torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(uin_padded,[1,2]))))**2     # Propagation of the field to the focal plane


    
    
    def SetMask(self,mask):
        """
        Sets the phase mask by converting the input mask to a complex exponential and normalizing it.
    
        Args:
            mask (torch tensor): Input phase mask (real-valued)
        Returns:
            None
        """
     
        phase_mask = torch.exp(1j * mask)
        phase_mask = phase_mask/ torch.sum(abs(phase_mask))
        self.mask = phase_mask
        self.BuildReferenceIntensity()
        
    

    def BuildPyramidMask(self):
        """
        Builds a pyramid phase mask and sets it using the SetMask function.
    
        Args:
            None
        Returns:
            None
        """
        x_mask = torch.linspace(-self.Npix/2, self.Npix/2-1, self.Npix).to(self.device)
        [x_mask,y_mask] = torch.meshgrid(x_mask,x_mask) 

        pyramid_mask = (torch.pi/2 +self.param[1] ) * (torch.abs(x_mask) + torch.abs(y_mask))
        
        self.SetMask(pyramid_mask+ self.param[0])
    
    
    
    def BuildReferenceIntensity(self):
        """
        Builds the reference intensity by propagating a zero-phase aberration.
    
        Args:
            None
        Returns:
            None
        """
        self.useNoise = False
        self.reference_intensity = self.Propagator(torch.zeros((1,1,1),dtype=torch.float64))
        self.useNoise = True
    
    def BuildReconstructionMatrix(self, modes, mask):
        """
        Builds the reconstruction matrix by computing the signals for each mode using finite differences.
    
        Args:
            modes (torch tensor): Modes (3D array with shape (Npix, Npix, Nmodes)) representing different phase aberrations
            mask (torch tensor): Phase mask used in the propagation (not directly used in this function)
        Returns:
            None
        """
        self.useNoise = False
        delta = 1e-5
        Nmodes = modes.shape[2]
        iMat = torch.zeros((self.Npix**2, Nmodes),dtype=torch.float64).to(mask.device)
        
        for ii in range(Nmodes):
            push = self.Propagator(modes[None,:,:,ii] * delta)
            pull = self.Propagator(-modes[None,:,:,ii] * delta)
            
            signal = (push - pull) / 2. / delta
            
            iMat[:,ii] = signal.flatten()
            
        
        self.useNoise = True
        self.reconstructionMatrix = torch.linalg.pinv(iMat)
        
    def GetReconstructedPhase(self, intensity):
        """
        Reconstructs the phase aberration from the intensity measurement by applying the reconstruction matrix.
    
        Args:
            intensity (torch tensor): Measured intensity (with noise, if applicable)
        Returns:
            torch tensor: Reconstructed phase aberration
        """
        
        
        reduced_intensity = intensity - self.reference_intensity
        
       
        
        temp = torch.matmul(reduced_intensity.flatten(start_dim = -2), self.reconstructionMatrix.T)  
       
        return temp

#%% Set general parameter and build classes
if __name__ == "__main__":

    ## WFS parameters
    Nres = 50                                                                       # Number of pixels in the aperture of the telescope
    sampling = 3                                                                    # Zero-padding factor (2 is Shannon)
    Npix = Nres * sampling                                                          # Total number of pixels
    D = 1                                                                           # Telescope diameter (m)
    Nphotons = 1e7                                                                  # Number of photons in measurement    
    RON = 0                                                                         # Read-out noise in photons per pixel per frame
    Nzernike = 50                                                                   # Number of Zernike modes to reconstruct
    Nactuator = 10                                                                  # Number of actuators across the diameter of the DM    
    useNoise  = True                                                                # Add Noise or not
                                                                
    ## Atmosphere parameters
    r0 = 0.15                                                                       # Fried parameter (m)
    l0 = 1e-10                                                                      # Inner scale (m)
    L0 = 20                                                                         # Outter scale (m)
    Nphases = 10                                                                # Number of independent phase screens to simulate
    
    
    ## Loop parameters
    loopFrequency = 1000
    delayFrames = 1
    windSpeedVector = torch.tensor([5,10])
    
    ## device ('cuda' or 'cpu')
    
    device  = 'cpu'
    
    ## intialization of the parameter vector 
    param = torch.tensor([0.0,0.0],dtype=torch.float64)
    ## Generate the wfs object
    wfs = WFS(Nres, sampling, D, Nphotons, RON,useNoise,param,device)
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
    temporalErrorPSD = GetTemporalErrorPSD(fx, fy, dF, loopFrequency, delayFrames, windSpeedVector)  
    
    
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
        line2.set_ydata(estimated_phase[ii,:].cpu().detach().numpy())
        plt.xlabel('Zernike mode index')
        plt.ylabel('Zernike mode Amplitude')
        plt.pause(0.1)
        plt.show()
        
    [outPhaseMap_test, outZe_test] = GetMultiplePhaseMapAndZernike(atmosphere_PSD * fitting_PSD + temporalErrorPSD * atmosphere_PSD, wfs.pupil, wfs.pupil_logical, invZ, Nphases)
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


    




