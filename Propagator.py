# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:02:43 2024

@author: franc
"""
import math
import numpy as np
np.math = math
import pylab as plt
import aotools as ao


def Zernike(pupil, pupil_logical, resolution, j):
    """
     Creates the Zernike polynomial basis

     Args:
        pupil (numpy array): Aperture of the telescope
        pupil_logical (numpy array): Logical values of the pupil for vectorization
        resolution (int): pixels in the diameter
        j (int): Number of zernike modes to use
     Returns:
        out (numpy array): 2D Matrix in which each column corresponds to a zernike mode
        outFullRes (numpy array): 3D Matrix in which each 2D slice corresponds to a zernike mode
     """
    X, Y = np.where(pupil > 0)
                        
    X = ( X-(resolution + resolution%2-1)/2 ) / resolution
    Y = ( Y-(resolution + resolution%2-1)/2 ) / resolution
    R = np.sqrt(X**2 + Y**2)
    R = R/R.max()
    theta = np.arctan2(Y, X)
    out = np.zeros([np.sum(pupil),j])
    outFullRes = np.zeros([resolution**2, j])
    

    for i in range(1, j+1):
        n, m = ao.zernike.zernIndex(i+1)
        if m == 0:
            Z = np.sqrt(n+1) * ao.zernike.zernikeRadialFunc(n, 0, R)
        else:
            if m > 0: # j is even
                Z = np.sqrt(2*(n+1)) * ao.zernike.zernikeRadialFunc(n, m, R) * np.cos(m * theta)
            else:   #i is odd
                m = abs(m)
                Z = np.sqrt(2*(n+1)) * ao.zernike.zernikeRadialFunc(n, m, R) * np.sin(m * theta)
        
        Z -= Z.mean()
        Z *= (1/np.std(Z))

        # clip
        out[:, i-1] = Z
       
        outFullRes[pupil_logical, i-1] = Z
     
    outFullRes = np.reshape( outFullRes, [resolution, resolution, j] )
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
            - fx (numpy array): Spatial frequency components in the x direction
            - fy (numpy array): Spatial frequency components in the y direction
    """
    dF = 1 / (D)
    fx = np.linspace(-resolution/2, resolution/2-1, resolution) * dF
    [fx,fy] = np.meshgrid(fx,fx)
    return dF, fx, fy


def GetAtmospherePSD(fx, fy, dF, r0, L0, pupil, pupilLogical):
    """
    Computes the atmospheric power spectral density (PSD) for phase aberrations based on the spatial frequencies.

    Args:
        fx (numpy array): Spatial frequency components in the x direction
        fy (numpy array): Spatial frequency components in the y direction
        dF (float): Frequency step size
        r0 (float): Fried parameter (m)
        L0 (float): Outer scale of turbulence (m)
        pupil (numpy array): Pupil function of the system
        pupilLogical (numpy array): Logical pupil mask indicating valid regions of the pupil

    Returns:
        numpy array: Atmospheric power spectral density (PSD) for phase aberrations
    """
    resolution = fx.shape[0]
    l0 = 1e-10 # Default value for the inner scale
    f = np.sqrt(fx**2 + fy**2)
    fm = 5.92/l0/(2*np.pi); 		# frecuencia de escala interna [1/m]
    f0 = 1/L0; 			          # frecuencia de escala externa [1/m]
    PSD_phi = 0.023*r0**(-5/3) * np.exp(-(f/fm)**2) / (f**2 + f0**2)**(11/6) * dF**2 * resolution**2;
    PSD_phi[resolution//2,resolution//2] = 0;
    return PSD_phi

def GetFittingPSD(fx, fy, dF, D, Nactuator, levelOfCorrection = 1):
    """
    Computes a fitting power spectral density (PSD) filter, including both low-pass and high-pass components.

    Args:
        fx (numpy array): Spatial frequency components in the x direction
        fy (numpy array): Spatial frequency components in the y direction
        dF (float): Frequency step size
        D (float): Diameter of the telescope
        Nactuator (int): Number of actuators in the diameter of the deformable mirror
        levelOfCorrection (float, optional): Correction factor for high-pass filter (default is 1)

    Returns:
        numpy array: High-pass filter for the fitting PSD
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
    Hdm = (1-np.exp(-Tp))/Tp;
    Hwfs = (1-np.exp(-Tp))/Tp;
    Hdelay = np.exp(-Nd*Tp);
    Hcorr = ki/ (1 - np.exp(-Tp))
    f_ol = Hdm*Hwfs*Hdelay*Hcorr;
    f_cl = 1/(1+f_ol);
    n_cl = f_cl*Hdm*Hcorr*Hdelay;
    return f_cl

def GetTemporalErrorPSD(fx, fy, dF, freq, delayFrames, windSpeedVector):
    """
    Computes the temporal error power spectral density (PSD) given the spatial frequencies and other parameters.

    Args:
        fx (numpy array): Spatial frequency components in the x direction
        fy (numpy array): Spatial frequency components in the y direction
        dF (float): Frequency step size
        freq (float): Temporal frequency of the system
        delayFrames (int): Number of frames for delay
        windSpeedVector (numpy array): Wind speed vector [vx, vy]

    Returns:
        numpy array: Temporal error power spectral density
    """
    fx_temporal = fx * windSpeedVector[0] + 1e-7
    fy_temporal = fy * windSpeedVector[1] + 1e-7
    
    f_temporal = np.sqrt(fx_temporal**2 + fy_temporal**2)
    
    T_delay = delayFrames / freq
    T_integration = 1 / freq
    
    
    return 1 - 2*np.cos(2 * np.pi * T_delay * f_temporal) * np.sinc(T_integration * f_temporal) + np.sinc(T_integration * f_temporal)**2


def GetMultiplePhaseMapAndZernike(PSD, pupil, pupilLogical, CM, Nphases):
    """
     Creates multiple atmospheric phase screens considering the according PSD and computes the zernike modes corresponding to them

     Args:
        PSD (numpy array): Power spectral density to be used
        pupil (numpy array): Aperture of the telescope
        invZ (numpy array): Inverse of the zernike 2D matrix to compute the true coefficients
        Nphases (int): Number of phases to simulate
     Returns:
        phaseMap (numpy array): Output phase map
        Ze (vector): True zernike coefficients of the phaseMap
     """
     
   
    resolution = PSD.shape[0]
    outPhaseMap = np.zeros((resolution,resolution,Nphases))
    outZe = np.zeros((CM.shape[0],Nphases))
    
    sqrt_fftshift_PSD = np.sqrt(np.fft.fftshift(PSD))
    
    for ii in range(Nphases):
        randMap = np.random.randn(resolution,resolution)
        phaseMap = np.fft.ifft2(sqrt_fftshift_PSD * np.fft.fft2(randMap))
        phaseMap = phaseMap.real
        phaseMap = pupil * phaseMap
       
        Ze = np.matmul(CM,phaseMap[pupil])
        outPhaseMap[:,:,ii] = phaseMap
        outZe[:,ii] = Ze
      
    return outPhaseMap, outZe



class WFS:
    def __init__(self, resolution, sampling, diameter, Nphotons, RON, useNoise):
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

        x = np.linspace(-self.Nres/2, self.Nres/2, self.Nres)                                          # Build the mesh
        [x,y] = np.meshgrid(x,x)                                        
        self.pupil = (x**2 + y**2) <= ((self.Nres+1)/2)**2
        self.pupil_logical = np.where(np.reshape(self.pupil,self.Nres*self.Nres)>0)
        
        self.BuildZernikeMask(2,0.78)
        
    def Propagator(self, phase):
        """
         Simulates the propagation considering a input phase aberration and a phase mask

         Args:
            phase (numpy array): Input phase aberration
         Returns:
            numpy array: Sensor measurement
         """
        uin = self.pupil * np.exp(1j * phase)
        uin_padded = np.pad(uin, self.Nres * (self.sampling - 1)//2)                # Pad the pupil 
       
        ufocal = np.fft.fft2(np.fft.fftshift(uin_padded))                           # Propagation of the field to the focal plane
        upupil = np.fft.fft2(ufocal * np.fft.fftshift(self.mask))                        # Multiplication to the phase mask and propagation to the detector
        frame_no_noise = abs(np.fft.fftshift(upupil))**2
        
        if not self.useNoise:
            return frame_no_noise / np.sum(frame_no_noise)
        
        frame_no_noise = frame_no_noise / np.sum(frame_no_noise) * self.Nphotons    # Set the number of photons in the frame
        frame_with_noise = np.random.poisson(frame_no_noise) + self.RON * np.random.randn(*frame_no_noise.shape)
        return frame_with_noise / np.sum(frame_with_noise)                          # Return the noisy image, normalized the the number of counts

    def GetPSF(self, phase):
        """
        Computes the Point Spread Function (PSF) for a given phase aberration.
    
        Args:
            phase (complex numpy array): Input phase aberration
        Returns:
            numpy array: Point Spread Function (PSF) in the focal plane
        """
        uin = self.pupil * np.exp(1j * phase)
        uin_padded = np.pad(uin, self.Nres * (self.sampling - 1)//2)                # Pad the pupil 
        return np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(uin_padded))))**2     # Propagation of the field to the focal plane

    def SetMask(self,mask):
        """
        Sets the phase mask by converting the input mask to a complex exponential and normalizing it.
    
        Args:
            mask (numpy array): Input phase mask (real-valued)
        Returns:
            None
        """
        phase_mask = np.exp(1j * mask)
        phase_mask /= np.sum(abs(phase_mask))
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
        x_mask = np.linspace(-self.Npix/2, self.Npix/2-1, self.Npix)
        [x_mask,y_mask] = np.meshgrid(x_mask,x_mask) 

        pyramid_mask = np.pi/2 * (abs(x_mask) + abs(y_mask))
        self.SetMask(pyramid_mask)
    
    def BuildZernikeMask(self, dot_diameter, dot_depth):
       """
       Builds a Zernike mask and sets it using the SetMask function.
   
       Args:
           dot_diameter (float): Diameter of the dot in units of lambda/d
           dot_depth (float): Depth of the dot in radians
       Returns:
           None
       """
       x_mask = np.linspace(-self.Npix/2, self.Npix/2-1, self.Npix)
       [x_mask,y_mask] = np.meshgrid(x_mask,x_mask)
       rho = np.sqrt(x_mask ** 2 + y_mask ** 2)
       
       diameter_in_pixels = dot_diameter * self.sampling
       
       zernike_mask = dot_depth * (rho < diameter_in_pixels / 2.)
       self.SetMask(zernike_mask)
    
    def BuildReferenceIntensity(self):
        """
        Builds the reference intensity by propagating a zero-phase aberration.
    
        Args:
            None
        Returns:
            None
        """
        self.useNoise = False
        self.reference_intensity = self.Propagator(0)
        self.useNoise = True
    
    def BuildReconstructionMatrix(self, modes, mask):
        """
        Builds the reconstruction matrix by computing the signals for each mode using finite differences.
    
        Args:
            modes (numpy array): Modes (3D array with shape (Npix, Npix, Nmodes)) representing different phase aberrations
            mask (numpy array): Phase mask used in the propagation (not directly used in this function)
        Returns:
            None
        """
        self.useNoise = False
        delta = 1e-5
        Nmodes = modes.shape[2]
        iMat = np.zeros((self.Npix**2, Nmodes))
        
        for ii in range(Nmodes):
            push = self.Propagator(modes[:,:,ii] * delta)
            pull = self.Propagator(-modes[:,:,ii] * delta)
            
            signal = (push - pull) / 2. / delta
            
            iMat[:,ii] = signal.flatten()
            
        
        self.useNoise = True
        self.reconstructionMatrix = np.linalg.pinv(iMat)
        
    def GetReconstructedPhase(self, intensity):
        """
        Reconstructs the phase aberration from the intensity measurement by applying the reconstruction matrix.
    
        Args:
            intensity (numpy array): Measured intensity (with noise, if applicable)
        Returns:
            numpy array: Reconstructed phase aberration
        """
        reduced_intensity = intensity - self.reference_intensity
        return self.reconstructionMatrix @ reduced_intensity.flatten()

#%% Set general parameter and build classes
if __name__ == "__main__":
    plt.close('all')
    ## WFS parameters
    Nres = 100                                                                       # Number of pixels in the aperture of the telescope
    sampling = 2                                                                    # Zero-padding factor (2 is Shannon)
    Npix = Nres * sampling                                                          # Total number of pixels
    D = 1                                                                           # Telescope diameter (m)
    Nphotons = 1e5                                                                  # Number of photons in measurement    
    RON = 1                                                                         # Read-out noise in photons per pixel per frame
    Nzernike = 50                                                                   # Number of Zernike modes to reconstruct
    Nactuator = 10                                                                  # Number of actuators across the diameter of the DM    
    useNoise  = True                                                                # Add Noise or not
                                                                
    ## Atmosphere parameters
    r0 = 0.03                                                                       # Fried parameter (m)
    l0 = 1e-10                                                                      # Inner scale (m)
    L0 = 20                                                                         # Outter scale (m)
    Nphases = 30                                                                    # Number of independent phase screens to simulate
    
    
    ## Loop parameters
    loopFrequency = 1000
    delayFrames = 1
    windSpeedVector = np.array([5,10])
    
    
    
    ## Generate the wfs object
    wfs = WFS(Nres, sampling, D, Nphotons, RON,useNoise)
    ## By default this generates the mask for the pyramid waferont sensor. Use the method wfs.SetMask(mask) to change to the desired mask
   
    
    ## Compute the first Nzernike Zernike polynomials and the inverse to obtain the perfect reconstructor
    [z, z_FullRes] = Zernike(wfs.pupil, wfs.pupil_logical, wfs.Nres, Nzernike)
    invZ = np.linalg.pinv(z)
  
    ## Build the reconstruction matrix
    wfs.BuildReconstructionMatrix(z_FullRes, wfs.mask)
    
    ## Compute some example PSDs
    [dF, fx, fy] = GetSpatialFrequencies(D, Nres)
    atmosphere_PSD = GetAtmospherePSD(fx, fy, dF, r0, L0, wfs.pupil, wfs.pupil_logical)
    fitting_PSD = GetFittingPSD(fx, fy, dF, D, Nactuator, 1)
    temporalErrorPSD = GetTemporalErrorPSD(fx, fy, dF, loopFrequency, delayFrames, windSpeedVector)  
    
    
    #%% This section is just to test how the images look like, what would be the perfect estimation and what is the phase estimation using a linear reconstructor 
    
    [outPhaseMap_test, outZe_test] = GetMultiplePhaseMapAndZernike(atmosphere_PSD, wfs.pupil, wfs.pupil_logical, invZ, Nphases)  
    test_frame = wfs.Propagator(outPhaseMap_test[:,:,0])
    test_psf = wfs.GetPSF(outPhaseMap_test[:,:,0])
    
   
    #%% Generate datasets
   
    ## Set initial data for figures
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle('Full atmospheric turbulence', fontsize=16)
    img1 = axes[0].imshow(test_psf)
    img2 = axes[1].imshow(test_frame)
    line1, = axes[2].plot(outZe_test[:,0])
    line2, = axes[2].plot(outZe_test[:,0])
    axes[2].legend(['Ground truth', 'Linear reconstructor'])
    
    ## run through the small dataset to observe the system in action
    for ii in  range(Nphases):
        test_frame = wfs.Propagator(outPhaseMap_test[:,:,ii])
        test_psf = wfs.GetPSF(outPhaseMap_test[:,:,ii])
        plt.figure(1)
        
        plt.subplot(1,3,1)
        img1.set_data(test_psf)
        
        plt.subplot(1,3,2)
        img2.set_data(test_frame)

        plt.subplot(1,3,3)
        line1.set_ydata(outZe_test[:,ii])       
        line2.set_ydata(wfs.GetReconstructedPhase(test_frame))
         
        plt.xlabel('Zernie mode index')
        plt.ylabel('Zernike mode Amplitude')
        plt.pause(0.1)
        plt.show()
        
    [outPhaseMap_test, outZe_test] = GetMultiplePhaseMapAndZernike(atmosphere_PSD * fitting_PSD + temporalErrorPSD * atmosphere_PSD, wfs.pupil, wfs.pupil_logical, invZ, Nphases)
    fig.suptitle('Residual turbulence after the AO loop', fontsize=16)
    
   
    
    for ii in  range(Nphases):
        test_frame = wfs.Propagator(outPhaseMap_test[:,:,ii])
        test_psf = wfs.GetPSF(outPhaseMap_test[:,:,ii])
        plt.figure(1)
        
        plt.subplot(1,3,1)
        img1.set_data(test_psf)
        
        plt.subplot(1,3,2)
        img2.set_data(test_frame)

        plt.subplot(1,3,3)
        line1.set_ydata(outZe_test[:,ii])       
        line2.set_ydata(wfs.GetReconstructedPhase(test_frame))
         
        plt.xlabel('Zernie mode index')
        plt.ylabel('Zernike mode Amplitude')
        plt.pause(0.1)
        plt.show()
      
    #%% Generate datasets
    
    Ndataset_each = 500
    ## Generate full turbulence data
    ## We have to simulate as many different atmospheric conditions as possible. To do this, we have to change r0 and L0
    ## r0 could have values like np.logspace(-2,-0.5,5)
    ## L0 could have values like np.linspace(20,40)
    ## Some examples...
    #atmosphere_PSD = GetAtmospherePSD(fx, fy, dF, 0.01, 30, wfs.pupil, wfs.pupil_logical)
    #atmosphere_PSD = GetAtmospherePSD(fx, fy, dF, 0.05, 40, wfs.pupil, wfs.pupil_logical)
    #atmosphere_PSD = GetAtmospherePSD(fx, fy, dF, 0.1, 20, wfs.pupil, wfs.pupil_logical)
    #atmosphere_PSD = GetAtmospherePSD(fx, fy, dF, 0.2, 30, wfs.pupil, wfs.pupil_logical)
    
    [outPhaseMap_fullTurbulence, outZe_fullTurbulence] = GetMultiplePhaseMapAndZernike(atmosphere_PSD, wfs.pupil, wfs.pupil_logical, invZ, Ndataset_each) 
    ## Generate perfect correction data
    ## We have to test different levels of correction
    ## Some examples...
    ## fitting_PSD = GetFittingPSD(fx, fy, dF, D, Nactuator, 0)
    ## fitting_PSD = GetFittingPSD(fx, fy, dF, D, Nactuator, 0.5)
    ## fitting_PSD = GetFittingPSD(fx, fy, dF, D, Nactuator, 1)
    [outPhaseMap_correction, outZe_correction] = GetMultiplePhaseMapAndZernike(atmosphere_PSD*fitting_PSD, wfs.pupil, wfs.pupil_logical, invZ, Ndataset_each) 


    # Check of the phase mask compared to the torch version
    
    # plt.figure()
    # plt.imshow(wfs.mask.imag)
    # plt.show()
    
    # plt.figure()
    # plt.imshow(wfs.mask.real)
    # plt.show()




