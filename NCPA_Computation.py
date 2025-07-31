# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 04:12:29 2025

@author: foyarzun
"""

import numpy as np
import pylab as plt
import aotools as ao

import time
from tqdm import tqdm

#import dao

class Pyramid:
    def __init__(self, Nres, sampling, modes, modulation):
        self.Nres = Nres
        self.sampling = sampling
        self.modes = modes
        self.modulation = modulation
        
        self.Npix = int(Nres * sampling)

        self.pupil = MakePupil(Nres)
        
        xmask = np.linspace(-self.Npix//2, self.Npix//2-1, self.Npix)
        self.xmask,self.ymask = np.meshgrid(xmask,xmask)
        
        self.mask = np.exp(1j * (np.pi/2 * (np.abs(self.xmask) + np.abs(self.ymask))) )

        
    def Propagator(self, phase):
        
        pad = int(self.Nres * (self.sampling - 1))//2            

        uin = self.pupil * np.exp(1j * phase)
        uin_padded = np.pad(uin,pad)       # Pad the pupil 
      
        ufocal = np.fft.fft2(np.fft.fftshift(uin_padded)) 

        if self.modulation != 0:
            nSteps = round(6.28*self.modulation / 4) * 4
            frame = np.abs(np.zeros_like(ufocal))
            
            for i in range(nSteps):
                modulation_phase = 2 * np.pi * i / nSteps
                tip_tilt_mirror_phase = np.exp(1j * 2 * np.pi * self.modulation / self.Npix * (self.xmask * np.cos(modulation_phase) + self.ymask * np.sin(modulation_phase)))
                ufocal_step = np.fft.fft2(np.fft.fftshift(uin_padded * tip_tilt_mirror_phase))
                upupil_step = np.fft.fft2(ufocal_step * np.fft.fftshift(self.mask))
                frame += np.abs(np.fft.fftshift(upupil_step))**2 / nSteps
                
                                  # Propagation of the field to the focal plane
        else:        
            upupil = np.fft.fft2(ufocal * np.fft.fftshift(self.mask))                        # Multiplication to the phase mask and propagation to the detector
            frame = np.abs(np.fft.fftshift(upupil))**2

        return frame / frame.sum()                # Return the noisy image, normalized the the number of counts

        
    def BuildReferenceIntensity(self):
        """
        Builds the reference intensity by propagating a zero-phase aberration.
    
        Args:
            None
        Returns:
            None
        """
        self.reference_intensity = self.Propagator(np.zeros((self.Nres, self.Nres)))


    
    def BuildReconstructionMatrix(self):
        """
        Builds the reconstruction matrix by computing the signals for each mode using finite differences.
    
        Args:
            modes (torch tensor): Modes (3D array with shape (Npix, Npix, Nmodes)) representing different phase aberrations
            mask (torch tensor): Phase mask used in the propagation (not directly used in this function)
        Returns:
            None
        """
        delta = 1e-6
        
        Nmodes = self.modes.shape[0]
        iMat_parts = []
        
        for i in range(0, Nmodes):
            
            # reshape to (1, Npix, Npix, batch_size) if needed by Propagator
            push = self.Propagator(self.modes[i] * delta)
            pull = self.Propagator(-self.modes[i] * delta)
            
            signal = (push - pull) / (2. * delta)
            signal_flat = signal.flatten()  # shape: (batch_size, Npix^2)
    
            iMat_parts.append(signal_flat)

        self.iMat = np.array(iMat_parts)  # shape: (Nmodes, Npix^2)
            
        self.reconstructionMatrix = np.linalg.pinv(self.iMat)
    
    def GetReconstructedPhase(self, intensity):
        """
        Reconstructs the phase aberration from the intensity measurement by applying the reconstruction matrix.
    
        Args:
            intensity (torch tensor): Measured intensity (with noise, if applicable)
        Returns:
            torch tensor: Reconstructed phase aberration
        """
            
        reduced_intensity = intensity - self.reference_intensity

  
        temp = np.matmul(reduced_intensity.flatten(), self.reconstructionMatrix)  
       
        return temp  

    def CloseLoop(self, phase):
        
        self.BuildReferenceIntensity()
        
        Nmodes = self.iMat.shape[0]
        
        z_est = np.zeros((Nmodes, 1))
        
        reconstructed_phase = np.zeros_like(phase)
        residual_phase = np.copy(phase)
        
        z_matrix = self.modes.reshape(Nmodes, -1).T
        
        gain = 0.2
        
        for i in tqdm(range(100)):
            frame = self.Propagator(residual_phase)
            z_out = np.expand_dims(self.GetReconstructedPhase(frame), axis = 1)
            
            z_est = z_est + gain * z_out
            
            reconstructed_phase = (z_matrix @ z_est).reshape(self.Nres, self.Nres)
            
            residual_phase = phase - reconstructed_phase
            
        return reconstructed_phase, z_est
            

        

        
def MakePupil(N, shape = "circ"):
    x = np.linspace(-1, 1, N)
    x,y = np.meshgrid(x,x)
    rr = np.sqrt(x**2 + y**2)
    
    if shape == "circ":
        return (rr <= 1.0 + 1/N)
    
    return np.exp(-rr**2 / 4.)


def zero_pad_center(image, target_shape):
    """
    Zero-pad a 2D tensor (H, W) to target shape (H_new, W_new), centered.
    """
    H, W = image.shape
    H_new, W_new = target_shape
    pad = (W_new - W) // 2
    return np.pad(image, pad)
 
def center_crop(image, target_shape: tuple[int, int]):
    """
    Crop the center region of a 2D tensor to the target shape.

    Args:
        tensor (torch.Tensor): 2D tensor to crop (H_pad, W_pad).
        target_shape (tuple[int, int]): Desired output shape (H, W).

    Returns:
        torch.Tensor: Cropped tensor of shape (H, W).
    """
    H_pad, W_pad = image.shape
    H, W = target_shape

    if H > H_pad or W > W_pad:
        raise ValueError("Target shape must be smaller than or equal to input tensor shape.")

    start_y = (H_pad - H) // 2
    start_x = (W_pad - W) // 2

    return image[start_y:start_y + H, start_x:start_x + W]

def gerchberg_saxton_padded(psf, starting_guess = None, iterations: int = 100, s: int = 180, pupil_shape = 'circ'):
    """
    GS phase retrieval using a circular pupil and **zero-padded** Fourier domain for oversampling.

    Args:
        psf (torch.Tensor): Normalized intensity in Fourier domain, shape [H, W], psf.sum() = 1
        iterations (int): Number of GS iterations.
        s (int): Sampling factor (e.g. 2 means 2x upsampling via zero-padding).
        device (str): Torch device.

    Returns:
        torch.Tensor: Complex pupil field (real + imag parts).
    """
    if psf.ndim != 2:
        raise ValueError("PSF must be 2D.")

    N = psf.shape[0]
    
    # Zero-pad PSF to (padH, padW)
    psf_padded = zero_pad_center(psf, (s, s))
    amplitude_f = np.sqrt(psf_padded)

    # Generate circular mask in pupil plane
    pupil_mask = MakePupil(N, pupil_shape)
    pupil_mask = zero_pad_center(pupil_mask, (s,s))
    
    # Initial random phase
    if starting_guess is None:
        phase = np.random.rand(s, s) * 2 * np.pi
    else:
        phase = zero_pad_center(starting_guess, (s,s))
        
    field_pupil = pupil_mask * np.exp(1j * phase)

    for _ in tqdm(range(iterations)):
        field_f = np.fft.fftshift(np.fft.fft2(field_pupil, norm="ortho"))
        phase_f = np.angle(field_f)
        field_f = amplitude_f * np.exp(1j * phase_f)
        field_pupil = np.fft.ifft2(np.fft.ifftshift(field_f), norm="ortho")
        field_pupil = pupil_mask * np.exp(1j * np.angle(field_pupil))
        
        

    phase = center_crop(np.angle(field_pupil) * np.abs(field_pupil), (N, N))
    reconstructed_psf = np.fft.fftshift(np.fft.fft2(field_pupil, norm="ortho"))
    reconstructed_psf = center_crop(np.abs(reconstructed_psf)**2, (N, N))
    
    return phase, reconstructed_psf  # shape: [padH, padW]



def GS_StartingGuess(modes, coefficients):
    phase = np.sum(modes * coefficients, axis = 0)
    return phase


def GetExposure(number_of_exposures = 10):
    pass


def CoG(img: np.ndarray, threshhold: np.float64 = 0.01):
    N, M = img.shape
    y = np.linspace(0, N-1, N)
    x = np.linspace(0, M-1, M)
    x,y = np.meshgrid(x,y)
    
    threshhold_map = img > (threshhold * img.max())
    
    xcog = (img[threshhold_map] * x[threshhold_map]).sum() / img[threshhold_map].sum()
    ycog = (img[threshhold_map] * y[threshhold_map]).sum() / img[threshhold_map].sum()
    return [int(ycog), int(xcog)]


def bin_image(image: np.ndarray, bin_size: int) -> np.ndarray:
    """
    Bin an image by summing over non-overlapping bin_size × bin_size blocks.

    Parameters:
        image (np.ndarray): 2D input image to be binned.
        bin_size (int): Size of the binning window.

    Returns:
        np.ndarray: Binned image.
    """
    H, W = image.shape
    if H % bin_size != 0 or W % bin_size != 0:
        raise ValueError(f"Image dimensions {H}x{W} must be divisible by bin_size {bin_size}.")

    # Reshape and sum over binning blocks
    reshaped = image.reshape(H // bin_size, bin_size, W // bin_size, bin_size)
    binned = reshaped.sum(axis=(1, 3))
    return binned
    

def CenterPSFImage(img, final_resoultion):
    
    cog1 = CoG(img)
    referencePSF = img[cog1[0] - final_resoultion: cog1[0] + final_resoultion, cog1[1] - final_resoultion: cog1[1] + final_resoultion]
    
    cog2 = CoG(referencePSF)
    referencePSF = referencePSF[cog2[0] - final_resoultion // 2: cog2[0] + final_resoultion // 2, cog2[1] - final_resoultion // 2: cog2[1] + final_resoultion // 2]
    
    
    referencePSF = np.abs(referencePSF)
    
    referencePSF = referencePSF / referencePSF.sum()
    
    return referencePSF


def Zernike(resolution, j):
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
    pupil = MakePupil(resolution)
    pupil_logical = np.where(np.reshape(pupil,resolution*resolution)>0)
     
    X, Y = np.where(pupil > 0)
                        
    X = ( X-(resolution + resolution%2-1)/2 ) / resolution
    Y = ( Y-(resolution + resolution%2-1)/2 ) / resolution
    R = np.sqrt(X**2 + Y**2)
    R = R/R.max()
    theta = np.arctan2(Y, X)
    out = np.zeros([np.sum(pupil),j])
    outFullRes = np.zeros([j, resolution**2])
    

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
       
        outFullRes[i-1, pupil_logical] = Z
     
    outFullRes = np.reshape( outFullRes, [j, resolution, resolution] )
    return outFullRes

def PapyrusDM(res, modes_to_use):
    M2C = np.load('../M2C.npy')[:,:modes_to_estimate]
    papyrus_dm = np.load("../papyrus_dm.npy").astype(np.float32) * 1e7
    papyrus_modal_dm = (papyrus_dm @ M2C).reshape(80,80,-1)[1:-1, 1:-1, :]
    papyrus_modal_dm = np.flip(papyrus_modal_dm, axis=[0,1])
    return np.transpose(papyrus_modal_dm, (2,0,1))
    

if __name__ == "__main__":
    
    

    modes_to_estimate = 50
    sampling = 2.8
    resolution = 78
    
    N = int(resolution * sampling)
    
    modes = PapyrusDM(resolution, modes_to_estimate)
    
    pyr = Pyramid(Nres = resolution,
                  sampling = sampling,
                  modes = modes,
                  modulation = 1
                  )
    
    pyr.BuildReconstructionMatrix()
    
    #%% Set Path for dark and psf
    
    dark = np.load("../cred3Avg202.npy")
    psf = np.load("../cred3Avg203.npy")
    
    psf -= dark
    
    # Check if referencePSF actually found the PSF and is not centered around a hot pixel :)
    referencePSF = CenterPSFImage(psf, resolution)
    
    # plt.imshow(referencePSF)
    # plt.show()
    
    

    #%% 
    
    starting_guess = np.zeros((modes_to_estimate, 1 , 1))
    
    starting_guess[2] = 0.2
    
    starting_phase = GS_StartingGuess(modes, starting_guess)
    
    
    print("Starting GS algorithm")
    # Use this if you have a starting guess
    phase_to_estimate, estimated_psf = gerchberg_saxton_padded(referencePSF, starting_phase, iterations = 100, s = N)
    
    # Use this if you don't have a starting guess
    #phase_to_estimate, estimated_psf = gerchberg_saxton_padded(referencePSF, iterations = 100, s = N)


    print("Starting unwrapping algorithm")
    reconstructed_phase, modal_decomposition = pyr.CloseLoop(phase_to_estimate)
    
    
    #np.save("NCPA_Papyrus.npy", modal_decomposition)

