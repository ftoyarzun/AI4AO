# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:02:43 2024

@author: foyarzun
"""

import torch # type: ignore[import]
import torch.nn as nn  # type: ignore[import]

import math
import numpy as np

np.math = math

from torch.fft import fft2, fftshift, ifft2, ifftshift # type: ignore[import]

from .Utils import MakePupil


def PoissonNoise(x):
    """From M. Dufraisse PhD : differentiable Poisson Noise Model using Gaussian approx for each pixel and reparametrization tricks"""

    return x + torch.sqrt(torch.clamp(x, min=1e-9)) * torch.randn(
        x.shape, device=x.device, dtype=x.dtype
    )


class WFS(nn.Module):
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
        super().__init__()
        self.Nres = ParamsDict["Nres"]
        self.sampling = ParamsDict["sampling"]
        self.Npix = int(self.Nres * self.sampling)
        self.crop_size = self.Npix  # 2 * self.Nres
        self.D = ParamsDict["D"]
        self.useNoise = ParamsDict["useNoise"]
        self.central_obstruction = ParamsDict["centralObstruction"]
        self.pupil_shift_x = ParamsDict.get("pupilShiftX", 0.0)
        self.pupil_shift_y = ParamsDict.get("pupilShiftY", 0.0)
        self.pupil_upscale = ParamsDict.get("pupilUpscale", 1)
        self.device = device
        self.reference_intensity = None
        self.pupil_centers = None
        self.beamSplitProportionForWFSDetector = 1 # ParamsDict["beamSplitProportionForWFSDetector"]

        self.Nphotons = 1e7
        self.RON = 2
        self.focalPlaneRON = 4

        # x = torch.linspace(
        #     -self.Nres / 2, self.Nres / 2, self.Nres, dtype=torch.float32
        # ).to(device)
        # [self.x, self.y] = torch.meshgrid(x, x)

        x_mask = torch.linspace(
            -self.Npix / 2, self.Npix / 2 - 1, self.Npix, dtype=torch.float32
        ).to(device)
        [self.x_mask, self.y_mask] = torch.meshgrid(x_mask, x_mask)

        self.rho_mask = torch.sqrt(self.x_mask**2 + self.y_mask**2)
        self.abs_x_mask = torch.abs(self.x_mask)
        self.abs_y_mask = torch.abs(self.y_mask)

        self.MakePupil()

    def MakePupil(self):
        self.pupil = MakePupil(
            self.Nres, self.device,
            central_obstruction=self.central_obstruction,
            shift_x=self.pupil_shift_x, shift_y=self.pupil_shift_y,
            upscale=self.pupil_upscale,
        )
        self.pupil_logical = torch.where(self.pupil.reshape(self.Nres * self.Nres) > 0)

    def FFTPropagator(self, uin_padded):
        ufocal = fft2(fftshift(uin_padded, [-2, -1]))
        upupil = ifft2(ufocal * fftshift(self.mask, [-2, -1]), norm="forward")  # Multiplication to the phase mask and propagation to the detector
        self.frame_no_noise = (torch.abs(fftshift(upupil, [-2, -1])) ** 2)  # Return the noisy image, normalized the the number of counts
        self.psf_no_noise = torch.abs(ifftshift(ufocal, [-2, -1])) ** 2

    def MakeMTFMatrices(self, fourier_extension):
        fourier_sampling = self.sampling * self.MTF_focal_upscale
        pupil_sampling = self.Nres
        pupil_extension = self.Npix
        # Pupil plane size
        Np = self.Nres

        # Number of focal plane pixels
        Nf = int(fourier_extension * fourier_sampling)

        # Pupil coordinates (in D units)
        x = (
            torch.arange(Np, device=self.device, dtype=torch.float32) - Np / 2
        ) / pupil_sampling
        # Focal plane coordinates (in lambda/D units)
        u = (
            torch.arange(Nf, device=self.device, dtype=torch.float32) - (Nf - 1) / 2
        ) / fourier_sampling

        # Detector coordinates (in D units)
        d = (
            torch.arange(
                self.Npix,
                device=self.device,
                dtype=torch.float32,
            )
            - self.Npix / 2
        ) / self.Nres

        # Fourier kernels
        self.Mx = torch.exp(-1j * 2 * torch.pi * torch.outer(x, u))  # (Np, Nf)
        self.My = torch.exp(-1j * 2 * torch.pi * torch.outer(x, u))  # (Np, Nf)

        # Inverse Fourier kernels
        self.iMx = torch.exp(1j * 2 * torch.pi * torch.outer(u, d))  # (Nf, Np)
        self.iMy = torch.exp(1j * 2 * torch.pi * torch.outer(u, d))  # (Nf, Np)

    def MFT_pupil_to_focal(self, E):
        """
        Perform a Matrix Fourier Transform (MFT) from pupil plane to focal plane.

        Parameters
        ----------
        E : ndarray
            Complex electric field in the pupil plane.
        Returns
        -------
        Ef : ndarray
            Complex electric field in the focal plane.
        """
        # Matrix Fourier Transform
        Ef = self.Mx.T @ E @ self.My

        # Normalization
        Ef *= 1 / self.Nres**2

        return Ef

    def iMFT_focal_to_pupil(self, E):
        """
        Perform an Inverse Matrix Fourier Transform (iMFT) from focal plane
        back to pupil plane.

        Parameters
        ----------
        E : ndarray
            Complex electric field in the focal plane.

        Returns
        -------
        Ep : ndarray
            Complex electric field reconstructed in the pupil plane.
        """

        # Inverse Matrix Fourier Transform
        Ep = self.iMx.T @ E @ self.iMy

        # Normalization
        Ep *= 1 / (self.sampling * self.MTF_focal_upscale) ** 2

        return Ep

    def MTFPropagator(self, uin, uin_padded):
        self.psi_f = self.MFT_pupil_to_focal(uin)
        psi_ref = self.psi_f * self.transmisionMask
        psi_zwfs = uin_padded + (
            torch.exp(1j * self.phaseMask) - 1
        ) * self.iMFT_focal_to_pupil(psi_ref)
        self.frame_no_noise = torch.abs(psi_zwfs) ** 2
        for i in range(self.frame_no_noise.shape[1]):
            self.frame_no_noise[:, i] = torch.roll(
                self.frame_no_noise[:, i], shifts=self.pupil_shifts[i].item(), dims=-2
            )  # up
            # self.frame_no_noise[:, 1] = torch.roll(
            #     self.frame_no_noise[:, 1], shifts=-self.pupil_shifts[1], dims=-2
            # )

    def forward(self, phase, pupil = None):
        return self.Propagator(phase, pupil)

    def Propagator(self, phase, pupil = None):
        """
        Simulates the propagation considering a input phase aberration and a phase mask

        Args:
           phase (torch tensor): Input phase aberration dim (NphasesxNresxNres)
        Returns:
           torch tensor: Sensor measurement NphasesxNresxNres

        """
        if pupil is None:
            pupil = self.pupil.unsqueeze(0)

        pad = int(np.round(self.Nres * (self.sampling - 1)) // 2)
        uin = (
            self.pupil.unsqueeze(0)
            * pupil
            * torch.exp(1j * phase)
            / torch.sqrt(self.pupil.sum())
        )
        uin = uin.unsqueeze(1)
        uin_padded = torch.nn.functional.pad(
            uin, (pad, pad, pad, pad)
        )  # Pad the pupil

        self.frame_no_noise = torch.abs(torch.zeros_like(uin_padded))
        self.psf_no_noise = torch.abs(torch.zeros_like(uin_padded))

        ufocal = fft2(fftshift(uin_padded, [-2, -1]))
        self.psf_no_noise = torch.abs(ifftshift(ufocal, [-2, -1])) ** 2

        self.PropagateField(uin, uin_padded)

        self.frame_no_noise = self.frame_no_noise.sum(dim=1)
        self.frame_no_noise /= self.frame_no_noise.sum(dim=(-2, -1), keepdim=True)
        # self.psf_no_noise /= self.psf_no_noise.sum(dim=(-2, -1), keepdim=True)

        if not self.useNoise:
            return self.frame_no_noise

        self.AddNoiseToFrame()
        return self.frame_with_noise
    
    def PropagateField(self, uin, uin_padded):
        if self.use_MTF:
            self.MTFPropagator(uin, uin_padded)
        else:
            self.FFTPropagator(uin_padded)

    def SetPhotonsAndRON(self, Nphotons, RON):
        self.Nphotons = Nphotons
        self.RON = RON

    def AddNoiseToFrame(self):
        self.frame_with_noise = PoissonNoise(
            self.frame_no_noise * self.Nphotons * self.beamSplitProportionForWFSDetector
        ) + self.RON * torch.randn_like(self.frame_no_noise)
        self.frame_with_noise /= self.frame_with_noise.sum(dim=(-2, -1), keepdim=True)

        if self.beamSplitProportionForWFSDetector < 1.0:
            self.psf_with_noise = PoissonNoise(
                self.psf_no_noise
                * self.Nphotons
                * (1.0 - self.beamSplitProportionForWFSDetector)
            ) + self.focalPlaneRON * torch.randn_like(self.psf_no_noise)
            self.psf_with_noise /= self.psf_with_noise.sum(dim=(-2, -1), keepdim=True)
        else:
            self.psf_with_noise = self.psf_no_noise

    def GetPSF(self, phase, pupil = None, sampling = None, fov = None):
        """
        Computes the Point Spread Function (PSF) for a given phase aberration.

        Args:
            phase (complex torch tensor): Input phase aberration
            fov (float, optional): Output field of view, in lambda/D. The frame is
                center-cropped to fov * sampling pixels. If None, the full frame
                is returned.
        Returns:
            torch tensor: Point Spread Function (PSF) in the focal plane
        """

        if sampling is None:
            sampling = self.sampling

        if pupil is None:
            pupil = self.pupil.unsqueeze(0)

        pad = int(np.round(self.Nres * (sampling - 1)) // 2)
        uin = (
            self.pupil.unsqueeze(0)
            * pupil
            * torch.exp(1j * phase)
            / torch.sqrt(self.pupil.sum())
        )
        uin_padded = torch.nn.functional.pad(uin, (pad, pad, pad, pad))  # Pad the pupil

        ufocal = torch.fft.fft2(
            torch.fft.fftshift(uin_padded, [-2, -1])
        )  # Pad the pupil
        psf = (
            torch.abs(torch.fft.fftshift(ufocal, [-2, -1])) ** 2
        )  # Propagation of the field to the focal plane

        if fov is None:
            return psf

        fov_pix = int(np.round(fov * sampling))
        Ny, Nx = psf.shape[-2], psf.shape[-1]
        y0 = (Ny - fov_pix) // 2
        x0 = (Nx - fov_pix) // 2
        return psf[..., y0 : y0 + fov_pix, x0 : x0 + fov_pix]

    def SetMask(self, phaseMask=None, transmisionMask=None):
        """
        Sets the phase mask by converting the input mask to a complex exponential and normalizing it.

        Args:
            phaseMask (torch tensor): Input phase mask (real-valued)
            transmisionMask (torch tensor): Input transmision mask (real-valued)
        Returns:
            None
        """
        self.phaseMask = phaseMask
        self.transmisionMask = transmisionMask
        if phaseMask is not None:
            self.mask = (
                torch.ones(
                    1,
                    1,
                    phaseMask.shape[-2],
                    phaseMask.shape[-1],
                    device=self.device,
                    dtype=torch.cfloat,
                )
                / self.Npix**2
            )
        else:
            self.mask = (
                torch.ones(
                    1,
                    1,
                    transmisionMask.shape[-2],
                    transmisionMask.shape[-1],
                    device=self.device,
                    dtype=torch.cfloat,
                )
                / self.Npix**2
            )

        if phaseMask is not None:
            if phaseMask.dim() == 2:
                self.mask = self.mask * torch.exp(
                    1j * phaseMask.unsqueeze(0).unsqueeze(0)
                )
            elif phaseMask.dim() == 3:
                self.mask = self.mask * torch.exp(1j * phaseMask.unsqueeze(0))
            else:
                self.mask = self.mask * torch.exp(1j * phaseMask)

        if transmisionMask is not None:
            if transmisionMask.dim() == 2:
                self.mask = self.mask * transmisionMask.unsqueeze(0).unsqueeze(0)
            elif transmisionMask.dim() == 3:
                self.mask = self.mask * transmisionMask.unsqueeze(0)
            else:
                self.mask = self.mask * transmisionMask

    def BuildReferenceIntensity(self, phaseOffset=0, pupil = None):
        """
        Builds the reference intensity by propagating a zero-phase aberration.

        Args:
            None
        Returns:
            None
        """
        tempUseNoise = self.useNoise
        self.useNoise = False
        self.reference_intensity = self.Propagator(
            torch.zeros(
                (1, self.Nres, self.Nres), dtype=torch.float32, device=self.device
            ) + phaseOffset, pupil
        )
        self.reference_intensity = self.reference_intensity.squeeze()
        self.useNoise = tempUseNoise

    def BuildReconstructionMatrix(self, modes, pupil = None, batch_size=30, phaseOffset=0):
        """
        Builds the reconstruction matrix as the inverse of the interaction matrix

        Args:
            modes (torch tensor): Modes (3D array with shape (Npix, Npix, Nmodes)) representing different phase aberrations
            mask (torch tensor): Phase mask used in the propagation (not directly used in this function)
        Returns:
            None
        """
        self.BuildInteractionMatrix(modes, pupil, batch_size, phaseOffset)
        self.reconstructionMatrix = torch.linalg.pinv(self.iMat.flatten(start_dim=-2))

    def BuildInteractionMatrix(self, modes, pupil = None, batch_size=30, phaseOffset=0, single_pass = False):
        """
        Builds the interaction matrix by computing the signals for each mode using finite differences.

        Args:
            modes (torch tensor): Modes (3D array with shape (Npix, Npix, Nmodes)) representing different phase aberrations
            mask (torch tensor): Phase mask used in the propagation (not directly used in this function)
        Returns:
            None
        """
        tempUseNoise = self.useNoise
        self.useNoise = False
        delta = 1e-2

        if pupil is None:
            pupil = self.pupil.unsqueeze(0)

        
        if single_pass and self.reference_intensity is None:
            self.BuildReferenceIntensity(phaseOffset, pupil)

        Nmodes = modes.shape[0]
        iMat_parts = []

        for i in range(0, Nmodes, batch_size):
            modes_batch = modes[i : i + batch_size]  # (Npix^2, batch_size)

            # reshape to (1, Npix, Npix, batch_size) if needed by Propagator
            push = self.Propagator(modes_batch * delta + phaseOffset, pupil)
            if single_pass:
                pull = self.reference_intensity
                signal = (push - pull) / (1.0 * delta)
            else:
                pull = self.Propagator(-modes_batch * delta + phaseOffset, pupil)
                signal = (push - pull) / (2.0 * delta)

            iMat_parts.append(signal)

        self.iMat = torch.cat(iMat_parts, dim=0).squeeze()  # shape: (Nmodes, Npix^2)
        self.useNoise = tempUseNoise

    def GetReconstructedPhase(self, intensity):
        """
        Reconstructs the phase aberration from the intensity measurement by applying the reconstruction matrix.

        Args:
            intensity (torch tensor): Measured intensity (with noise, if applicable)
        Returns:
            torch tensor: Reconstructed phase aberration
        """

        reduced_intensity = intensity - self.reference_intensity

        temp = torch.matmul(
            reduced_intensity.flatten(start_dim=-2), self.reconstructionMatrix
        )

        return temp

    def LoadCalibration(self, file_path):
        checkpoint = torch.load(file_path)
        model = checkpoint["model"]
        self.load_state_dict(model)

        with torch.no_grad():
            self.BuildMask()
            self.BuildReferenceIntensity()

    def SaveCalibration(self, file_path):
        torch.save({"model": self.state_dict()}, file_path)

    def train(self, mode=True):
        # Let PyTorch handle the normal train/eval behavior
        super().train(mode)

        if not mode:
            self.requires_grad_(False)
            with torch.no_grad():
                self.BuildMask()
                self.BuildReferenceIntensity()
        return self