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
        The wavefront sensor object is in charge of the propagation and reconstruction of the OPD aberrations.

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
        self.device = device
        self.wavelength = ParamsDict["Wavelength"]
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
        self.reference_intensity = None
        self.pupil_centers = None

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
        for i in range(self.frame_no_noise.shape[-3]):
            self.frame_no_noise[..., i, :, :] = torch.roll(
                self.frame_no_noise[..., i, :, :], shifts=self.pupil_shifts[i].item(), dims=-2
            )  # up
            # self.frame_no_noise[:, 1] = torch.roll(
            #     self.frame_no_noise[:, 1], shifts=-self.pupil_shifts[1], dims=-2
            # )

    def forward(self, opd, pupil = None):
        return self.Propagator(opd, pupil)

    def Propagator(self, opd, pupil = None, collapse_wvl = True):
        """
        Simulates the propagation considering a input OPD aberration and a phase mask

        Args:
           opd (torch tensor): Input optical path difference, in meters, dim (NphasesxNresxNres).
               If `self.wavelength` is a 1-D tensor of Nwavelength sensing wavelengths, propagation
               for every wavelength is computed in parallel and the wavelength channel is summed
               away before returning (mirroring how the pre-existing mask/modulation channel is
               summed), so the return shape is always NphasesxNresxNres regardless of Nwavelength.
        Returns:
           torch tensor: Sensor measurement NphasesxNresxNres

        """
        if pupil is None:
            pupil = self.pupil.unsqueeze(0)

        if opd.dim() == 3:
            opd = opd.unsqueeze(1)
        if pupil.dim() == 3:
            pupil = pupil.unsqueeze(1)

        phase = self.wavenumber * opd

        pad = int(np.round(self.Nres * (self.sampling - 1)) // 2)
        uin = (self.pupil.unsqueeze(0) * pupil * torch.exp(1j * phase) / torch.sqrt(self.pupil.sum()))
        uin = uin.unsqueeze(-3)
        uin_padded = torch.nn.functional.pad(
            uin, (pad, pad, pad, pad)
        )  # Pad the pupil

        self.PropagateField(uin, uin_padded)

        if collapse_wvl:
            self.frame_no_noise = self.frame_no_noise.sum(dim=(1, -3))  # collapse mask/modulation channel
        else:
            self.frame_no_noise = self.frame_no_noise.sum(dim=(-3)) # Collapse only mask and maintain color info
        self.frame_no_noise /= self.frame_no_noise.sum(dim=(-2, -1), keepdim=True)

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
            self.frame_no_noise * self.Nphotons
        ) + self.RON * torch.randn_like(self.frame_no_noise)
        self.frame_with_noise /= self.frame_with_noise.sum(dim=(-2, -1), keepdim=True)

    def GetPSF(self, opd, pupil = None, sampling = None, fov = None, wl = None, collapse_wvl = True):
        """
        Computes the Point Spread Function (PSF) for a given OPD aberration.

        Args:
            opd (torch tensor): Input optical path difference, in meters. If
                `self.wavelength` is a 1-D tensor of Nwavelength sensing wavelengths
                (assumed sorted ascending), the PSF is computed for every wavelength
                in parallel. Because the physical angular pixel scale of an FFT-based
                PSF is (1/sampling)*(wavelength/D), different wavelengths are NOT
                natively on the same physical grid: the shared FFT is run at the
                oversampling needed for the *longest* wavelength (self.wavelength[-1])
                so every other channel comes out finer than the target, then each
                channel is resampled (never upsampled) via grid_sample onto the
                output grid implied by the *shortest* wavelength (self.wavelength[0])
                before being summed away. Return shape is always NphasesxNresxNres.
            fov (float, optional): Output field of view, in lambda/D. The frame is
                center-cropped to fov * sampling pixels. If None, the full frame
                is returned.
            wl (float/array, optional): wavelength to use for the PSF computation. It
                can be a single float, or an array (list, np.array, torch.tensor) for a 
                polychromatic simulation. If defaults to the same as for the sensing.
        Returns:
            torch tensor: Point Spread Function (PSF) in the focal plane
        """

        if sampling is None:
            sampling = self.sampling

        if pupil is None:
            pupil = self.pupil.unsqueeze(0)

        if wl is None:
            wl = self.wavelength
        else:
            # Coerce the same way the wavelength setter does: the docstring
            # promises a plain float/list/array works, not just a tensor
            # already shaped the way _GetPolychromaticPSF needs (1-D, indexable).
            wl = torch.as_tensor(wl, device=self.device, dtype=torch.float32).reshape(-1)

        psf = self._GetPolychromaticPSF(opd, pupil, sampling, wl, collapse_wvl)

        if fov is None:
            return psf

        fov_pix = int(np.round(fov * sampling))
        Ny, Nx = psf.shape[-2], psf.shape[-1]
        y0 = (Ny - fov_pix) // 2
        x0 = (Nx - fov_pix) // 2
        return psf[..., y0 : y0 + fov_pix, x0 : x0 + fov_pix]

    def _GetPolychromaticPSF(self, opd, pupil, sampling, wl, collapse_wvl):
        """Multi-wavelength branch of GetPSF (see its docstring). Runs one shared
        FFT oversampled for self.wavelength[-1], then resamples each wavelength
        channel via grid_sample onto the self.wavelength[0]-scale output grid
        before summing, since the channels are not natively on the same physical
        angular grid (see class-level discussion in GetPSF's docstring).
        """

        assert torch.all(wl[1:] >= wl[:-1]), (
            "GetPSF's multi-wavelength path assumes self.wavelength is sorted "
            "ascending (wavelength[0] smallest, wavelength[-1] largest)."
        )

        if opd.dim() == 3:
            opd = opd.unsqueeze(1)
        if pupil.dim() == 3:
            pupil = pupil.unsqueeze(1)

        # Oversample the shared simulation enough that even the longest
        # wavelength reaches wavelength[0]'s target physical pixel scale.
        sim_sampling = float((sampling * wl[-1] / wl[0]).item())
        pad_sim = int(np.round(self.Nres * (sim_sampling - 1)) // 2)
        Npix_sim = self.Nres + 2 * pad_sim
        Npix_out = self.Nres + 2 * int(np.round(self.Nres * (sampling - 1)) // 2)

        phase = 2 * torch.pi / wl.reshape(1,-1,1,1) * opd
        uin = (
            self.pupil.unsqueeze(0)
            * pupil
            * torch.exp(1j * phase)
            / torch.sqrt(self.pupil.sum())
        )
        uin_padded = torch.nn.functional.pad(uin, (pad_sim, pad_sim, pad_sim, pad_sim))

        ufocal = torch.fft.fft2(torch.fft.fftshift(uin_padded, [-2, -1]))
        psf_sim = torch.abs(torch.fft.fftshift(ufocal, [-2, -1])) ** 2  # (Nphases, Nwavelength, Npix_sim, Npix_sim)

        # sim-pixels per output-pixel, per wavelength: >=1 everywhere, ==1 at
        # wavelength[-1] (a no-op resample -- that channel already IS the sim grid).
        scale = wl[-1] / wl
        grid = self._BuildWavelengthResampleGrid(scale, Npix_sim, Npix_out)  # (Nwavelength, Npix_out, Npix_out, 2)

        Nphases_, Nwave_ = psf_sim.shape[:2]
        psf_flat = psf_sim.reshape(Nphases_ * Nwave_, 1, Npix_sim, Npix_sim)
        grid_flat = grid.unsqueeze(0).expand(Nphases_, -1, -1, -1, -1).reshape(
            Nphases_ * Nwave_, Npix_out, Npix_out, 2
        )
        resampled = torch.nn.functional.grid_sample(
            psf_flat, grid_flat, mode="bilinear", align_corners=True, padding_mode="zeros"
        )
        psf = resampled.reshape(Nphases_, Nwave_, Npix_out, Npix_out)
        if collapse_wvl:
            psf = psf.sum(dim=1)

        return psf

    def _BuildWavelengthResampleGrid(self, scale, Npix_sim, Npix_out):
        """Builds grid_sample coordinates (align_corners=True) mapping each of
        Npix_out centered output pixels to its corresponding location in an
        Npix_sim x Npix_sim simulation grid, separately for each wavelength's
        sim-pixels-per-output-pixel factor in `scale` (shape (Nwavelength,)).
        Returns (Nwavelength, Npix_out, Npix_out, 2), fully vectorized (no loop).
        """
        j = torch.arange(Npix_out, device=self.device, dtype=torch.float32)
        center_out = (Npix_out - 1) / 2
        center_sim = (Npix_sim - 1) / 2
        offset = j - center_out  # (Npix_out,)

        p_sim = center_sim + offset.view(1, -1) * scale.view(-1, 1)  # (Nwavelength, Npix_out)
        norm = 2 * p_sim / (Npix_sim - 1) - 1  # (Nwavelength, Npix_out), in [-1, 1]

        grid_x = norm.unsqueeze(1).expand(-1, Npix_out, -1)  # (Nwavelength, Npix_out, Npix_out)
        grid_y = norm.unsqueeze(2).expand(-1, -1, Npix_out)  # (Nwavelength, Npix_out, Npix_out)
        return torch.stack([grid_x, grid_y], dim=-1)

    def SetMask(self, phaseMask=None, transmisionMask=None):
        """
        Sets the phase mask by converting the input mask to a complex exponential and normalizing it.

        Args:
            phaseMask (torch tensor): Input phase mask (real-valued). 2-D is a single
                mask; 3-D is always interpreted as (Nmask, H, W) -- several masks
                sharing one wavelength (e.g. modulation steps), broadcast across
                whatever wavelength axis Propagator/GetPSF's field carries. A mask
                that must instead vary *per wavelength* -- even with only one mask --
                needs an explicit 4-D tensor shaped (Nwavelength, Nmask, H, W), which
                this method passes through unchanged; ordinary broadcasting against
                the (Nphases, Nwavelength, Nmask, H, W) propagated field then pairs
                wavelength indices instead of sharing them. See ZernikeWFS.BuildZernikeMaskFFT
                for the existing precedent of building such a tensor with an explicit
                unsqueeze rather than relying on this method's 3-D auto-unsqueeze.
                Note this derived self.mask only drives the FFT path (FFTPropagator);
                the MTF path (MTFPropagator) uses self.phaseMask/self.transmisionMask
                directly and relies on the same broadcasting rules independently.
            transmisionMask (torch tensor): Input transmision mask (real-valued), same
                shape convention as phaseMask.
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
                    1j * phaseMask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                )
            elif phaseMask.dim() == 3:
                self.mask = self.mask * torch.exp(1j * phaseMask.unsqueeze(0).unsqueeze(0))
            elif phaseMask.dim() == 4:
                self.mask = self.mask * torch.exp(1j * phaseMask.unsqueeze(0))
            else:
                raise ValueError(f"Phase mask has too many dimentions: {phaseMask.dim()}")

        if transmisionMask is not None:
            if transmisionMask.dim() == 2:
                self.mask = self.mask * transmisionMask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif transmisionMask.dim() == 3:
                self.mask = self.mask * transmisionMask.unsqueeze(0).unsqueeze(0)
            elif transmisionMask.dim() == 4:
                self.mask = self.mask * transmisionMask.unsqueeze(0)
            else:
                raise ValueError(f"Transmision mask has too many dimentions: {transmisionMask.dim()}")

    def BuildReferenceIntensity(self, opdOffset=0, pupil = None):
        """
        Builds the reference intensity by propagating a zero-OPD aberration.

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
            ) + opdOffset, pupil
        )
        self.reference_intensity = self.reference_intensity.squeeze()
        self.useNoise = tempUseNoise

    def BuildReconstructionMatrix(self, modes, pupil = None, batch_size=30, opdOffset=0):
        """
        Builds the reconstruction matrix as the inverse of the interaction matrix

        Args:
            modes (torch tensor): Modes (3D array with shape (Npix, Npix, Nmodes)) representing different OPD aberrations
            mask (torch tensor): Phase mask used in the propagation (not directly used in this function)
        Returns:
            None
        """
        self.BuildInteractionMatrix(modes, pupil, batch_size, opdOffset)
        self.reconstructionMatrix = torch.linalg.pinv(self.iMat.flatten(start_dim=-2))

    def BuildInteractionMatrix(self, modes, pupil = None, batch_size=30, opdOffset=0, single_pass = False):
        """
        Builds the interaction matrix by computing the signals for each mode using finite differences.

        Args:
            modes (torch tensor): Modes (3D array with shape (Npix, Npix, Nmodes)) representing different OPD aberrations
            mask (torch tensor): Phase mask used in the propagation (not directly used in this function)
        Returns:
            None
        """
        tempUseNoise = self.useNoise
        self.useNoise = False
        # Small OPD perturbation for the finite difference
        delta = 1 / 100

        if pupil is None:
            pupil = self.pupil.unsqueeze(0)


        if single_pass and self.reference_intensity is None:
            self.BuildReferenceIntensity(opdOffset, pupil)

        Nmodes = modes.shape[0]
        iMat_parts = []

        for i in range(0, Nmodes, batch_size):
            modes_batch = modes[i : i + batch_size]  # (Npix^2, batch_size)

            # reshape to (1, Npix, Npix, batch_size) if needed by Propagator
            push = self.Propagator(modes_batch * delta + opdOffset, pupil)
            if single_pass:
                pull = self.reference_intensity
                signal = (push - pull) / (1.0 * delta)
            else:
                pull = self.Propagator(-modes_batch * delta + opdOffset, pupil)
                signal = (push - pull) / (2.0 * delta)

            iMat_parts.append(signal)

        self.iMat = torch.cat(iMat_parts, dim=0).squeeze()  # shape: (Nmodes, Npix^2)
        self.useNoise = tempUseNoise

    def GetReconstructedOPD(self, intensity):
        """
        Reconstructs the OPD aberration from the intensity measurement by applying the reconstruction matrix.

        Args:
            intensity (torch tensor): Measured intensity (with noise, if applicable)
        Returns:
            torch tensor: Reconstructed OPD aberration, in meters
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
        else:
            self.requires_grad_(True)
            self.BuildMask()
            self.BuildReferenceIntensity()
        return self

    @property
    def wavelength(self):
        return self._wavelength
    @wavelength.setter
    def wavelength(self, value):
        """Scalar sensing wavelength (meters), or a 1-D tensor of Nwavelength
        wavelengths for parallel polychromatic propagation (see Propagator/GetPSF).
        """
        value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        assert not torch.is_complex(value) and torch.all(value > 0), "wavelength must be real and positive"
        assert value.dim() <= 1, "wavelength must be scalar (0-d) or a 1-D tensor of wavelengths"
        self._wavelength = value.reshape(-1)
        self.wavenumber = 2 * torch.pi / value.reshape(1, -1, 1, 1)
        # if wavenumber.dim() > 0:
        #     # dim 1 (not dim 0) so it lines up with the batch-first Propagator/GetPSF
        #     # convention: dim 0 is strictly Nphases, dim 1 is wavelength.
        #     wavenumber = wavenumber
        # self.wavenumber = wavenumber