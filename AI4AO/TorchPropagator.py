# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:02:43 2024

@author: foyarzun
"""

import torch # type: ignore[import]
import torch.nn as nn  # type: ignore[import]

import numpy as np

from torch.fft import fft2, fftshift, ifft2, ifftshift # type: ignore[import]

from .Utils import MakePupil, set_frozen
from .paths import ensure_parent


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
        self.use_MTF = ParamsDict.get("Use_MTF", False)
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
        [self.x_mask, self.y_mask] = torch.meshgrid(x_mask, x_mask, indexing="ij")

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
        self.frame_no_noise = self.ShiftPupilImages(torch.abs(psi_zwfs) ** 2)

    def BuildShiftGrid(self, pupil_shift_row, pupil_shift_col):
        """
        Precomputes and caches the per-mask-channel grid_sample sampling grid used
        by ShiftPupilImages (self.shift_grid, shape (Nmask, Npix, Npix, 2)), so it
        isn't rebuilt on every forward call -- only when the mask geometry (hence
        the per-channel pixel shift) changes. Call this once at mask-build time
        (e.g. from BuildZernikeMaskMFT, alongside MakeMTFMatrices), after the row/
        col pixel shifts for each mask channel are known.

        Args:
            pupil_shift_row (array-like): shape (Nmask,), row (dims=-2) pixel offset per mask channel.
            pupil_shift_col (array-like): shape (Nmask,), col (dims=-1) pixel offset per mask channel.
        Returns:
            None
        """
        H = W = self.Npix
        ys = torch.linspace(-1, 1, H, device=self.device, dtype=torch.float32)
        xs = torch.linspace(-1, 1, W, device=self.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

        row_shift = torch.as_tensor(pupil_shift_row, device=self.device, dtype=torch.float32)  # (Nmask,)
        col_shift = torch.as_tensor(pupil_shift_col, device=self.device, dtype=torch.float32)  # (Nmask,)
        grid_y = grid_y.unsqueeze(0) - (2 * row_shift / (H - 1)).view(-1, 1, 1)  # (Nmask, H, W)
        grid_x = grid_x.unsqueeze(0) - (2 * col_shift / (W - 1)).view(-1, 1, 1)  # (Nmask, H, W)
        self.shift_grid = torch.stack([grid_x, grid_y], dim=-1)  # (Nmask, H, W, 2), grid_sample wants (x, y) last

    def ShiftPupilImages(self, frame):
        """
        Shifts each mask channel of a (..., Nmask, H, W) frame to its own detector
        position with a single batched grid_sample call, using the per-channel
        grid cached by BuildShiftGrid. Replaces a per-channel torch.roll loop that
        only ever applied the row component and rounded the shift to whole pixels.

        Args:
            frame (torch tensor): real-valued frame, shape (..., Nmask, H, W).
        Returns:
            torch tensor: same shape, each mask channel independently shifted.
        """
        lead_shape = frame.shape[:-3]
        Nmask, H, W = frame.shape[-3:]
        flat = frame.reshape(-1, Nmask, H, W).movedim(1, 0)  # (Nmask, B, H, W)
        B = flat.shape[1]

        grid = self.shift_grid.unsqueeze(1).expand(-1, B, -1, -1, -1).reshape(Nmask * B, H, W, 2)
        shifted = torch.nn.functional.grid_sample(
            flat.reshape(Nmask * B, 1, H, W), grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )
        return shifted.reshape(Nmask, B, H, W).movedim(0, 1).reshape(*lead_shape, Nmask, H, W)

    def BuildMask(self):
        pass

    def forward(self, opd, pupil = None):
        if self.training: 
            self.BuildMask()
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

        self.frame_with_noise = self.AddNoiseToFrame()
        return self.frame_with_noise
    
    def PropagateField(self, uin, uin_padded):
        if self.use_MTF:
            self.MTFPropagator(uin, uin_padded)
        else:
            self.FFTPropagator(uin_padded)

    def SetPhotonsAndRON(self, Nphotons, RON):
        self.Nphotons = Nphotons
        self.RON = RON

    def AddNoiseToFrame(self, frame = None):
        if frame is None:
            frame = self.frame_no_noise
        frame_with_noise = PoissonNoise(
            frame * self.Nphotons
        ) + self.RON * torch.randn_like(frame)
        frame_with_noise /= frame_with_noise.sum(dim=(-2, -1), keepdim=True)
        return frame_with_noise

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
                channel is area-integrated (never upsampled; flux-conserving) onto the
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
        FFT oversampled for self.wavelength[-1], then area-integrates each
        wavelength channel onto the self.wavelength[0]-scale output grid
        (_AreaResample, flux-conserving) before summing, since the channels are not natively on the same physical
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

        if wl.numel() == 1:
            # Single wavelength: sim_sampling == sampling, so the sim grid already
            # is the output grid -- nothing to resample.
            return psf_sim if not collapse_wvl else psf_sim.sum(dim=1)

        # Sim pixels per output pixel, per wavelength (>= ~1; ~1 at wavelength[-1]).
        # An FFT pixel at wavelength w spans w * Nres / (D * Npix) radians, so
        # use the actual (pad-rounded) grid sizes rather than wl[-1] / wl.
        scale = (wl[0] * Npix_sim) / (wl * Npix_out)
        psf = self._AreaResample(psf_sim, scale, Npix_out)  # (Nphases, Nwavelength, Npix_out, Npix_out)
        # _AreaResample returns each output pixel's integral over scale**2 sim
        # pixels; divide by wavelength[0]'s footprint so the result keeps
        # GetPSF's monochromatic units (a wavelength[0] channel matches a
        # monochromatic GetPSF on the same grid). Longer wavelengths then come
        # out (wl[0] / w)**2 dimmer per pixel -- same energy spread over a
        # larger angular PSF -- and every channel carries equal flux, up to what
        # its wider wings lose outside the field of view.
        psf = psf * (Npix_out / Npix_sim) ** 2
        if collapse_wvl:
            psf = psf.sum(dim=1)

        return psf

    def _AreaResample(self, psf_sim, scale, Npix_out):
        """Area-integrating resample of each wavelength channel of psf_sim
        (..., Nwavelength, Npix_sim, Npix_sim) onto a centered Npix_out grid whose
        pixels are `scale` (shape (Nwavelength,), >= 1) sim pixels wide.

        Each output pixel gets the integral of the sim image over its footprint
        (treating sim pixels as constant-valued squares), so every channel keeps
        its flux and decimation doesn't alias -- unlike bilinear point-sampling,
        which picks up one sim pixel's worth of intensity per output pixel.
        Implemented with a summed-area table: bilinear interpolation of the
        cumulative integral is exact for piecewise-constant pixels, so one
        grid_sample at the output pixel *edges* plus a 2x2 difference gives the
        exact footprint integrals, fully vectorized and differentiable w.r.t.
        both the image and `scale`. The table is built in float64 because the
        corner differences of a cumulative sum would otherwise lose the faint
        PSF wings to float32 cancellation.
        """
        lead_shape = psf_sim.shape[:-3]
        Nwave, Npix_sim = psf_sim.shape[-3], psf_sim.shape[-1]
        flat = psf_sim.reshape(-1, Nwave, Npix_sim, Npix_sim).double()
        B = flat.shape[0]

        # sat[..., a, b] = sum of pixels [0, a) x [0, b): the integral up to sim
        # coordinate a - 0.5 (pixel i spans [i - 0.5, i + 0.5]).
        sat = torch.nn.functional.pad(flat.cumsum(dim=-2).cumsum(dim=-1), (1, 0, 1, 0))

        # Output pixel edges, in sim coordinates. fftshift puts the optical axis
        # (zero frequency) at index N // 2, not the geometric center (N - 1) / 2,
        # so scale about N // 2 on both grids to keep every channel on-axis.
        k = torch.arange(Npix_out + 1, device=psf_sim.device, dtype=torch.float64)
        edges = Npix_sim // 2 + (k.view(1, -1) - Npix_out // 2 - 0.5) * scale.double().view(-1, 1)  # (Nwavelength, Npix_out + 1)
        # sat index a = edge + 0.5; align_corners=True maps index 0..Npix_sim to [-1, 1].
        # Border padding is exact outside the table: the integral stops growing.
        norm = 2 * (edges + 0.5) / Npix_sim - 1
        grid = torch.stack([
            norm.unsqueeze(1).expand(-1, Npix_out + 1, -1),  # x varies along the last axis
            norm.unsqueeze(2).expand(-1, -1, Npix_out + 1),  # y varies along rows
        ], dim=-1)  # (Nwavelength, Npix_out + 1, Npix_out + 1, 2)

        corners = torch.nn.functional.grid_sample(
            sat.reshape(B * Nwave, 1, Npix_sim + 1, Npix_sim + 1),
            grid.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * Nwave, Npix_out + 1, Npix_out + 1, 2),
            mode="bilinear", align_corners=True, padding_mode="border",
        ).reshape(B, Nwave, Npix_out + 1, Npix_out + 1)

        out = corners[..., 1:, 1:] - corners[..., :-1, 1:] - corners[..., 1:, :-1] + corners[..., :-1, :-1]
        return out.to(psf_sim.dtype).reshape(*lead_shape, Nwave, Npix_out, Npix_out)

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
                PyramidWFS.BuildMask and ZernikeWFS.BuildZernikeMaskFFT always pass
                this 4-D form, with Nwavelength = len(self.wavelength), since their
                masks scale with lambda_c / lambda (see WFS.ChromaticRatio).
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
        # map_location so a checkpoint saved on CUDA still loads on a CPU-only machine
        checkpoint = torch.load(file_path, map_location=self.device)
        model = checkpoint["model"]
        self.load_state_dict(model)

        with torch.no_grad():
            self.BuildMask()
            self.BuildReferenceIntensity()

    def SaveCalibration(self, file_path):
        ensure_parent(file_path)
        torch.save({"model": self.state_dict()}, file_path)

    def train(self, mode=True):
        # Let PyTorch handle the normal train/eval behavior
        super().train(mode)

        # eval() freezes every parameter; train() restores the per-parameter
        # requires_grad flags from before the freeze rather than forcing all True
        set_frozen(self, not mode)
        if not mode:
            with torch.no_grad():
                self.BuildMask()
                self.BuildReferenceIntensity()
        else:
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

        # The mask may depend on the band (see ChromaticRatio), so rebuild it and
        # the reference once construction is done -- this setter also runs inside
        # __init__, before any mask exists.
        if getattr(self, "initialized", False):
            self.RebuildMaskAndReference()

    def RebuildMaskAndReference(self):
        """Rebuild the mask and the reference intensity after a mask setting
        changed, without a graph when not training (mirrors train())."""
        if self.training:
            self.BuildMask()
            self.BuildReferenceIntensity()
        else:
            with torch.no_grad():
                self.BuildMask()
                self.BuildReferenceIntensity()

    def ChromaticRatio(self):
        """lambda_c / lambda for each sensing wavelength, shape (Nwavelength,),
        with lambda_c = (min + max) / 2 of the current band.

        A physical focal-plane mask is fixed in angle, while a shared FFT grid
        has lambda / (D * sampling) per pixel at wavelength lambda. A mask
        feature specified in lambda_c/D (modulation radius, rooftop width, dot
        diameter) therefore spans `sampling * ChromaticRatio()` pixels at each
        wavelength. Exactly 1.0 for a single wavelength, so monochromatic masks
        are unchanged.
        """
        wl = self.wavelength
        return (wl.min() + wl.max()) / 2 / wl
        # if wavenumber.dim() > 0:
        #     # dim 1 (not dim 0) so it lines up with the batch-first Propagator/GetPSF
        #     # convention: dim 0 is strictly Nphases, dim 1 is wavelength.
        #     wavenumber = wavenumber
        # self.wavenumber = wavenumber