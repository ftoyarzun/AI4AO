# -*- coding: utf-8 -*-
"""
Created on Tue  Mar 17 15:58 2026
@author: Matias Marambio-Jimenez
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from TorchPropagator import PoissonNoise

class ShackHartmann(nn.Module):
    """
    Shack-Hartmann wavefront sensor class based on OOPAO implementation.
    """
    def __init__(self, WFSParams, device=None):
        super().__init__()
        self.device = device
        # Telescope and detector parameters
        self.Nres = WFSParams["Nres"]
        self.D = WFSParams["D"]  # telescope diameter in meters
        self.c_obs = WFSParams["c_obs"]
        self.useNoise = WFSParams["useNoise"]
        self.beamSplitProportionForWFSDetector = WFSParams["beamSplitProportionForWFSDetector"]
        # Shack-Hartmann parameters
        self.wavelength = WFSParams["wavelength"]
        self.nSubap = WFSParams["nSubap"]
        self.lightRatio = WFSParams["lightRatio"]
        self.binning_factor = WFSParams["binning_factor"]
        self.pixel_scale_requested = WFSParams["pixel_scale"]  # in arcsec / unbinned pixel
        self.shannon_sampling = WFSParams["shannon_sampling"]
        self.n_pixel_per_subap = WFSParams["n_pixel_per_subap"]
        self.half_pixel_shift = WFSParams["half_pixel_shift"]

        if self.binning_factor < 1 or int(self.binning_factor) != self.binning_factor:
            raise ValueError(
                "binning_factor must be a positive integer"
                )
        self.binning_factor = int(self.binning_factor)

        if self.Nres % self.nSubap != 0:
            raise ValueError(
                f"Nres ({self.Nres}) must be divisible by nSubap ({self.nSubap})"
                )
        
        self.n_pix_subap_init = self.Nres // self.nSubap 
        self.d_subap = self.D / self.nSubap
        
        self.zero_padding = 2

        # pixel scale implementation
        self.pixel_scale_init, self.pixel_scale, self.binning_pixel_scale = self.configure_pixel_scale()
        # FFT stamp size at propagation sampling
        self.n_pix_lenslet_init = self.n_pix_subap_init * self.zero_padding
        self.center_init = self.n_pix_lenslet_init // 2

        # detector-side stamp size after pixel-scale resampling
        native_n_pix_subap = int(np.ceil(self.n_pix_lenslet_init / self.binning_pixel_scale))
        if native_n_pix_subap % 2 != 0:
            native_n_pix_subap += 1
        
        if self.n_pixel_per_subap is None:
            self.n_pix_subap = native_n_pix_subap
        else:
            if int(self.n_pixel_per_subap) != self.n_pixel_per_subap or self.n_pixel_per_subap <= 0:
                raise ValueError(
                    "n_pixel_per_subap must be a positive integer"
                    )
            if int(self.n_pixel_per_subap) % 2 != 0:
                raise ValueError(
                    "n_pixel_per_subap must be an even integer"
                    )
            self.n_pix_subap = int(self.n_pixel_per_subap)

        if self.n_pix_subap % self.binning_factor != 0:
            raise ValueError(
                f"n_pixel_per_subap ({self.n_pix_subap}) must be divisible by "
                f"binning_factor ({self.binning_factor})"
            )

        self.detector_pitch = self.n_pix_subap // self.binning_factor
        self.detector_size = self.nSubap * self.detector_pitch

        self.max_fov_arcsec = 0.5 * self.n_pix_subap * self.pixel_scale
        self.fov_pixel_binned_arcsec = self.pixel_scale * self.binning_factor

        tech_fov_arcsec = self.n_pix_lenslet_init * self.pixel_scale_init
        req_fov_arcsec = self.n_pix_subap * self.pixel_scale
        if req_fov_arcsec > tech_fov_arcsec + 1e-12:
            warnings.warn(
                "Requested SH FoV exceeds the non-wrapped FFT FoV. "
                "The class will zero-pad detector stamps, but no extra signal exists outside the "
                "technical FoV; use a larger pupil resolution if you need more physical FoV."
            )
        # Noise defaults
        self.Nphotons = 1e7
        self.RON = 2.
        self.focalplaneRON = 4.

        # build pupil
        x = torch.linspace(-self.Nres / 2, self.Nres / 2, self.Nres, dtype=torch.float32, device=self.device)
        self.x, self.y = torch.meshgrid(x, x, indexing="ij")
        radius = ((self.Nres + 1) / 2)
        self.pupil = torch.bitwise_and(((self.x**2 + self.y**2) <= radius**2), (self.x**2 + self.y**2) >= (self.c_obs * radius)**2).to(torch.float32)
        self.pupil_logical = torch.where(self.pupil.reshape(self.Nres * self.Nres) > 0)

        # edge mask used for wrap detection on the native FFT stamp
        self.outerMask = torch.ones(
            (self.n_pix_lenslet_init, self.n_pix_lenslet_init),
            dtype=torch.float32,
            device=self.device
        )
        self.outerMask[1:-1, 1:-1] = 0.0

        # precompute lenslet indexing over the pupil
        self._build_lenslet_geometry()

        # build subaperture flux cube and valid lenslet mask from the pupil
        self._initialize_flux_from_pupil()
        self._set_valid_subapertures()

        # optional phasor
        self._build_phasor()

        # buffers updated during propagation
        self.frame_no_noise = None
        self.frame_with_noise = None
        self.maps_intensity = None
        self.psf_no_noise = None
        self.psf_with_noise = None

        self._SetPhotonsAndRON(WFSParams["Nphotons"][0], WFSParams["RON"][0])

        ## init matched filter for SNR calc
        self.matched_filter = nn.Conv2d(1, 1, 8, 8, bias=False, device=self.device, padding=0)
        kernel_size = 8
        fwhm = 0.73  # arcseconds
        fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))  # ≈ 0.4247
        sigma_physical = fwhm * fwhm_to_sigma
        sigma_pixels = sigma_physical / self.pixel_scale
        # Create coordinate grid centered at the kernel center
        center = kernel_size / 2 - 0.5
        x = np.arange(kernel_size) - center
        y = np.arange(kernel_size) - center
        xx, yy = np.meshgrid(x, y)
        # Create Gaussian kernel
        gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma_pixels**2))
        # Normalize so weights sum to 1
        gaussian = gaussian / gaussian.sum()
        # Convert to PyTorch tensor [out_channels, in_channels, height, width]
        kernel = torch.from_numpy(gaussian).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            self.matched_filter.weight.copy_(kernel)
        # 8x8 pupil:
        x = torch.linspace(-8 / 2, 8 / 2, 8, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, x, indexing="ij")
        radius = (8 + 1) / 2
        self.snr_pupil = torch.bitwise_and(((xx**2 + yy**2) <= radius**2), (xx**2 + yy**2) >= (self.c_obs * radius)**2).to(torch.float32)

    def configure_pixel_scale(self):
        self.pixel_scale_init = np.rad2deg(self.wavelength / self.d_subap / self.zero_padding) * 3600.
        if self.pixel_scale_requested is None:
            # OOPAO convention:
            # shannon_sampling=True  -> 1 * pixel_scale_init
            # shannon_sampling=False -> 2 * pixel_scale_init
            binning_pixel_scale = int(2 - int(self.shannon_sampling))
            pixel_scale = self.pixel_scale_init * binning_pixel_scale
            return self.pixel_scale_init, pixel_scale, binning_pixel_scale
        pixel_scale_requested = float(self.pixel_scale_requested)
        if pixel_scale_requested <= 0:
            raise ValueError("pixel_scale must be > 0")
        ratio = pixel_scale_requested / self.pixel_scale_init
        # If a finer pixel scale is requested, increase the FFT sampling first
        while ratio < .95:
            self.zero_padding += 1
            self.pixel_scale_init = np.rad2deg(self.wavelength / self.d_subap / self.zero_padding) / 3600
            ratio  = pixel_scale_requested / self.pixel_scale_init
        candidates = np.array([
            max(1, int(np.floor(pixel_scale_requested / self.pixel_scale_init))),
            max(1, int(np.ceil(pixel_scale_requested / self.pixel_scale_init)))
        ], dtype=int)
        candidates = np.unique(candidates)
        errors = np.abs(candidates * self.pixel_scale_init - pixel_scale_requested)
        binning_pixel_scale = int(candidates[np.argmin(errors)])
        pixel_scale = self.pixel_scale_init * binning_pixel_scale
        if abs(pixel_scale - pixel_scale_requested) > 1e-12:
            warnings.warn(
                f"Requested pixel_scale={pixel_scale_requested:.6g} arcsec/pix, "
                f"effective value is {pixel_scale:.6g} arcsec/pix."
            )
        return self.pixel_scale_init, pixel_scale, binning_pixel_scale

    def _build_phasor(self):
        """
        OOPAO-like phasor used to center spots within each lenslet stamp.
        """
        coords = torch.linspace(0, self.n_pix_lenslet_init - 1, self.n_pix_lenslet_init, dtype=torch.float32, device=self.device)
        xx, yy = torch.meshgrid(coords, coords, indexing="ij")
        half_shift = (self.pixel_scale / self.pixel_scale_init) * float(self.half_pixel_shift)
        phase_term = (
            -1j * np.pi * (self.n_pix_lenslet_init + 1 + half_shift) / self.n_pix_lenslet_init * (xx + yy)
            )
        self.phasor = torch.exp(phase_term).to(torch.complex64)

    def sum_bin_last2(self, x, factor):
        """
        Sum-bin the last two dimensions by an integer factor.

        Input:
            x: (B, L, H, W) or (B, H, W)
        """
        if factor == 1:
            return x

        if x.dim() == 4:
            B, L, H, W = x.shape
            if H % factor != 0 or W % factor != 0:
                raise ValueError(f"Cannot bin shape {(H, W)} by factor {factor}.")
            x = x.view(B, L, H // factor, factor, W // factor, factor).sum(dim=(3, 5))
            return x

        if x.dim() == 3:
            B, H, W = x.shape
            if H % factor != 0 or W % factor != 0:
                raise ValueError(f"Cannot bin shape {(H, W)} by factor {factor}.")
            x = x.view(B, H // factor, factor, W // factor, factor).sum(dim=(2, 4))
            return x

        raise ValueError("Input must have 3 or 4 dimensions.")



    def _build_lenslet_geometry(self):
        self.index_x = []
        self.index_y = []
        self.lenslet_slices = []

        for i in range(self.nSubap):
            x0 = i * self.n_pix_subap_init
            x1 = (i + 1) * self.n_pix_subap_init
            for j in range(self.nSubap):
                y0 = j * self.n_pix_subap_init
                y1 = (j + 1) * self.n_pix_subap_init
                self.index_x.append(i)
                self.index_y.append(j)
                self.lenslet_slices.append((slice(x0, x1), slice(y0, y1)))
        self.index_x = torch.tensor(self.index_x, device=self.device, dtype=torch.long)
        self.index_y = torch.tensor(self.index_y, device=self.device, dtype=torch.long)  

    def _initialize_flux_from_pupil(self):
        """
        OOPAO-like initialize_flux using the telescope flux map.
        """
        n_lenslets = self.nSubap ** 2
        cube_flux = torch.zeros((n_lenslets, self.n_pix_lenslet_init, self.n_pix_lenslet_init),
                                 device=self.device, dtype=torch.float32)
        k = 0
        # Insert the subaperture flux into the center of each zero-padded
        # lenslet stamp. The inserted block has size `n_pix_subap_init` (the
        # subaperture size), not `n_pix_lenslet_init` (the zero-padded stamp
        # size). Using the subaperture size ensures shapes match when
        # assigning `sub_flux` (which is `n_pix_subap_init x n_pix_subap_init`).
        
        x_insert_0 = self.center_init - self.n_pix_subap_init // 2
        x_insert_1 = self.center_init + self.n_pix_subap_init // 2
        y_insert_0 = self.center_init - self.n_pix_subap_init // 2
        y_insert_1 = self.center_init + self.n_pix_subap_init // 2

        for xs, ys in self.lenslet_slices:
            sub_flux = self.pupil[xs, ys]
            cube_flux[k, x_insert_0:x_insert_1, y_insert_0:y_insert_1] = sub_flux
            k += 1

        self.cube_flux = cube_flux
        self.photon_per_subaperture = cube_flux.sum(dim=(1, 2))

    def _set_valid_subapertures(self):
        """
        OOPAO-like valid lenslet selection based on relative flux threshold.
        """
        max_flux = torch.max(self.photon_per_subaperture)
        if max_flux <= 0:
            valid_id = torch.zeros_like(self.photon_per_subaperture, dtype=torch.bool)
        else:
            valid_id = self.photon_per_subaperture >= self.lightRatio * max_flux
        self.valid_subapertures_id = valid_id
        self.valid_subapertures = valid_id.view(self.nSubap, self.nSubap)
        self.nValidSubap = int(valid_id.sum().item())
        valid_2d_idx = torch.nonzero(self.valid_subapertures, as_tuple=False)
        if valid_2d_idx.numel() == 0:
            self.validLenslets_x = torch.empty(0, dtype=torch.long, device=self.device)
            self.validLenslets_y = torch.empty(0, dtype=torch.long, device=self.device)
        else:
            self.validLenslets_x = valid_2d_idx[:, 0]
            self.validLenslets_y = valid_2d_idx[:, 1]

    def get_lenslet_em_field(self, phase):
        if phase.dim() != 3:
            raise ValueError("phase must have shape (B, Nres, Nres).")

        B = phase.shape[0]
        n_lenslets = self.nSubap ** 2

        uin = self.pupil.unsqueeze(0) * torch.exp(1j * phase)
        uin = uin / torch.sqrt(self.pupil.sum())

        em_field = torch.zeros(
            (B, n_lenslets, self.n_pix_lenslet_init, self.n_pix_lenslet_init),
            dtype=torch.complex64,
            device=self.device
        )

        x_insert_0 = self.center_init - self.n_pix_subap_init // 2
        x_insert_1 = self.center_init + self.n_pix_subap_init // 2
        y_insert_0 = self.center_init - self.n_pix_subap_init // 2
        y_insert_1 = self.center_init + self.n_pix_subap_init // 2

        for k, (xs, ys) in enumerate(self.lenslet_slices):
            sub_field = uin[:, xs, ys]
            em_field[:, k, x_insert_0:x_insert_1, y_insert_0:y_insert_1] = sub_field

        em_field = em_field * self.phasor.unsqueeze(0).unsqueeze(0)
        return em_field
    
    def crop_or_pad_to_size(self, stamps, target_size):
        """
        Center crop or symmetric zero-pad each lenslet stamp to target_size.
        """
        B, L, H, W = stamps.shape
        if H != W:
            raise ValueError("Lenslet stamps must be square.")
        if H == target_size:
            return stamps
        if H > target_size:
            start = (H - target_size) // 2
            end = start + target_size
            return stamps[:, :, start:end, start:end]
        pad_total = target_size - H
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return F.pad(stamps, (pad_before, pad_after, pad_before, pad_after))
    
    def resample_lenslet_stamps(self, intensity):
        """
        Convert native FFT stamps to the requested detector pixel scale, then to the
        requested FoV per subaperture, then optionally apply detector binning.
        """
        if self.pixel_scale < self.pixel_scale_init - 1e-12:
            raise ValueError(
                f"The smallest achievable pixel scale is {self.pixel_scale_init:.6g} arcsec/pix."
            )

        # 1) Convert FFT sampling -> requested unbinned detector pixel scale
        if self.binning_pixel_scale > 1:
            remainder = intensity.shape[-1] % self.binning_pixel_scale
            if remainder != 0:
                pad_total = self.binning_pixel_scale - remainder
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                intensity = F.pad(
                    intensity,
                    (pad_before, pad_after, pad_before, pad_after),
                    mode="constant",
                    value=0.0
                )
            intensity = self.sum_bin_last2(intensity, self.binning_pixel_scale)

        # 2) Match requested FoV per subaperture
        intensity = self.crop_or_pad_to_size(intensity, self.n_pix_subap)

        # 3) Optional camera binning
        if self.binning_factor > 1:
            intensity = self.sum_bin_last2(intensity, self.binning_factor)

        return intensity


    def add_stamp_raw_data(self, ind_x, ind_y, stamps, detector_shape=None):
        """
        Parameters
        ----------
        ind_x, ind_y : torch.Tensor
            Shape (nStamp,), lenslet indices on the SH grid.
        stamps : torch.Tensor
            Shape (B, nStamp, sy, sx), sy and sx are coordinates within subaperture 
        detector_shape : tuple or None
            Optional (Ny, Nx). If None, uses the standard SH detector size.

        Returns
        -------
        raw_data : torch.Tensor
            Shape (B, Ny, Nx)
        """
        if stamps.dim() != 4:
            raise ValueError("stamps must have shape (B, nStamp, sy, sx).")

        B, nStamp, sy, sx = stamps.shape
        device = stamps.device
        dtype = stamps.dtype

        if not torch.is_tensor(ind_x):
            ind_x = torch.as_tensor(ind_x, device=device, dtype=torch.long)
        else:
            ind_x = ind_x.to(device=device, dtype=torch.long)

        if not torch.is_tensor(ind_y):
            ind_y = torch.as_tensor(ind_y, device=device, dtype=torch.long)
        else:
            ind_y = ind_y.to(device=device, dtype=torch.long)

        if ind_x.numel() != nStamp or ind_y.numel() != nStamp:
            raise ValueError("ind_x and ind_y must have shape (nStamp,) matching stamps.")

        pitch = self.n_pix_subap // self.binning_factor

        if detector_shape is None:
            Ny = self.nSubap * pitch
            Nx = self.nSubap * pitch
        else:
            Ny, Nx = detector_shape

        cx = ind_x * pitch + pitch // 2  # central coordinates for each stamp
        cy = ind_y * pitch + pitch // 2
        x0 = cx - sy // 2
        y0 = cy - sx // 2
        dx = torch.arange(sy, device=device, dtype=torch.long)
        dy = torch.arange(sx, device=device, dtype=torch.long)
        gx = x0[:, None, None] + dx[None, :, None] # (nStamp, sy, 1)
        gy = y0[:, None, None] + dy[None, None, :] # (nStamp, 1, sx)
        gx = gx.expand(nStamp, sy, sx)             # (nStamp, sy, sx)
        gy = gy.expand(nStamp, sy, sx)             # (nStamp, sy, sx)
        valid = (gx >= 0) & (gx < Ny) & (gy >= 0) & (gy < Nx)
        flat_idx = (gx * Nx + gy).reshape(nStamp, sy * sx)
        flat_idx = flat_idx.clamp(0, Ny * Nx - 1)
        src = stamps.reshape(B, nStamp, sy * sx)
        src = src * valid.reshape(1, nStamp, sy * sx).to(dtype)
        raw_data_stamps_flat = torch.zeros((B, nStamp, Ny * Nx),
                                            dtype=dtype,  device=device)
        index = flat_idx.unsqueeze(0).expand(B, -1, -1)
        raw_data_stamps_flat.scatter_add_(dim=2, index=index, src=src)
        raw_data_stamps = raw_data_stamps_flat.view(B, nStamp, Ny, Nx)
        raw_data = raw_data_stamps.sum(dim=1)
        return raw_data

    def split_raw_data(self, input_frame=None, valid_only=False):
        """
        The detector frame is split into nSubap x nSubap tiles of size
        (n_pix_subap // binning_factor), and each tile is embedded in the center
        of an analysis window of size (n_pix_subap, n_pix_subap).
        """
        if input_frame is None:
            input_frame = self.frame_no_noise

        if input_frame.dim() == 2:
            input_frame = input_frame.unsqueeze(0)

        B = input_frame.shape[0]
        det_pitch = self.n_pix_subap // self.binning_factor
        center = self.n_pix_subap // 2

        maps_intensity = torch.zeros(
            (B, self.nSubap ** 2, self.n_pix_subap, self.n_pix_subap),
            dtype=input_frame.dtype, device=input_frame.device,
        )

        xw0 = center - det_pitch // 2
        xw1 = center + det_pitch // 2
        yw0 = center - det_pitch // 2
        yw1 = center + det_pitch // 2

        k = 0
        row_chunks = torch.chunk(input_frame, self.nSubap, dim=-2)
        for i in range(self.nSubap):
            col_chunks = torch.chunk(row_chunks[i], self.nSubap, dim=-1)
            for j in range(self.nSubap):
                maps_intensity[:, k, xw0:xw1, yw0:yw1] = col_chunks[j]
                k += 1

        if valid_only:
            maps_intensity = maps_intensity[:, self.valid_subapertures_id]

        return maps_intensity

    @torch.no_grad()
    def Propagator(self, phase):
        """
        Diffractive Shack-Hartmann WFS propagation.
        Steps:
        1) build full FFT spot stamps per lenslet,
        2) resample to the requested detector pixel scale,
        3) optionally crop/pad to the requested subaperture FoV,
        4) optionally bin detector pixels,
        5) add each stamp to the global detector with overlap,
        6) re-window the detector into subaperture analysis cubes.
        """
        if phase.dim() == 2:
            phase = phase.unsqueeze(0)
        if phase.shape[-2:] != (self.Nres, self.Nres):
            raise ValueError(f"phase must have shape (B, {self.Nres}, {self.Nres}),\n\tgot {tuple(phase.shape)}.")
        
        # Full lenslet EM fields, zero-padded on the lenslet FFT grid
        lenslet_em_field = self.get_lenslet_em_field(phase)
        norma = float(self.cube_flux.shape[1])  # lenslet grid size as FFT normalization
        # Full lenslet stamps before detector-plane assembly
        intensity = torch.abs(torch.fft.fft2(lenslet_em_field, dim=(-2, -1)) / norma) **2
        # intensity is shape (B, nSubap^2, H, W)
        valid_cube = intensity[:, self.valid_subapertures_id]
        self.sum_intensity = valid_cube.sum(dim=1)

        denom = valid_cube.sum(dim=(1, 2, 3)).clamp_min(1e-12)
        numer = (valid_cube * self.outerMask.unsqueeze(0).unsqueeze(0)).sum(dim=(1, 2, 3))
        self.edge_subaperture_criterion = numer / denom

        # Convert to requested detector sampling/FoV
        
        intensity_detector = self.resample_lenslet_stamps(intensity)

        
        self.raw_data = self.add_stamp_raw_data(
            ind_x=self.index_x,
            ind_y=self.index_y,
            stamps=intensity_detector,
            detector_shape=(self.detector_size, self.detector_size),
        )

        self.frame_no_noise = self.raw_data
        flux = self.frame_no_noise.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        self.frame_no_noise = self.frame_no_noise / flux
        self.psf_no_noise = self.frame_no_noise.clone()

        if not self.useNoise:
            self.maps_intensity = self.split_raw_data(input_frame=self.frame_no_noise)
            return self.frame_no_noise

        self._AddNoiseToFrame()
        self.maps_intensity = self.split_raw_data(input_frame=self.frame_with_noise)
        return self.frame_with_noise
    
    def _SetPhotonsAndRON(self, Nphotons, RON):
        self.Nphotons = torch.pow(10, torch.Tensor([Nphotons], device=self.device))
        self.RON = RON
    
    def _AddNoiseToFrame(self):
        self.frame_with_noise = PoissonNoise(
            self.frame_no_noise * self.Nphotons * self.beamSplitProportionForWFSDetector
            ) + self.RON * torch.randn_like(self.frame_no_noise)
        flux = self.frame_with_noise.sum(dim=(-2, -1), keepdim=True)
        flux = torch.clamp(flux, min=1e-12)
        self.frame_with_noise = self.frame_with_noise / flux
        if self.beamSplitProportionForWFSDetector < 1.0:
            self.psf_with_noise = PoissonNoise(
                self.psf_no_noise * self.Nphotons * (1.0 - self.beamSplitProportionForWFSDetector) + self.focalplaneRON * torch.randn_like(self.psf_no_noise)
                )
        else:
            self.psf_with_noise = self.psf_no_noise


    def calcSNR(self):
        with torch.no_grad():
            signal = self.matched_filter(self.frame_with_noise.unsqueeze(1)) * self.snr_pupil
            noise = torch.sqrt(signal)  
            return signal / noise        
