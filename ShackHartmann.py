# -*- coding: utf-8 -*-
"""
Created on Tue  Mar 17 15:58 2026
@author: Matias Marambio-Jimenez
"""

import warnings
import poppy
import torch
import torch.nn as nn
import torch.nn.functional as F
import astropy.units as u
import numpy as np
from TorchPropagator import PoissonNoise
from torchvision.transforms import Resize, InterpolationMode

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
        self.separateNoise = WFSParams["separateNoise"]
        self.beamSplitProportionForWFSDetector = WFSParams["beamSplitProportionForWFSDetector"]
        self.QE = WFSParams["QE"]
        self.FWC = WFSParams["FWC"]
        self.gain = WFSParams["gain"]
        self.bits = WFSParams["bits"]
        self.sensor = WFSParams["sensor"]
        self.darkCurrent = WFSParams["darkCurrent"]
        self.applyDigitalization = WFSParams["applyDigitalization"]
        self.exposureTime = WFSParams["exposureTime"]
        self.backgroundRange = WFSParams["backgroundRange"]
        if not (0.0 <= self.QE <= 1.0):
            raise ValueError(f"QE must be between 0 and 1, got {self.QE}")
        if self.FWC is not None and self.FWC <= 0:
            raise ValueError(f"FWC must be > 0, got {self.FWC}")
        if self.sensor not in ["EMCCD", "CCD", "CMOS"]:
            raise ValueError("sensor must be 'EMCCD', 'CCD', or 'CMOS'")
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

        self.pixel_area_arcsec2 = self.fov_pixel_binned_arcsec ** 2

        # Noise defaults
        self.Nphotons = 1e7
        self.RON = 2.

        # IRCS+AO188 pupil parameters
        sp_offset = 1.278  # Spider offset [m]
        sp_angle = 51.75   # Spider angle [deg]
        sp_thick = 0.224   # Spider thickness [m]

        # Subaru IRCS+AO188
        ap = poppy.CircularAperture(radius=self.D/2.0*u.m)
        sec = poppy.AsymmetricSecondaryObscuration(
                secondary_radius=self.c_obs/2.0*u.m,
                support_angle=(90-sp_angle, 90+sp_angle, 270-sp_angle, 270+sp_angle),
                support_width=[sp_thick, sp_thick, sp_thick, sp_thick],
                support_offset_x=[-sp_offset/2., -sp_offset/2., sp_offset/2., sp_offset/2.])
        pupil = poppy.CompoundAnalyticOptic(opticslist=[ap,sec], name='AO188+IRCS')
        mask = pupil.sample(npix=self.Nres, grid_size=self.D, what='amplitude')
        self.pupil = torch.from_numpy(mask).to(self.device)

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
        self.frame_photons_no_noise = None
        self.frame_photons_with_noise = None
        self.frame_electrons_no_noise = None
        self.frame_electrons_with_noise = None
        self.saturation = 0.0
        self.quantification_noise = 0.0
        self.photon_noise = 0.0
        self.dark_shot_noise = 0.0
        

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
        # pupils
        resizer = Resize(size=(self.nSubap, self.nSubap), interpolation=InterpolationMode.BILINEAR)
        # pupil for bg flux
        self.bg_pupil = resizer(self.pupil.unsqueeze(0)).squeeze()
        # valid pupil for snr calc
        self.snr_pupil = self.bg_pupil > 0.2


    def photons_to_electrons(self, frame_photons):
        return frame_photons * self.QE

    def apply_saturation(self, frame_electrons):
        if self.FWC is None:
            self.saturation = 0.0
            return frame_electrons

        max_val = float(frame_electrons.max().detach().item())
        self.saturation = 100.0 * max_val / self.FWC
        if max_val > self.FWC:
            warnings.warn(f"The detector is saturating, {self.saturation:.1f} %")
        return torch.clamp(frame_electrons, min=0.0, max=self.FWC)

    def apply_dark_current(self, frame_electrons):
        if self.darkCurrent == 0:
            self.dark_shot_noise = 0.0
            return frame_electrons
        mean_dark = self.darkCurrent * self.exposureTime
        dark_map = torch.full_like(frame_electrons, mean_dark)
        dark_noise = PoissonNoise(dark_map)
        self.dark_shot_noise = float(np.sqrt(self.darkCurrent))
        return frame_electrons + dark_noise

    def apply_readout_noise(self, frame_electrons):
        if self.RON == 0:
            return frame_electrons
        return frame_electrons + self.RON * torch.randn_like(frame_electrons)

    def apply_gain(self, frame):
        return frame * self.gain

    def digitalize(self, frame):
        if self.bits is None:
            self.quantification_noise = 0.0
            return frame

        if self.FWC is None:
            max_val = frame.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            self.quantification_noise = 0.0
            out = frame / max_val * (2**self.bits)
            return torch.clamp(out, min=0.0, max=float(2**self.bits - 1))

        self.quantification_noise = self.FWC * (2**(-self.bits)) / np.sqrt(12)
        out = frame / self.FWC * (2**self.bits - 1)
        return torch.clamp(out, min=0.0, max=float(2**self.bits - 1)).to(torch.uint16)


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
            self.pixel_scale_init = np.rad2deg(self.wavelength / self.d_subap / self.zero_padding) * 3600
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
        s = self.n_pix_subap_init
        L = self.nSubap ** 2

        sub_flux = self.pupil.unfold(0, s, s).unfold(1, s, s)
        sub_flux = sub_flux.contiguous().view(L, s, s)

        pad_total = self.n_pix_lenslet_init - s
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before

        cube_flux = F.pad(
            sub_flux,
            (pad_before, pad_after, pad_before, pad_after),
            mode="constant",
            value=0.0,
        )

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

        device = phase.device
        pupil = self.pupil.to(device)
        phasor = self.phasor.to(device)

        B = phase.shape[0]
        s = self.n_pix_subap_init
        L = self.nSubap ** 2

        # Complex field in the pupil plane
        uin = pupil.unsqueeze(0) * torch.exp(1j * phase)
        uin = uin / torch.sqrt(pupil.sum())

        # Extract all subapertures at once:
        # (B, nSubap, nSubap, s, s)
        sub_fields = uin.unfold(1, s, s).unfold(2, s, s)

        # Reorder to (B, L, s, s)
        sub_fields = sub_fields.contiguous().view(B, L, s, s)

        # Symmetric center padding from s -> n_pix_lenslet_init
        pad_total = self.n_pix_lenslet_init - s
        if pad_total < 0:
            raise ValueError(
                f"n_pix_lenslet_init ({self.n_pix_lenslet_init}) must be >= "
                f"n_pix_subap_init ({s})."
            )
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before

        em_field = F.pad(
            sub_fields,
            (pad_before, pad_after, pad_before, pad_after),
            mode="constant",
            value=0.0,
        )

        em_field = em_field * phasor.unsqueeze(0).unsqueeze(0)
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
        if input_frame is None:
            input_frame = self.frame_no_noise

        if input_frame.dim() == 2:
            input_frame = input_frame.unsqueeze(0)

        B, H, W = input_frame.shape
        det_pitch = self.n_pix_subap // self.binning_factor

        expected_H = self.nSubap * det_pitch
        expected_W = self.nSubap * det_pitch
        if (H, W) != (expected_H, expected_W):
            raise ValueError(
                f"input_frame must have shape (B, {expected_H}, {expected_W}), "
                f"got {(B, H, W)}."
            )

        tiles = input_frame.view(B, self.nSubap, det_pitch, self.nSubap, det_pitch)
        tiles = tiles.permute(0, 1, 3, 2, 4).contiguous()
        tiles = tiles.view(B, self.nSubap ** 2, det_pitch, det_pitch)

        pad_total = self.n_pix_subap - det_pitch
        if pad_total < 0:
            raise ValueError(
                f"n_pix_subap ({self.n_pix_subap}) must be >= det_pitch ({det_pitch})."
            )
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before

        maps_intensity = F.pad(
            tiles,
            (pad_before, pad_after, pad_before, pad_after),
            mode="constant",
            value=0.0,
        )

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
        norma = self.cube_flux.shape[1]  # lenslet grid size as FFT normalization
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
        flux = self.raw_data.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        raw_data_norm = self.raw_data / flux

        self._SetUniformBackgroundFromSky()
        bg_subap_photons = self.bg_per_pixel * (self.detector_pitch ** 2) * self.bg_pupil
        bg_subap_per_pixel = bg_subap_photons / (self.detector_pitch ** 2)
        bg_map = bg_subap_per_pixel.repeat_interleave(self.detector_pitch, dim=0).repeat_interleave(self.detector_pitch, dim=1)
        bg_map = bg_map.unsqueeze(0)  # shape (1, detector_size, detector_size)
        if bg_map.shape[0] == 1 and raw_data_norm.shape[0] > 1:
            bg_map = bg_map.expand(raw_data_norm.shape[0], -1, -1)

        source_photons = raw_data_norm * self.Nphotons * self.beamSplitProportionForWFSDetector

        self.source_photons_no_noise = source_photons
        if self.separateNoise:
            self.source_electrons_no_noise = self.photons_to_electrons(self.source_photons_no_noise)
            self.frame_no_noise_no_bg = self.apply_saturation(self.source_electrons_no_noise)
            self.frame_no_noise_no_bg = self.apply_gain(self.frame_no_noise_no_bg)
            self.frame_no_noise_no_bg = self.digitalize(self.frame_no_noise_no_bg)
        self.background_photons_no_noise = bg_map
        self.frame_photons_no_noise = source_photons + bg_map

        self.frame_electrons_no_noise = self.photons_to_electrons(self.frame_photons_no_noise)
        self.frame_no_noise = self.frame_electrons_no_noise

        if not self.useNoise:
            if self.FWC is not None:
                self.frame_electrons_no_noise = self.apply_saturation(self.frame_electrons_no_noise)
                self.frame_no_noise = self.frame_electrons_no_noise
            self.maps_intensity = self.split_raw_data(input_frame=self.frame_no_noise)
            return self.frame_no_noise

        self._AddNoiseToFrame()
        self.maps_intensity = self.split_raw_data(input_frame=self.frame_with_noise)
        return self.frame_with_noise
    
    def _SetPhotonsAndRON(self, Nphotons, RON):
        self.Nphotons = torch.pow(10, torch.tensor([Nphotons], device=self.device))
        self.RON = RON

    def _SetUniformBackgroundFromSky(self):
        self.sky_brightness = torch.empty(1, device=self.device).uniform_(*self.backgroundRange)


        self.bg_per_pixel = (
            self.sky_brightness * self.exposureTime * self.pixel_area_arcsec2 * self.beamSplitProportionForWFSDetector
        )
    
    def _AddNoiseToFrame(self):
        frame_photons = self.frame_photons_no_noise.clone()
        self.frame_photons_with_noise = PoissonNoise(frame_photons)
        self.photon_noise = torch.sqrt(frame_photons.clamp_min(0.0))
        frame_electrons = self.photons_to_electrons(self.frame_photons_with_noise)
        frame_electrons = self.apply_dark_current(frame_electrons)
        frame_electrons = self.apply_saturation(frame_electrons)
        if self.sensor == "EMCCD":
            frame_electrons = self.apply_gain(frame_electrons)
        frame_electrons = self.apply_readout_noise(frame_electrons)
        if self.sensor in ["CCD", "CMOS"]:
            frame_electrons = self.apply_gain(frame_electrons)
        self.frame_electrons_with_noise = torch.clamp(frame_electrons, min=0.0)
        self.frame_with_noise = self.frame_electrons_with_noise
        if self.applyDigitalization and self.bits is not None:
            self.frame_with_noise = self.digitalize(self.frame_with_noise)


    def calcSNR(self):
        with torch.no_grad():
            signal = self.matched_filter(self.frame_with_noise.unsqueeze(1).to(torch.float32)) * self.snr_pupil
            noise = torch.sqrt(signal) # torch.sqrt(signal + background + dark_signal + ron^2) 
            return signal / noise        
