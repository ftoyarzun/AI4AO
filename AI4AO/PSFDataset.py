# -*- coding: utf-8 -*-
"""
AO point-spread functions computed from the residual-phase PSD, without
simulating turbulent phase screens (AOPERA/maoppy-style).

PSFModel is the differentiable physics: explicit parameter dict -> residual
OPD PSD -> OTF -> PSF (long exposure), plus single short-exposure draws.
PSFDataset draws random parameters from the same params dicts and
distributions as PhaseDataset and returns batches of PSFs.

The PSF grid, pupil and normalization are those of WFS.GetPSF, so with every
aberration switched off PSFModel.PSF reproduces GetPSF exactly.

See Ideas/14-PSFDataset.md for the design and the math.
"""
import warnings

import numpy as np
import torch  # type: ignore[import]
from torch.utils.data import Dataset  # type: ignore[import]

from .PSD import (
    REFERENCE_WAVELENGTH,
    CenterCrop,
    DrawAOParameters,
    GetAliasedAtmospherePSD,
    GetFittingPSD,
    GetNoisePSD,
    GetSpatialFrequencies,
    GetTemporalErrorPSD,
    GetTemporalNoisePSD,
    JitterOTF,
    NoiseTransferGain,
    PistonFilter,
    PowerLawPSD,
    PSDToCovariance,
    PSDToScreen,
    VonKarmanOPDPSD,
    VonKarmanTail,
    WFSNoiseVariance,
)

ALL_TERMS = ("fitting", "servo", "aliasing", "noise", "static", "jitter")

MAS_TO_RAD = np.pi / (180 * 3600 * 1000)


class PSFModel:
    """
    Differentiable PSD-based PSF model of one instrument.

    Holds only fixed instrument state (grids, the WFS pupil, loop constants);
    every per-PSF quantity comes in through a params dict of tensors, so fitting
    code can pass nn.Parameters (with its own reparameterization) and
    backpropagate through PSF(params).

    Params dict (B PSFs, L layers); only the keys the enabled terms need are read:
        r0 (B,)                    Fried parameter at 500 nm (m)
        L0 (B,)                    outer scale (m)
        fractional_r0 (L, B)       layer weights, summing to 1 over L
        wind (2, L, B)             layer wind (m/s); component 0 along array dim -2
        loop_gain, loop_leak (B,)  integrator gain and leak
        level_of_correction (B,)   c in [0, 1], scales the DM-corrected band
        nphotons, ron (B,)         WFS photons per frame, read noise (e-/pix)
        static_opd (B, Nres, Nres) static aberration (m) on the WFS pupil grid, optional
        jitter (B, 2)              residual jitter rms along the principal axes (rad)
        jitter_angle (B,)          rotation of the jitter axes (rad)

    Args:
        wfs (WFS): supplies the pupil (with central obstruction), D, Nres and
            the GetPSF convention. Any WFS subclass works; only GetPSF is used.
        WFSParams, AtmosParams, LoopParams, DMParams, PSFParams (dict): instrument
            params dicts (see Ideas/14-PSFDataset.md for the PSFParams keys).
        terms (tuple): error terms to include, a subset of ALL_TERMS.
        psd_integral (str): "grid" takes the residual variance as the sum over
            the PSD grid, like GetPSF(PhaseDataset) screens; "infinite" adds the
            analytic Kolmogorov tail beyond the grid (maoppy), better for
            fitting real data.
    """

    def __init__(self, wfs, WFSParams, AtmosParams, LoopParams, DMParams, PSFParams, device,
                 terms=ALL_TERMS, psd_integral="grid"):
        unknown = set(terms) - set(ALL_TERMS)
        if unknown:
            raise ValueError(f"Unknown PSF terms {sorted(unknown)}; choose from {ALL_TERMS}")
        if psd_integral not in ("grid", "infinite"):
            raise ValueError("psd_integral must be 'grid' or 'infinite'")

        self.wfs = wfs
        self.device = device
        self.terms = tuple(terms)
        self.psd_integral = psd_integral

        self.D = wfs.D
        self.Nres = wfs.Nres
        self.Nactuator = DMParams["Nactuator"]
        self.f_slope = AtmosParams["f_slope"]
        self.loopFrequency = LoopParams["loopFrequency"]
        self.delayFrames = LoopParams["delayFrames"]

        self.wavelength = float(PSFParams.get("Wavelength", WFSParams["Wavelength"]))
        if np.ndim(PSFParams.get("Wavelength", 0.0)) > 0:
            raise ValueError("PSFModel is monochromatic: PSFParams['Wavelength'] must be a scalar")
        self.wavenumber = 2 * np.pi / self.wavelength
        self.sampling = PSFParams.get("sampling", WFSParams["sampling"])
        self.fov = PSFParams.get("fov", None)

        # WFS sensing wavelength (band centre for a polychromatic WFS), used only
        # to convert the WFS noise from radians to OPD.
        wl_wfs = wfs.wavelength
        self.sensing_wavelength = float((wl_wfs.min() + wl_wfs.max()) / 2)

        # WFS measurement pitch in the pupil and number of illuminated pixels
        bin_factor = WFSParams.get("Bin_factor", 1)
        self.wfs_pitch = self.D * bin_factor / self.Nres
        self.n_valid_pixels = float(wfs.pupil.sum()) / bin_factor**2

        self.noise_photon_factor = PSFParams.get("NoisePhotonFactor", None)
        self.noise_ron_factor = PSFParams.get("NoiseRONFactor", None)
        if "noise" in self.terms and (self.noise_photon_factor is None or self.noise_ron_factor is None):
            raise ValueError(
                "The 'noise' term needs PSFParams['NoisePhotonFactor'] and PSFParams['NoiseRONFactor'] "
                "(WFS sensitivity constants); set them or drop 'noise' from `terms`."
            )

        # Same grid as WFS.GetPSF: the pupil (Nres pixels across D) zero-padded to N
        self.N = self.Nres + 2 * int(np.round(self.Nres * (self.sampling - 1)) // 2)
        if self.N < 2 * self.Nres:
            raise ValueError(
                f"PSF sampling {self.sampling} gives a {self.N}-pixel grid for a {self.Nres}-pixel pupil; "
                "the OTF support needs at least 2 * Nres (sampling >= 2)."
            )
        self.dx = self.D / self.Nres
        self.dF, self.fx, self.fy = GetSpatialFrequencies(self.N * self.dx, self.N, device)
        self.fsqr = self.fx**2 + self.fy**2
        self.f_max = (self.N // 2) * self.dF
        self.inside_f_max = (self.fsqr < self.f_max**2).to(torch.float32)

        # Pupil-plane lags matching the unshifted OTF (zero lag at index 0)
        lag = (torch.arange(self.N, device=device, dtype=torch.float32) - self.N // 2) * self.dx
        rho_x, rho_y = torch.meshgrid(lag, lag, indexing="ij")
        self.rho_x = torch.fft.ifftshift(rho_x)
        self.rho_y = torch.fft.ifftshift(rho_y)

        # Pupil coordinates (m) for short-exposure jitter tilts, centred like MakePupil
        x = (torch.arange(self.Nres, device=device, dtype=torch.float32) - (self.Nres - 1) / 2) * self.dx
        self.pupil_x, self.pupil_y = torch.meshgrid(x, x, indexing="ij")

        # Diffraction-limited (aberration-free) telescope OTF, cached
        with torch.no_grad():
            self.otf_tel0 = self.TelescopeOTF(None)
            self.diffraction_peak = self.otf_tel0.real.sum(dim=(-2, -1))

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @staticmethod
    def ParametersFromPhaseDataset(dataset):
        """Canonical params dict for the conditions a PhaseDataset last drew,
        read from its unsqueezed attributes (its batch dict squeezes away
        singleton layer/batch dims)."""
        B = dataset.Nphases
        L = dataset.nLayers
        return {
            "r0": dataset.r0_moving.reshape(B),
            "L0": dataset.L0.reshape(B),
            "level_of_correction": dataset.levelOfCorrection.reshape(B),
            "loop_gain": dataset.loopGain.reshape(B),
            "loop_leak": dataset.loopLeak.reshape(B),
            "nphotons": dataset.Nphotons.reshape(B),
            "ron": dataset.RON.reshape(B),
            "fractional_r0": dataset.fractionalr0.reshape(L, B),
            "wind": torch.stack((dataset.windSpeedVector_x, dataset.windSpeedVector_y)).reshape(2, L, B),
        }

    def DrawStaticOPD(self, rms, slope=2.2):
        """
        Random static aberration maps (B, Nres, Nres) in meters: a
        piston-filtered f^(-slope) PSD screen on the WFS pupil, piston-removed
        and rescaled so its rms over the pupil equals `rms` (B,) exactly. The
        default slope 2.2 is AOPERA's psd_ncpa default. Uses the global torch RNG.
        """
        rms = torch.as_tensor(rms, device=self.device, dtype=torch.float32).reshape(-1)
        psd = PowerLawPSD(self.fsqr, slope) * PistonFilter(self.fsqr, self.D)
        screen = CenterCrop(PSDToScreen(psd.expand(rms.shape[0], -1, -1), self.dF), self.Nres)
        pupil = self.wfs.pupil
        n = pupil.sum()
        screen = pupil * (screen - (screen * pupil).sum(dim=(-2, -1), keepdim=True) / n)
        current = torch.sqrt((screen**2 * pupil).sum(dim=(-2, -1)) / n)
        return screen * (rms / current).view(-1, 1, 1)

    # ------------------------------------------------------------------
    # PSD
    # ------------------------------------------------------------------

    def CorrectedBand(self, params):
        """c * LP: the level of correction inside the square DM band, (B, N, N)."""
        c = params["level_of_correction"].view(-1, 1, 1)
        return 1 - GetFittingPSD(self.fx, self.fy, self.dF, self.D, self.Nactuator, c)

    def ResidualPSD(self, params, return_terms=False):
        """
        Residual OPD PSD (m^2 m^2), (B, N, N), centred (fftshift convention).

        fitting:  (1 - c LP) Phi_atm
        servo:    c LP Phi_atm sum_l fr0_l |CLTF(f . v_l)|^2
        aliasing: c LP sum_l fr0_l |NTF(f . v_l)|^2 sum_{shifts} Phi_atm(f - shift)
        noise:    c LP G_n sigma_n^2 (lambda_wfs / 2 pi)^2 d_wfs^2

        With return_terms, also returns a dict of the individual terms.
        """
        r0 = params["r0"].view(-1, 1, 1)
        L0 = params["L0"].view(-1, 1, 1)
        corrected = self.CorrectedBand(params)
        terms = {}

        needs_atmosphere = {"fitting", "servo"} & set(self.terms)
        if needs_atmosphere:
            atmosphere = VonKarmanOPDPSD(self.fsqr, r0, L0, self.f_slope)
        if "fitting" in self.terms:
            terms["fitting"] = atmosphere * (1 - corrected)

        if {"servo", "aliasing"} & set(self.terms):
            fr0 = params["fractional_r0"]
            L, B = fr0.shape
            fr0 = fr0.view(L, B, 1, 1)
            wind_x = params["wind"][0].view(L, B, 1, 1)
            wind_y = params["wind"][1].view(L, B, 1, 1)
            gain = params["loop_gain"].view(1, B, 1, 1)
            leak = params["loop_leak"].view(1, B, 1, 1)
            if "servo" in self.terms:
                rejection = GetTemporalErrorPSD(self.fx, self.fy, self.loopFrequency, gain, leak,
                                                self.delayFrames, wind_x, wind_y)
                terms["servo"] = corrected * atmosphere * (fr0 * rejection).sum(dim=0)
            if "aliasing" in self.terms:
                ntf = GetTemporalNoisePSD(self.fx, self.fy, self.loopFrequency, gain, leak,
                                          self.delayFrames, wind_x, wind_y)
                aliased = GetAliasedAtmospherePSD(self.fx, self.fy, r0, L0, self.f_slope, self.wfs_pitch)
                terms["aliasing"] = corrected * (fr0 * ntf).sum(dim=0) * aliased

        if "noise" in self.terms:
            gain_n = NoiseTransferGain(self.loopFrequency, params["loop_gain"], params["loop_leak"],
                                       self.delayFrames)
            variance = WFSNoiseVariance(params["nphotons"], params["ron"], self.n_valid_pixels,
                                        self.noise_photon_factor, self.noise_ron_factor)
            terms["noise"] = GetNoisePSD(corrected, gain_n.view(-1, 1, 1), variance.view(-1, 1, 1),
                                         self.sensing_wavelength, self.wfs_pitch)

        if terms:
            total = sum(terms.values())
        else:
            total = torch.zeros_like(corrected)
        if return_terms:
            return total, terms
        return total

    def ResidualVariance(self, params, psd):
        """Residual OPD variance (m^2) per PSF, (B,): the grid sum, plus the
        Kolmogorov tail beyond the grid when psd_integral == "infinite"."""
        if self.psd_integral == "grid":
            return psd.sum(dim=(-2, -1)) * self.dF**2
        variance = (psd * self.inside_f_max).sum(dim=(-2, -1)) * self.dF**2
        if "fitting" in self.terms:
            variance = variance + VonKarmanTail(params["r0"], self.f_max)
        return variance

    # ------------------------------------------------------------------
    # OTF and PSF
    # ------------------------------------------------------------------

    def TelescopeOTF(self, static_opd):
        """Static/diffraction OTF (unshifted, zero lag at [0, 0]) from
        WFS.GetPSF, so it shares GetPSF's pupil, padding and normalization."""
        if static_opd is None:
            static_opd = torch.zeros(1, self.Nres, self.Nres, device=self.device)
        psf = self.wfs.GetPSF(static_opd, sampling=self.sampling, wl=self.wavelength)
        return torch.fft.ifft2(torch.fft.ifftshift(psf, dim=(-2, -1)), dim=(-2, -1))

    def OTF(self, params, psd=None):
        """Long-exposure OTF, (B, N, N) complex, unshifted (zero lag at [0, 0]):
        OTF_tel * exp(-k^2 (sigma^2 - B(rho))) * OTF_jitter."""
        if psd is None:
            psd = self.ResidualPSD(params)
        covariance = PSDToCovariance(psd, self.dF)
        variance = self.ResidualVariance(params, psd).view(-1, 1, 1)
        otf = torch.exp(-self.wavenumber**2 * (variance - covariance))

        static_opd = params.get("static_opd") if "static" in self.terms else None
        otf = otf * (self.TelescopeOTF(static_opd) if static_opd is not None else self.otf_tel0)

        if "jitter" in self.terms and params.get("jitter") is not None:
            otf = otf * JitterOTF(self.rho_x, self.rho_y, params["jitter"],
                                  params["jitter_angle"], self.wavelength)
        return otf

    def PSFFromOTF(self, otf):
        """Centred PSF from an unshifted OTF, cropped to the field of view."""
        psf = torch.fft.fftshift(torch.fft.fft2(otf, dim=(-2, -1)), dim=(-2, -1)).real
        return self.CropFOV(psf.clamp_min(0))

    def CropFOV(self, psf):
        """Centre crop to fov (in lambda/D), exactly as WFS.GetPSF does."""
        if self.fov is None:
            return psf
        fov_pix = int(np.round(self.fov * self.sampling))
        Ny, Nx = psf.shape[-2], psf.shape[-1]
        y0 = (Ny - fov_pix) // 2
        x0 = (Nx - fov_pix) // 2
        return psf[..., y0 : y0 + fov_pix, x0 : x0 + fov_pix]

    def PSF(self, params):
        """Long-exposure PSF (B, H, W), in WFS.GetPSF units."""
        return self.PSFFromOTF(self.OTF(params))

    def Strehl(self, otf):
        """On-axis Strehl ratio (B,) relative to the aberration-free pupil."""
        return otf.real.sum(dim=(-2, -1)) / self.diffraction_peak

    def ShortExposurePSF(self, params, psd=None, noise=None):
        """
        One instantaneous PSF per sample (B, H, W): a random residual screen
        drawn from the residual PSD, plus the static map and a random jitter
        tilt, imaged with WFS.GetPSF. Draws are independent (not a time series);
        on average they converge to PSF(params) for psd_integral="grid".
        Uses the global torch RNG; pass `noise` (complex (B, N, N)) to fix the screen.
        """
        if psd is None:
            psd = self.ResidualPSD(params)
        opd = CenterCrop(PSDToScreen(psd, self.dF, noise), self.Nres)

        if "static" in self.terms and params.get("static_opd") is not None:
            opd = opd + params["static_opd"]

        if "jitter" in self.terms and params.get("jitter") is not None:
            B = opd.shape[0]
            theta = torch.randn(B, 2, device=self.device) * params["jitter"]
            cos = torch.cos(params["jitter_angle"])
            sin = torch.sin(params["jitter_angle"])
            tilt_x = (theta[:, 0] * cos - theta[:, 1] * sin).view(-1, 1, 1)
            tilt_y = (theta[:, 0] * sin + theta[:, 1] * cos).view(-1, 1, 1)
            opd = opd + tilt_x * self.pupil_x + tilt_y * self.pupil_y

        psf = self.wfs.GetPSF(opd, sampling=self.sampling, wl=self.wavelength)
        return self.CropFOV(psf)


class PSFDataset(Dataset):
    """
    Batches of AO PSFs computed from the residual PSD (no turbulent screens).

    Draws the same atmosphere/loop/detector conditions as PhaseDataset (same
    params dicts, same distributions, and for a given seed the same values),
    plus the PSF-only static aberration and jitter, and returns the long-exposure
    PSF (and optionally one short exposure) for each.

    Unlike PhaseDataset there is no sequential-idx contract: every
    __getitem__ call draws a fresh, independent batch.

    Batch dict:
        psf (B, H, W), psf_short (B, H, W) if return_short_exposure,
        strehl (B,), psd (B, N, N), plus every drawn parameter (see PSFModel).

    Args:
        wfs, WFSParams, AtmosParams, LoopParams, DMParams, PSFParams, device,
        terms, psd_integral: see PSFModel.
        return_short_exposure (bool): also return one short-exposure PSF per sample.
    """

    def __init__(self, wfs, WFSParams, AtmosParams, LoopParams, DMParams, PSFParams, device,
                 terms=ALL_TERMS, return_short_exposure=False, psd_integral="grid"):
        self.model = PSFModel(wfs, WFSParams, AtmosParams, LoopParams, DMParams, PSFParams, device,
                              terms=terms, psd_integral=psd_integral)
        self.device = device
        self.return_short_exposure = return_short_exposure

        if AtmosParams.get("Scintillation", False):
            warnings.warn("PSFDataset does not model scintillation; AtmosParams['Scintillation'] is ignored.")

        self.Nphases = AtmosParams["Nphases"]
        self.photonRange = WFSParams["Nphotons"]
        self.RONRange = WFSParams["RON"]
        self.r0Range = AtmosParams["r0"]
        self.L0Range = AtmosParams["L0"]
        self.nLayersRange = AtmosParams["Layers"]
        self.levelOfCorrectionRange = LoopParams["levelOfCorrection"]
        self.loopGainRange = LoopParams["loopGain"]
        self.loopLeakRange = LoopParams["loopLeak"]
        self.windSpeedRange = LoopParams["windSpeedVector"]
        self.height_exp_dist_lambda = 0.0003  # same as PhaseDataset

        self.ncpaRange = PSFParams.get("NCPA", [0.0, 0.0])
        self.ncpaSlope = PSFParams.get("NCPASlope", 2.2)
        self.jitterRange = PSFParams.get("Jitter", [0.0, 0.0])

    def __len__(self):
        return self.Nphases

    @torch.no_grad()
    def DrawRandomParameters(self):
        """One batch of random parameters (canonical PSFModel params dict).
        The atmosphere/loop draws come first, identical to PhaseDataset's."""
        params = DrawAOParameters(
            self.Nphases, self.nLayersRange, self.r0Range, self.L0Range,
            self.levelOfCorrectionRange, self.loopGainRange, self.loopLeakRange,
            self.photonRange, self.RONRange, self.windSpeedRange,
            self.height_exp_dist_lambda, self.device,
        )
        B = self.Nphases
        terms = self.model.terms

        params["ncpa_rms"] = torch.empty(B, device=self.device).uniform_(*self.ncpaRange)
        if "static" in terms and max(self.ncpaRange) > 0:
            params["static_opd"] = self.model.DrawStaticOPD(params["ncpa_rms"], self.ncpaSlope)

        jitter_mas = torch.empty(B, 2, device=self.device).uniform_(*self.jitterRange)
        params["jitter"] = jitter_mas * MAS_TO_RAD
        params["jitter_angle"] = torch.empty(B, device=self.device).uniform_(0, np.pi)
        return params

    @torch.no_grad()
    def __getitem__(self, idx):
        params = self.DrawRandomParameters()
        psd = self.model.ResidualPSD(params)
        otf = self.model.OTF(params, psd)
        batch = {
            "psf": self.model.PSFFromOTF(otf),
            "strehl": self.model.Strehl(otf),
            "psd": psd,
            **params,
        }
        if self.return_short_exposure:
            batch["psf_short"] = self.model.ShortExposurePSF(params, psd)
        return batch
