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


def _AsRange(value, name):
    """A scalar or a [min, max] pair as a (min, max) tuple of floats; a scalar
    is the degenerate range (value, value)."""
    if np.ndim(value) == 0:
        return float(value), float(value)
    if len(value) != 2 or float(value[0]) > float(value[1]):
        raise ValueError(f"PSFParams['{name}'] must be a number or a [min, max] range, got {value}")
    return float(value[0]), float(value[1])


class PSFGrid:
    """
    Frequency/lag grids and the diffraction OTF for one PSF sampling: the
    WFS.GetPSF grid, i.e. the Nres-pixel pupil zero-padded to N pixels.

    Attributes:
        N (int): grid size; sampling (float): effective sampling N / Nres;
        dF (float): frequency step (1/m); fx, fy, fsqr: centred frequency grids;
        f_max, inside_f_max: largest inscribed frequency and its disk mask;
        rho_x, rho_y: pupil-plane lags (m) of the unshifted OTF;
        otf_tel0: aberration-free telescope OTF; diffraction_peak: its on-axis PSF value.
    """

    def __init__(self, model, N):
        device = model.device
        self.N = N
        self.sampling = N / model.Nres
        self.dF, self.fx, self.fy = GetSpatialFrequencies(N * model.dx, N, device)
        self.fsqr = self.fx**2 + self.fy**2
        self.f_max = (N // 2) * self.dF
        self.inside_f_max = (self.fsqr < self.f_max**2).to(torch.float32)

        # Pupil-plane lags matching the unshifted OTF (zero lag at index 0)
        lag = (torch.arange(N, device=device, dtype=torch.float32) - N // 2) * model.dx
        rho_x, rho_y = torch.meshgrid(lag, lag, indexing="ij")
        self.rho_x = torch.fft.ifftshift(rho_x)
        self.rho_y = torch.fft.ifftshift(rho_y)

        with torch.no_grad():
            self.otf_tel0 = model.TelescopeOTF(None, self)
            self.diffraction_peak = self.otf_tel0.real.sum(dim=(-2, -1))


class PSFModel:
    """
    Differentiable PSD-based PSF model of one instrument.

    Holds only fixed instrument state (the WFS pupil, loop constants, cached
    grids); every per-PSF quantity comes in through a params dict of tensors,
    so fitting code can pass nn.Parameters (with its own reparameterization)
    and backpropagate through PSF(params).

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
        sampling (float)           PSF pixels per lambda/D, shared by the batch; optional
        fov (float or None)        field of view in lambda/D (None: full grid); optional

    `sampling` and `fov` default to PSFParams["sampling"] / PSFParams["fov"]
    when those are single values. When PSFParams gives [min, max] ranges
    (PSFDataset then draws one value per batch) the params dict must carry
    them.

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
        self.dx = self.D / self.Nres
        self.Nactuator = DMParams["Nactuator"]
        self.f_slope = AtmosParams["f_slope"]
        self.loopFrequency = LoopParams["loopFrequency"]
        self.delayFrames = LoopParams["delayFrames"]

        self.wavelength = float(PSFParams.get("Wavelength", WFSParams["Wavelength"]))
        if np.ndim(PSFParams.get("Wavelength", 0.0)) > 0:
            raise ValueError("PSFModel is monochromatic: PSFParams['Wavelength'] must be a scalar")
        self.wavenumber = 2 * np.pi / self.wavelength

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

        # PSF sampling and field of view: single values or [min, max] ranges
        self.sampling_range = _AsRange(PSFParams.get("sampling", WFSParams["sampling"]), "sampling")
        fov = PSFParams.get("fov", None)
        self.fov_range = None if fov is None else _AsRange(fov, "fov")
        self.allowed_samplings = self.AllowedSamplings()
        if self.fov_range is not None:
            for s in self.allowed_samplings:
                if int(np.round(self.fov_range[1] * s)) > self.GridSize(s):
                    raise ValueError(
                        f"fov {self.fov_range[1]} lambda/D does not fit the {self.GridSize(s)}-pixel grid at "
                        f"sampling {s:.3f}; the largest field is Nres = {self.Nres} lambda/D."
                    )

        # Pupil coordinates (m) for short-exposure jitter tilts, centred like MakePupil
        x = (torch.arange(self.Nres, device=device, dtype=torch.float32) - (self.Nres - 1) / 2) * self.dx
        self.pupil_x, self.pupil_y = torch.meshgrid(x, x, indexing="ij")

        # One PSFGrid per grid size, built on first use
        self._grids = {}
        for s in self.allowed_samplings:
            self.Grid(s)

    # ------------------------------------------------------------------
    # Grids, sampling and field of view
    # ------------------------------------------------------------------

    def GridSize(self, sampling):
        """Grid size N that WFS.GetPSF uses at this sampling."""
        return self.Nres + 2 * int(np.round(self.Nres * (sampling - 1)) // 2)

    def AllowedSamplings(self):
        """
        The samplings PSFDataset draws from. For a single configured value,
        that value. For a [min, max] range, the effective sampling N / Nres of
        every grid size N = Nres + 2k inside the range, so the reported
        sampling is exactly the pixel scale of the PSF.
        """
        lo, hi = self.sampling_range
        if lo < 2:
            raise ValueError(
                f"PSF sampling {lo} is below 2: the OTF support needs a grid of at least 2 * Nres "
                "pixels (sampling >= 2)."
            )
        if lo == hi:
            return [lo]
        k_min = int(np.ceil(self.Nres * (lo - 1) / 2 - 1e-9))
        k_max = int(np.floor(self.Nres * (hi - 1) / 2 + 1e-9))
        samplings = [(self.Nres + 2 * k) / self.Nres for k in range(k_min, k_max + 1)]
        if not samplings:
            raise ValueError(
                f"No PSF grid falls inside the sampling range [{lo}, {hi}]: with Nres = {self.Nres} "
                f"the sampling moves in steps of 2 / Nres = {2 / self.Nres:.4f}."
            )
        return samplings

    def Grid(self, sampling):
        """The (cached) PSFGrid for this sampling."""
        return self._GridForSize(self.GridSize(sampling))

    def _GridForSize(self, N):
        if N not in self._grids:
            self._grids[N] = PSFGrid(self, N)
        return self._grids[N]

    def Sampling(self, params):
        """The sampling for this params dict: params['sampling'], or the
        configured value when PSFParams['sampling'] is a single value."""
        if params is not None and params.get("sampling") is not None:
            return float(params["sampling"])
        lo, hi = self.sampling_range
        if lo != hi:
            raise ValueError("PSFParams['sampling'] is a range: pass the sampling as params['sampling'].")
        return lo

    def FOV(self, params):
        """The field of view (lambda/D, or None for the full grid) for this
        params dict: params['fov'], or the configured single value."""
        if params is not None and "fov" in params:
            return None if params["fov"] is None else float(params["fov"])
        if self.fov_range is None:
            return None
        lo, hi = self.fov_range
        if lo != hi:
            raise ValueError("PSFParams['fov'] is a range: pass the field of view as params['fov'].")
        return lo

    # Shortcuts to the grid of a single configured sampling (raise for a range)
    @property
    def sampling(self):
        return self.Sampling(None)

    @property
    def fov(self):
        return self.FOV(None)

    @property
    def N(self):
        return self.Grid(self.sampling).N

    @property
    def dF(self):
        return self.Grid(self.sampling).dF

    @property
    def fx(self):
        return self.Grid(self.sampling).fx

    @property
    def fy(self):
        return self.Grid(self.sampling).fy

    @property
    def fsqr(self):
        return self.Grid(self.sampling).fsqr

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
        The map lives on the pupil grid, so it does not depend on the PSF sampling;
        it is synthesized on the 2 * Nres (sampling 2) grid.
        """
        rms = torch.as_tensor(rms, device=self.device, dtype=torch.float32).reshape(-1)
        grid = self._GridForSize(2 * self.Nres)
        psd = PowerLawPSD(grid.fsqr, slope) * PistonFilter(grid.fsqr, self.D)
        screen = CenterCrop(PSDToScreen(psd.expand(rms.shape[0], -1, -1), grid.dF), self.Nres)
        pupil = self.wfs.pupil
        n = pupil.sum()
        screen = pupil * (screen - (screen * pupil).sum(dim=(-2, -1), keepdim=True) / n)
        current = torch.sqrt((screen**2 * pupil).sum(dim=(-2, -1)) / n)
        return screen * (rms / current).view(-1, 1, 1)

    # ------------------------------------------------------------------
    # PSD
    # ------------------------------------------------------------------

    def CorrectedBand(self, params, grid=None):
        """c * LP: the level of correction inside the square DM band, (B, N, N)."""
        if grid is None:
            grid = self.Grid(self.Sampling(params))
        c = params["level_of_correction"].view(-1, 1, 1)
        return 1 - GetFittingPSD(grid.fx, grid.fy, grid.dF, self.D, self.Nactuator, c)

    def ResidualPSD(self, params, return_terms=False):
        """
        Residual OPD PSD (m^2 m^2), (B, N, N), centred (fftshift convention),
        on the grid of params' sampling.

        fitting:  (1 - c LP) Phi_atm
        servo:    c LP Phi_atm sum_l fr0_l |CLTF(f . v_l)|^2
        aliasing: c LP sum_l fr0_l |NTF(f . v_l)|^2 sum_{shifts} Phi_atm(f - shift)
        noise:    c LP G_n sigma_n^2 (lambda_wfs / 2 pi)^2 d_wfs^2

        With return_terms, also returns a dict of the individual terms.
        """
        grid = self.Grid(self.Sampling(params))
        fx, fy = grid.fx, grid.fy
        r0 = params["r0"].view(-1, 1, 1)
        L0 = params["L0"].view(-1, 1, 1)
        corrected = self.CorrectedBand(params, grid)
        terms = {}

        needs_atmosphere = {"fitting", "servo"} & set(self.terms)
        if needs_atmosphere:
            atmosphere = VonKarmanOPDPSD(grid.fsqr, r0, L0, self.f_slope)
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
                rejection = GetTemporalErrorPSD(fx, fy, self.loopFrequency, gain, leak,
                                                self.delayFrames, wind_x, wind_y)
                terms["servo"] = corrected * atmosphere * (fr0 * rejection).sum(dim=0)
            if "aliasing" in self.terms:
                ntf = GetTemporalNoisePSD(fx, fy, self.loopFrequency, gain, leak,
                                          self.delayFrames, wind_x, wind_y)
                aliased = GetAliasedAtmospherePSD(fx, fy, r0, L0, self.f_slope, self.wfs_pitch)
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
        grid = self._GridForSize(psd.shape[-1])
        if self.psd_integral == "grid":
            return psd.sum(dim=(-2, -1)) * grid.dF**2
        variance = (psd * grid.inside_f_max).sum(dim=(-2, -1)) * grid.dF**2
        if "fitting" in self.terms:
            variance = variance + VonKarmanTail(params["r0"], grid.f_max)
        return variance

    # ------------------------------------------------------------------
    # OTF and PSF
    # ------------------------------------------------------------------

    def TelescopeOTF(self, static_opd, grid):
        """Static/diffraction OTF (unshifted, zero lag at [0, 0]) on `grid`,
        from WFS.GetPSF, so it shares GetPSF's pupil, padding and normalization."""
        if static_opd is None:
            static_opd = torch.zeros(1, self.Nres, self.Nres, device=self.device)
        psf = self.wfs.GetPSF(static_opd, sampling=grid.sampling, wl=self.wavelength)
        return torch.fft.ifft2(torch.fft.ifftshift(psf, dim=(-2, -1)), dim=(-2, -1))

    def OTF(self, params, psd=None):
        """Long-exposure OTF, (B, N, N) complex, unshifted (zero lag at [0, 0]):
        OTF_tel * exp(-k^2 (sigma^2 - B(rho))) * OTF_jitter."""
        grid = self.Grid(self.Sampling(params))
        if psd is None:
            psd = self.ResidualPSD(params)
        elif psd.shape[-1] != grid.N:
            raise ValueError(f"psd is on a {psd.shape[-1]}-pixel grid, but the sampling needs {grid.N}")
        covariance = PSDToCovariance(psd, grid.dF)
        variance = self.ResidualVariance(params, psd).view(-1, 1, 1)
        otf = torch.exp(-self.wavenumber**2 * (variance - covariance))

        static_opd = params.get("static_opd") if "static" in self.terms else None
        otf = otf * (self.TelescopeOTF(static_opd, grid) if static_opd is not None else grid.otf_tel0)

        if "jitter" in self.terms and params.get("jitter") is not None:
            otf = otf * JitterOTF(grid.rho_x, grid.rho_y, params["jitter"],
                                  params["jitter_angle"], self.wavelength)
        return otf

    def PSFFromOTF(self, otf, params=None):
        """Centred PSF from an unshifted OTF, cropped to params' field of view
        (the configured one when params is None)."""
        psf = torch.fft.fftshift(torch.fft.fft2(otf, dim=(-2, -1)), dim=(-2, -1)).real
        return self.CropFOV(psf.clamp_min(0), params)

    def CropFOV(self, psf, params=None):
        """Centre crop to the field of view (lambda/D), exactly as WFS.GetPSF does."""
        fov = self.FOV(params)
        if fov is None:
            return psf
        fov_pix = int(np.round(fov * self.Sampling(params)))
        Ny, Nx = psf.shape[-2], psf.shape[-1]
        y0 = (Ny - fov_pix) // 2
        x0 = (Nx - fov_pix) // 2
        return psf[..., y0 : y0 + fov_pix, x0 : x0 + fov_pix]

    def PSF(self, params):
        """Long-exposure PSF (B, H, W), in WFS.GetPSF units."""
        return self.PSFFromOTF(self.OTF(params), params)

    def Strehl(self, otf):
        """On-axis Strehl ratio (B,) relative to the aberration-free pupil on
        the same grid."""
        return otf.real.sum(dim=(-2, -1)) / self._GridForSize(otf.shape[-1]).diffraction_peak

    def ShortExposurePSF(self, params, psd=None, noise=None):
        """
        One instantaneous PSF per sample (B, H, W): a random residual screen
        drawn from the residual PSD, plus the static map and a random jitter
        tilt, imaged with WFS.GetPSF at params' sampling and field of view.
        Draws are independent (not a time series); on average they converge to
        PSF(params) for psd_integral="grid". Uses the global torch RNG; pass
        `noise` (complex (B, N, N)) to fix the screen.
        """
        sampling = self.Sampling(params)
        grid = self.Grid(sampling)
        if psd is None:
            psd = self.ResidualPSD(params)
        opd = CenterCrop(PSDToScreen(psd, grid.dF, noise), self.Nres)

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

        psf = self.wfs.GetPSF(opd, sampling=sampling, wl=self.wavelength)
        return self.CropFOV(psf, params)


class PSFDataset(Dataset):
    """
    Batches of AO PSFs computed from the residual PSD (no turbulent screens).

    Draws the same atmosphere/loop/detector conditions as PhaseDataset (same
    params dicts, same distributions, and for a given seed the same values),
    plus the PSF-only static aberration and jitter, and returns the long-exposure
    PSF (and optionally one short exposure) for each.

    PSFParams["sampling"] and PSFParams["fov"] may be single values or
    [min, max] ranges. With ranges, each call draws one sampling (uniformly
    among the grid sizes in range, see PSFModel.AllowedSamplings) and one
    field of view (uniform in lambda/D), shared by the whole batch, so the
    PSF size round(fov * sampling) changes from batch to batch.

    Unlike PhaseDataset there is no sequential-idx contract: every
    __getitem__ call draws a fresh, independent batch.

    Batch dict:
        psf (B, H, W), psf_short (B, H, W) if return_short_exposure,
        strehl (B,), psd (B, N, N), plus every drawn parameter (see PSFModel),
        including sampling and fov as Python floats. The batch dict is itself a
        complete params dict: model.PSF(batch) reproduces batch["psf"].

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
        The atmosphere/loop draws come first, identical to PhaseDataset's; the
        batch-wide sampling and fov come last, and are drawn only when they
        are ranges."""
        params = DrawAOParameters(
            self.Nphases, self.nLayersRange, self.r0Range, self.L0Range,
            self.levelOfCorrectionRange, self.loopGainRange, self.loopLeakRange,
            self.photonRange, self.RONRange, self.windSpeedRange,
            self.height_exp_dist_lambda, self.device,
        )
        B = self.Nphases
        model = self.model

        params["ncpa_rms"] = torch.empty(B, device=self.device).uniform_(*self.ncpaRange)
        if "static" in model.terms and max(self.ncpaRange) > 0:
            params["static_opd"] = model.DrawStaticOPD(params["ncpa_rms"], self.ncpaSlope)

        jitter_mas = torch.empty(B, 2, device=self.device).uniform_(*self.jitterRange)
        params["jitter"] = jitter_mas * MAS_TO_RAD
        params["jitter_angle"] = torch.empty(B, device=self.device).uniform_(0, np.pi)

        samplings = model.allowed_samplings
        if len(samplings) == 1:
            params["sampling"] = samplings[0]
        else:
            params["sampling"] = samplings[int(torch.randint(len(samplings), (1,), device=self.device))]

        if model.fov_range is None:
            params["fov"] = None
        elif model.fov_range[0] == model.fov_range[1]:
            params["fov"] = model.fov_range[0]
        else:
            params["fov"] = float(torch.empty(1, device=self.device).uniform_(*model.fov_range))
        return params

    @torch.no_grad()
    def __getitem__(self, idx):
        params = self.DrawRandomParameters()
        psd = self.model.ResidualPSD(params)
        otf = self.model.OTF(params, psd)
        batch = {
            "psf": self.model.PSFFromOTF(otf, params),
            "strehl": self.model.Strehl(otf),
            "psd": psd,
            **params,
        }
        if self.return_short_exposure:
            batch["psf_short"] = self.model.ShortExposurePSF(params, psd)
        return batch
