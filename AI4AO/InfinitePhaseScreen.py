# -*- coding: utf-8 -*-
"""
InfinitePhaseScreen: a drop-in alternative to PhaseDataset whose turbulent
layers never repeat and carry the right low-order (tip/tilt) power.

PhaseDataset moves each layer with a Fourier phase ramp on a 2D-wide FFT
screen. That screen is a torus, so the atmosphere repeats exactly after 2D/|v|,
and it has no spatial frequencies below 1/(2D), so tip/tilt is ~0.4 of the
von Karman value. This class keeps PhaseDataset's constructor, batch dict and
public attributes, but generates the layers with the engine designed in
Ideas/16-infinite-frozen-flow.md (prototype: prototyping/InfiniteFrozenFlow.ipynb):

- Frozen flow by an accumulated float64 displacement s(t) = sum v dt, as an
  exact Fourier shift (any sub-pixel value).
- Multi-scale screens ("subharmonic levels"): a full-resolution FFT torus
  with a square hole of (2 m + 1)^2 central frequency cells, filled by
  `subharmonicLevels` coarser lattices (q times finer frequency cells per
  level) evaluated on the output grid by a matrix Fourier transform. The
  levels tile the frequency plane exactly.
- No repetition: each level holds three independent screens blended with
  weights attached to the moving air (sum of squared weights = 1). A screen is
  redrawn only while its weight is exactly zero over the whole visible region,
  so the pupil sees one infinite frozen field.
- Optional boiling: a per-coefficient AR(1) update that keeps the PSD exact,
  with coherence time boilingTime * (|f| D)^(-2/3).

Differences from PhaseDataset's contract:
- `dataset[idx]` must be called with idx = 0 (new sequence) or with idx >= the
  previous index (frames in between are stepped through, so a jump equals
  stepping one by one). A backward idx raises.
- `layeredOPD` is (nLayers, Nphases, Nres, Nres) without scintillation (the
  pupil window, not PhaseDataset's 2 Nres torus) and (nLayers, Nphases,
  4 Nres, 4 Nres) with it. Only the pupil (or, with scintillation, a 2D-wide
  disc around it) is guaranteed to be exact frozen flow; the corners of the
  window may change when a screen is redrawn.
- Wind direction and conditions are fixed for a sequence, as in PhaseDataset.
- The full-resolution level is sized at dataset[0] for the fastest drawn layer
  (its blend zone must hold the pupil plus one frame of motion), so its grid,
  and memory, can change between sequences.
"""
import math

import numpy as np
import torch  # type: ignore[import]
from torch.utils.data import Dataset  # type: ignore[import]

from .Utils import MakePupil
# Re-exported so `from AI4AO.InfinitePhaseScreen import ...` can replace
# `from AI4AO.PhaseDataset import ...` without other changes.
from .PhaseDataset import Zernike  # noqa: F401
from .PSD import (  # noqa: F401
    GetSpatialFrequencies,
    GetAtmospherePSD,
    GetFittingPSD,
    openLoopTransferFunction,
    closedLoopTransferFunction,
    GetTemporalErrorPSD,
    CenterCrop,
    DrawAOParameters,
    VonKarmanPSD,
)


def FFTFriendlyOdd(n):
    """Smallest odd integer >= n whose prime factors are all in {3, 5, 7}.
    Odd avoids a Nyquist term (its real part flickers under sub-pixel shifts)."""
    n = max(int(n), 1)
    while True:
        if n % 2:
            k = n
            for p in (3, 5, 7):
                while k % p == 0:
                    k //= p
            if k == 1:
                return n
        n += 1


def CrossfadeWeights(u):
    """
    The three crossfade weights at moving-frame coordinate u = xi / R (any
    shape) -> (3, *u.shape). With a = (u - k) mod 3, w_k = sin(pi a / 2) on
    [0, 1), cos(pi (a - 1) / 2) on [1, 2) and 0 on [2, 3), so sum_k w_k^2 = 1.
    """
    k = torch.arange(3, device=u.device, dtype=u.dtype).reshape(3, *([1] * u.dim()))
    a = torch.remainder(u.unsqueeze(0) - k, 3.0)
    return torch.sin(math.pi / 2 * torch.minimum(a, 2 - a).clamp_min(0))


def CrossfadeSum(u, fields, axis):
    """
    sum_k CrossfadeWeights(u)[k] * fields[k along `axis`] from the two nonzero
    weights only: with t = u mod 3, j = floor(t) and tau = t - j, screen j has
    weight sin(pi tau / 2), screen j - 1 has cos(pi tau / 2) and screen j + 1
    contributes exactly 0. `u` has `fields`' number of dims with size 1 along
    `axis` (size 3 in `fields`) and broadcasts against it. Returns the sum with
    `axis` removed.
    """
    t = torch.remainder(u, 3.0)
    j = torch.floor(t).clamp(max=2)  # t can round to exactly 3.0; tau = 1 then gives the same weights
    tau = (t - j) * (math.pi / 2)
    j = j.long()
    shape = list(torch.broadcast_shapes(u.shape, fields.shape))
    shape[axis] = 1
    rising = torch.gather(fields, axis, j.expand(shape))
    falling = torch.gather(fields, axis, torch.remainder(j - 1, 3).expand(shape))
    return (torch.sin(tau) * rising + torch.cos(tau) * falling).squeeze(axis)


class InfinitePhaseScreen(Dataset):
    """
    Non-repeating, multi-scale frozen-flow OPD screens with PhaseDataset's API.

    Usage (same as PhaseDataset):
        dataset = InfinitePhaseScreen(WFSParams, AtmosParams, LoopParams, DMParams, device)
        batch = dataset[0]            # new conditions and new screens
        batch = dataset[1]            # next frame, ...

    Extra keyword arguments (algorithmic options, not params-dict keys). They
    are plain attributes and take effect at the next dataset[0]:
        subharmonicLevels (int or None): number of coarse levels below the
            full-resolution screen. None (default) chooses the smallest number
            whose outer torus is at least 3 * max(AtmosParams["L0"]). 0 gives a
            single crossfaded full-band torus (non-repeating, but with the
            torus's tip/tilt deficit and a few % high-order blend bias).
        boiling (bool): enable the AR(1) boiling update. Default False.
        boilingTime (float or [min, max]): coherence time tau_b (s) at spatial
            frequency 1/D; tau(f) = tau_b (|f| D)^(-2/3). A [min, max] range is
            drawn uniformly per batch element at dataset[0]. Required when
            boiling is True.

    Further attributes: `subharmonicHole` (m, default 4) and
    `subharmonicRefinement` (q, odd, default 3) set the level geometry.
    """

    def __init__(self, WFSParams, AtmosParams, LoopParams, DMParams, device,
                 subharmonicLevels=None, boiling=False, boilingTime=None):
        self.D = WFSParams['D']
        self.Nres = WFSParams['Nres']
        self.photonRange = WFSParams['Nphotons']
        self.RONRange = WFSParams['RON']
        self.wavelength = WFSParams["Wavelength"]
        self.wavenumber = 2 * torch.pi / self.wavelength

        # Wavelength at which r0 is quoted; converts the rad^2 Kolmogorov PSD to OPD (m^2).
        self.referenceWavelength = 500e-9

        self.L0Range = AtmosParams['L0']
        self.r0Range = AtmosParams['r0']
        self.Nphases = AtmosParams['Nphases']
        self.nLayersRange = AtmosParams["Layers"]
        self.f_slope = AtmosParams["f_slope"]
        self.useScintillation = AtmosParams["Scintillation"]

        self.levelOfCorrectionRange = LoopParams['levelOfCorrection']
        self.loopFrequency = LoopParams['loopFrequency']
        self.delayFrames = LoopParams['delayFrames']
        self.loopGainRange = LoopParams['loopGain']
        self.loopLeakRange = LoopParams['loopLeak']
        self.windSpeedRange = LoopParams['windSpeedVector']

        self.Nactuator = DMParams['Nactuator']

        self.device = device

        self.subharmonicLevels = subharmonicLevels
        self.subharmonicHole = 4
        self.subharmonicRefinement = 3
        self.boiling = boiling
        self.boilingTime = boilingTime
        if boiling and boilingTime is None:
            raise ValueError("boiling=True needs a boilingTime (s), scalar or [min, max]")

        self.movingCount = 0
        self.frame = 0
        self.height_exp_dist_lambda = 0.0003
        self.generateClosedLoop = False

        self.pupil = MakePupil(self.Nres, self.device)
        self.pupilSum = self.pupil.sum()
        self.pupil_logical = torch.where(self.pupil.view(-1, 1) > 0)

        [self.dF, self.fx, self.fy] = GetSpatialFrequencies(self.D, self.Nres, self.device)
        self.fsqr = self.fx**2 + self.fy**2

        # PhaseDataset's torus grid: only used by BuildAtmospherePSD and the scintillation propagator.
        upSize = 4 if self.useScintillation else 2
        [self.dF_moving, self.fx_moving, self.fy_moving] = GetSpatialFrequencies(self.D * upSize, self.Nres * upSize, self.device)
        self.fsqr_moving = self.fx_moving**2 + self.fy_moving**2

        if self.useScintillation:
            self.pupilASP = MakePupil(self.Nres * upSize, self.device, Rpx=(self.Nres * upSize + 1) / 2.2)
            self.masterPropagatorPhase = torch.fft.fftshift(torch.exp(-1j * torch.pi * self.wavelength * self.fsqr_moving), dim=(-2, -1))

        # Screen-engine geometry. The output window is the pupil (or the 4 Nres scintillation field);
        # the visible region is the pupil disc plus one pixel each side (2D-wide disc with scintillation).
        self.px = self.D / self.Nres
        self.dt = 1 / self.loopFrequency
        self.outputSize = self.Nres * 4 if self.useScintillation else self.Nres
        self.visibleWidth = 2 * self.D if self.useScintillation else self.D + 2 * self.px
        start = self.outputSize // 2 - self.Nres // 2              # CenterCrop offset of the pupil
        self.visibleCenter = (start + (self.Nres - 1) / 2) * self.px
        x = torch.arange(self.outputSize, device=self.device, dtype=torch.float32) * self.px
        self.X, self.Y = torch.meshgrid(x, x, indexing="ij")
        self._levelCache = {}
        self.maxFieldElements = 2**24  # bounds the coarse-level temporaries (scintillation grids)

        self.r0_moving = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.L0 = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.levelOfCorrection = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.Nphotons = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.RON = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.nLayers = np.random.randint(*self.nLayersRange)
        self.fractionalr0 = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device)
        self.windSpeedVector_x = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device)
        self.windSpeedVector_y = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device)
        self.loopGain = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.loopLeak = torch.empty(self.Nphases, 1, 1, device=self.device)
        self.layerHeights = torch.empty(self.nLayers, self.Nphases, 1, 1, device=self.device)

    def __len__(self):
        return self.Nphases

    def __getitem__(self, idx):
        if idx == 0:
            self.ResetMovingWavefront()
        return self.GetMovingWavefront(idx)

    # ------------------------------------------------------------------
    # PhaseDataset API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def GetMovingWavefront(self, idx):
        """
        Frame `idx` of the current sequence. idx == 0 draws new conditions and
        new screens; idx > 0 advances frame by frame from the last index.

        Returns the same dict as PhaseDataset.GetMovingWavefront.
        """
        if idx == 0:
            self.DrawRandomParameters()
            self.ResetScreens()
        elif idx < self.frame:
            raise ValueError(f"InfinitePhaseScreen only moves forward: idx {idx} after frame {self.frame} "
                             "(use idx = 0 to start a new sequence)")
        while self.frame < idx:
            self.AdvanceFrame()

        layered = self.LayeredOPD()
        self.layeredOPD = layered[0]
        opdMap = self.CompressAtmosphere()

        if self.useScintillation:
            # layered[-1] is the open-loop atmosphere: the same screens as layered[0] in open loop
            self.scintillationLayeredOPD = layered[-1]
            pupilMap = CenterCrop(self.ComputeScintillation(self.scintillationLayeredOPD).abs(), self.Nres)
        else:
            pupilMap = self.pupil.repeat(self.Nphases, 1, 1)

        self.movingCount += 1

        return {
                "opd": opdMap,
                "pupil": pupilMap,
                "nphotons": self.Nphotons,
                "ron": self.RON,
                "r0": self.r0_moving.squeeze(),
                "L0": self.L0.reshape(-1),
                "wind": torch.stack((self.windSpeedVector_x, self.windSpeedVector_y)).squeeze(),
                "fractional_r0": self.fractionalr0.squeeze(),
                "loop_gain": self.loopGain.reshape(-1, 1),
                "loop_leak": self.loopLeak.reshape(-1, 1),
                "level_of_correction": self.levelOfCorrection.reshape(-1, 1),
                }

    def RemovePiston(self, opdMap):
        mask = self.pupil.unsqueeze(0)  # shape (1, H, W)
        masked_mean = (opdMap * mask).sum(dim=(-2, -1), keepdim=True) / self.pupilSum
        return opdMap - masked_mean * mask

    def CompressAtmosphere(self):
        """Sum the layers of `layeredOPD`, crop to the pupil, mask and remove piston (meters)."""
        croppedLayeredOPD = CenterCrop(self.layeredOPD, self.Nres)
        opdMap = croppedLayeredOPD.sum(dim=0)
        opdMap = self.pupil * opdMap
        opdMap = self.RemovePiston(opdMap)
        return opdMap

    def ComputeScintillation(self, layeredOPD):
        scintillationSupport = self.pupilASP.repeat(self.Nphases, 1, 1).to(dtype=torch.complex64)

        for i in range(self.nLayers):
            dist = self.layerHeights[i] - self.layerHeights[i + 1] if i < self.nLayers - 1 else self.layerHeights[-1]
            # OPD (m) -> phase (rad) at the sensing wavelength
            scintillationSupport = scintillationSupport * torch.exp(1j * self.wavenumber * layeredOPD[i])
            scintillationSupport = self.ASP(scintillationSupport, dist)

        return scintillationSupport

    def ASP(self, input_field, distance):
        field_freq = torch.fft.fft2(torch.fft.fftshift(input_field, dim=(-2, -1)), dim=(-2, -1))
        field_filtered = field_freq * self.masterPropagatorPhase ** distance
        return torch.fft.ifftshift(torch.fft.ifft2(field_filtered, dim=(-2, -1)), dim=(-2, -1))

    def ResetMovingWavefront(self):
        """Reset the frame counters (the screens themselves are redrawn by ResetScreens)."""
        self.movingCount = 0
        self.frame = 0

    def DrawRandomParameters(self):
        """Draw new random conditions (r0, L0, correction, noise, Cn2 profile, wind), as PhaseDataset."""
        params = DrawAOParameters(
            self.Nphases, self.nLayersRange, self.r0Range, self.L0Range,
            self.levelOfCorrectionRange, self.loopGainRange, self.loopLeakRange,
            self.photonRange, self.RONRange, self.windSpeedRange,
            self.height_exp_dist_lambda, self.device,
        )
        self.aoParameters = params  # canonical shapes, see DrawAOParameters

        def column(t):
            # (B,) -> (B, 1, 1) and (L, B) -> (L, B, 1, 1), broadcasting against (..., H, W)
            return t.reshape(*t.shape, 1, 1)

        self.nLayers = params["fractional_r0"].shape[0]
        self.r0_moving = column(params["r0"])
        self.L0 = column(params["L0"])
        self.levelOfCorrection = column(params["level_of_correction"])
        self.loopGain = column(params["loop_gain"])
        self.loopLeak = column(params["loop_leak"])
        self.Nphotons = column(params["nphotons"])
        self.RON = column(params["ron"])
        self.fractionalr0 = column(params["fractional_r0"])
        self.layerHeights = column(params["layer_heights"])
        self.windSpeed = column(params["wind_speed"])
        self.windSpeedVector_x = column(params["wind"][0])
        self.windSpeedVector_y = column(params["wind"][1])

    def ResidualOPDPSD(self, fx, fy):
        """
        Physical OPD PSDs (m^2 m^2) of every layer at arbitrary frequencies
        fx (along dim -2), fy (1/m), before the fractional-r0 weighting.

        Returns (total, atmosphere): total is (nLayers, Nphases, *fx.shape),
        the open-loop atmosphere or, with generateClosedLoop, the fitting +
        servo-lag residual; atmosphere is the open-loop PSD, (Nphases, *fx.shape).
        """
        atmosphere = VonKarmanPSD(fx**2 + fy**2, self.r0_moving, self.L0, self.f_slope)
        atmosphere = atmosphere * (self.referenceWavelength / (2 * torch.pi)) ** 2
        if not self.generateClosedLoop:
            return atmosphere.expand(self.nLayers, *atmosphere.shape), atmosphere

        fitting = GetFittingPSD(fx, fy, None, self.D, self.Nactuator, self.levelOfCorrection)
        temporalError = GetTemporalErrorPSD(fx, fy, self.loopFrequency, self.loopGain, self.loopLeak,
                                            self.delayFrames, self.windSpeedVector_x, self.windSpeedVector_y)
        total = atmosphere * fitting + temporalError * (1. - fitting) * atmosphere
        return total, atmosphere

    def BuildAtmospherePSD(self):
        """
        PhaseDataset-compatible PSDs on its 2 Nres (4 Nres with scintillation)
        torus grid, in its FFT-generator normalization (physical PSD * dF^2 N^2).
        Not used to generate screens here; kept for API compatibility.
        """
        total, atmosphere = self.ResidualOPDPSD(self.fx_moving, self.fy_moving)
        norm = self.dF_moving**2 * self.fx_moving.shape[-1] ** 2
        total = total * norm
        if not self.generateClosedLoop:
            return total, total
        return total, atmosphere * norm

    # ------------------------------------------------------------------
    # Screen engine
    # ------------------------------------------------------------------
    # State, with nL layers, B = Nphases, 3 crossfaded screens per level and
    # nPSD = 2 only for closed loop + scintillation (residual and open-loop
    # atmosphere driven by the same noise), else 1:
    #   s (nL, B, 2) float64 displacement (m); wind, windDirection (nL, B, 2)
    #   level 0 (FFT torus):  coefficients0 (nPSD, 3, nL, B, N0, N0), amplitude0 (nPSD, nL, B, N0, N0)
    #   coarse levels, stacked (all Nc x Nc), level axis first so the transforms batch over it:
    #       coarseCoefficients (nC, nPSD, 3, nL, B, Nc, Nc), coarseAmplitude (nC, nPSD, nL, B, Nc, Nc)
    #   passages (nLevels, 3, nL, B): blend period each screen was last drawn for

    def BuildLevels(self, stepPx):
        """
        Geometry of the nested frequency lattices for a maximum motion of
        `stepPx` pixels per frame (cached). Returns a dict with
            levels: list of per-level dicts -- index, L (torus side, m),
                R = L / 2, N, f1d (N,) float64 1/m (FFT order for level 0),
                keep (N, N) bool (the lattice points this level carries)
            sides: (nLevels,) float64 torus sides
            coarseFreq: (nC, Nc) float64 coarse-level frequencies (None if nC = 0)
            coarseET: (nC, Nc, outputSize) complex64, E^T with E[n, a] = exp(2 pi i x_n f_a)
            coarseK: (nC, 2 Nc, outputSize) float32, [Re E, -Im E]^T interleaved, so that
                view_as_real(H) @ coarseK = Re(H E^T) for complex H
        """
        hole, q = self.subharmonicHole, self.subharmonicRefinement
        if q % 2 == 0:
            raise ValueError("subharmonicRefinement must be odd for the levels to tile exactly")
        key = (stepPx, self.subharmonicLevels, hole, q)
        if key in self._levelCache:
            return self._levelCache[key]

        visiblePx = math.ceil(self.visibleWidth / self.px)
        N = FFTFriendlyOdd(max(self.outputSize, 2 * (visiblePx + stepPx + 1) + 1))  # R0 >= visible + one step
        L = N * self.px
        nLevels = None if self.subharmonicLevels is None else self.subharmonicLevels + 1
        L0Max = max(self.L0Range)
        levels = []
        while True:
            index = len(levels)
            last = (index == nLevels - 1) if nLevels is not None else (index > 0 and L >= 3 * L0Max)
            i = torch.arange(N, device=self.device) - N // 2
            if index == 0:
                i = torch.fft.ifftshift(i)  # level 0 is evaluated by FFT: coefficients in FFT order
            I, J = torch.meshgrid(i, i, indexing="ij")
            keep = (I != 0) | (J != 0)
            if not last:
                keep &= torch.maximum(I.abs(), J.abs()) > hole
            levels.append({"index": index, "L": L, "R": L / 2, "N": N, "f1d": i.double() / L, "keep": keep})
            if last:
                break
            L, N = L * q, (2 * hole + 1) * q

        geometry = {"levels": levels,
                    "sides": torch.tensor([level["L"] for level in levels], dtype=torch.float64, device=self.device),
                    "coarseFreq": None, "coarseET": None, "coarseK": None}
        if len(levels) > 1:
            x = torch.arange(self.outputSize, device=self.device, dtype=torch.float64) * self.px
            f = torch.stack([level["f1d"] for level in levels[1:]])
            E = torch.exp(2j * math.pi * x[:, None] * f[:, None, :]).to(torch.complex64)  # (nC, n, Nc)
            geometry["coarseFreq"] = f
            geometry["coarseET"] = E.transpose(-1, -2).contiguous()
            geometry["coarseK"] = torch.stack([E.real, -E.imag], -1).flatten(-2).transpose(-1, -2).contiguous()
        self._levelCache[key] = geometry
        return geometry

    def LevelAmplitude(self, index):
        """(nPSD, nL, B, N, N) coefficient standard deviations of level `index`."""
        return self.amplitude0 if index == 0 else self.coarseAmplitude[index - 1]

    @torch.no_grad()
    def ResetScreens(self):
        """Size the levels for the current winds, draw every screen and set s = 0."""
        wind = torch.stack((self.windSpeedVector_x, self.windSpeedVector_y), -1).reshape(self.nLayers, self.Nphases, 2)
        self.wind = wind.double()
        speed = self.wind.norm(dim=-1, keepdim=True)
        default = torch.tensor([1.0, 0.0], device=self.device, dtype=torch.float64)
        self.windDirection = torch.where(speed > 0, self.wind / speed.clamp_min(1e-300), default)
        self.s = torch.zeros_like(self.wind)  # accumulated displacement (m), float64

        geometry = self.BuildLevels(math.ceil(speed.max().item() * self.dt / self.px))
        self.levels = geometry["levels"]
        self.levelSides = geometry["sides"]
        self.coarseFreq, self.coarseET, self.coarseK = geometry["coarseFreq"], geometry["coarseET"], geometry["coarseK"]
        self.nCoarse = len(self.levels) - 1

        nPSD = 2 if (self.useScintillation and self.generateClosedLoop) else 1
        amplitudes = []
        for level in self.levels:
            f = level["f1d"].float()
            FX, FY = torch.meshgrid(f, f, indexing="ij")
            total, atmosphere = self.ResidualOPDPSD(FX, FY)
            psds = torch.stack([total, atmosphere.expand_as(total)][:nPSD])  # (nPSD, nL, B, N, N)
            psds = psds.clamp_min(0) * level["keep"] * self.fractionalr0
            amplitudes.append((torch.sqrt(psds) / level["L"]).float())

        self.amplitude0 = amplitudes[0]
        self.coefficients0 = self.amplitude0.unsqueeze(1) * self.DrawNoise((3, *self.amplitude0.shape[1:]))
        self.coarseAmplitude = self.coarseCoefficients = None
        if self.nCoarse:
            self.coarseAmplitude = torch.stack(amplitudes[1:])
            noise = self.DrawNoise((self.nCoarse, 1, 3, *self.coarseAmplitude.shape[2:]))
            self.coarseCoefficients = self.coarseAmplitude.unsqueeze(2) * noise
        self.passages = self.WindowState()[1]
        self.redrawCounts = torch.zeros(len(self.levels), dtype=torch.long)

        self.rho = None
        if self.boiling:
            if self.boilingTime is None:
                raise ValueError("boiling=True needs a boilingTime (s), scalar or [min, max]")
            tau = torch.as_tensor(self.boilingTime, dtype=torch.float32, device=self.device).reshape(-1)
            low, high = tau.min().item(), tau.max().item()
            self.boilingTimes = torch.empty(self.Nphases, device=self.device).uniform_(low, high)
            rho = []
            for level in self.levels:
                f = level["f1d"].float()
                FX, FY = torch.meshgrid(f, f, indexing="ij")
                rate = (torch.sqrt(FX**2 + FY**2) * self.D) ** (2 / 3) / self.boilingTimes.reshape(-1, 1, 1)
                rho.append(torch.exp(-self.dt * rate))  # (B, N, N)
            # Per level: rho and the innovation scale sqrt(2 (1 - rho^2)) * amplitude, broadcast against
            # the coefficients (level 0: (nPSD, 3, nL, B, N0, N0); coarse: (nC, nPSD, 3, nL, B, Nc, Nc)).
            self.rho = [(rho[0], self.amplitude0.unsqueeze(1) * torch.sqrt(2 * (1 - rho[0] ** 2)))]
            if self.nCoarse:
                rhoC = torch.stack(rho[1:]).reshape(self.nCoarse, 1, 1, 1, *rho[1].shape)
                self.rho.append((rhoC, self.coarseAmplitude.unsqueeze(2) * torch.sqrt(2 * (1 - rhoC ** 2))))

    def DrawNoise(self, shape):
        """Complex white noise with N(0, 1) real and imaginary parts."""
        return torch.randn(shape, dtype=torch.complex64, device=self.device) * math.sqrt(2)

    def WindowState(self):
        """
        For each level and screen: is the screen's weight zero over the whole
        visible region, and which blend period (passage) is the region in?
        Both (nLevels, 3, nL, B), from the unwrapped float64 displacement.
        """
        R = (self.levelSides / 2).reshape(-1, 1, 1, 1)
        p = (self.s * self.windDirection).sum(-1)
        edge = self.visibleCenter * self.windDirection.sum(-1) - p - self.visibleWidth / 2  # along-wind start (m)
        a = edge / R - torch.arange(3, device=self.device, dtype=torch.float64).reshape(1, 3, 1, 1)
        r = torch.remainder(a, 3)
        return (r >= 2) & (r + self.visibleWidth / R <= 3), torch.floor(a / 3)

    @torch.no_grad()
    def AdvanceFrame(self):
        """s += v dt, redraw the screens that just became invisible, then boil."""
        self.s = self.s + self.wind * self.dt
        self.frame += 1

        zero, passage = self.WindowState()
        redraw = zero & (passage != self.passages)
        self.passages = torch.where(redraw, passage, self.passages)
        index = redraw.nonzero().cpu()  # (n, 4): level, screen, layer, batch -- one host sync per frame
        if len(index):
            self.redrawCounts += torch.bincount(index[:, 0], minlength=len(self.levels))
            first = index[:, 0] == 0
            _, k, layer, b = index[first].T
            if len(k):
                noise = self.DrawNoise((len(k), *self.amplitude0.shape[-2:]))
                self.coefficients0[:, k, layer, b] = self.amplitude0[:, layer, b] * noise
            c, k, layer, b = index[~first].T
            if len(k):
                noise = self.DrawNoise((len(k), *self.coarseAmplitude.shape[-2:]))
                # advanced indices around a slice: both sides are (n, nPSD, Nc, Nc)
                self.coarseCoefficients[c - 1, :, k, layer, b] = self.coarseAmplitude[c - 1, :, layer, b] * noise[:, None]

        if self.rho is not None:
            # c <- rho c + sqrt(1 - rho^2) * amplitude * fresh noise: keeps the PSD exact
            self.coefficients0 = self.Boil(self.coefficients0, *self.rho[0], psdAxis=0)
            if self.nCoarse:
                self.coarseCoefficients = self.Boil(self.coarseCoefficients, *self.rho[1], psdAxis=1)

    def Boil(self, C, rho, innovation, psdAxis):
        """AR(1) step. The fresh noise is shared along the PSD axis, as in a redraw; `innovation`
        already holds sqrt(2), so the noise is drawn with N(0, 1/2) real and imaginary parts."""
        shape = list(C.shape)
        shape[psdAxis] = 1
        noise = torch.randn(shape, dtype=torch.complex64, device=self.device)
        return torch.addcmul(rho * C, innovation, noise)

    @staticmethod
    def PhaseRamps(f, s):
        """exp(-2 pi i f s_x) and exp(-2 pi i f s_y): f (..., 1, 1, N) float64, s (..., nL, B, 2) float64
        -> two (..., nL, B, N) complex64 ramps, along dims -2 and -1."""
        return (torch.exp(-2j * math.pi * f * s[..., 0:1]).to(torch.complex64),
                torch.exp(-2j * math.pi * f * s[..., 1:2]).to(torch.complex64))

    @torch.no_grad()
    def LayeredOPD(self):
        """(nPSD, nLayers, Nphases, out, out) layered OPD (m) at the current displacement."""
        # Everything below is evaluated on the transposed grid (both transforms then end on a
        # contiguous axis) and transposed back once at the end.
        n = self.outputSize
        ux = self.windDirection[..., 0].float()[..., None, None]
        uy = self.windDirection[..., 1].float()[..., None, None]
        xiT = self.Y * ux + self.X * uy                                  # along-wind coordinate at [m, n] = (x_n, y_m)
        sides = self.levelSides
        p = torch.remainder((self.s * self.windDirection).sum(-1), 1.5 * sides[:, None, None]).float()  # mod 3R
        R = (sides / 2).float()
        s = torch.remainder(self.s, sides[:, None, None, None])          # a shift by L is the identity on a torus

        # Level 0: pruned inverse FFT (no 1/N scaling) -- transform along f_y, keep the n output
        # columns, apply the f_x ramp while transposing, then transform along f_x.
        phaseX, phaseY = self.PhaseRamps(self.levels[0]["f1d"], s[0])  # (nL, B, N0)
        G = torch.fft.ifft(self.coefficients0 * phaseY[..., None, :], dim=-1, norm="forward")[..., :n]
        G = G.transpose(-1, -2) * phaseX[..., None, :]                   # (nPSD, 3, nL, B, n, N0)
        fieldT = torch.fft.ifft(G, dim=-1, norm="forward")[..., :n].real
        u = ((xiT - p[0, ..., None, None]) / R[0]).unsqueeze(0)          # (1, nL, B, m, n)
        outT = CrossfadeSum(u, fieldT, axis=1)                           # (nPSD, nL, B, m, n)

        if self.nCoarse:
            # Coarse levels: matrix Fourier transform E G E^T as two GEMMs batched over the level
            # axis (the second one real, giving Re directly), a few levels at a time to bound memory.
            Nc = self.coarseFreq.shape[-1]
            M = self.coarseCoefficients[0, ..., 0, 0].numel()           # nPSD * 3 * nL * B
            chunk = max(1, self.maxFieldElements // (M * n * n))
            for start in range(0, self.nCoarse, chunk):
                c = slice(start, start + chunk)
                C = self.coarseCoefficients[c]                            # (nc, nPSD, 3, nL, B, Nc, Nc)
                nc = C.shape[0]
                phaseX, phaseY = self.PhaseRamps(self.coarseFreq[c, None, None, :], s[1:][c])  # (nc, nL, B, Nc)
                G = C * phaseX[:, None, None, ..., :, None] * phaseY[:, None, None, ..., None, :]
                H = G.reshape(nc, M * Nc, Nc) @ self.coarseET[c]          # sum over f_y: (nc, M * Nc, n)
                H = H.reshape(nc, M, Nc, n).transpose(-1, -2).contiguous()  # (nc, M, m, Nc)
                fieldT = torch.view_as_real(H).reshape(nc, M * n, 2 * Nc) @ self.coarseK[c]  # Re, sum over f_x
                fieldT = fieldT.reshape(*C.shape[:5], n, n)               # (nc, nPSD, 3, nL, B, m, n)
                u = (xiT - p[1:][c, ..., None, None]) / R[1:][c].reshape(-1, 1, 1, 1, 1)
                outT = outT + CrossfadeSum(u[:, None, None], fieldT, axis=2).sum(0)
        return outT.transpose(-1, -2).contiguous()
