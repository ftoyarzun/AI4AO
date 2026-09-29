# -*- coding: utf-8 -*-
"""
Power spectral densities (PSDs) and loop transfer functions of an AO system.

Shared by PhaseDataset (which synthesizes OPD screens from these PSDs) and
PSFModel/PSFDataset (which turn the residual PSD directly into a PSF). All
functions are pure and differentiable torch code.

Units: `VonKarmanPSD` is the phase PSD in rad^2 m^2 at the wavelength r0 is
quoted at; every other PSD here is an OPD PSD in m^2 m^2 (see the OPD-first
convention in .claude/CLAUDE.md). `GetAtmospherePSD` is the one exception: it
carries PhaseDataset's FFT-generator normalization (times dF^2 N^2).

Several formulas follow AOPERA (R. Fetick et al.,
https://gitlab.lam.fr/lam-grd-public/aopera) and maoppy
(https://gitlab.lam.fr/lam-grd-public/maoppy); each function says which.
"""
import math

import torch  # type: ignore[import]


# Wavelength at which r0 is conventionally quoted.
REFERENCE_WAVELENGTH = 500e-9

# Kolmogorov PSD constant, 0.023 r0^(-5/3) f^(-11/3).
KOLMOGOROV_CONSTANT = 0.023


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def GetSpatialFrequencies(D, resolution, device="cpu"):
    """
    Computes the spatial frequencies of an FFT grid.

    Args:
        D (float): Physical extent of the grid (m); the frequency step is 1/D.
        resolution (int): Number of samples along each axis.

    Returns:
        tuple:
            - dF (float): Frequency step size (1/m)
            - fx (torch array): Spatial frequencies along dim -2 ("ij" meshgrid)
            - fy (torch array): Spatial frequencies along dim -1

    Zero frequency sits at index resolution // 2 for both odd and even
    resolutions (fftshift convention).
    """
    dF = 1 / (D)
    fx = (
        torch.arange(resolution, dtype=torch.float32, device=device) - resolution // 2
    ) * dF
    [fx, fy] = torch.meshgrid(fx, fx, indexing="ij")
    return dF, fx, fy


def CenterCrop(x, size):
    """Crops the central size x size window of the last two dims, keeping the
    element at index N // 2 at index size // 2 (the fftshift center of both)."""
    N = x.shape[-1]
    start = N // 2 - size // 2
    return x[..., start:start + size, start:start + size]


# ---------------------------------------------------------------------------
# Atmosphere
# ---------------------------------------------------------------------------

def VonKarmanPSD(fsqr, r0, L0, f_slope=11.0 / 6.0):
    """
    Physical von Karman phase PSD, in rad^2 m^2 at the wavelength r0 is quoted
    at. Zero at f = 0 (piston).

    Args:
        fsqr (torch array): Squared spatial frequency (1/m^2).
        r0 (float or torch array): Fried parameter (m), broadcastable to fsqr.
        L0 (float or torch array): Outer scale (m), broadcastable to fsqr.
        f_slope (float): PSD exponent (11/6 for Kolmogorov).
    """
    l0 = 1e-10  # inner scale (m); effectively no inner-scale cutoff
    fm = 5.92 / l0 / (2 * torch.pi)
    f0 = 1 / L0
    psd = (
        KOLMOGOROV_CONSTANT
        * r0 ** (-5 / 3)
        / (fsqr + f0**2) ** (f_slope)
        * torch.exp(-fsqr / fm**2)
    )
    return torch.where(fsqr == 0, torch.zeros_like(psd), psd)


def VonKarmanOPDPSD(fsqr, r0, L0, f_slope=11.0 / 6.0):
    """Von Karman PSD as an OPD PSD (m^2 m^2), wavelength-independent. r0 is
    the Fried parameter at REFERENCE_WAVELENGTH (500 nm)."""
    return VonKarmanPSD(fsqr, r0, L0, f_slope) * (REFERENCE_WAVELENGTH / (2 * torch.pi)) ** 2


def GetAtmospherePSD(fsqr, dF, r0, L0, f_slope=11.0 / 6.0):
    """
    Von Karman phase PSD in PhaseDataset's FFT-generator normalization: the
    physical PSD (VonKarmanPSD, rad^2 m^2) times dF^2 * resolution^2, so that
    fft2(sqrt(PSD) * complex white noise, norm="ortho") has the right variance.

    Args:
        fsqr (torch array): Squared spatial frequency (1/m^2), (..., N, N)
        dF (float): Frequency step size (1/m)
        r0 (float): Fried parameter (m)
        L0 (float): Outer scale of turbulence (m)
        f_slope (float): PSD exponent (11/6 for Kolmogorov)

    Returns:
        torch array: Generator-normalized atmospheric phase PSD
    """
    resolution = fsqr.shape[-1]
    return VonKarmanPSD(fsqr, r0, L0, f_slope) * dF**2 * resolution**2


def VonKarmanTail(r0, f_max):
    """OPD variance (m^2) of the Kolmogorov PSD outside the disk |f| > f_max,
    0.023 (6 pi / 5) r0^(-5/3) f_max^(-5/3), with L0 neglected. Source:
    maoppy Psfao.psd ("integral to infinity")."""
    phase = KOLMOGOROV_CONSTANT * 6 * torch.pi / 5 * r0 ** (-5 / 3) * f_max ** (-5 / 3)
    return phase * (REFERENCE_WAVELENGTH / (2 * torch.pi)) ** 2


def GetAliasedAtmospherePSD(fx, fy, r0, L0, f_slope, pitch, n_shift=1):
    """
    Sum of the von Karman OPD PSD evaluated at every alias of (fx, fy) for a
    sampler of pitch `pitch` (m): sum over (i, j) in [-n, n]^2 \\ (0, 0) of
    Phi(fx - i / pitch, fy - j / pitch). Since the PSD is analytic it is
    evaluated directly at the shifted frequencies (no roll or interpolation,
    unlike AOPERA's psd_aliasing). Returns an OPD PSD (m^2 m^2).
    """
    out = 0.0
    fs = 1 / pitch
    for i in range(-n_shift, n_shift + 1):
        for j in range(-n_shift, n_shift + 1):
            if i == 0 and j == 0:
                continue
            fsqr = (fx - i * fs) ** 2 + (fy - j * fs) ** 2
            out = out + VonKarmanOPDPSD(fsqr, r0, L0, f_slope)
    return out


def PowerLawPSD(fsqr, exponent):
    """Unnormalized f^(-exponent) PSD, zero at f = 0."""
    safe = torch.where(fsqr > 0, fsqr, torch.ones_like(fsqr))
    return torch.where(fsqr > 0, safe ** (-exponent / 2), torch.zeros_like(fsqr))


def PistonFilter(fsqr, D):
    """1 - (2 J1(pi D f) / (pi D f))^2: removes the pupil-piston component of
    a PSD. Source: AOPERA aopsd.piston_filter."""
    x = torch.pi * D * torch.sqrt(fsqr)
    safe = torch.where(x > 0, x, torch.ones_like(x))
    airy = (2 * torch.special.bessel_j1(safe) / safe) ** 2
    return torch.where(x > 0, 1 - airy, torch.zeros_like(x))


# ---------------------------------------------------------------------------
# DM correction and loop transfer functions
# ---------------------------------------------------------------------------

def GetFittingPSD(fx, fy, dF, D, Nactuator, levelOfCorrection=1):
    """
    Computes a fitting power spectral density (PSD) filter, including both low-pass and high-pass components.

    Args:
        fx (torch array): Spatial frequency components in the x direction
        fy (torch array): Spatial frequency components in the y direction
        dF (float): Frequency step size
        D (float): Diameter of the telescope
        Nactuator (int): Number of actuators in the diameter of the deformable mirror
        levelOfCorrection (float, optional): Correction factor for high-pass filter (default is 1)

    Returns:
        torch array: High-pass filter for the fitting PSD, 1 - levelOfCorrection
        inside the square corrected band |fx|, |fy| < fc, 1 outside.
    """
    # DM pitch is D / (Nactuator - 1) (see DeformableMirror.MakeActGrid)
    fc = (Nactuator - 1) / 2 / D

    low_pass_filter = (fx < fc) & (fy > -fc) & (fy < fc) & (fx > -fc)
    high_pass_filter = 1 - low_pass_filter * levelOfCorrection

    return high_pass_filter


def openLoopTransferFunction(freq, ao_freq, ki, leak, nb_frame_delay):
    """
    Return the temporal open loop transfer function for a integrator controller.
    Source: AOPERA (R. Fetick)
    Parameters
    ----------
    freq : np.array
        Array of temporal frequencies to evaluate the CLTF on.
    ao_freq : float
        The sampling temporal frequency of the AO loop.
    ki : float
        Integrator gain.
    leak : float
        Leaky integrator.
    nb_frame_delay : float
        Number of frame delay.
        Must include: RTC, pixel transfert, DM rise.
        Must not include: WFS integration, DM zero-order-hold.
    """
    z = torch.exp(2j*torch.pi*freq/ao_freq) # it is one method to pass from Tp to z
    not_zero_issue = 1 - 1e-8 # avoid issue to divide by zero at leak/z = 1
    H = ki/(1-not_zero_issue*leak/z) # controler
    H *= 1/z**(nb_frame_delay+1) # delay + WFS + zero order hold
    H *= torch.sinc(freq/ao_freq)

    return H


def closedLoopTransferFunction(*args, **kwargs):
    """
    Return the temporal closed loop transfer function.
    See the open_loop_transfer arguments.
    Source: AOPERA (R. Fetick)
    """
    return 1/(1+openLoopTransferFunction(*args, **kwargs))


def noiseTransferFunction(*args, **kwargs):
    """
    Return the noise transfer function H_ol / (1 + H_ol) = 1 - CLTF.
    See the open_loop_transfer arguments.
    Source: AOPERA (R. Fetick), control.noise_transfer
    """
    H = openLoopTransferFunction(*args, **kwargs)
    return H / (1 + H)


def NoiseTransferGain(ao_freq, ki, leak, nb_frame_delay, n_freq=1024):
    """
    Noise propagation gain of the loop, (2 / F) * integral_0^{F/2} |NTF(nu)|^2 dnu:
    the factor by which white measurement noise reaches the residual. Source:
    AOPERA simulation (2-D mode, `ntf_integral`).

    Args:
        ao_freq (float): Loop frequency F (Hz).
        ki, leak (torch array): Integrator gain and leak, shape (B,) (or scalars).
        nb_frame_delay (float): Loop delay in frames (see openLoopTransferFunction).

    Returns:
        torch array: shape (B,) (or (1,) for scalar ki and leak).
    """
    ki = torch.as_tensor(ki, dtype=torch.float32)
    leak = torch.as_tensor(leak, dtype=torch.float32, device=ki.device)
    nu = torch.linspace(0, ao_freq / 2, n_freq, device=ki.device)
    # Evaluate slightly off nu = 0, where a leak-free integrator's pole makes
    # H_ol infinite (in float32, 1 - 0.99999999 * leak / z is exactly 0 there).
    ntf = noiseTransferFunction(nu + 1e-7, ao_freq, ki.unsqueeze(-1), leak.unsqueeze(-1), nb_frame_delay)
    return torch.trapezoid(ntf.abs() ** 2, nu, dim=-1) * 2 / ao_freq


def TemporalFrequency(fx, fy, windSpeedVector_x, windSpeedVector_y):
    """Temporal frequency (Hz) f . v that frozen flow at wind (vx, vy) maps
    spatial frequency (fx, fy) to. Offset slightly from 0 to avoid the
    integrator pole."""
    fx_temporal = fx * windSpeedVector_x + 1e-7
    fy_temporal = fy * windSpeedVector_y + 1e-7
    return fx_temporal + fy_temporal


def GetTemporalErrorPSD(
    fx, fy, freq, ki, leak, delayFrames, windSpeedVector_x, windSpeedVector_y
):
    """
    Computes the temporal error power spectral density (PSD) given the spatial frequencies and other parameters.

    Args:
        fx (torch array): Spatial frequency components in the x direction
        fy (torch array): Spatial frequency components in the y direction
        freq (float): Temporal frequency of the system
        ki, leak: Integrator gain and leak
        delayFrames (int): Number of frames for delay
        windSpeedVector_x, windSpeedVector_y (torch array): Wind speed vector [vx, vy]

    Returns:
        torch array: |CLTF(f . v)|^2, the servo-lag rejection of each frequency
    """
    f_temporal = TemporalFrequency(fx, fy, windSpeedVector_x, windSpeedVector_y)

    ETF = closedLoopTransferFunction(f_temporal, freq, ki, leak, delayFrames)
    ETF = torch.abs(ETF) ** 2

    return ETF


def GetTemporalNoisePSD(
    fx, fy, freq, ki, leak, delayFrames, windSpeedVector_x, windSpeedVector_y
):
    """|NTF(f . v)|^2 under frozen flow, the temporal filter AOPERA applies to
    the aliasing PSD (2-D mode). Same arguments as GetTemporalErrorPSD."""
    f_temporal = TemporalFrequency(fx, fy, windSpeedVector_x, windSpeedVector_y)
    return torch.abs(noiseTransferFunction(f_temporal, freq, ki, leak, delayFrames)) ** 2


# ---------------------------------------------------------------------------
# WFS noise
# ---------------------------------------------------------------------------

def WFSNoiseVariance(nphotons, ron, n_valid_pixels, photon_factor, ron_factor):
    """
    Phase measurement-noise variance per WFS pixel (rad^2, at the sensing
    wavelength): s_ph / n + s_ron (RON / n)^2 with n = nphotons / n_valid_pixels
    photons per illuminated pixel. The constants s_ph, s_ron carry the WFS
    sensitivity; AOPERA's PWFS uses s_ph = (1 + emccd) / nfaces and s_ron = 1
    (ffwfs.PWFS.var_photon_subap / var_ron_subap).
    """
    n = nphotons / n_valid_pixels
    return photon_factor / n + ron_factor * (ron / n) ** 2


def GetNoisePSD(corrected, noise_gain, variance_rad, sensing_wavelength, pitch):
    """
    White WFS-noise OPD PSD (m^2 m^2) propagated through the loop:
    corrected * G_n * variance * (lambda_wfs / 2 pi)^2 * pitch^2. Per-pixel
    variance sigma^2 sampled at pitch d is white with PSD sigma^2 d^2 over
    |f| < 1/(2d); only the corrected band (`corrected`, c * LP) reaches the DM.
    Source: AOPERA aopsd.psd_noise (2-D mode).

    Args:
        corrected (torch array): c * LP mask, (B, N, N).
        noise_gain (torch array): NoiseTransferGain, broadcastable to (B, 1, 1).
        variance_rad (torch array): WFSNoiseVariance, broadcastable to (B, 1, 1).
        sensing_wavelength (float): WFS wavelength (m), converts rad to OPD.
        pitch (float): WFS measurement pitch in the pupil (m).
    """
    variance_opd = variance_rad * (sensing_wavelength / (2 * torch.pi)) ** 2
    return corrected * noise_gain * variance_opd * pitch**2


# ---------------------------------------------------------------------------
# PSD -> covariance / screens / OTF pieces
# ---------------------------------------------------------------------------

def PSDToCovariance(psd, dF):
    """
    Autocovariance B(rho) = dF^2 Re FFT2[ifftshift PSD](rho) of a centered
    (fftshift-convention) PSD. Zero lag lands at index [0, 0] (unshifted).
    An OPD PSD gives an OPD covariance in m^2. Source: AOPERA otfpsf.psd2otf,
    maoppy ParametricPSFfromPSD._otfTurbulent.
    """
    shifted = torch.fft.ifftshift(psd, dim=(-2, -1))
    return torch.fft.fft2(shifted, dim=(-2, -1)).real * dF**2


def _SafeSqrt(x):
    """sqrt with a zero (not infinite) gradient at x = 0."""
    positive = x > 0
    safe = torch.where(positive, x, torch.ones_like(x))
    return torch.where(positive, torch.sqrt(safe), torch.zeros_like(x))


def PSDToScreen(psd, dF, noise=None):
    """
    One random screen with covariance PSDToCovariance(psd, dF), periodic over
    the grid. Same normalization as PhaseDataset's generator (physical PSD times
    dF^2 N^2, complex white noise times sqrt(2), ortho FFT, real part), so an
    OPD PSD gives an OPD screen in m. Differentiable w.r.t. psd
    (reparameterization); pass `noise` (complex, same shape) to fix the draw.
    """
    N = psd.shape[-1]
    if noise is None:
        noise = torch.randn(psd.shape, dtype=torch.complex64, device=psd.device)
    amplitude = _SafeSqrt(torch.fft.ifftshift(psd, dim=(-2, -1)) * dF**2 * N**2)
    return torch.fft.fft2(amplitude * noise * math.sqrt(2), dim=(-2, -1), norm="ortho").real


def JitterOTF(rho_x, rho_y, sigma, angle, wavelength):
    """
    Gaussian tip-tilt jitter OTF exp(-2 pi^2 (s1^2 r1^2 + s2^2 r2^2) / lambda^2),
    where (r1, r2) is the lag (rho_x, rho_y) rotated by `angle`.

    Args:
        rho_x, rho_y (torch array): Pupil-plane lag (m), (N, N).
        sigma (torch array): Jitter rms along the two principal axes (rad), (B, 2).
        angle (torch array): Rotation of the principal axes (rad), (B,).
        wavelength (float): Imaging wavelength (m).
    """
    cos = torch.cos(angle).view(-1, 1, 1)
    sin = torch.sin(angle).view(-1, 1, 1)
    r1 = rho_x * cos + rho_y * sin
    r2 = -rho_x * sin + rho_y * cos
    s1 = sigma[:, 0].view(-1, 1, 1)
    s2 = sigma[:, 1].view(-1, 1, 1)
    return torch.exp(-2 * torch.pi**2 * ((s1 * r1) ** 2 + (s2 * r2) ** 2) / wavelength**2)


# ---------------------------------------------------------------------------
# Random AO conditions
# ---------------------------------------------------------------------------

def DrawAOParameters(Nphases, nLayersRange, r0Range, L0Range, levelOfCorrectionRange,
                     loopGainRange, loopLeakRange, photonRange, RONRange, windSpeedRange,
                     height_exp_dist_lambda, device):
    """
    Draws one batch of random atmosphere/loop/detector conditions from the
    global torch RNG. Shared by PhaseDataset and PSFDataset, so for a given seed
    both draw identical conditions.

    Returns a dict of canonical shapes (B = Nphases, L = number of layers):
        r0, L0, level_of_correction, loop_gain, loop_leak, nphotons, ron: (B,)
        fractional_r0, layer_heights, wind_speed: (L, B)
        wind: (2, L, B), component 0 along array dim -2 (fx)
    """
    nLayers = int(torch.randint(*nLayersRange, (1,)))
    r0 = torch.empty(Nphases, device=device).uniform_(*r0Range)
    L0 = torch.empty(Nphases, device=device).uniform_(*L0Range)

    levelOfCorrection = torch.empty(Nphases, device=device).uniform_(*levelOfCorrectionRange)
    loopGain = torch.empty(Nphases, device=device).uniform_(*loopGainRange)
    loopLeak = torch.empty(Nphases, device=device).uniform_(*loopLeakRange)

    Nphotons = torch.pow(10, torch.empty(Nphases, device=device).uniform_(*photonRange))
    RON = torch.empty(Nphases, device=device).uniform_(*RONRange)

    fractionalr0 = torch.empty(nLayers, Nphases, device=device).uniform_(0., 1.)
    random_to_sort = torch.empty(nLayers, Nphases, device=device).uniform_(0., 0.5)
    _, index_sorted = torch.sort(fractionalr0 + random_to_sort, dim=0)
    fractionalr0 = torch.gather(fractionalr0, dim=0, index=index_sorted)
    fractionalr0 = fractionalr0 / torch.sum(fractionalr0, dim=0)

    layerHeights = torch.empty(nLayers, Nphases, device=device).exponential_(lambd=height_exp_dist_lambda)
    layerHeights, _ = torch.sort(layerHeights, dim=0, descending=True)

    windSpeed = torch.empty(nLayers, Nphases, device=device).uniform_(*windSpeedRange)
    windSpeedVector_x = torch.empty(nLayers, Nphases, device=device).uniform_(*[-1, 1])
    windSpeedVector_y = torch.empty(nLayers, Nphases, device=device).uniform_(*[-1, 1])

    currentIntegratedWindSpeed = torch.sum(
        fractionalr0 * torch.sqrt(windSpeedVector_x**2 + windSpeedVector_y**2) ** (5 / 3), dim=0
    ) ** (3 / 5)

    normalization = windSpeed / currentIntegratedWindSpeed
    windSpeedVector_x = windSpeedVector_x * normalization
    windSpeedVector_y = windSpeedVector_y * normalization

    return {
        "r0": r0,
        "L0": L0,
        "level_of_correction": levelOfCorrection,
        "loop_gain": loopGain,
        "loop_leak": loopLeak,
        "nphotons": Nphotons,
        "ron": RON,
        "fractional_r0": fractionalr0,
        "layer_heights": layerHeights,
        "wind_speed": windSpeed,
        "wind": torch.stack((windSpeedVector_x, windSpeedVector_y)),
    }
