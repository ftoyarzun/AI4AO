"""Tests for AI4AO.PyramidWFS."""
import numpy as np
import pytest
import torch

from AI4AO.PhaseDataset import Zernike
from AI4AO.PyramidWFS import PyramidWFS


# ---------------------------------------------------------------------------
# Pure/deterministic pieces
# ---------------------------------------------------------------------------

def test_get_pupil_center_default_params(pyramid_wfs):
    # mainSlope=pi/2, maskShifts=ones(4,2) (the __init__ defaults) place the
    # 4 pupil centers at the symmetric quarter-points of the Npix x Npix frame.
    centers = pyramid_wfs.GetPupilCenter()
    Npix = pyramid_wfs.Npix
    q = Npix // 4

    assert centers.shape == (4, 2)
    expected = np.array([[q, q], [Npix - q, q], [Npix - q, Npix - q], [q, Npix - q]])
    assert np.array_equal(centers, expected)


def test_pyramid_mask_scalar_vs_array_offsets(pyramid_wfs):
    mask_scalar = pyramid_wfs.PyramidMask(x_offset=0, y_offset=0)
    assert mask_scalar.shape == (pyramid_wfs.Npix, pyramid_wfs.Npix)

    x_offset = torch.tensor([0.0, 1.0, 2.0])
    y_offset = torch.tensor([0.0, -1.0, -2.0])
    mask_batch = pyramid_wfs.PyramidMask(x_offset=x_offset, y_offset=y_offset)
    assert mask_batch.shape == (3, pyramid_wfs.Npix, pyramid_wfs.Npix)


# ---------------------------------------------------------------------------
# Forward pass (mask already built at construction time)
# ---------------------------------------------------------------------------

def test_forward_static_mask_shape_and_normalization(pyramid_wfs, device):
    Nphases = 3
    opd = torch.randn(Nphases, pyramid_wfs.Nres, pyramid_wfs.Nres, device=device)

    frame = pyramid_wfs(opd)

    assert frame.shape == (Nphases, pyramid_wfs.Npix, pyramid_wfs.Npix)
    assert torch.all(frame >= 0)
    sums = frame.sum(dim=(-2, -1))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_forward_modulated_mask_shape_and_normalization(tiny_wfs_params, device):
    wfs = PyramidWFS(tiny_wfs_params(modulation=1.0), device)
    Nphases = 2
    opd = torch.randn(Nphases, wfs.Nres, wfs.Nres, device=device)

    frame = wfs(opd)

    assert frame.shape == (Nphases, wfs.Npix, wfs.Npix)
    assert torch.all(frame >= 0)
    sums = frame.sum(dim=(-2, -1))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


# ---------------------------------------------------------------------------
# Calibration chain
# ---------------------------------------------------------------------------

def test_calibration_chain_recovers_reference(pyramid_wfs):
    pyramid_wfs.BuildReferenceIntensity()

    _, modes = Zernike(pyramid_wfs.pupil, j=3)


    pyramid_wfs.BuildReconstructionMatrix(modes)

    reconstructed = pyramid_wfs.GetReconstructedOPD(pyramid_wfs.reference_intensity)
    assert torch.allclose(reconstructed, torch.zeros_like(reconstructed), atol=1e-4)


# ---------------------------------------------------------------------------
# Chromatic modulation and rooftop (polychromatic sensing)
# ---------------------------------------------------------------------------

_BAND = [600e-9, 700e-9, 800e-9]  # lambda_c = 700 nm


def _legacy_pyramid_mask(wfs):
    """Monochromatic mask computed the pre-chromatic way: modulation offsets
    modulation * sampling and rooftop width rooftop * sampling, in pixels.
    Returns (C, H, W), C = number of modulation steps (1 when unmodulated)."""
    if wfs.modulation == 0:
        x_off = torch.zeros(1, device=wfs.device)
        y_off = torch.zeros(1, device=wfs.device)
    else:
        nSteps = min(wfs.maxModulationSteps, max(round(6.28 * wfs.modulation / 4) * 4, 8))
        steps = torch.linspace(0, 2 * torch.pi * (1 - 1 / nSteps), nSteps, device=wfs.device, dtype=torch.float32)
        x_off = wfs.modulation * wfs.sampling * torch.cos(steps)
        y_off = wfs.modulation * wfs.sampling * torch.sin(steps)
    r = wfs.rooftop * wfs.sampling / np.sqrt(2)
    s = wfs.maskShifts
    x = wfs.x_mask.unsqueeze(0) + x_off.view(-1, 1, 1)
    y = wfs.y_mask.unsqueeze(0) + y_off.view(-1, 1, 1)
    P1 = (x + r / 2) * s[0, 0] + (y + r / 2) * s[0, 1]
    P2 = -x * s[1, 0] + y * s[1, 1]
    P3 = -(x - r / 2) * s[2, 0] - (y - r / 2) * s[2, 1]
    P4 = x * s[3, 0] - y * s[3, 1]
    return torch.max(torch.stack([P1, P2, P3, P4]) * wfs.mainSlope, dim=0).values


def _pyramid(tiny_wfs_params, device, wavelength, modulation, rooftop):
    params = tiny_wfs_params(modulation=modulation)
    params["Wavelength"] = wavelength
    wfs = PyramidWFS(params, device)
    with torch.no_grad():
        wfs.rooftop.fill_(rooftop)
    wfs.BuildMask()
    return wfs


@pytest.mark.parametrize("modulation", [0.0, 1.0, 3.0])
def test_single_wavelength_mask_and_frame_match_legacy(tiny_wfs_params, device, modulation):
    """For one wavelength lambda_c / lambda is exactly 1, so the mask and frame
    must be bit-identical to the pre-chromatic computation."""
    wfs = _pyramid(tiny_wfs_params, device, 635e-9, modulation, rooftop=0.7)
    legacy = _legacy_pyramid_mask(wfs)

    assert wfs.phaseMask.shape == (1, *legacy.shape)
    assert torch.equal(wfs.phaseMask[0], legacy)

    reference = PyramidWFS(tiny_wfs_params(modulation=modulation), device)
    reference.SetMask(phaseMask=legacy)
    opd = 1e-7 * torch.randn(2, wfs.Nres, wfs.Nres, device=device)
    assert torch.equal(wfs.Propagator(opd), reference.Propagator(opd))


def _monochromatic_equivalents(tiny_wfs_params, device, modulation, rooftop):
    """One monochromatic Pyramid per band wavelength, with modulation and
    rooftop rescaled from lambda_c/D to that wavelength's own lambda/D."""
    lambda_c = (min(_BAND) + max(_BAND)) / 2
    return [
        _pyramid(tiny_wfs_params, device, wl, modulation * lambda_c / wl, rooftop * lambda_c / wl)
        for wl in _BAND
    ]


def test_polychromatic_channel_equals_rescaled_monochromatic(tiny_wfs_params, device):
    """Each wavelength channel of a polychromatic Pyramid (modulation r and
    rooftop rho in lambda_c/D) must equal a monochromatic Pyramid at that
    wavelength with modulation r * lambda_c / lambda and rooftop
    rho * lambda_c / lambda."""
    modulation, rooftop = 1.0, 0.8
    wfs = _pyramid(tiny_wfs_params, device, 635e-9, modulation, rooftop)
    wfs.wavelength = torch.tensor(_BAND, device=device)
    monos = _monochromatic_equivalents(tiny_wfs_params, device, modulation, rooftop)

    # Chosen so every channel uses the same number of modulation steps.
    assert all(m.phaseMask.shape[1] == wfs.phaseMask.shape[1] for m in monos)

    opd = 1e-7 * torch.randn(2, wfs.Nres, wfs.Nres, device=device)
    frames = wfs.Propagator(opd, collapse_wvl=False)
    assert frames.shape == (2, len(_BAND), wfs.Npix, wfs.Npix)
    for i, mono in enumerate(monos):
        assert torch.allclose(frames[:, i], mono.Propagator(opd), rtol=1e-4, atol=1e-8)

    # Sanity check: the channels really differ from the achromatic (unscaled) mask.
    achromatic = _pyramid(tiny_wfs_params, device, _BAND[0], modulation, rooftop)
    assert not torch.allclose(frames[:, 0], achromatic.Propagator(opd), rtol=1e-4, atol=1e-8)


def test_polychromatic_collapsed_frame_is_normalized_sum_of_equivalents(tiny_wfs_params, device):
    modulation, rooftop = 1.0, 0.8
    wfs = _pyramid(tiny_wfs_params, device, 635e-9, modulation, rooftop)
    wfs.wavelength = torch.tensor(_BAND, device=device)
    monos = _monochromatic_equivalents(tiny_wfs_params, device, modulation, rooftop)

    opd = 1e-7 * torch.randn(2, wfs.Nres, wfs.Nres, device=device)
    expected = torch.stack([m.Propagator(opd) for m in monos]).sum(dim=0)
    expected = expected / expected.sum(dim=(-2, -1), keepdim=True)

    assert torch.allclose(wfs.Propagator(opd), expected, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_wavelength_setter_rebuilds_mask_and_reference(tiny_wfs_params, device, mode):
    wfs = _pyramid(tiny_wfs_params, device, 635e-9, modulation=1.0, rooftop=0.8)
    wfs.train(mode == "train")
    reference_before = wfs.reference_intensity.clone()

    wfs.wavelength = torch.tensor(_BAND, device=device)

    assert wfs.phaseMask.shape[0] == len(_BAND)
    assert wfs.reference_intensity.shape == reference_before.shape
    assert not torch.allclose(wfs.reference_intensity, reference_before)
    # eval() mirrors WFS.train(False): the rebuild runs without a graph.
    assert wfs.phaseMask.requires_grad == (mode == "train")


def test_gradients_flow_through_chromatic_pyramid_mask(tiny_wfs_params, device):
    wfs = _pyramid(tiny_wfs_params, device, 635e-9, modulation=1.0, rooftop=0.8)
    wavelength = torch.tensor(_BAND, device=device).requires_grad_()
    wfs.wavelength = wavelength

    opd = 1e-7 * torch.randn(2, wfs.Nres, wfs.Nres, device=device)
    torch.var(wfs(opd), dim=(-2, -1)).sum().backward()

    for grad in (wfs.mainSlope.grad, wfs.rooftop.grad, wavelength.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert torch.any(grad != 0)

    # With a flat wavefront the phase carries no wavelength dependence, so a
    # nonzero wavelength gradient must come through the chromatic mask itself.
    # wavenumber is derived once in the setter, so re-assign for a fresh graph.
    wavelength.grad = None
    wfs.wavelength = wavelength
    flat = torch.zeros(1, wfs.Nres, wfs.Nres, device=device)
    torch.var(wfs(flat), dim=(-2, -1)).sum().backward()
    assert wavelength.grad is not None
    assert torch.isfinite(wavelength.grad).all()
    assert torch.any(wavelength.grad != 0)
