"""Tests for AI4AO.ZernikeWFS."""
import pytest
import torch

from AI4AO.PhaseDataset import Zernike
from AI4AO.ZernikeWFS import ZernikeWFS


def test_build_zernike_mask_fft_single_mask(zernike_wfs):
    assert zernike_wfs.phaseMask.shape == (1, 1, zernike_wfs.Npix, zernike_wfs.Npix)
    assert zernike_wfs.pupil_centers.shape == (1, 2)


def test_build_zernike_mask_fft_double_mask(tiny_zernike_wfs_params, device):
    wfs = ZernikeWFS(tiny_zernike_wfs_params(mask_type="DoubleZernike", use_mtf=False), device)

    assert wfs.number_of_masks == 2
    assert wfs.phaseMask.shape == (1, 2, wfs.Npix, wfs.Npix)
    assert wfs.pupil_centers.shape == (2, 2)


def test_unrecognized_mask_type_raises_attribute_error(tiny_zernike_wfs_params, device):
    # "zernike" (lowercase) matches neither the case-sensitive "Zernike" branch
    # nor the case-insensitive double/vector-Zernike alias list, so depths/
    # diameters/positions/number_of_masks are never set and the BuildMask()
    # call at the end of __init__ fails with an unhelpful AttributeError
    # rather than a clear config error. This documents today's behavior.
    with pytest.raises(AttributeError):
        ZernikeWFS(tiny_zernike_wfs_params(mask_type="not zernike", use_mtf=False), device)


# ---------------------------------------------------------------------------
# Forward pass: FFT path (Use_MTF=False) vs MFT path (Use_MTF=True) are
# materially different code paths (TorchPropagator.FFTPropagator vs.
# MTFPropagator), so both are exercised explicitly.
# ---------------------------------------------------------------------------

def test_forward_fft_path_shape_and_normalization(zernike_wfs, device):
    Nphases = 2
    opd = torch.randn(Nphases, zernike_wfs.Nres, zernike_wfs.Nres, device=device)

    frame = zernike_wfs(opd)

    assert frame.shape == (Nphases, zernike_wfs.Npix, zernike_wfs.Npix)
    assert torch.all(frame >= 0)
    sums = frame.sum(dim=(-2, -1))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_forward_mft_path_shape_and_normalization(tiny_zernike_wfs_params, device):
    wfs = ZernikeWFS(tiny_zernike_wfs_params(mask_type="Zernike", use_mtf=True), device)
    Nphases = 2
    opd = torch.randn(Nphases, wfs.Nres, wfs.Nres, device=device)

    frame = wfs(opd)

    assert frame.shape == (Nphases, wfs.Npix, wfs.Npix)
    assert torch.all(frame >= 0)
    sums = frame.sum(dim=(-2, -1))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


@pytest.mark.parametrize("mtf_upscale", [1, 2, 4, 8])
def test_fft_vs_mtf_agreement_across_upscales(tiny_zernike_wfs_params, device, mtf_upscale):
    """FFT and MTF should stay in reasonable agreement across MTF_upscale
    values. Low-order modes (tip/tilt especially) carry a larger,
    upscale-dependent amplitude mismatch -- documented in
    prototyping/ZWFS_FFT_vs_MTF_investigation.ipynb, not something this test
    tries to pin down -- so the amplitude (norm ratio) check is restricted to
    the higher-order (>10) modes, where agreement is tight at every upscale.
    Spatial-pattern agreement (cosine similarity) and MFT mask sizing are
    checked at every mode/upscale regardless.
    """
    wfs_fft = ZernikeWFS(tiny_zernike_wfs_params(mask_type="Zernike", use_mtf=False), device)
    wfs_mtf = ZernikeWFS(
        tiny_zernike_wfs_params(mask_type="Zernike", use_mtf=True, mtf_upscale=mtf_upscale), device
    )

    # MFT focal-plane window/mask sizing: Nf = sampling * MTF_upscale * dot diameter.
    expected_N = int(wfs_mtf.sampling * wfs_mtf.MTF_focal_upscale * wfs_mtf.diameters[0].item())
    assert wfs_mtf.transmisionMask.shape[-2:] == (expected_N, expected_N)
    assert wfs_mtf.phaseMask.shape == (1, wfs_mtf.number_of_masks, 1, 1)

    Nmodes = 20
    amp = 20e-9  # meters
    _, modes_full = Zernike(wfs_fft.pupil, j=Nmodes)
    opd_modes = modes_full * amp

    wfs_fft.BuildReferenceIntensity()
    wfs_mtf.BuildReferenceIntensity()
    with torch.no_grad():
        signal_fft = (wfs_fft.Propagator(opd_modes) - wfs_fft.Propagator(-opd_modes)) / 2
        signal_mtf = (wfs_mtf.Propagator(opd_modes) - wfs_mtf.Propagator(-opd_modes)) / 2

    sf = signal_fft.flatten(start_dim=1)
    sm = signal_mtf.flatten(start_dim=1)
    cosine_similarity = (sf * sm).sum(dim=1) / (sf.norm(dim=1) * sm.norm(dim=1) + 1e-30)
    norm_ratio = sf.norm(dim=1) / (sm.norm(dim=1) + 1e-30)

    # Spatial pattern agreement holds for every mode, low- or high-order.
    assert torch.all(cosine_similarity > 0.95)

    # Amplitude agreement is only expected to be tight for higher-order (index
    # >10, i.e. Noll >= 12) modes.
    assert torch.all(norm_ratio[10:] > 0.9)
    assert torch.all(norm_ratio[10:] < 1.1)


# ---------------------------------------------------------------------------
# Chromatic dot (FFT path, polychromatic sensing)
# ---------------------------------------------------------------------------

_BAND = [600e-9, 700e-9, 800e-9]  # lambda_c = 700 nm


def _legacy_zernike_mask_fft(wfs):
    """FFT mask computed the pre-chromatic way: dot diameter diameters * sampling
    pixels and depth `depths` radians. Returns (1, Nmask, H, W)."""
    coords = torch.stack([-wfs.x_mask, -wfs.y_mask], dim=0)
    ramps = torch.einsum("ck,kwh->cwh", wfs.positions, coords)
    diameters_in_pixels = (wfs.diameters * wfs.sampling).unsqueeze(1).unsqueeze(1)
    annular = torch.tanh(10 * (diameters_in_pixels / 2.0 - wfs.rho_mask.unsqueeze(0))) / 2 + 0.5
    return (ramps + wfs.depths.unsqueeze(1).unsqueeze(1) * annular).reshape(1, wfs.number_of_masks, wfs.Npix, wfs.Npix)


def _zernike(tiny_zernike_wfs_params, device, wavelength, diameter_scale=1.0, depth_scale=1.0, mask_type="Zernike"):
    params = tiny_zernike_wfs_params(mask_type=mask_type, use_mtf=False)
    params["Wavelength"] = wavelength
    wfs = ZernikeWFS(params, device)
    with torch.no_grad():
        wfs.diameters.mul_(diameter_scale)
        wfs.depths.mul_(depth_scale)
    wfs.BuildMask()
    return wfs


@pytest.mark.parametrize("mask_type", ["Zernike", "DoubleZernike"])
def test_single_wavelength_fft_mask_and_frame_match_legacy(tiny_zernike_wfs_params, device, mask_type):
    wfs = _zernike(tiny_zernike_wfs_params, device, 635e-9, diameter_scale=1.3, mask_type=mask_type)
    legacy = _legacy_zernike_mask_fft(wfs)

    assert torch.equal(wfs.phaseMask, legacy)

    reference = ZernikeWFS(tiny_zernike_wfs_params(mask_type=mask_type, use_mtf=False), device)
    reference.SetMask(phaseMask=legacy)
    opd = 1e-7 * torch.randn(2, wfs.Nres, wfs.Nres, device=device)
    assert torch.equal(wfs.Propagator(opd), reference.Propagator(opd))


def _zernike_monochromatic_equivalents(tiny_zernike_wfs_params, device, mask_type):
    """One monochromatic ZWFS per band wavelength, with dot diameter and depth
    rescaled from lambda_c to that wavelength."""
    lambda_c = (min(_BAND) + max(_BAND)) / 2
    return [
        _zernike(tiny_zernike_wfs_params, device, wl, lambda_c / wl, lambda_c / wl, mask_type)
        for wl in _BAND
    ]


@pytest.mark.parametrize("mask_type", ["Zernike", "DoubleZernike"])
def test_polychromatic_fft_channel_equals_rescaled_monochromatic(tiny_zernike_wfs_params, device, mask_type):
    """Each wavelength channel of a polychromatic ZWFS (dot diameter in
    lambda_c/D, depth in radians at lambda_c) must equal a monochromatic ZWFS at
    that wavelength with both scaled by lambda_c / lambda."""
    wfs = _zernike(tiny_zernike_wfs_params, device, 635e-9, mask_type=mask_type)
    wfs.wavelength = torch.tensor(_BAND, device=device)
    monos = _zernike_monochromatic_equivalents(tiny_zernike_wfs_params, device, mask_type)

    opd = 1e-7 * torch.randn(2, wfs.Nres, wfs.Nres, device=device)
    frames = wfs.Propagator(opd, collapse_wvl=False)
    assert frames.shape == (2, len(_BAND), wfs.Npix, wfs.Npix)
    for i, mono in enumerate(monos):
        assert torch.allclose(frames[:, i], mono.Propagator(opd), rtol=1e-4, atol=1e-8)

    achromatic = _zernike(tiny_zernike_wfs_params, device, _BAND[0], mask_type=mask_type)
    assert not torch.allclose(frames[:, 0], achromatic.Propagator(opd), rtol=1e-4, atol=1e-8)


def test_polychromatic_fft_collapsed_frame_is_normalized_sum_of_equivalents(tiny_zernike_wfs_params, device):
    wfs = _zernike(tiny_zernike_wfs_params, device, 635e-9)
    wfs.wavelength = torch.tensor(_BAND, device=device)
    monos = _zernike_monochromatic_equivalents(tiny_zernike_wfs_params, device, "Zernike")

    opd = 1e-7 * torch.randn(2, wfs.Nres, wfs.Nres, device=device)
    expected = torch.stack([m.Propagator(opd) for m in monos]).sum(dim=0)
    expected = expected / expected.sum(dim=(-2, -1), keepdim=True)

    assert torch.allclose(wfs.Propagator(opd), expected, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_wavelength_setter_rebuilds_fft_mask_and_reference(zernike_wfs, device, mode):
    zernike_wfs.train(mode == "train")
    reference_before = zernike_wfs.reference_intensity.clone()

    zernike_wfs.wavelength = torch.tensor(_BAND, device=device)

    assert zernike_wfs.phaseMask.shape == (len(_BAND), 1, zernike_wfs.Npix, zernike_wfs.Npix)
    assert not torch.allclose(zernike_wfs.reference_intensity, reference_before)
    assert zernike_wfs.phaseMask.requires_grad == (mode == "train")


def test_gradients_flow_through_chromatic_zernike_fft_mask(zernike_wfs, device):
    wavelength = torch.tensor(_BAND, device=device).requires_grad_()
    zernike_wfs.wavelength = wavelength

    # Flat wavefront: the only wavelength dependence is the chromatic dot.
    flat = torch.zeros(1, zernike_wfs.Nres, zernike_wfs.Nres, device=device)
    torch.var(zernike_wfs(flat), dim=(-2, -1)).sum().backward()

    for grad in (zernike_wfs.depths.grad, zernike_wfs.diameters.grad, wavelength.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert torch.any(grad != 0)
