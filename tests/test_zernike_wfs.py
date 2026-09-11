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
