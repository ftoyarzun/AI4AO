"""Tests for AI4AO.ZernikeWFS."""
import pytest
import torch

from AI4AO.ZernikeWFS import ZernikeWFS


def test_make_pupil_shape_and_circularity(zernike_wfs):
    Rpx = 4.0
    nPx = 12
    pupil = zernike_wfs.make_pupil(Rpx, nPx)

    assert pupil.shape == (nPx, nPx)
    assert pupil.dtype == torch.bool
    assert pupil.sum() > 0
    assert pupil.sum() < nPx * nPx  # not everything is inside the pupil


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
    phase = torch.randn(Nphases, zernike_wfs.Nres, zernike_wfs.Nres, device=device)

    frame = zernike_wfs(phase)

    assert frame.shape == (Nphases, zernike_wfs.Npix, zernike_wfs.Npix)
    assert torch.all(frame >= 0)
    sums = frame.sum(dim=(-2, -1))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_forward_mft_path_shape_and_normalization(tiny_zernike_wfs_params, device):
    wfs = ZernikeWFS(tiny_zernike_wfs_params(mask_type="Zernike", use_mtf=True), device)
    Nphases = 2
    phase = torch.randn(Nphases, wfs.Nres, wfs.Nres, device=device)

    frame = wfs(phase)

    assert frame.shape == (Nphases, wfs.Npix, wfs.Npix)
    assert torch.all(frame >= 0)
    sums = frame.sum(dim=(-2, -1))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
