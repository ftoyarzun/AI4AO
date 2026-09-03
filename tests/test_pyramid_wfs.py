"""Tests for AI4AO.PyramidWFS."""
import numpy as np
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
    phase = torch.randn(Nphases, pyramid_wfs.Nres, pyramid_wfs.Nres, device=device)

    frame = pyramid_wfs(phase)

    assert frame.shape == (Nphases, pyramid_wfs.Npix, pyramid_wfs.Npix)
    assert torch.all(frame >= 0)
    sums = frame.sum(dim=(-2, -1))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_forward_modulated_mask_shape_and_normalization(tiny_wfs_params, device):
    wfs = PyramidWFS(tiny_wfs_params(modulation=1.0), device)
    Nphases = 2
    phase = torch.randn(Nphases, wfs.Nres, wfs.Nres, device=device)

    frame = wfs(phase)

    assert frame.shape == (Nphases, wfs.Npix, wfs.Npix)
    assert torch.all(frame >= 0)
    sums = frame.sum(dim=(-2, -1))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


# ---------------------------------------------------------------------------
# Calibration chain
# ---------------------------------------------------------------------------

def test_calibration_chain_recovers_reference(pyramid_wfs):
    pyramid_wfs.BuildReferenceIntensity()

    _, modes_full_res = Zernike(pyramid_wfs.pupil, j=3)
    modes = modes_full_res.permute(2, 0, 1)  # (Nres,Nres,j) -> (j,Nres,Nres)

    pyramid_wfs.BuildReconstructionMatrix(modes)

    reconstructed = pyramid_wfs.GetReconstructedPhase(pyramid_wfs.reference_intensity)
    assert torch.allclose(reconstructed, torch.zeros_like(reconstructed), atol=1e-4)
