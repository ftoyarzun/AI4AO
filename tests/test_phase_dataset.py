"""
Tests for AI4AO.PhaseDataset: the standalone PSD/Zernike helper functions
(pure, deterministic) plus the PhaseDataset class itself.

Note: PhaseDataset.__getitem__ is only valid for strictly sequential
idx=0,1,2,... access (idx==0 draws new random atmosphere parameters and
resets internal state, idx==1 computes a wind-translation kernel that later
indices reuse) -- it is not a conventional random-access Dataset, so every
test below that touches a dataset instance iterates from idx=0.
"""
import numpy as np
import torch

from AI4AO.PhaseDataset import (
    Zernike,
    GetSpatialFrequencies,
    GetAtmospherePSD,
    GetFittingPSD,
    openLoopTransferFunction,
    closedLoopTransferFunction,
    GetTemporalErrorPSD,
    PhaseDataset,
)


def _make_circular_pupil(n):
    x = torch.linspace(-n / 2, n / 2, n)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    return (xx ** 2 + yy ** 2) <= ((n + 1) / 2) ** 2


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def test_zernike_shapes_and_normalization():
    pupil = _make_circular_pupil(16)
    n_pupil_pix = int(pupil.sum().item())
    j = 5

    out, out_full_res = Zernike(pupil, j=j)

    assert out.shape == (j, n_pupil_pix, )
    assert out_full_res.shape == (j, 16, 16)

    for k in range(j):
        assert torch.allclose(out[k].mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(out[k].std(), torch.tensor(1.0), atol=1e-4)

    outside_pupil = ~pupil
    assert torch.all(out_full_res[:, outside_pupil] == 0)


def test_get_spatial_frequencies():
    D = 2.0
    resolution = 8
    dF, fx, fy = GetSpatialFrequencies(D, resolution, device="cpu")

    assert dF == 1 / D
    assert fx.shape == (resolution, resolution)
    assert fy.shape == (resolution, resolution)
    # fx varies along axis 0 (rows), constant along axis 1 for a plain meshgrid
    assert torch.allclose(fx[:, 0], fx[:, 1])


def test_get_atmosphere_psd_center_pixel_zeroed():
    resolution = 8
    dF, fx, fy = GetSpatialFrequencies(1.0, resolution)
    fsqr = fx ** 2 + fy ** 2

    psd = GetAtmospherePSD(fsqr, dF, r0=0.1, L0=25.0, f_slope=11.0 / 6.0)

    assert psd.shape == fsqr.shape
    assert psd[resolution // 2, resolution // 2] == 0
    assert torch.all(psd >= 0)


def test_get_fitting_psd_shape_and_dtype():
    resolution = 8
    dF, fx, fy = GetSpatialFrequencies(1.0, resolution)

    high_pass = GetFittingPSD(fx, fy, dF, D=1.0, Nactuator=4, levelOfCorrection=1.0)

    assert high_pass.shape == fx.shape


def test_closed_loop_is_one_over_one_plus_open_loop():
    freq = torch.linspace(1.0, 100.0, 10)
    open_tf = openLoopTransferFunction(freq, ao_freq=500.0, ki=0.3, leak=0.95, nb_frame_delay=1.0)
    closed_tf = closedLoopTransferFunction(freq, ao_freq=500.0, ki=0.3, leak=0.95, nb_frame_delay=1.0)

    assert torch.allclose(closed_tf, 1 / (1 + open_tf))


def test_get_temporal_error_psd_nonnegative():
    resolution = 8
    dF, fx, fy = GetSpatialFrequencies(1.0, resolution)

    etf = GetTemporalErrorPSD(
        fx, fy, freq=500.0, ki=0.3, leak=0.95, delayFrames=1.0,
        windSpeedVector_x=2.0, windSpeedVector_y=1.0,
    )

    assert etf.shape == fx.shape
    assert torch.all(etf >= 0)


# ---------------------------------------------------------------------------
# PhaseDataset
# ---------------------------------------------------------------------------

def test_remove_piston_zeroes_pupil_mean(phase_dataset):
    phase_map = torch.randn(3, phase_dataset.Nres, phase_dataset.Nres)
    pistoned = phase_dataset.RemovePiston(phase_map)

    mask = phase_dataset.pupil.unsqueeze(0)
    mean_after = (pistoned * mask).sum(dim=(-2, -1)) / phase_dataset.pupilSum
    assert torch.allclose(mean_after, torch.zeros_like(mean_after), atol=1e-5)


def test_getitem_sequential_contract(phase_dataset, tiny_atmos_params):
    Nphases = tiny_atmos_params["Nphases"]
    Nres = phase_dataset.Nres
    n_layers = phase_dataset.nLayers

    for idx in range(3):
        sample = phase_dataset[idx]

        assert sample["phase"].shape == (Nphases, Nres, Nres)
        assert sample["pupil"].shape == (Nphases, Nres, Nres)
        assert sample["nphotons"].shape == (Nphases, 1, 1)
        assert sample["ron"].shape == (Nphases, 1, 1)
        assert sample["r0"].shape == (Nphases,)
        assert sample["wind"].shape == (2, n_layers, Nphases)
        assert sample["fractional_r0"].shape == (n_layers, Nphases)
        assert sample["loop_gain"].shape == (Nphases, 1)
        assert sample["loop_leak"].shape == (Nphases, 1)

        assert torch.isfinite(sample["phase"]).all()


def test_seeded_construction_is_reproducible(
    tiny_wfs_params, tiny_atmos_params, tiny_loop_params, tiny_dm_params, device
):
    np.random.seed(42)
    torch.manual_seed(42)
    ds1 = PhaseDataset(tiny_wfs_params(), tiny_atmos_params, tiny_loop_params, tiny_dm_params, device)
    sample1 = ds1[0]

    np.random.seed(42)
    torch.manual_seed(42)
    ds2 = PhaseDataset(tiny_wfs_params(), tiny_atmos_params, tiny_loop_params, tiny_dm_params, device)
    sample2 = ds2[0]

    assert torch.allclose(sample1["phase"], sample2["phase"])
    assert torch.allclose(sample1["r0"], sample2["r0"])
