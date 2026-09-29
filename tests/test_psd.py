"""
Tests for AI4AO.PSD: the PSD / transfer-function helpers shared by
PhaseDataset and PSFModel. (The helpers that already existed in
PhaseDataset.py are also covered, through the PhaseDataset re-exports, in
test_phase_dataset.py.)
"""
import importlib

import numpy as np
import pytest
import torch

from AI4AO import PSD
from AI4AO.PSD import (
    REFERENCE_WAVELENGTH,
    CenterCrop,
    DrawAOParameters,
    GetAliasedAtmospherePSD,
    GetAtmospherePSD,
    GetNoisePSD,
    GetSpatialFrequencies,
    JitterOTF,
    NoiseTransferGain,
    PistonFilter,
    PSDToCovariance,
    PSDToScreen,
    VonKarmanOPDPSD,
    VonKarmanPSD,
    WFSNoiseVariance,
)


@pytest.mark.parametrize("N", [7, 8])
def test_spatial_frequencies_zero_at_center_for_any_parity(N):
    dF, fx, fy = GetSpatialFrequencies(2.0, N)
    assert fx[N // 2, 0] == 0 and fy[0, N // 2] == 0
    assert torch.allclose(fx[1:, 0] - fx[:-1, 0], torch.full((N - 1,), dF))


@pytest.mark.parametrize("N,size", [(8, 4), (8, 5), (9, 4), (9, 5)])
def test_center_crop_keeps_fftshift_center(N, size):
    x = torch.zeros(N, N)
    x[N // 2, N // 2] = 1
    crop = CenterCrop(x, size)
    assert crop.shape == (size, size)
    assert crop[size // 2, size // 2] == 1


def test_phase_dataset_reexports_are_the_psd_functions():
    # AI4AO/__init__ rebinds the name AI4AO.PhaseDataset to the class, so fetch the module explicitly
    phase_dataset_module = importlib.import_module("AI4AO.PhaseDataset")
    for name in ["GetSpatialFrequencies", "GetAtmospherePSD", "GetFittingPSD",
                 "openLoopTransferFunction", "closedLoopTransferFunction", "GetTemporalErrorPSD"]:
        assert getattr(phase_dataset_module, name) is getattr(PSD, name)


def test_generator_psd_is_physical_psd_times_df2_n2():
    N = 16
    dF, fx, fy = GetSpatialFrequencies(3.0, N)
    fsqr = fx**2 + fy**2
    physical = VonKarmanPSD(fsqr, 0.1, 20.0)
    assert torch.allclose(GetAtmospherePSD(fsqr, dF, 0.1, 20.0), physical * dF**2 * N**2)
    assert physical[N // 2, N // 2] == 0
    opd = VonKarmanOPDPSD(fsqr, 0.1, 20.0)
    assert torch.allclose(opd, physical * (REFERENCE_WAVELENGTH / (2 * np.pi)) ** 2)


def test_structure_function_matches_hankel_integral():
    """Guards the PSD normalization: the grid structure function
    2 (B(0) - B(rho)) equals the 1-D Hankel integral of the same von Karman PSD
    over the band the grid samples, 2 int_0^fmax 2 pi f Phi(f) (1 - J0(2 pi f rho)) df.
    L0 = 5 m keeps the power below the grid's lowest frequency negligible."""
    N, extent, r0, L0 = 128, 8.0, 0.2, 5.0
    dF, fx, fy = GetSpatialFrequencies(extent, N)
    cov = PSDToCovariance(VonKarmanPSD(fx**2 + fy**2, r0, L0), dF)
    D_grid = 2 * (cov[0, 0] - cov[:, 0])

    f = torch.logspace(-5, np.log10((N // 2) * dF), 200000, dtype=torch.float64)
    phi = VonKarmanPSD(f**2, r0, L0).double()
    dx = extent / N
    for i in [4, 16, 32]:
        rho = i * dx
        integrand = 4 * np.pi * f * phi * (1 - torch.special.bessel_j0(2 * np.pi * f * rho))
        D_ref = torch.trapezoid(integrand, f).item()
        assert D_grid[i].item() == pytest.approx(D_ref, rel=0.02)


def test_screen_variance_matches_covariance():
    N = 32
    dF, fx, fy = GetSpatialFrequencies(4.0, N)
    psd = VonKarmanOPDPSD(fx**2 + fy**2, 0.1, 10.0).expand(512, N, N)
    screens = PSDToScreen(psd, dF)
    expected = PSDToCovariance(psd[0], dF)[0, 0]
    assert screens.var(dim=0).mean().item() == pytest.approx(expected.item(), rel=0.05)


def test_noise_transfer_gain_limits():
    gains = torch.tensor([0.05, 0.2, 0.4])
    G = NoiseTransferGain(1000.0, gains, torch.ones(3), 1)
    assert torch.isfinite(G).all()
    assert torch.all(G[1:] > G[:-1])
    # Small-gain limit of an integrator: g / (2 - g) (delay matters little)
    assert G[0].item() == pytest.approx(0.05 / 1.95, rel=0.1)


def test_wfs_noise_variance_scaling():
    n_valid = 100.0
    photon_only = WFSNoiseVariance(torch.tensor([1e4, 2e4]), torch.zeros(2), n_valid, 0.25, 1.0)
    assert photon_only[0] / photon_only[1] == pytest.approx(2.0)
    ron_dominated = WFSNoiseVariance(torch.tensor([100.0, 100.0]), torch.tensor([10.0, 20.0]), n_valid, 0.0, 1.0)
    assert ron_dominated[1] / ron_dominated[0] == pytest.approx(4.0)


def test_noise_psd_confined_to_corrected_band():
    corrected = torch.zeros(1, 8, 8)
    corrected[:, 2:6, 2:6] = 0.5
    psd = GetNoisePSD(corrected, torch.tensor(0.3), torch.tensor(0.1), 635e-9, 0.05)
    assert torch.all(psd[:, corrected[0] == 0] == 0)
    expected = 0.5 * 0.3 * 0.1 * (635e-9 / (2 * np.pi)) ** 2 * 0.05**2
    assert psd[0, 3, 3].item() == pytest.approx(expected, rel=1e-5)


def test_aliased_psd_nonnegative_and_decreasing_with_finer_sampling():
    dF, fx, fy = GetSpatialFrequencies(4.0, 32)
    coarse = GetAliasedAtmospherePSD(fx, fy, 0.1, 20.0, 11 / 6, pitch=0.2)
    fine = GetAliasedAtmospherePSD(fx, fy, 0.1, 20.0, 11 / 6, pitch=0.05)
    assert torch.all(coarse >= 0) and torch.all(fine >= 0)
    assert fine.sum() < coarse.sum()


def test_piston_filter_limits():
    fsqr = torch.tensor([0.0, 1e-6, 100.0])
    pf = PistonFilter(fsqr, D=1.0)
    assert pf[0] == 0
    assert pf[1] < 1e-4
    assert pf[2] == pytest.approx(1.0, abs=1e-3)


def test_jitter_otf_gaussian():
    rho = torch.linspace(-1, 1, 5)
    rho_x, rho_y = torch.meshgrid(rho, rho, indexing="ij")
    sigma = torch.tensor([[1e-6, 0.0]])
    otf = JitterOTF(rho_x, rho_y, sigma, torch.zeros(1), 1e-6)
    assert otf[0, 2, 2] == 1  # zero lag
    assert torch.allclose(otf[0, :, 2], torch.exp(-2 * np.pi**2 * rho**2))
    assert torch.allclose(otf[0, 2, :], torch.ones(5))  # no jitter along the second axis


def test_draw_ao_parameters_matches_phase_dataset(phase_dataset, tiny_atmos_params, tiny_loop_params,
                                                  tiny_wfs_params):
    torch.manual_seed(3)
    phase_dataset.DrawRandomParameters()
    wfs_params = tiny_wfs_params()
    torch.manual_seed(3)
    params = DrawAOParameters(
        tiny_atmos_params["Nphases"], tiny_atmos_params["Layers"], tiny_atmos_params["r0"],
        tiny_atmos_params["L0"], tiny_loop_params["levelOfCorrection"], tiny_loop_params["loopGain"],
        tiny_loop_params["loopLeak"], wfs_params["Nphotons"], wfs_params["RON"],
        tiny_loop_params["windSpeedVector"], phase_dataset.height_exp_dist_lambda, phase_dataset.device,
    )
    assert torch.equal(params["loop_gain"], phase_dataset.loopGain.reshape(-1))
    assert torch.equal(params["wind"][0], phase_dataset.windSpeedVector_x.reshape(params["wind"][0].shape))
    assert torch.equal(params["fractional_r0"], phase_dataset.fractionalr0.reshape(params["fractional_r0"].shape))
