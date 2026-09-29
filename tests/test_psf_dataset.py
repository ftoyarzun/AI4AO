"""
Tests for AI4AO.PSFDataset: PSFModel (residual PSD -> OTF -> PSF, differentiable)
and PSFDataset (random parameter sampling).

The physics checks compare against independent references: WFS.GetPSF for
the diffraction/static limit, the Marechal approximation, Tokovinin's von
Karman seeing FWHM, and the ensemble average of GetPSF over PhaseDataset
closed-loop screens drawn from the same PSD.
"""
import numpy as np
import pytest
import torch

from AI4AO import PhaseDataset, PSFDataset, PSFModel
from AI4AO.TorchPropagator import WFS

FIT_PARAMS = ("r0", "L0", "fractional_r0", "wind", "loop_gain", "loop_leak",
              "level_of_correction", "nphotons", "ron", "static_opd", "jitter", "jitter_angle")


def make_params(Nres=16, sampling=3.0, D=1.5, obstruction=0.2, r0=(0.08, 0.2), L0=(10.0, 30.0),
                correction=(0.8, 1.0), layers=(2, 4), wfs_wavelength=635e-9, Nphases=4, psf=None):
    W = {"Nres": Nres, "sampling": sampling, "D": D, "centralObstruction": obstruction,
         "useNoise": False, "Wavelength": wfs_wavelength, "Nphotons": [3.0, 5.0], "RON": [0.5, 3.0],
         "Bin_factor": 1}
    A = {"L0": list(L0), "r0": list(r0), "Nphases": Nphases, "Layers": list(layers),
         "f_slope": 11.0 / 6.0, "Scintillation": False}
    L = {"levelOfCorrection": list(correction), "loopFrequency": 500.0, "delayFrames": 1,
         "loopGain": [0.2, 0.5], "loopLeak": [0.9, 1.0], "windSpeedVector": [1.0, 10.0]}
    Dm = {"Nactuator": 9}
    P = {"Wavelength": 1600e-9, "NCPA": [20e-9, 60e-9], "Jitter": [2.0, 10.0],
         "NoisePhotonFactor": 0.25, "NoiseRONFactor": 1.0}
    if psf is not None:
        P.update(psf)
    return W, A, L, Dm, P


def build(device, terms=None, **kwargs):
    W, A, L, Dm, P = make_params(**kwargs)
    wfs = WFS(W, device)
    extra = {} if terms is None else {"terms": terms}
    return wfs, PSFDataset(wfs, W, A, L, Dm, P, device, **extra)


def fwhm(profile):
    """FWHM in pixels of a centred 1-D profile, with linear edge interpolation."""
    half = profile.max() / 2
    above = (profile >= half).nonzero().squeeze(-1)
    lo, hi = above.min().item(), above.max().item()
    left = lo - 1 + (half - profile[lo - 1]) / (profile[lo] - profile[lo - 1])
    right = hi + (profile[hi] - half) / (profile[hi] - profile[hi + 1])
    return (right - left).item()


# ---------------------------------------------------------------------------
# Construction and batch contract
# ---------------------------------------------------------------------------

def test_batch_shapes(device):
    _, ds = build(device, psf={"fov": 12})
    ds.return_short_exposure = True
    batch = ds[0]
    B, N = 4, ds.model.N
    assert batch["psf"].shape == (B, 36, 36)
    assert batch["psf_short"].shape == (B, 36, 36)
    assert batch["psd"].shape == (B, N, N)
    assert batch["strehl"].shape == (B,)
    assert batch["static_opd"].shape == (B, 16, 16)
    L = batch["fractional_r0"].shape[0]
    assert batch["wind"].shape == (2, L, B)
    for key in ("psf", "psf_short", "psd", "strehl"):
        assert torch.isfinite(batch[key]).all()
        assert (batch[key] >= 0).all()
    assert ((batch["strehl"] > 0) & (batch["strehl"] <= 1)).all()


def test_noise_term_requires_sensitivity_constants(device):
    W, A, L, Dm, P = make_params()
    del P["NoisePhotonFactor"]
    with pytest.raises(ValueError, match="NoisePhotonFactor"):
        PSFModel(WFS(W, device), W, A, L, Dm, P, device)
    PSFModel(WFS(W, device), W, A, L, Dm, P, device, terms=("fitting", "servo"))  # fine without noise


def test_sampling_below_two_raises(device):
    W, A, L, Dm, P = make_params(psf={"sampling": 1.5})
    with pytest.raises(ValueError, match="sampling"):
        PSFModel(WFS(W, device), W, A, L, Dm, P, device)


def test_scintillation_is_ignored_with_warning(device):
    W, A, L, Dm, P = make_params()
    A["Scintillation"] = True
    with pytest.warns(UserWarning, match="scintillation"):
        PSFDataset(WFS(W, device), W, A, L, Dm, P, device)


def test_atmosphere_draws_match_phase_dataset(device):
    W, A, L, Dm, P = make_params()
    torch.manual_seed(7)
    psf_params = PSFDataset(WFS(W, device), W, A, L, Dm, P, device).DrawRandomParameters()
    phase_dataset = PhaseDataset(W, A, L, Dm, device)
    torch.manual_seed(7)
    phase_dataset.DrawRandomParameters()
    phase_params = PSFModel.ParametersFromPhaseDataset(phase_dataset)
    for key, value in phase_params.items():
        assert torch.equal(psf_params[key], value), key


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fov", [None, 10])
def test_diffraction_limit_matches_getpsf(device, fov):
    wfs, ds = build(device, terms=(), psf={"fov": fov})
    params = ds.DrawRandomParameters()
    expected = wfs.GetPSF(torch.zeros(4, 16, 16, device=device), sampling=3.0, fov=fov, wl=1600e-9)
    psf = ds.model.PSF(params)
    assert torch.allclose(psf, expected, atol=1e-5 * expected.max().item())


def test_static_only_matches_getpsf(device):
    wfs, ds = build(device, terms=("static",))
    params = ds.DrawRandomParameters()
    expected = wfs.GetPSF(params["static_opd"], sampling=3.0, wl=1600e-9)
    psf = ds.model.PSF(params)
    assert torch.allclose(psf, expected, atol=1e-5 * expected.max().item())


def test_marechal_strehl(device):
    """High-Strehl regime: Strehl ~ exp(-k^2 sigma^2), and the OTF-sum Strehl
    equals the centre-pixel ratio of the PSFs. Full correction (c = 1) leaves
    only the small-scale fitting residual, where Marechal holds; residual power
    at scales >~ D acts partly like piston and would not cost that much Strehl."""
    wfs, ds = build(device, terms=("fitting",), r0=(0.08, 0.08), L0=(20.0, 20.0), correction=(1.0, 1.0))
    model = ds.model
    params = ds.DrawRandomParameters()
    psd = model.ResidualPSD(params)
    otf = model.OTF(params, psd)
    strehl = model.Strehl(otf)
    phase_variance = model.wavenumber**2 * model.ResidualVariance(params, psd)
    assert torch.allclose(strehl, torch.exp(-phase_variance), rtol=0.03)
    assert (phase_variance > 0.05).all()

    psf = model.PSFFromOTF(otf)
    diffraction = wfs.GetPSF(torch.zeros(1, 16, 16, device=device), sampling=3.0, wl=1600e-9)
    c = model.N // 2
    assert torch.allclose(strehl, psf[:, c, c] / diffraction[0, c, c], rtol=1e-4)


def test_seeing_fwhm_matches_tokovinin(device):
    """Open loop (c = 0), D/r0 = 20: FWHM = 0.98 lambda/r0 sqrt(1 - 2.183 (r0/L0)^0.356)."""
    r0_500, L0, wavelength = 0.1, 20.0, 500e-9
    W, A, L, Dm, P = make_params(Nres=64, sampling=4.0, D=2.0, obstruction=0.0, r0=(r0_500, r0_500),
                                 L0=(L0, L0), correction=(0.0, 0.0), Nphases=1,
                                 psf={"Wavelength": wavelength})
    wfs = WFS(W, device)
    model = PSFModel(wfs, W, A, L, Dm, P, device, terms=("fitting",), psd_integral="infinite")
    params = PSFDataset(wfs, W, A, L, Dm, P, device).DrawRandomParameters()
    psf = model.PSF(params)[0]
    c = model.N // 2
    pixel = wavelength / (model.N * model.dx)
    expected = 0.98 * wavelength / r0_500 * np.sqrt(1 - 2.183 * (r0_500 / L0) ** 0.356)
    assert fwhm(psf[c]) * pixel == pytest.approx(expected, rel=0.03)
    assert fwhm(psf[:, c]) * pixel == pytest.approx(expected, rel=0.03)


def test_infinite_psd_integral_lowers_strehl(device):
    W, A, L, Dm, P = make_params(r0=(0.3, 0.3))
    wfs = WFS(W, device)
    grid = PSFModel(wfs, W, A, L, Dm, P, device, terms=("fitting",))
    infinite = PSFModel(wfs, W, A, L, Dm, P, device, terms=("fitting",), psd_integral="infinite")
    params = PSFDataset(wfs, W, A, L, Dm, P, device).DrawRandomParameters()
    assert (infinite.Strehl(infinite.OTF(params)) < grid.Strehl(grid.OTF(params))).all()


def test_jitter_elongates_along_its_axis(device):
    """Jitter-only PSF: the principal axis of the intensity covariance follows
    jitter_angle, and major - minor variance equals sigma1^2 - sigma2^2 (the
    isotropic diffraction part cancels)."""
    _, ds = build(device, terms=("jitter",), Nres=16, sampling=6.0, obstruction=0.0, Nphases=1)
    model = ds.model
    params = ds.DrawRandomParameters()
    lam_over_d = model.wavelength / model.D
    params["jitter"] = torch.tensor([[3.0, 1.0]], device=device) * lam_over_d
    params["jitter_angle"] = torch.tensor([np.pi / 6], device=device)
    psf = model.PSF(params)[0]
    psf = psf / psf.sum()

    pixel = model.wavelength / (model.N * model.dx)
    coords = (torch.arange(model.N, device=device) - model.N // 2) * pixel
    X, Y = torch.meshgrid(coords, coords, indexing="ij")
    cov = torch.tensor([[(psf * X * X).sum(), (psf * X * Y).sum()],
                        [(psf * X * Y).sum(), (psf * Y * Y).sum()]])
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    major = eigenvectors[:, 1]
    angle = torch.atan2(major[1], major[0]).item() % np.pi
    assert angle == pytest.approx(np.pi / 6, abs=np.radians(2))
    expected = (9.0 - 1.0) * lam_over_d**2
    assert (eigenvalues[1] - eigenvalues[0]).item() == pytest.approx(expected, rel=0.05)


def test_matches_phase_dataset_ensemble(device):
    """The analytic long-exposure PSF equals the ensemble average of GetPSF over
    PhaseDataset closed-loop screens with the same (pinned) conditions. At
    sampling 2 the PSF grid samples exactly PhaseDataset's frequencies
    (dF = 1/(2D)). Two layers with orthogonal winds make the servo-lag halo
    anisotropic, so an axis transpose would show up in the second moments."""
    W, A, L, Dm, _ = make_params(Nres=16, sampling=2.0, obstruction=0.0, r0=(0.1, 0.1), L0=(20.0, 20.0),
                                 correction=(1.0, 1.0), layers=(2, 3), wfs_wavelength=1600e-9, Nphases=64)
    L.update({"loopGain": [0.3, 0.3], "loopLeak": [0.99, 0.99]})

    class Pinned(PhaseDataset):
        def DrawRandomParameters(self):
            super().DrawRandomParameters()
            shape = (2, self.Nphases, 1, 1)
            self.fractionalr0 = torch.tensor([0.7, 0.3], device=device).view(2, 1, 1, 1).expand(shape).clone()
            self.windSpeedVector_x = torch.tensor([15.0, 0.0], device=device).view(2, 1, 1, 1).expand(shape).clone()
            self.windSpeedVector_y = torch.tensor([0.0, 8.0], device=device).view(2, 1, 1, 1).expand(shape).clone()

    phase_dataset = Pinned(W, A, L, Dm, device)
    phase_dataset.generateClosedLoop = True
    wfs = WFS(W, device)
    model = PSFModel(wfs, W, A, L, Dm, {}, device, terms=("fitting", "servo"))

    n_draws = 40
    ensemble = 0
    for _ in range(n_draws):
        ensemble = ensemble + wfs.GetPSF(phase_dataset[0]["opd"], sampling=2.0).mean(0)
    ensemble = ensemble / n_draws
    analytic = model.PSF(PSFModel.ParametersFromPhaseDataset(phase_dataset)).mean(0)

    assert ensemble.max().item() == pytest.approx(analytic.max().item(), rel=0.02)
    assert ((ensemble - analytic).abs().sum() / analytic.sum()).item() < 0.03

    coords = torch.arange(model.N, device=device) - model.N // 2
    X, Y = torch.meshgrid(coords, coords, indexing="ij")
    for psf in (ensemble, analytic):
        psf /= psf.sum()
    for weight in (X**2, Y**2):
        assert (ensemble * weight).sum().item() == pytest.approx((analytic * weight).sum().item(), rel=0.02)
    assert (analytic * X**2).sum() > 1.05 * (analytic * Y**2).sum()  # anisotropy is real


@pytest.mark.slow
def test_short_exposures_average_to_long_exposure(device):
    _, ds = build(device, Nphases=4)
    model = ds.model
    params = ds.DrawRandomParameters()
    psd = model.ResidualPSD(params)
    long_exposure = model.PSF(params)
    n_draws = 1000
    mean = sum(model.ShortExposurePSF(params, psd) for _ in range(n_draws)) / n_draws
    rel_l1 = (mean - long_exposure).abs().sum(dim=(-2, -1)) / long_exposure.sum(dim=(-2, -1))
    assert (rel_l1 < 0.05).all()  # residual speckle noise biases L1 upward
    # Compare the on-axis pixel: the max of a noisy average is biased upward
    c = model.N // 2
    assert torch.allclose(mean[:, c, c], long_exposure[:, c, c], rtol=0.03)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_gradients_reach_every_parameter(device):
    _, ds = build(device)
    params = ds.DrawRandomParameters()
    leaves = {k: v.clone().requires_grad_(True) for k, v in params.items() if k in FIT_PARAMS}
    loss = (ds.model.PSF(leaves) ** 2).sum() + (ds.model.ShortExposurePSF(leaves) ** 2).sum()
    loss.backward()
    for key, leaf in leaves.items():
        assert torch.isfinite(leaf.grad).all(), key
        assert leaf.grad.abs().sum() > 0, key


def test_fitting_recovers_r0(device):
    """A few Adam steps on log r0 through the differentiable model recover the
    r0 that generated a target PSF (the PSF-fitting use case)."""
    _, ds = build(device, terms=("fitting", "servo"), r0=(0.12, 0.12), Nphases=1)
    params = ds.DrawRandomParameters()
    target = ds.model.PSF(params)

    log_r0 = torch.tensor([np.log(0.25)], device=device, requires_grad=True)
    optimizer = torch.optim.Adam([log_r0], lr=0.05)
    for _ in range(150):
        fit = dict(params, r0=torch.exp(log_r0))
        loss = ((ds.model.PSF(fit) - target) ** 2).mean() / (target**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert torch.exp(log_r0).item() == pytest.approx(0.12, rel=0.02)
