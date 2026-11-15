"""Tests for AI4AO.GDReconstructor (gradient-descent inversion of the frozen twin).

The tiny Pyramid twin of conftest.py (Nres 12, 21 actuators) with an 8-mode Zernike
basis is small enough that a 150-iteration fit runs in a fraction of a second on CPU
and recovers a noise-free aberration to well below 1 nm.
"""
import numpy as np
import pytest
import torch

from AI4AO.GDReconstructor import GDReconstructor

N_ITER = 150
LR = {"dm": 30.0, "pixel": 3.0, "dm+pixel": 30.0}
# Zernike modes with zero mean over the (obstructed) WFS pupil, so the injected OPD
# has no piston for the WFS to miss.
PISTON_FREE_MODES = (1, 4, 6)


@pytest.fixture
def twin(pyramid_wfs, deformable_mirror, phase_dataset):
    wfs, dm = pyramid_wfs.eval(), deformable_mirror.eval()
    M2C = dm.MakeZernikeM2C()                                     # (21, 8)
    mode_scale = GDReconstructor.mode_scale_from_dataset(phase_dataset, dm, M2C, n_batches=3)
    return wfs, dm, M2C, mode_scale


def make_gd(twin, opd_mode="dm", **kwargs):
    wfs, dm, M2C, mode_scale = twin
    kwargs.setdefault("n_iter", N_ITER)
    kwargs.setdefault("lr", LR[opd_mode])
    return GDReconstructor(wfs, dm, M2C, opd_mode=opd_mode,
                           mode_scale=None if opd_mode == "pixel" else mode_scale, **kwargs)


def true_coefs(n_modes, amplitude=0.3, device="cpu"):
    """One piston-free mode per frame, alternating sign: (3, n_modes)."""
    c = torch.zeros(len(PISTON_FREE_MODES), n_modes, device=device)
    for i, k in enumerate(PISTON_FREE_MODES):
        c[i, k] = amplitude * (-1) ** i
    return c


def frames_of(twin, c):
    wfs, dm, M2C, _ = twin
    opd = dm.GetDMShape(c @ M2C.T)
    with torch.no_grad():
        return wfs(opd), opd


def pupil_rms_nm(x, pupil):
    v = x[:, pupil.bool()] * 1e9
    return (v - v.mean(dim=-1, keepdim=True)).pow(2).mean(dim=-1).sqrt()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_rejects_unknown_opd_mode(twin):
    with pytest.raises(ValueError, match="opd_mode"):
        GDReconstructor(*twin[:3], opd_mode="zonal")


@pytest.mark.parametrize("opd_mode", ["dm", "dm+pixel"])
def test_dm_modes_need_mode_scale(twin, opd_mode):
    wfs, dm, M2C, _ = twin
    with pytest.raises(ValueError, match="mode_scale"):
        GDReconstructor(wfs, dm, M2C, opd_mode=opd_mode)


def test_mode_scale_length_must_match_basis(twin):
    wfs, dm, M2C, mode_scale = twin
    with pytest.raises(ValueError, match="8 modes"):
        GDReconstructor(wfs, dm, M2C, mode_scale=mode_scale[:5])


def test_pixel_mode_needs_no_mode_scale_and_zonal_default(twin):
    wfs, dm, _, _ = twin
    gd = GDReconstructor(wfs, dm, opd_mode="pixel")
    assert gd.M2C.shape == (int(dm.totalAct), int(dm.totalAct))
    assert torch.equal(gd.M2C, torch.eye(int(dm.totalAct)))


def test_requires_frozen_twin(twin):
    wfs, dm, M2C, _ = twin
    frames, _ = frames_of(twin, true_coefs(M2C.shape[1]))
    gd = make_gd(twin)
    for module in (wfs, dm):
        module.train()
        with pytest.raises(RuntimeError, match="frozen twin"):
            gd(frames)
        module.eval()


# ---------------------------------------------------------------------------
# mode_scale_from_dataset
# ---------------------------------------------------------------------------

def test_mode_scale_from_dataset_shape_and_values(twin, phase_dataset):
    _, dm, M2C, mode_scale = twin
    assert mode_scale.shape == (M2C.shape[1],)
    assert torch.isfinite(mode_scale).all() and (mode_scale > 0).all()

    zonal = GDReconstructor.mode_scale_from_dataset(phase_dataset, dm, n_batches=2)
    assert zonal.shape == (int(dm.totalAct),)


def test_mode_scale_from_dataset_is_reproducible_and_amplitude_free(twin, phase_dataset):
    _, dm, M2C, _ = twin
    torch.manual_seed(3)
    np.random.seed(3)
    a = GDReconstructor.mode_scale_from_dataset(phase_dataset, dm, M2C, n_batches=2)
    torch.manual_seed(3)
    np.random.seed(3)
    b = GDReconstructor.mode_scale_from_dataset(phase_dataset, dm, M2C, n_batches=2)
    assert torch.equal(a, b)

    # each screen is normalized to 1 nm RMS first, so a global OPD gain cancels out
    class Scaled:
        def __getitem__(self, i):
            batch = phase_dataset[i]
            return dict(batch, opd=batch["opd"] * 7.0)

    torch.manual_seed(3)
    np.random.seed(3)
    c = GDReconstructor.mode_scale_from_dataset(Scaled(), dm, M2C, n_batches=2)
    torch.testing.assert_close(a, c, rtol=1e-4, atol=0)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("opd_mode", GDReconstructor.OPD_MODES)
@pytest.mark.parametrize("amplitude", [0.05, 0.3])
def test_recovers_noise_free_aberration(twin, opd_mode, amplitude):
    wfs, dm, M2C, _ = twin
    c_true = true_coefs(M2C.shape[1], amplitude)
    frames, opd = frames_of(twin, c_true)
    gd = make_gd(twin, opd_mode)

    c = gd(frames)

    assert c.shape == c_true.shape and not c.requires_grad
    torch.testing.assert_close(c, c_true, rtol=0, atol=0.01 * amplitude / 0.05)
    assert gd.loss.shape == (3,) and (gd.loss < 1e-5).all()
    # the fitted OPD matches the true one to < 0.5 nm RMS over the WFS pupil
    assert (pupil_rms_nm(gd.opd - opd, wfs.pupil) < 0.5).all()


@pytest.mark.parametrize("opd_mode", GDReconstructor.OPD_MODES)
def test_fitted_parts_are_kept_detached(twin, opd_mode):
    wfs, dm, M2C, _ = twin
    frames, _ = frames_of(twin, true_coefs(M2C.shape[1]))
    gd = make_gd(twin, opd_mode, n_iter=20)
    c = gd(frames)

    Nres = wfs.Nres
    assert gd.opd.shape == (3, Nres, Nres)
    for x in (gd.opd, gd.dm_opd, gd.pixel_opd, gd.loss):
        assert x is None or not x.requires_grad
    if opd_mode == "dm":
        assert gd.pixel_opd is None
        torch.testing.assert_close(gd.opd, gd.dm_opd)
        # returned coefficients are exactly the DM part
        torch.testing.assert_close(dm.GetDMShape(c @ M2C.T), gd.dm_opd)
    elif opd_mode == "pixel":
        assert gd.dm_opd is None
        torch.testing.assert_close(gd.opd, gd.pixel_opd)
        torch.testing.assert_close(c, gd._project_to_dm(gd.pixel_opd))
    else:
        torch.testing.assert_close(gd.opd, gd.dm_opd + gd.pixel_opd)
        # returned coefficients = DM part + projection of the per-pixel part
        dm_coefs = c - gd._project_to_dm(gd.pixel_opd)
        torch.testing.assert_close(dm.GetDMShape(dm_coefs @ M2C.T), gd.dm_opd,
                                   rtol=1e-4, atol=1e-12)


def test_project_to_dm_inverts_piston_free_dm_shapes(twin):
    _, dm, M2C, _ = twin
    gd = make_gd(twin, "pixel")
    c = true_coefs(M2C.shape[1])
    torch.testing.assert_close(gd._project_to_dm(dm.GetDMShape(c @ M2C.T)), c,
                               rtol=0, atol=1e-5)
    # a pure piston projects to nothing: the WFS cannot see it
    piston = torch.full((1, dm.Nres, dm.Nres), 50e-9)
    assert gd._project_to_dm(piston).abs().max() < 1e-6


def test_is_deterministic_and_draws_no_random_numbers(twin):
    wfs, _, M2C, _ = twin
    frames, _ = frames_of(twin, true_coefs(M2C.shape[1]))
    gd = make_gd(twin, "dm+pixel")
    wfs.useNoise = True                       # the model frame must stay noise-free anyway
    state = torch.get_rng_state()
    np_state = np.random.get_state()[1].copy()

    a = gd(frames)
    b = gd(frames)

    assert torch.equal(a, b)
    assert torch.equal(torch.get_rng_state(), state)
    assert np.array_equal(np.random.get_state()[1], np_state)
    assert (gd.loss < 1e-5).all()             # a noisy model frame could not fit this well


def test_restores_use_noise_even_on_error(twin):
    wfs, _, M2C, _ = twin
    frames, _ = frames_of(twin, true_coefs(M2C.shape[1]))
    gd = make_gd(twin, n_iter=5)
    for flag in (True, False):
        wfs.useNoise = flag
        gd(frames)
        assert wfs.useNoise is flag
    wfs.useNoise = True
    with pytest.raises(RuntimeError):
        gd(frames[:, :-1])                    # wrong frame size fails inside the fit
    assert wfs.useNoise is True


def test_runs_under_no_grad_and_leaves_twin_untouched(twin):
    """Trainer._rollout calls reconstructors inside torch.no_grad()."""
    wfs, dm, M2C, _ = twin
    c_true = true_coefs(M2C.shape[1])
    frames, _ = frames_of(twin, c_true)
    frames = frames.clone().requires_grad_(True)
    params = {name: p.detach().clone() for m in (wfs, dm) for name, p in m.named_parameters()}

    with torch.no_grad():
        c = make_gd(twin)(frames)

    torch.testing.assert_close(c, c_true, rtol=0, atol=0.05)
    assert frames.grad is None
    for m in (wfs, dm):
        for name, p in m.named_parameters():
            assert p.grad is None, name
            assert torch.equal(p, params[name]), name


# ---------------------------------------------------------------------------
# In the closed loop
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_closes_the_loop_in_evaluate_many(tiny_wfs_params, tiny_atmos_params, tiny_loop_params,
                                          tiny_dm_params, device):
    from AI4AO.DeformableMirror import DeformableMirror
    from AI4AO.FramePreprocess import FramePreprocess
    from AI4AO.LossFunctions import LogResidualVarianceLoss
    from AI4AO.PhaseDataset import PhaseDataset
    from AI4AO.PyramidWFS import PyramidWFS
    from AI4AO.Trainer import Trainer

    params = tiny_wfs_params()
    wfs = PyramidWFS(params, device).eval()
    dm = DeformableMirror(params, tiny_dm_params, device).eval()
    fp = FramePreprocess(params, wfs, device)
    fp.ProcessReference(wfs.reference_intensity)
    dataset = PhaseDataset(params, tiny_atmos_params, tiny_loop_params, tiny_dm_params, device)
    M2C = dm.MakeZernikeM2C()
    trainer = Trainer(wfs, fp, dm, M2C, dataset=dataset,
                      loss=LogResidualVarianceLoss(dataset.pupil, wfs.wavelength))

    mode_scale = GDReconstructor.mode_scale_from_dataset(dataset, dm, M2C, n_batches=3)
    gd = GDReconstructor(wfs, dm, M2C, mode_scale=mode_scale, n_iter=50)
    report = trainer.evaluate_many({"gd": gd, "gd_again": gd}, seeds=[0], n_steps=10,
                                   references=("linear", "oracle"), psf_fov=8)

    assert np.isfinite(report.nm_rms["gd"]).all()
    # no random numbers drawn, so a repeated rollout is identical
    assert report.nm_rms["gd"] == report.nm_rms["gd_again"]
    # the oracle is the floor
    assert report.nm_rms["oracle"][0] <= report.nm_rms["gd"][0] + 1e-6
