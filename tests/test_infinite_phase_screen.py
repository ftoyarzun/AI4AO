"""
Tests for AI4AO.InfinitePhaseScreen, the non-repeating multi-scale
alternative to PhaseDataset (design: Ideas/16-infinite-frozen-flow.md).

Index contract: dataset[0] starts a new sequence; later indices must not go
backwards, and a forward jump steps through every skipped frame.
"""
import math

import pytest
import torch

from AI4AO.InfinitePhaseScreen import (CrossfadeSum, CrossfadeWeights, FFTFriendlyOdd, InfinitePhaseScreen,
                                       Zernike)
from AI4AO.PhaseDataset import PhaseDataset
from AI4AO.PSD import VonKarmanOPDPSD


def make_screen(wfs, atmos, loop, dm, device, wind=None, **kwargs):
    """InfinitePhaseScreen, optionally with every layer's wind pinned to `wind` (m/s, (vx, vy)).
    `wind` may be a callable of the dataset, e.g. to express it in pixels per frame."""
    class Pinned(InfinitePhaseScreen):
        def DrawRandomParameters(self):
            super().DrawRandomParameters()
            if wind is not None:
                vx, vy = wind(self) if callable(wind) else wind
                self.windSpeedVector_x = torch.full_like(self.windSpeedVector_x, vx)
                self.windSpeedVector_y = torch.full_like(self.windSpeedVector_y, vy)
    return Pinned(wfs, atmos, loop, dm, device, **kwargs)


def pixels_per_frame(n):
    """Wind along dim -2 moving exactly n pixels per frame."""
    return lambda ds: (n * ds.px * ds.loopFrequency, 0.0)


def run(ds, n_frames):
    """(T, nL, B, H, W) layered OPD of frames 0 .. n_frames - 1."""
    frames = []
    for t in range(n_frames):
        ds[t]
        frames.append(ds.layeredOPD.clone())
    return torch.stack(frames)


def shift_error(frames, lag, pupil):
    """Max |frame(t + lag) shifted back by one row - frame(t)| / rms over the pupil overlap."""
    both = pupil[1:, :] & pupil[:-1, :]
    a = frames[lag:, ..., 1:, :][..., both]
    b = frames[:-lag, ..., :-1, :][..., both]
    return ((a - b).abs().max() / b.std()).item()


@pytest.fixture
def one_layer_atmos(tiny_atmos_params):
    return dict(tiny_atmos_params, Layers=[1, 2], r0=[0.1, 0.1])


# ---------------------------------------------------------------------------
# PhaseDataset API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scintillation", [False, True])
@pytest.mark.parametrize("closed_loop", [False, True])
def test_batch_matches_phase_dataset_contract(tiny_wfs_params, tiny_atmos_params, tiny_loop_params,
                                              tiny_dm_params, device, closed_loop, scintillation):
    wfs_params = dict(tiny_wfs_params(), Nres=13)  # odd Nres, as in test_phase_dataset
    atmos_params = dict(tiny_atmos_params, Scintillation=scintillation)
    B, Nres = atmos_params["Nphases"], 13
    ds = InfinitePhaseScreen(wfs_params, atmos_params, tiny_loop_params, tiny_dm_params, device)
    ds.generateClosedLoop = closed_loop
    for idx in range(3):
        sample = ds[idx]
        n_layers = ds.nLayers
        assert sample["opd"].shape == (B, Nres, Nres)
        assert sample["pupil"].shape == (B, Nres, Nres)
        assert sample["nphotons"].shape == (B, 1, 1)
        assert sample["ron"].shape == (B, 1, 1)
        assert sample["r0"].shape == (B,)
        assert sample["L0"].shape == (B,)
        assert sample["wind"].shape == (2, n_layers, B)
        assert sample["fractional_r0"].shape == (n_layers, B)
        assert sample["loop_gain"].shape == (B, 1)
        assert sample["loop_leak"].shape == (B, 1)
        assert sample["level_of_correction"].shape == (B, 1)
        assert torch.isfinite(sample["opd"]).all()
        assert torch.isfinite(sample["pupil"]).all()
        assert sample["opd"].abs().max() > 0
    # piston removed over the pupil, zero outside, as PhaseDataset
    mean = (sample["opd"] * ds.pupil).sum((-2, -1)) / ds.pupilSum
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5 * sample["opd"].abs().max().item())
    assert (sample["opd"][:, ~ds.pupil.bool()] == 0).all()


def test_seeded_construction_is_reproducible(tiny_wfs_params, tiny_atmos_params, tiny_loop_params,
                                             tiny_dm_params, device):
    samples = []
    for _ in range(2):
        torch.manual_seed(42)
        ds = InfinitePhaseScreen(tiny_wfs_params(), tiny_atmos_params, tiny_loop_params, tiny_dm_params, device)
        samples.append([ds[t]["opd"] for t in range(3)])
    for a, b in zip(*samples):
        assert torch.equal(a, b)


def test_draws_same_conditions_as_phase_dataset(tiny_wfs_params, tiny_atmos_params, tiny_loop_params,
                                                tiny_dm_params, device):
    """Same seed -> same conditions, and BuildAtmospherePSD agrees with PhaseDataset's
    on its torus grid (open and closed loop)."""
    for closed_loop in (False, True):
        out = []
        for cls in (PhaseDataset, InfinitePhaseScreen):
            torch.manual_seed(3)
            ds = cls(tiny_wfs_params(), tiny_atmos_params, tiny_loop_params, tiny_dm_params, device)
            ds.generateClosedLoop = closed_loop
            batch = ds[0]
            out.append((batch, ds.BuildAtmospherePSD()))
        (b_old, psd_old), (b_new, psd_new) = out
        for key in ("r0", "L0", "wind", "fractional_r0", "loop_gain", "nphotons"):
            assert torch.equal(b_old[key], b_new[key])
        for p_old, p_new in zip(psd_old, psd_new):
            assert p_old.shape == p_new.shape
            assert torch.allclose(p_old, p_new, rtol=1e-5, atol=0)


def test_backward_index_raises_and_jump_equals_stepping(tiny_wfs_params, tiny_atmos_params, tiny_loop_params,
                                                        tiny_dm_params, device):
    torch.manual_seed(1)
    stepped = InfinitePhaseScreen(tiny_wfs_params(), tiny_atmos_params, tiny_loop_params, tiny_dm_params, device)
    for t in range(9):
        last = stepped[t]["opd"]

    torch.manual_seed(1)
    jumped = InfinitePhaseScreen(tiny_wfs_params(), tiny_atmos_params, tiny_loop_params, tiny_dm_params, device)
    jumped[0]
    assert torch.equal(jumped[8]["opd"], last)
    assert torch.equal(jumped[8]["opd"], last)  # re-requesting the current frame is allowed
    with pytest.raises(ValueError):
        jumped[3]
    jumped[0]  # idx 0 always starts a new sequence


# ---------------------------------------------------------------------------
# Frozen flow
# ---------------------------------------------------------------------------

def test_exact_frozen_flow_across_redraws(tiny_wfs_params, one_layer_atmos, tiny_loop_params,
                                         tiny_dm_params, device):
    """At exactly 1 px/frame along dim -2, frame t+1 is frame t moved down one row on the
    pupil, including at every frame where a screen of any level was redrawn."""
    ds = make_screen(tiny_wfs_params(), one_layer_atmos, tiny_loop_params, tiny_dm_params, device,
                     wind=pixels_per_frame(1), subharmonicLevels=1)
    frames = run(ds, 300)
    assert all(n > 0 for n in ds.redrawCounts), ds.redrawCounts
    assert shift_error(frames, 1, ds.pupil.bool()) < 1e-4


def test_exact_sub_pixel_frozen_flow(tiny_wfs_params, one_layer_atmos, tiny_loop_params, tiny_dm_params, device):
    """At 0.5 px/frame (oblique-free, along dim -2), frame t+2 is frame t moved one row."""
    ds = make_screen(tiny_wfs_params(), one_layer_atmos, tiny_loop_params, tiny_dm_params, device,
                     wind=pixels_per_frame(0.5), subharmonicLevels=1)
    frames = run(ds, 300)
    assert ds.redrawCounts[0] > 0
    assert shift_error(frames, 2, ds.pupil.bool()) < 1e-4
    # and it really moves by half a pixel: frame t+1 is not a whole-row shift of frame t
    assert shift_error(frames, 1, ds.pupil.bool()) > 1e-2


def test_no_repetition_at_the_torus_period(tiny_wfs_params, one_layer_atmos, tiny_loop_params,
                                           tiny_dm_params, device):
    """With a single (crossfaded) level, the pupil content one torus side later is
    uncorrelated; a plain torus (PhaseDataset) would repeat exactly there."""
    ds = make_screen(tiny_wfs_params(), dict(one_layer_atmos, Nphases=8), tiny_loop_params, tiny_dm_params,
                     device, wind=pixels_per_frame(1), subharmonicLevels=0)
    ds[0]
    period = ds.levels[0]["N"]
    frames = run(ds, 2 * period + 20)[:, 0]  # (T, B, H, W)
    pupil = ds.pupil.bool()
    x = frames[..., pupil]
    x = x - x.mean(-1, keepdim=True)
    x = x / x.norm(dim=-1, keepdim=True)
    corr_period = (x[period:] * x[:-period]).sum(-1).mean().abs().item()
    corr_short = (x[1:] * x[:-1]).sum(-1).mean().item()
    assert corr_short > 0.9
    assert corr_period < 0.25


# ---------------------------------------------------------------------------
# Level geometry and crossfade
# ---------------------------------------------------------------------------

def test_crossfade_partition_of_unity():
    u = torch.linspace(-7, 7, 2001, dtype=torch.float64)
    w = CrossfadeWeights(u)
    assert w.shape == (3, 2001)
    assert torch.allclose((w**2).sum(0), torch.ones_like(u), atol=1e-12)
    assert (w >= 0).all()


def test_crossfade_sum_equals_weighted_sum():
    """The two-gather blend equals the explicit weighted sum, and the zero-weight screen
    contributes nothing at all (not even NaN/inf)."""
    u = torch.cat([torch.linspace(-7, 7, 999), torch.tensor([0.0, 1.0, 2.0, 3.0, -1e-9, 3 - 1e-9])])
    fields = torch.randn(2, 3, len(u))
    explicit = (CrossfadeWeights(u)[None] * fields).sum(1)
    assert torch.allclose(CrossfadeSum(u[None, None], fields, axis=1), explicit, atol=1e-6)
    t = torch.remainder(u, 3.0)
    off = (torch.floor(t).clamp(max=2).long() + 1) % 3                # the screen with zero weight
    poisoned = fields.clone()
    poisoned[:, off, torch.arange(len(u))] = float("inf")
    assert torch.isfinite(CrossfadeSum(u[None, None], poisoned, axis=1)).all()


def test_fft_friendly_odd():
    assert FFTFriendlyOdd(157) == 175
    assert FFTFriendlyOdd(165) == 175
    assert FFTFriendlyOdd(27) == 27
    for n in range(1, 400):
        m = FFTFriendlyOdd(n)
        assert m >= n and m % 2 == 1


def test_levels_tile_the_frequency_plane(tiny_wfs_params, tiny_atmos_params, tiny_loop_params,
                                         tiny_dm_params, device):
    """Every kept cell of level l sits exactly in the hole of level l-1, and the kept
    cell areas add up to the level-0 band minus the last level's DC cell."""
    ds = InfinitePhaseScreen(tiny_wfs_params(), tiny_atmos_params, tiny_loop_params, tiny_dm_params, device)
    ds[0]
    levels = ds.levels
    assert len(levels) > 1
    assert levels[-1]["L"] >= 3 * max(tiny_atmos_params["L0"])
    for a, b in zip(levels[:-1], levels[1:]):
        f = b["f1d"]
        FX, FY = torch.meshgrid(f, f, indexing="ij")
        outer = torch.maximum(FX.abs(), FY.abs())[b["keep"]].max() + 0.5 / b["L"]
        assert outer.item() == pytest.approx((ds.subharmonicHole + 0.5) / a["L"], rel=1e-12)
    area = sum(int(level["keep"].sum()) / level["L"] ** 2 for level in levels)
    expected = (levels[0]["N"] / levels[0]["L"]) ** 2 - 1 / levels[-1]["L"] ** 2
    assert area == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("n", [0, 2])
def test_subharmonic_levels_option(tiny_wfs_params, tiny_atmos_params, tiny_loop_params, tiny_dm_params,
                                   device, n):
    ds = InfinitePhaseScreen(tiny_wfs_params(), tiny_atmos_params, tiny_loop_params, tiny_dm_params, device,
                             subharmonicLevels=n)
    ds[0]
    assert len(ds.levels) == n + 1
    # only the last level keeps its centre (minus DC)
    assert int((~ds.levels[-1]["keep"]).sum()) == 1
    # changing the attribute takes effect at the next sequence
    ds.subharmonicLevels = 1
    ds[0]
    assert len(ds.levels) == 2


def test_level_zero_is_sized_for_the_fastest_layer(tiny_wfs_params, one_layer_atmos, tiny_loop_params,
                                                   tiny_dm_params, device):
    """The zero zone R0 = L0/2 must hold the visible width plus one frame of motion."""
    ds = make_screen(tiny_wfs_params(), one_layer_atmos, tiny_loop_params, tiny_dm_params, device,
                     wind=pixels_per_frame(7.3))
    ds[0]
    assert ds.levels[0]["R"] >= ds.visibleWidth + 7.3 * ds.px
    ds[1]
    assert torch.isfinite(ds.layeredOPD).all()


# ---------------------------------------------------------------------------
# Statistics (exact covariance, no Monte Carlo)
# ---------------------------------------------------------------------------

def test_exact_modal_variances_match_von_karman(tiny_wfs_params, one_layer_atmos, tiny_loop_params,
                                                tiny_dm_params, device):
    """The pupil covariance of the multi-scale stack, including the crossfade blend at
    several positions, gives Zernike variances close to an (almost) infinite von Karman
    screen. A single torus misses about half the tip/tilt."""
    r0, L0 = 0.1, 25.0
    ds = InfinitePhaseScreen(dict(tiny_wfs_params(), Nres=16), dict(one_layer_atmos, Nphases=1, L0=[L0, L0]),
                             tiny_loop_params, tiny_dm_params, device)
    ds[0]
    px, Nres = ds.px, ds.Nres
    pupil = ds.pupil.bool()
    rows, cols = torch.where(pupil)
    dR, dC = rows[:, None] - rows[None, :], cols[:, None] - cols[None, :]
    lags = torch.arange(-Nres + 1, Nres, device=device, dtype=torch.float64) * px
    Z, _ = Zernike(pupil, 20)  # j = 2..21
    Z = Z.double()

    def modal_variance(C):
        return torch.einsum("jp,pq,jq->j", Z, C, Z) / Z.shape[1] ** 2

    def level_covariance(level, amplitude):
        f = level["f1d"]
        FX, FY = torch.meshgrid(f, f, indexing="ij")
        k = level["keep"]
        ex = torch.exp(2j * math.pi * torch.outer(FX[k], lags))
        ey = torch.exp(2j * math.pi * torch.outer(FY[k], lags))
        Cg = torch.einsum("k,ka,kb->ab", amplitude.double()[k].to(torch.complex128) ** 2, ex, ey).real
        return Cg[dR + Nres - 1, dC + Nres - 1]

    # reference: a 2048-pixel torus (128 m), effectively infinite for L0 = 25 m
    N_ref = 2048
    f = torch.fft.fftfreq(N_ref, d=px, dtype=torch.float64, device=device)
    FX, FY = torch.meshgrid(f, f, indexing="ij")
    C_ref = torch.fft.fft2(VonKarmanOPDPSD(FX**2 + FY**2, r0, L0)).real / (N_ref * px) ** 2
    reference = modal_variance(C_ref[dR % N_ref, dC % N_ref])

    covariances = [level_covariance(level, ds.LevelAmplitude(level["index"])[0, 0, 0]) for level in ds.levels]
    unblended = modal_variance(sum(covariances)) / reference
    groups = {"tip/tilt": slice(0, 2), "j4-10": slice(2, 9), "j11-21": slice(9, 20)}
    for name, g in groups.items():
        assert unblended[g].mean().item() == pytest.approx(1.0, abs=0.02), name

    xi = rows.double() * px  # wind along dim -2
    for u0 in torch.linspace(0, 3, 13)[:-1].tolist():
        C = 0
        for level, Cl in zip(ds.levels, covariances):
            w = CrossfadeWeights(xi / level["R"] + u0)
            C = C + torch.einsum("kp,kq->pq", w, w) * Cl
        blended = modal_variance(C) / reference
        for name, g in groups.items():
            assert blended[g].mean().item() == pytest.approx(1.0, abs=0.04), (name, u0)

    # for contrast: PhaseDataset's 2 Nres torus misses about half of the tip/tilt
    N_old = 2 * Nres
    f = (torch.arange(N_old, device=device, dtype=torch.float64) - N_old // 2) / (N_old * px)
    FX, FY = torch.meshgrid(f, f, indexing="ij")
    keep = (FX != 0) | (FY != 0)
    amp = torch.sqrt(VonKarmanOPDPSD(FX**2 + FY**2, r0, L0)) / (N_old * px) * keep
    old_level = {"f1d": f, "keep": keep}
    old = modal_variance(level_covariance(old_level, amp)) / reference
    assert old[:2].mean().item() < 0.6


# ---------------------------------------------------------------------------
# Boiling
# ---------------------------------------------------------------------------

def test_boiling_requires_a_time(tiny_wfs_params, tiny_atmos_params, tiny_loop_params, tiny_dm_params, device):
    with pytest.raises(ValueError):
        InfinitePhaseScreen(tiny_wfs_params(), tiny_atmos_params, tiny_loop_params, tiny_dm_params, device,
                            boiling=True)
    ds = InfinitePhaseScreen(tiny_wfs_params(), tiny_atmos_params, tiny_loop_params, tiny_dm_params, device)
    ds.boiling = True
    with pytest.raises(ValueError):
        ds[0]


def test_boiling_breaks_frozen_flow_but_keeps_the_statistics(tiny_wfs_params, one_layer_atmos, tiny_loop_params,
                                                            tiny_dm_params, device):
    ds = make_screen(tiny_wfs_params(), dict(one_layer_atmos, Nphases=6), tiny_loop_params, tiny_dm_params,
                     device, wind=pixels_per_frame(1), boiling=True, boilingTime=[0.01, 0.05])
    frames = run(ds, 200)
    assert ((ds.boilingTimes >= 0.01) & (ds.boilingTimes <= 0.05)).all()
    assert shift_error(frames, 1, ds.pupil.bool()) > 1e-2
    # the AR(1) update keeps every coefficient's variance, hence the PSD
    for C, amplitude in [(ds.coefficients0, ds.amplitude0), (ds.coarseCoefficients, ds.coarseAmplitude)]:
        amplitude = amplitude.unsqueeze(1).expand_as(C)
        ratio = (C.abs() / amplitude)[amplitude > 0] ** 2
        assert ratio.mean().item() == pytest.approx(2.0, rel=0.05)

    # boiling off: same seed gives an exact translation again
    ds.boiling = False
    frames = run(ds, 50)
    assert shift_error(frames, 1, ds.pupil.bool()) < 1e-4
