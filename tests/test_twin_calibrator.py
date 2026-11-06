"""Tests for AI4AO.TwinCalibrator."""
import numpy as np
import pytest
import torch

from AI4AO.TwinCalibrator import TwinCalibrator


# ---------------------------------------------------------------------------
# Pure NumPy tiling helpers -- no torch, no instrument state needed.
# ---------------------------------------------------------------------------

def test_tile_and_untile_pyramid_frame_round_trip():
    full_shape = (40, 40)
    pupil_size, pupil_separation = 6, 1
    coords = [(10, 10), (10, 28), (28, 28), (28, 10)]  # top-left, bottom-left, bottom-right, top-right

    valid_pix_map = np.zeros(full_shape)
    crop = pupil_size // 2 + pupil_separation
    for (x, y) in coords:
        valid_pix_map[x - crop:x + crop, y - crop:y + crop] = 1

    n_valid = int(valid_pix_map.sum())
    rng = np.random.default_rng(0)
    flat_frames = rng.standard_normal((3, n_valid))

    tiled = TwinCalibrator.tile_pyramid_frame(
        flat_frames, valid_pix_map, coords, full_shape, pupil_size, pupil_separation
    )
    crop_size = pupil_size + 2 * pupil_separation
    assert tiled.shape == (3, 2 * crop_size, 2 * crop_size)

    for i in range(3):
        reconstructed_full = TwinCalibrator.untile_pyramid_image(
            tiled[i], coords, full_shape, pupil_size, pupil_separation
        )
        assert np.allclose(
            reconstructed_full[valid_pix_map != 0], flat_frames[i]
        )


# ---------------------------------------------------------------------------
# save/load round trip (delegates to wfs.SaveCalibration/dm.SaveCalibration)
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(pyramid_wfs, deformable_mirror, device, tmp_path):
    deformable_mirror.rotationAngle = torch.tensor([9.0])
    deformable_mirror.MakeZonalModes()

    calibrator = TwinCalibrator(pyramid_wfs, deformable_mirror, device)
    # save() creates the <data_dir>/<instrument>/ directory itself
    wfs_path, dm_path = calibrator.save("TestInstrument", data_dir=str(tmp_path))

    assert (tmp_path / "TestInstrument" / "TestInstrumentWFS.pth").exists()
    assert (tmp_path / "TestInstrument" / "TestInstrumentDM.pth").exists()

    from AI4AO.PyramidWFS import PyramidWFS
    from AI4AO.DeformableMirror import DeformableMirror

    # Build fresh instances from the same params, then load the saved state.
    fresh_wfs_params = {
        "Nres": pyramid_wfs.Nres, "sampling": pyramid_wfs.sampling, "D": pyramid_wfs.D,
        "centralObstruction": pyramid_wfs.central_obstruction, "useNoise": False,
        "Wavelength": 635e-9, "Modulation": 0.0,
    }
    fresh_wfs = PyramidWFS(fresh_wfs_params, device)
    fresh_dm_params = {
        "Nactuator": deformable_mirror.Nact, "Nmodes": 8, "moffatParam": 2.0,
        "signedAmplitude": 1e-5, "MechCoupling": 0.36,
        "FlipLeftRight": False, "FlipTopBottom": False,
    }
    fresh_dm = DeformableMirror(fresh_wfs_params, fresh_dm_params, device)

    fresh_calibrator = TwinCalibrator(fresh_wfs, fresh_dm, device)
    fresh_calibrator.load("TestInstrument", data_dir=str(tmp_path))

    misreg_orig, _ = deformable_mirror.GetMisreg()
    misreg_loaded, _ = fresh_dm.GetMisreg()
    assert misreg_orig["rotationAngle"] == pytest.approx(misreg_loaded["rotationAngle"], abs=1e-3)


# ---------------------------------------------------------------------------
# Slow end-to-end fit smoke tests. No repo-committed bench data exists, so
# these fit toward a *synthetic* target (the twin's own reference frame /
# interaction matrix at its current, uncalibrated state) purely to exercise
# the optimization loops without raising. live_plot=False is required --
# live_plot=True calls IPython.display, which hangs/fails headless.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_fit_pupil_to_reference_smoke(pyramid_wfs, deformable_mirror, device):
    pyramid_wfs.BuildReferenceIntensity()
    target_frame = pyramid_wfs.reference_intensity.clone().detach()

    calibrator = TwinCalibrator(pyramid_wfs, deformable_mirror, device)
    final_loss = calibrator.fit_pupil_to_reference(
        target_frame, pyramid_wfs.parameters(), lr=1e-3, n_iter=3, live_plot=False
    )

    assert final_loss is not None
    assert np.isfinite(final_loss)


@pytest.mark.slow
def test_fit_dm_and_offsets_smoke(pyramid_wfs, deformable_mirror, device):
    total_act = int(deformable_mirror.totalAct.item())
    M2C = torch.eye(total_act, device=device)
    mode_index = torch.arange(0, min(3, total_act), device=device)

    modes = deformable_mirror(M2C.T)
    pyramid_wfs.BuildInteractionMatrix(modes)
    bench_iMat = pyramid_wfs.iMat.detach().clone()

    calibrator = TwinCalibrator(pyramid_wfs, deformable_mirror, device)
    final_loss, original_positions, transformed_positions = calibrator.fit_dm_and_offsets(
        bench_iMat, M2C, mode_index,
        n_iter=2, lr_dm=1e-3, lr_wfs=1e-4,
        fit_static_offsets=False, batch_size=10, live_plot=False,
    )

    assert final_loss is not None
    assert np.isfinite(final_loss)
    assert original_positions.shape == transformed_positions.shape


@pytest.mark.slow
@pytest.mark.parametrize("static_opd_mode", ["pixel", "dm", "dm+pixel"])
def test_fit_dm_and_offsets_creates_and_fits_static_offsets(pyramid_wfs, deformable_mirror, device, static_opd_mode):
    total_act = int(deformable_mirror.totalAct.item())
    M2C = torch.eye(total_act, device=device)
    mode_index = torch.arange(0, min(3, total_act), device=device)
    modes = deformable_mirror(M2C.T)
    pyramid_wfs.BuildInteractionMatrix(modes)
    bench_iMat = pyramid_wfs.iMat.detach().clone() * 1.1   # something left to fit

    calibrator = TwinCalibrator(pyramid_wfs, deformable_mirror, device)
    assert calibrator.ref_pupil is None and calibrator.static_opd_mode is None

    final_loss, _, _ = calibrator.fit_dm_and_offsets(
        bench_iMat, M2C, mode_index, n_iter=3, lr_offset_start=-2, lr_offset_end=-2,
        fit_static_offsets=True, static_opd_mode=static_opd_mode,
        batch_size=10, live_plot=False,
    )

    assert np.isfinite(final_loss)
    assert calibrator.static_opd_mode == static_opd_mode
    assert calibrator.ref_pupil.shape == pyramid_wfs.pupil.shape
    assert torch.is_tensor(calibrator.ref_opd)
    assert torch.isfinite(calibrator.ref_opd).all()
    # the static maps were actually fitted
    assert not torch.equal(calibrator._log_ref_pupil.detach(), torch.zeros_like(calibrator._log_ref_pupil))


# ---------------------------------------------------------------------------
# Static offsets: representation, positivity, projection, DM command export
# ---------------------------------------------------------------------------

def _pupil_rms(x, pupil):
    v = x[pupil] - x[pupil].mean()
    return v.pow(2).mean().sqrt()


def test_static_offsets_absent_until_requested(pyramid_wfs, deformable_mirror, device):
    calibrator = TwinCalibrator(pyramid_wfs, deformable_mirror, device)
    assert calibrator.ref_pupil is None
    assert calibrator.ref_opd == 0
    with pytest.raises(ValueError):
        calibrator.static_correction_command()
    with pytest.raises(ValueError):
        calibrator.project_static_opd_to_dm()


def test_static_amplitude_is_positive(pyramid_wfs, deformable_mirror, device):
    calibrator = TwinCalibrator(pyramid_wfs, deformable_mirror, device)
    calibrator._prepare_static_offsets("pixel", None)
    assert torch.allclose(calibrator.ref_pupil, torch.ones_like(calibrator.ref_pupil))
    with torch.no_grad():
        calibrator._log_ref_pupil.fill_(-50.0)
    assert (calibrator.ref_pupil > 0).all()


def test_invalid_static_opd_mode_raises(pyramid_wfs, deformable_mirror, device):
    calibrator = TwinCalibrator(pyramid_wfs, deformable_mirror, device)
    with pytest.raises(ValueError):
        calibrator._prepare_static_offsets("zonal", None)


def _set_pixel_opd(calibrator, opd):
    calibrator._prepare_static_offsets("pixel", None)
    with torch.no_grad():
        calibrator._ref_opd_pixel.copy_(opd / 1e-6)


def test_project_dm_shaped_opd_recovers_command(pyramid_wfs, deformable_mirror, device):
    """A static OPD the DM can produce exactly is projected with ~no residual,
    and the correction command cancels it."""
    dm = deformable_mirror
    dm.eval()
    pupil = pyramid_wfs.pupil.bool()
    command = 0.01 * torch.randn(int(dm.totalAct), device=device)
    target = dm.GetDMShape(command[None])[0]

    calibrator = TwinCalibrator(pyramid_wfs, dm, device)
    _set_pixel_opd(calibrator, target)
    uncaptured = calibrator.project_static_opd_to_dm(keep_residual=False)

    assert calibrator.static_opd_mode == "dm"
    assert uncaptured < 1e-3 * _pupil_rms(target, pupil).item()
    with torch.no_grad():
        corrected = target + dm.GetDMShape(calibrator.static_correction_command()[None])[0]
    assert _pupil_rms(corrected, pupil) < 1e-3 * _pupil_rms(target, pupil)


def test_mode_conversions_preserve_total_static_opd(pyramid_wfs, deformable_mirror, device):
    dm = deformable_mirror
    dm.eval()
    pupil = pyramid_wfs.pupil.bool()
    target = 50e-9 * torch.randn(*pupil.shape, device=device)

    calibrator = TwinCalibrator(pyramid_wfs, dm, device)
    _set_pixel_opd(calibrator, target)

    uncaptured = calibrator.project_static_opd_to_dm(keep_residual=True)
    assert calibrator.static_opd_mode == "dm+pixel"
    assert uncaptured > 0      # white noise is not DM-shaped
    with torch.no_grad():
        assert torch.allclose(calibrator.ref_opd[pupil], target[pupil], atol=1e-12)
        dm_opd, pixel_opd = calibrator.static_opd_parts()
        assert torch.allclose((dm_opd + pixel_opd)[pupil], target[pupil], atol=1e-12)

    # back to pixel keeps everything, the DM part folded into the per-pixel map
    calibrator._prepare_static_offsets("pixel", None)
    assert calibrator._static_coefs is None
    with torch.no_grad():
        assert torch.allclose(calibrator.ref_opd[pupil], target[pupil], atol=1e-12)


def test_modal_static_basis(pyramid_wfs, deformable_mirror, device):
    dm = deformable_mirror
    dm.eval()
    pupil = pyramid_wfs.pupil.bool()
    total_act = int(dm.totalAct)
    static_M2C = torch.randn(total_act, 4, device=device)
    target = dm.GetDMShape((static_M2C @ torch.tensor([0.01, -0.02, 0.005, 0.0], device=device))[None])[0]

    calibrator = TwinCalibrator(pyramid_wfs, dm, device)
    _set_pixel_opd(calibrator, target)
    calibrator.project_static_opd_to_dm(M2C=static_M2C, keep_residual=False)

    assert calibrator._static_coefs.shape == (4,)
    command = calibrator.static_dm_command()
    assert command.shape == (total_act,)
    with torch.no_grad():
        assert _pupil_rms(dm.GetDMShape(command[None])[0] - target, pupil) < 1e-3 * _pupil_rms(target, pupil)

    # the same mode with a different basis re-projects instead of reusing the coefficients
    calibrator._prepare_static_offsets("dm", None)
    assert calibrator._static_M2C is None
    assert calibrator._static_coefs.shape == (total_act,)


def test_static_correction_command_stroke_limit(pyramid_wfs, deformable_mirror, device):
    dm = deformable_mirror
    dm.eval()
    calibrator = TwinCalibrator(pyramid_wfs, dm, device)
    calibrator._prepare_static_offsets("dm", None)
    with torch.no_grad():
        calibrator._static_coefs.copy_(torch.linspace(-1, 1, calibrator._static_coefs.numel()) / calibrator._static_coef_scale)
    command = calibrator.static_correction_command(stroke_limit=0.5)
    assert command.abs().max() <= 0.5
    assert not command.requires_grad
    assert torch.allclose(calibrator.static_correction_command(), -calibrator.static_dm_command().detach())


def test_dm_static_opd_is_differentiable(pyramid_wfs, deformable_mirror, device):
    """In "dm" mode the WFS frame depends on the static command coefficients,
    the amplitude, and (through the influence functions) the DM parameters."""
    dm = deformable_mirror
    dm.train()
    calibrator = TwinCalibrator(pyramid_wfs, dm, device)
    calibrator._prepare_static_offsets("dm", None)
    with torch.no_grad():
        calibrator._static_coefs.normal_(0, 0.05)

    dm.MakeZonalModes()
    pyramid_wfs.BuildReferenceIntensity(calibrator.ref_opd, calibrator.ref_pupil)
    pyramid_wfs.reference_intensity.pow(2).sum().backward()

    assert calibrator._static_coefs.grad is not None and calibrator._static_coefs.grad.abs().sum() > 0
    assert calibrator._log_ref_pupil.grad is not None and calibrator._log_ref_pupil.grad.abs().sum() > 0
    assert dm._sign.grad is not None and dm._sign.grad.abs().sum() > 0


@pytest.mark.slow
@pytest.mark.parametrize("static_opd_mode", ["pixel", "dm", "dm+pixel"])
def test_fit_static_offsets_to_reference_reduces_loss(pyramid_wfs, deformable_mirror, device, static_opd_mode):
    """Fitting only the static maps to a frame generated with known offsets
    lowers the loss, without touching the WFS/DM parameters."""
    dm = deformable_mirror
    dm.eval()
    pupil = pyramid_wfs.pupil.bool()
    true_opd = dm.GetDMShape((0.005 * torch.randn(int(dm.totalAct), device=device))[None])[0]
    true_amplitude = 1 + 0.1 * torch.randn(*pupil.shape, device=device).abs()
    with torch.no_grad():
        pyramid_wfs.BuildReferenceIntensity(true_opd, true_amplitude)
    reference = pyramid_wfs.reference_intensity.clone()
    wfs_before = [p.detach().clone() for p in pyramid_wfs.parameters()]

    calibrator = TwinCalibrator(pyramid_wfs, dm, device)
    first = calibrator.fit_static_offsets_to_reference(reference, static_opd_mode, n_iter=1, live_plot=False)
    calibrator.reset_static_offsets()
    last = calibrator.fit_static_offsets_to_reference(reference, static_opd_mode, n_iter=40, live_plot=False)

    assert np.isfinite(last) and last < first
    assert calibrator.static_opd_mode == static_opd_mode
    assert not pyramid_wfs.reference_intensity.requires_grad
    for before, p in zip(wfs_before, pyramid_wfs.parameters()):
        assert torch.equal(before, p.detach())


def test_fit_does_not_decay_parameters_without_gradient_signal(pyramid_wfs, deformable_mirror, device):
    """With a loss that carries no gradient, fitted physical parameters must
    not move: AdamW's default weight_decay=0.01 would otherwise pull them
    toward zero (e.g. maskShifts, initialized at 1)."""
    calibrator = TwinCalibrator(pyramid_wfs, deformable_mirror, device)
    before = pyramid_wfs.maskShifts.detach().clone()

    calibrator.fit_pupil_to_reference(
        torch.zeros(pyramid_wfs.Npix, pyramid_wfs.Npix, device=device), [pyramid_wfs.maskShifts],
        lr=0.1, n_iter=5, live_plot=False,
        loss_fn=lambda ref, dig: 0.0 * dig.sum(),
    )

    assert torch.equal(pyramid_wfs.maskShifts.detach(), before)


# ---------------------------------------------------------------------------
# PSF path: psf_model, fit_dm_to_psfs, rough search, estimate_ncpa
# ---------------------------------------------------------------------------

PSF_WAVELENGTH = 700e-9


def _psf_twin(device, Nres=24, Nact=7):
    """Bare WFS (only GetPSF is used) and a DM in eval mode, plus a Zernike M2C."""
    from AI4AO.TorchPropagator import WFS
    from AI4AO.DeformableMirror import DeformableMirror
    W = {"Nres": Nres, "sampling": 3.0, "D": 1.0, "centralObstruction": 0.3,
         "useNoise": False, "Wavelength": PSF_WAVELENGTH}
    D = {"Nactuator": Nact, "Nmodes": None, "moffatParam": 2.0, "signedAmplitude": -5e-6,
         "MechCoupling": 0.36, "FlipLeftRight": False, "FlipTopBottom": False}
    wfs = WFS(W, device)
    dm = DeformableMirror(W, D, device)
    dm.eval()
    return wfs, dm, dm.MakeZernikeM2C(nModes=12)


def _probes(M2C, columns, amplitude=1.0):
    """Zero command plus +-amplitude pokes of the given M2C columns, (n, totalAct)."""
    rows = [torch.zeros(M2C.shape[0], device=M2C.device)]
    for col in columns:
        rows += [amplitude * M2C[:, col], -amplitude * M2C[:, col]]
    return torch.stack(rows)


def _camera_psfs(wfs, dm, commands, sampling, size, static_opd=0.0):
    with torch.no_grad():
        psf = wfs.GetPSF(dm(commands) + static_opd, sampling=sampling, wl=PSF_WAVELENGTH, fov_pix=size)
    return psf / psf.sum(dim=(-2, -1), keepdim=True)


def _pupil_tiptilt(calibrator, opd):
    """The least-squares piston/tip/tilt part of opd over the WFS pupil."""
    return opd - calibrator._remove_tiptilt(opd)


def test_psf_model_normalized_and_blind_to_static_tiptilt(device):
    wfs, dm, M2C = _psf_twin(device)
    commands = _probes(M2C, [2, 3])
    psfs = _camera_psfs(wfs, dm, commands, 3.0, 48)

    calibrator = TwinCalibrator(wfs, dm, device)
    calibrator._prepare_static_offsets("pixel", None)
    calibrator._init_psf_nuisances(commands, psfs, 3.0, PSF_WAVELENGTH)
    ncpa = 0.05 * torch.randn(24, 24, device=device)               # microns
    ramps = calibrator._tiptilt_basis()
    with torch.no_grad():
        calibrator._ref_opd_pixel.copy_(ncpa)
        model = calibrator.psf_model(commands, 3.0, PSF_WAVELENGTH)
        calibrator._ref_opd_pixel.copy_(ncpa + 0.02 * ramps[1] - 0.03 * ramps[2])
        model_tilted = calibrator.psf_model(commands, 3.0, PSF_WAVELENGTH)

    assert torch.allclose(model.sum(dim=(-2, -1)), torch.ones(len(commands), device=device), atol=1e-5)
    assert torch.allclose(model, model_tilted, atol=1e-6)


def test_psf_model_tiptilt_moves_psf_by_lambda_over_d(device):
    """One unit of per-frame tip-tilt moves the PSF by `sampling` pixels:
    x along columns, y along rows."""
    wfs, dm, _ = _psf_twin(device)
    calibrator = TwinCalibrator(wfs, dm, device)
    calibrator._psf_tiptilt = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, -2.0]], device=device))
    calibrator._psf_window = 48
    with torch.no_grad():
        psf = calibrator.psf_model(torch.zeros(2, int(dm.totalAct), device=device), 3.0, PSF_WAVELENGTH)
    assert divmod(int(torch.argmax(psf[0])), 48) == (24, 24 + 3)
    assert divmod(int(torch.argmax(psf[1])), 48) == (24 - 6, 24)


def test_psf_fit_gradients_reach_every_fitted_parameter(device):
    wfs, dm, M2C = _psf_twin(device)
    commands = _probes(M2C, [2, 3])
    static = dm.GetDMShape(0.3 * M2C[:, 5][None])
    psfs = _camera_psfs(wfs, dm, commands, 3.0, 48, static_opd=static)

    calibrator = TwinCalibrator(wfs, dm, device)
    calibrator._prepare_static_offsets("dm", M2C[:, :6])
    calibrator._init_psf_nuisances(commands, psfs, 3.0, PSF_WAVELENGTH)
    dm.train()
    calibrator._psf_loss(psfs, calibrator._psf_model(commands, 3.0, PSF_WAVELENGTH)).backward()

    for p in (dm._rotationAngle, dm._grid_shift, dm._radialScaling,
              calibrator._static_coefs, calibrator._psf_tiptilt):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_fit_dm_to_psfs_freezes_intrinsics_and_amplitude(device):
    wfs, dm, M2C = _psf_twin(device)
    commands = _probes(M2C, [2, 3, 4])
    psfs = _camera_psfs(wfs, dm, commands, 3.0, 48)
    dm.train()
    names = TwinCalibrator.DM_INTRINSIC_PARAMETERS
    intrinsic = {name: getattr(dm, name).detach().clone() for name in names}
    flags = {name: getattr(dm, name).requires_grad for name in names}
    amplitude = 0.9 + 0.1 * torch.rand(24, 24, device=device)

    calibrator = TwinCalibrator(wfs, dm, device)
    loss, before, after = calibrator.fit_dm_to_psfs(
        commands, psfs, 3.0, PSF_WAVELENGTH, static_M2C=M2C[:, :6], ref_pupil=amplitude,
        n_iter=5, lr_geometry=1e-2, live_plot=False,
    )

    assert np.isfinite(loss) and before.shape == after.shape
    assert not torch.equal(before, after)                           # the geometry was fitted
    for name in names:
        assert torch.equal(getattr(dm, name).detach(), intrinsic[name])
        assert getattr(dm, name).requires_grad == flags[name]
    assert torch.allclose(calibrator.ref_pupil, amplitude)
    assert calibrator.psf_tiptilt.shape == (len(commands), 2)


@pytest.mark.parametrize("static_opd_mode", ["pixel", "dm", "dm+pixel"])
def test_fit_dm_to_psfs_removes_tiptilt_from_stored_ncpa(device, static_opd_mode):
    wfs, dm, M2C = _psf_twin(device)
    commands = _probes(M2C, [2, 3])
    psfs = _camera_psfs(wfs, dm, commands, 3.0, 48)

    calibrator = TwinCalibrator(wfs, dm, device)
    calibrator._prepare_static_offsets("pixel", None)
    with torch.no_grad():   # a stored NCPA (microns) with a large tilt in it
        calibrator._ref_opd_pixel.copy_(0.2 * calibrator._tiptilt_basis()[1] / 24
                                        + dm.GetDMShape(0.2 * M2C[:, 4][None])[0] / 1e-6)
    tiptilt_before = _pupil_tiptilt(calibrator, calibrator.ref_opd).detach()

    calibrator.estimate_ncpa(commands, psfs, 3.0, PSF_WAVELENGTH, static_opd_mode=static_opd_mode,
                             n_iter=2, live_plot=False)

    pupil = wfs.pupil.bool()
    tiptilt_after = _pupil_tiptilt(calibrator, calibrator.ref_opd).detach()
    ratio = tiptilt_after[pupil].norm() / tiptilt_before[pupil].norm()
    # "dm": only up to what the DM basis can express (a 7x7 DM gives ~1e-2 of a pure tilt back)
    assert ratio < (5e-2 if static_opd_mode == "dm" else 1e-4)


def test_rough_calibrate_dm_to_psfs_finds_flip_and_rotation(device):
    import copy
    wfs, dm, M2C = _psf_twin(device)
    dm_true = copy.deepcopy(dm)
    dm_true.flip_lr = True
    dm_true.rotationAngle = 130.0
    commands = _probes(M2C, [3, 4, 5, 6])          # astigmatism and coma: not rotation symmetric
    psfs = _camera_psfs(wfs, dm_true, commands, 3.0, 48)

    calibrator = TwinCalibrator(wfs, dm, device)
    calibrator.rough_calibrate_dm_to_psfs(commands, psfs, 3.0, PSF_WAVELENGTH, rotation_step=10.0)

    assert dm.flip_lr and not dm.flip_tb
    assert abs(float(dm.rotationAngle) - 130.0) <= 5.0


@pytest.mark.parametrize("train_mode", [True, False])
def test_estimate_ncpa_leaves_dm_unchanged(device, train_mode):
    wfs, dm, M2C = _psf_twin(device)
    dm.train(train_mode)
    if train_mode:
        dm._grid_shift.requires_grad_(False)       # a choice of the caller that must survive
    commands = _probes(M2C, [2, 3])
    static = dm.GetDMShape(0.3 * M2C[:, 5][None]).detach()
    psfs = _camera_psfs(wfs, dm, commands, 3.0, 48, static_opd=static)
    params = {n: p.detach().clone() for n, p in dm.named_parameters()}
    flags = {n: p.requires_grad for n, p in dm.named_parameters()}
    IF = dm.IF.detach().clone()

    calibrator = TwinCalibrator(wfs, dm, device)
    loss = calibrator.estimate_ncpa(commands, psfs, 3.0, PSF_WAVELENGTH,
                                    static_M2C=M2C[:, :6], n_iter=3, live_plot=False)

    assert dm.training == train_mode
    for n, p in dm.named_parameters():
        assert torch.equal(p.detach(), params[n]) and p.requires_grad == flags[n]
    assert torch.equal(dm.IF.detach(), IF)
    assert np.isfinite(loss) and torch.is_tensor(calibrator.ref_opd)
    assert calibrator.static_correction_command().shape == (int(dm.totalAct),)


@pytest.mark.slow
def test_psf_calibration_recovers_geometry_and_ncpa(device):
    """End-to-end synthetic check of the science-path calibration: a DM seen
    through a flipped, rotated, anamorphically scaled and shifted relay, an
    NCPA, a random tip-tilt per frame and photon noise."""
    import copy
    wfs, dm, M2C = _psf_twin(device, Nres=32, Nact=9)
    sampling, size = 3.0, 72

    dm_true = copy.deepcopy(dm)
    dm_true.flip_lr = True
    dm_true.rotationAngle = 37.0
    dm_true.radialScaling = 0.02
    dm_true.tangentialScaling = -0.015
    dm_true.grid_shift = torch.tensor([[0.3, -0.2]], device=device)[..., None, None]

    ncpa_coefs = torch.zeros(M2C.shape[1], device=device)
    ncpa_coefs[2:8] = 0.15 * torch.randn(6, device=device)
    ncpa_command = M2C @ ncpa_coefs
    low = _probes(M2C, range(2, 8))
    commands = torch.cat([low, _probes(M2C, range(8, 12))[1:]])
    tiptilt = 0.5 * torch.randn(len(commands), 2, device=device)

    calibrator = TwinCalibrator(wfs, copy.deepcopy(dm), device)
    with torch.no_grad():
        ramps = calibrator._tiptilt_basis()[1:]
        opd = dm_true(commands + ncpa_command) + PSF_WAVELENGTH / 32 * torch.einsum('nk,kwh->nwh', tiptilt, ramps)
        psf = wfs.GetPSF(opd, sampling=sampling, wl=PSF_WAVELENGTH, fov_pix=size)
        psfs = torch.poisson(psf / psf.sum(dim=(-2, -1), keepdim=True) * 1e8 + 2.0) - 2.0   # background-subtracted
        psfs = psfs / psfs.sum(dim=(-2, -1), keepdim=True)

    n_low = len(low)
    calibrator.rough_calibrate_dm_to_psfs(commands[:n_low], psfs[:n_low], sampling, PSF_WAVELENGTH)
    calibrator.fit_dm_to_psfs(commands[:n_low], psfs[:n_low], sampling, PSF_WAVELENGTH,
                              static_M2C=M2C[:, :8], live_plot=False)
    calibrator.fit_dm_to_psfs(commands, psfs, sampling, PSF_WAVELENGTH, static_M2C=M2C[:, :8], live_plot=False)
    calibrator.estimate_ncpa(commands, psfs, sampling, PSF_WAVELENGTH, static_M2C=M2C[:, :8], live_plot=False)

    fitted, true = calibrator.dm.GetMisreg()[0], dm_true.GetMisreg()[0]
    assert calibrator.dm.flip_lr
    assert abs(fitted["rotationAngle"] - 37.0) < 0.5
    for key in ("radialScaling", "tangentialScaling"):      # in percent
        assert abs(fitted[key] - true[key]) < 0.5
    for key in ("shiftX", "shiftY"):
        assert abs(fitted[key] - true[key]) < 0.01

    pupil = wfs.pupil.bool()
    with torch.no_grad():
        true_ncpa = calibrator._remove_tiptilt(dm_true.GetDMShape(ncpa_command[None])[0])
        residual = (true_ncpa - calibrator.ref_opd)[pupil]
    assert residual.std() < 0.1 * true_ncpa[pupil].std()
