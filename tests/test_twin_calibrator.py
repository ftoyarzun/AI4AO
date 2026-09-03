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

    (tmp_path / "TestInstrument").mkdir()  # save() does not create its own output directory

    calibrator = TwinCalibrator(pyramid_wfs, deformable_mirror, device)
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
