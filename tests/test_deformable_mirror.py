"""Tests for AI4AO.DeformableMirror."""
import pytest
import torch

from AI4AO.DeformableMirror import DeformableMirror


# ---------------------------------------------------------------------------
# @property scaling round-trips (see CLAUDE.md: all learnable misreg/DM
# quantities are reparameterized so they sit near unit order of magnitude
# for the optimizer).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attr,value", [
    ("rotationAngle", 37.0),
    ("radialScaling", 0.05),
    ("tangentialScaling", -0.03),
    ("anamorphosisAngle", -12.0),
])
def test_scalar_property_round_trip(deformable_mirror, attr, value):
    setattr(deformable_mirror, attr, torch.tensor([value]))
    result = getattr(deformable_mirror, attr)
    assert torch.allclose(result, torch.tensor([value]), atol=1e-4)


def test_sign_property_round_trip(deformable_mirror):
    deformable_mirror.sign = torch.tensor([3.5e-6])
    assert torch.allclose(deformable_mirror.sign, torch.tensor([3.5e-6]), atol=1e-9)


def test_grid_shift_property_round_trip(deformable_mirror):
    value = torch.tensor([[1.5, -2.0]]).unsqueeze(-1).unsqueeze(-1)
    deformable_mirror.grid_shift = value
    assert torch.allclose(deformable_mirror.grid_shift, value, atol=1e-4)


def test_moffat_parameter_round_trip_and_validation(deformable_mirror):
    deformable_mirror.moffatParameter = torch.tensor([3.0])
    assert torch.allclose(deformable_mirror.moffatParameter, torch.tensor([3.0]), atol=1e-4)

    with pytest.raises(ValueError):
        deformable_mirror.moffatParameter = torch.tensor([0.0])
    with pytest.raises(ValueError):
        deformable_mirror.moffatParameter = torch.tensor([-1.0])


def test_mech_coupling_round_trip_and_validation(deformable_mirror):
    deformable_mirror.mechCoupling = torch.tensor([0.4])
    assert torch.allclose(deformable_mirror.mechCoupling, torch.tensor([0.4]), atol=1e-4)

    with pytest.raises(ValueError):
        deformable_mirror.mechCoupling = torch.tensor([0.0])
    with pytest.raises(ValueError):
        deformable_mirror.mechCoupling = torch.tensor([1.0])


def test_apply_and_get_misreg_round_trip(deformable_mirror):
    misreg = {
        "rotationAngle": 15.0,
        "shiftX": 0.01,
        "shiftY": -0.02,
        "radialScaling": 2.0,
        "tangentialScaling": -1.0,
        "anamorphosisAngle": 8.0,
    }
    deformable_mirror.ApplyMisreg(misreg)
    readback, dm_dict = deformable_mirror.GetMisreg()

    assert readback["rotationAngle"] == pytest.approx(misreg["rotationAngle"], abs=1e-3)
    assert readback["shiftX"] == pytest.approx(misreg["shiftX"], abs=1e-3)
    assert readback["shiftY"] == pytest.approx(misreg["shiftY"], abs=1e-3)
    assert readback["radialScaling"] == pytest.approx(misreg["radialScaling"], abs=1e-3)
    assert readback["tangentialScaling"] == pytest.approx(misreg["tangentialScaling"], abs=1e-3)
    assert readback["anamorphosisAngle"] == pytest.approx(misreg["anamorphosisAngle"], abs=1e-3)
    assert dm_dict["FlipLeftRight"] == deformable_mirror.flip_lr
    assert dm_dict["FlipTopBottom"] == deformable_mirror.flip_tb


# ---------------------------------------------------------------------------
# Shapes / actuator grid / IF
# ---------------------------------------------------------------------------

def test_actuator_grid_and_if_shapes(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    Nres = deformable_mirror.Nres

    assert deformable_mirror.actuator_positions.shape == (total_act, 2)
    assert deformable_mirror.IF.shape == (total_act, Nres, Nres)


def test_if_is_zero_mean_over_pupil(deformable_mirror):
    pupil = deformable_mirror.pupil
    if_over_pupil = deformable_mirror.IF[:, pupil]
    assert torch.allclose(
        if_over_pupil.mean(dim=-1), torch.zeros(if_over_pupil.shape[0]), atol=1e-4
    )


def test_get_dm_shape_forward(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    coefs = torch.zeros(2, total_act)
    coefs[0, 0] = 1.0  # single-actuator command, batch 0
    # batch 1 stays all-zero

    shape = deformable_mirror.GetDMShape(coefs)

    assert shape.shape == (2, deformable_mirror.Nres, deformable_mirror.Nres)
    assert torch.allclose(shape[1], torch.zeros_like(shape[1]))
    assert not torch.allclose(shape[0], torch.zeros_like(shape[0]))


# ---------------------------------------------------------------------------
# Save/load round trip
# ---------------------------------------------------------------------------

def test_save_and_load_calibration_round_trip(deformable_mirror, tiny_wfs_params, tiny_dm_params, device, tmp_path):
    deformable_mirror.rotationAngle = torch.tensor([12.0])
    deformable_mirror.sign = torch.tensor([2e-5])
    # Property setters mutate the raw parameters but do NOT themselves rebuild
    # IF (only forward() in training mode, or an explicit call, does) -- so
    # a real comparison of "did save/load preserve the physical DM state"
    # needs IF rebuilt first, same as LoadCalibration does on the other side.
    deformable_mirror.MakeZonalModes()

    path = tmp_path / "dm.pth"
    deformable_mirror.SaveCalibration(str(path))

    fresh = DeformableMirror(tiny_wfs_params(), tiny_dm_params, device)
    fresh.LoadCalibration(str(path))

    misreg_orig, dmdict_orig = deformable_mirror.GetMisreg()
    misreg_loaded, dmdict_loaded = fresh.GetMisreg()

    assert misreg_orig == pytest.approx(misreg_loaded, abs=1e-4)
    # IF is now pure OPD (meters, no wavenumber factor), so its magnitude is
    # set directly by `sign` (~1e-5 m here) -- use a tighter atol than that scale.
    assert torch.allclose(deformable_mirror.IF, fresh.IF, atol=1e-9)


# ---------------------------------------------------------------------------
# Per-actuator calibration (moffatParameter/sign/mechCoupling are always
# length-totalAct vectors; per_actuator_calibration gates only how
# MakeZonalModes uses them -- see Ideas/09-per-actuator-dm-calibration.md).
# ---------------------------------------------------------------------------

def test_scalar_assignment_broadcasts_to_full_vector(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.sign = torch.tensor([4e-6])
    assert deformable_mirror.sign.shape == (total_act,)
    assert torch.allclose(deformable_mirror.sign, torch.full((total_act,), 4e-6), atol=1e-9)


def test_per_actuator_vector_stored_exactly(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    vector = torch.linspace(1.0, 2.0, total_act)
    deformable_mirror.moffatParameter = vector
    assert torch.allclose(deformable_mirror.moffatParameter, vector, atol=1e-4)


def test_wrong_length_vector_raises(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    with pytest.raises(RuntimeError):
        deformable_mirror.sign = torch.ones(total_act + 1)


def test_if_differs_per_actuator_when_flag_true(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.per_actuator_calibration = True
    deformable_mirror.sign = torch.linspace(0.8e-5, 1.2e-5, total_act)
    deformable_mirror.MakeZonalModes()

    peak = deformable_mirror.IF.reshape(total_act, -1).abs().max(dim=-1).values
    assert not torch.allclose(peak, peak.mean().expand(total_act), atol=1e-9)


def test_if_depends_only_on_mean_sign_when_flag_false(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.per_actuator_calibration = False

    vector = torch.linspace(0.8e-5, 1.2e-5, total_act)
    deformable_mirror.sign = vector
    # Setter stores the non-uniform vector regardless of the flag...
    assert not torch.allclose(deformable_mirror.sign, deformable_mirror.sign.mean().expand(total_act))
    deformable_mirror.MakeZonalModes()
    if_a = deformable_mirror.IF.clone()

    # ...but while per_actuator_calibration is False, only the MEAN feeds into
    # IF -- reassigning a different vector with the same mean (here, the same
    # values under a different actuator-to-value assignment) must leave IF
    # unchanged, since which actuator nominally "owns" which raw value is
    # irrelevant to the collapsed, tied value actually used.
    deformable_mirror.sign = torch.flip(vector, dims=[0])
    deformable_mirror.MakeZonalModes()
    if_b = deformable_mirror.IF.clone()

    assert torch.allclose(if_a, if_b, atol=1e-9)


def test_gradient_independent_per_actuator_when_flag_true(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.per_actuator_calibration = True
    deformable_mirror.MakeZonalModes()

    weights = torch.arange(1, total_act + 1, dtype=torch.float32)
    loss = (deformable_mirror.IF.reshape(total_act, -1).pow(2).sum(dim=-1) * weights).sum()
    loss.backward()

    grad = deformable_mirror._sign.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.unique().numel() > 1


def test_gradient_tied_per_actuator_when_flag_false(deformable_mirror):
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.per_actuator_calibration = False
    deformable_mirror.MakeZonalModes()

    weights = torch.arange(1, total_act + 1, dtype=torch.float32)
    loss = (deformable_mirror.IF.reshape(total_act, -1).pow(2).sum(dim=-1) * weights).sum()
    loss.backward()

    grad = deformable_mirror._sign.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.allclose(grad, grad[0].expand(total_act), atol=1e-6)


def test_offset_change_rebroadcasts_mean_into_new_vector(deformable_mirror):
    deformable_mirror.per_actuator_calibration = True
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.moffatParameter = torch.linspace(1.0, 2.0, total_act)
    expected_raw_mean = deformable_mirror._moffatParameter.detach().mean()

    deformable_mirror.offset_to_fit_number_of_actuators = 0.5

    new_total_act = int(deformable_mirror.totalAct.item())
    assert deformable_mirror._moffatParameter.shape == (new_total_act,)
    assert torch.allclose(deformable_mirror._moffatParameter, expected_raw_mean.expand(new_total_act), atol=1e-5)


def test_load_calibration_preserves_per_actuator_vector_and_flag(deformable_mirror, tiny_wfs_params, tiny_dm_params, device, tmp_path):
    deformable_mirror.per_actuator_calibration = True
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.sign = torch.linspace(1e-5, 2e-5, total_act)
    deformable_mirror.MakeZonalModes()

    path = tmp_path / "dm_per_actuator.pth"
    deformable_mirror.SaveCalibration(str(path))

    fresh = DeformableMirror(tiny_wfs_params(), tiny_dm_params, device)
    # per_actuator_calibration is saved as a top-level checkpoint key (not in
    # DMDict) and restored on load -- a fresh object starts False...
    assert fresh.per_actuator_calibration is False
    fresh.LoadCalibration(str(path))
    # ...but picks up the source's True after loading its checkpoint.
    assert fresh.per_actuator_calibration is True

    assert torch.allclose(fresh.sign, deformable_mirror.sign, atol=1e-9)


def test_load_calibration_defaults_flag_false_for_old_checkpoint_without_it(deformable_mirror, tiny_wfs_params, tiny_dm_params, device, tmp_path):
    # A checkpoint saved without the "per_actuator_calibration" key (e.g. from
    # before this key existed) must still load cleanly, defaulting to False.
    misreg, DMDict = deformable_mirror.GetMisreg()
    path = tmp_path / "dm_legacy_checkpoint.pth"
    torch.save({"model": deformable_mirror.state_dict(), "config": DMDict, "misreg": misreg}, path)

    fresh = DeformableMirror(tiny_wfs_params(), tiny_dm_params, device)
    fresh.per_actuator_calibration = True
    fresh.LoadCalibration(str(path))

    assert fresh.per_actuator_calibration is False


def test_load_calibration_with_mismatched_construction_offset(deformable_mirror, tiny_wfs_params, tiny_dm_params, device, tmp_path):
    deformable_mirror.per_actuator_calibration = True
    total_act = int(deformable_mirror.totalAct.item())
    deformable_mirror.sign = torch.linspace(1e-5, 2e-5, total_act)
    deformable_mirror.MakeZonalModes()

    path = tmp_path / "dm_mismatched_offset.pth"
    deformable_mirror.SaveCalibration(str(path))

    # Constructed with a different offset_to_fit_number_of_actuators than what
    # was saved (0.5 vs. the default 0.2), so totalAct differs from the
    # checkpoint's until LoadCalibration restores geometry before load_state_dict.
    fresh = DeformableMirror(tiny_wfs_params(), tiny_dm_params, device, offset_to_fit_number_of_actuators=0.5)
    assert fresh.totalAct != deformable_mirror.totalAct

    fresh.LoadCalibration(str(path))

    assert fresh.totalAct == deformable_mirror.totalAct
    assert torch.allclose(fresh.sign, deformable_mirror.sign, atol=1e-9)
