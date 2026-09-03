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
    assert torch.allclose(deformable_mirror.IF, fresh.IF, atol=1e-5)
