"""Tests for AI4AO.FramePreprocess.

FramePreprocess's `wfs` constructor argument is duck-typed -- only
`wfs.pupil_centers` is ever read -- so most tests here use a trivial stub
object instead of a real PyramidWFS/ZernikeWFS.
"""
import numpy as np
import torch

from AI4AO.FramePreprocess import FramePreprocess


class _StubWFS:
    def __init__(self, pupil_centers):
        self.pupil_centers = np.asarray(pupil_centers)


def _make_preprocessor(tiny_wfs_params, device, centers=((10, 10), (20, 20), (30, 15))):
    wfs = _StubWFS(centers)
    return FramePreprocess(tiny_wfs_params(), wfs, device), wfs


def test_get_training_pupils_shape_and_determinism(tiny_wfs_params, device):
    fp, wfs = _make_preprocessor(tiny_wfs_params, device)
    C = wfs.pupil_centers.shape[0]
    images = torch.randn(2, 40, 40, device=device)

    torch.manual_seed(1)
    patches_a = fp.GetTrainingPupils(images, add_position_noise=False, add_size_noise=False)
    torch.manual_seed(999)  # different RNG state -- must not matter, no noise involved
    patches_b = fp.GetTrainingPupils(images, add_position_noise=False, add_size_noise=False)

    assert patches_a.shape == (2, C, fp.Nout, fp.Nout)
    assert torch.equal(patches_a, patches_b)


def test_get_training_pupils_noise_changes_output(tiny_wfs_params, device):
    # Center_noise/Pupil_size_noise default to 0 in the shared fixture (so most
    # tests are exactly deterministic); this test needs actual jitter magnitude.
    params = tiny_wfs_params()
    params["Center_noise"] = 2.0
    params["Pupil_size_noise"] = 0.1
    wfs = _StubWFS([(10, 10), (20, 20), (30, 15)])
    fp = FramePreprocess(params, wfs, device)
    images = torch.randn(2, 40, 40, device=device)

    torch.manual_seed(1)
    patches_a = fp.GetTrainingPupils(images, add_position_noise=True, add_size_noise=True)
    torch.manual_seed(2)
    patches_b = fp.GetTrainingPupils(images, add_position_noise=True, add_size_noise=True)

    assert not torch.equal(patches_a, patches_b)


def test_reference_defaults_before_process_reference(tiny_wfs_params, device):
    fp, _ = _make_preprocessor(tiny_wfs_params, device)
    assert fp.reference == 1.0
    assert fp.normalization == 1.0


def test_process_reference_then_process_frame(tiny_wfs_params, device):
    fp, wfs = _make_preprocessor(tiny_wfs_params, device)
    C = wfs.pupil_centers.shape[0]

    reference_frame = torch.rand(40, 40, device=device)
    fp.ProcessReference(reference_frame)

    assert torch.is_tensor(fp.reference)
    assert fp.reference.shape == (1, C, fp.Nout, fp.Nout)

    frames = torch.rand(3, 40, 40, device=device)
    processed = fp.ProcessFrame(frames, add_pupil_noise=False)
    assert processed.shape == (3, C, fp.Nout, fp.Nout)
    assert torch.isfinite(processed).all()

    # Processing the reference frame itself should come back close to zero
    # once reference-subtracted (before normalization-driven amplification).
    self_processed = fp.ProcessFrame(reference_frame.unsqueeze(0), add_pupil_noise=False)
    assert torch.allclose(self_processed, torch.zeros_like(self_processed), atol=1e-4)


def test_substract_reference_flag_changes_behavior(tiny_wfs_params, device):
    wfsParams_sub = tiny_wfs_params()
    wfsParams_sub["Substract_Reference"] = True
    wfsParams_nosub = tiny_wfs_params()
    wfsParams_nosub["Substract_Reference"] = False

    wfs = _StubWFS([(20, 20)])
    fp_sub = FramePreprocess(wfsParams_sub, wfs, device)
    fp_nosub = FramePreprocess(wfsParams_nosub, wfs, device)

    reference_frame = torch.rand(40, 40, device=device)
    fp_sub.ProcessReference(reference_frame)
    fp_nosub.ProcessReference(reference_frame)

    frames = torch.rand(2, 40, 40, device=device)
    out_sub = fp_sub.ProcessFrame(frames, add_pupil_noise=False)
    out_nosub = fp_nosub.ProcessFrame(frames, add_pupil_noise=False)

    assert not torch.allclose(out_sub, out_nosub)
