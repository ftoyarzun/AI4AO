"""Differentiability is a first-class, non-negotiable constraint anywhere in
the phase -> WFS/mask -> DM -> loss chain per CLAUDE.md. These tests check
gradient flow explicitly, distinct from "does it run" coverage elsewhere."""
import torch
import torch.nn as nn

from AI4AO.FramePreprocess import FramePreprocess
from AI4AO.LossFunctions import RMSELoss
from AI4AO.ZernikeWFS import ZernikeWFS


def test_gradient_flows_through_pyramid_wfs(pyramid_wfs, device):
    phase = torch.randn(2, pyramid_wfs.Nres, pyramid_wfs.Nres, device=device, requires_grad=True)

    frame = pyramid_wfs(phase)
    # Each frame is flux-normalized to sum to 1 (see TorchPropagator.Propagator),
    # so frame.sum() is a phase-independent constant with an identically-zero
    # gradient -- not a useful differentiability probe. A WFS actually encodes
    # phase as how flux is spatially *redistributed*, so use the per-sample
    # spatial variance instead, which is sensitive to that redistribution.
    torch.var(frame, dim=(-2, -1)).sum().backward()

    assert phase.grad is not None
    assert torch.isfinite(phase.grad).all()
    assert not torch.allclose(phase.grad, torch.zeros_like(phase.grad))


def test_gradient_flows_through_zernike_wfs(zernike_wfs, device):
    phase = torch.randn(2, zernike_wfs.Nres, zernike_wfs.Nres, device=device, requires_grad=True)

    frame = zernike_wfs(phase)
    torch.var(frame, dim=(-2, -1)).sum().backward()

    assert phase.grad is not None
    assert torch.isfinite(phase.grad).all()
    assert not torch.allclose(phase.grad, torch.zeros_like(phase.grad))


def test_gradient_flows_through_deformable_mirror_to_coefs_and_misreg(deformable_mirror, device):
    deformable_mirror.train()
    total_act = int(deformable_mirror.totalAct.item())
    coefs = torch.randn(2, total_act, device=device, requires_grad=True)

    shape = deformable_mirror(coefs)
    shape.sum().backward()

    assert coefs.grad is not None
    assert torch.isfinite(coefs.grad).all()

    for name in ["_rotationAngle", "_sign", "_moffatParameter", "_mechCoupling"]:
        param = getattr(deformable_mirror, name)
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all()


def test_gradient_flows_end_to_end_phase_to_reconstructor_loss(
    pyramid_wfs, deformable_mirror, tiny_wfs_params, device
):
    frame_preprocessor = FramePreprocess(tiny_wfs_params(), pyramid_wfs, device)
    pyramid_wfs.BuildReferenceIntensity()
    frame_preprocessor.ProcessReference(pyramid_wfs.reference_intensity)

    total_act = int(deformable_mirror.totalAct.item())
    Nphases = 2
    phase = torch.randn(Nphases, pyramid_wfs.Nres, pyramid_wfs.Nres, device=device, requires_grad=True)

    wfs_frame = pyramid_wfs(phase)
    preprocessed = frame_preprocessor.ProcessFrame(wfs_frame, add_pupil_noise=False)

    reconstructor = nn.Linear(preprocessed[0].numel(), total_act).to(device)
    z_output = reconstructor(preprocessed.flatten(start_dim=1))
    phase_reconstructed = deformable_mirror(z_output)  # mirrors the Trainer.train chain

    loss_fn = RMSELoss()
    Ze = torch.zeros_like(z_output)
    loss = loss_fn(Ze, z_output, phase, phase_reconstructed, wfs_frame)
    loss.backward()

    assert phase.grad is not None
    assert torch.isfinite(phase.grad).all()
    assert not torch.allclose(phase.grad, torch.zeros_like(phase.grad))
    assert reconstructor.weight.grad is not None
    assert torch.isfinite(reconstructor.weight.grad).all()
