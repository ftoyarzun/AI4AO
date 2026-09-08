"""Differentiability is a first-class, non-negotiable constraint anywhere in
the OPD -> WFS/mask -> DM -> loss chain per CLAUDE.md. These tests check
gradient flow explicitly, distinct from "does it run" coverage elsewhere."""
import torch
import torch.nn as nn

from AI4AO.FramePreprocess import FramePreprocess
from AI4AO.LossFunctions import RMSELoss
from AI4AO.ZernikeWFS import ZernikeWFS


def test_gradient_flows_through_pyramid_wfs(pyramid_wfs, device):
    # Physically sane OPD magnitude (~tens of nm) rather than O(1) radians,
    # since the WFS now expects meters and converts to phase internally.
    opd = (1e-7 * torch.randn(2, pyramid_wfs.Nres, pyramid_wfs.Nres, device=device)).requires_grad_()

    frame = pyramid_wfs(opd)
    # Each frame is flux-normalized to sum to 1 (see TorchPropagator.Propagator),
    # so frame.sum() is an OPD-independent constant with an identically-zero
    # gradient -- not a useful differentiability probe. A WFS actually encodes
    # OPD as how flux is spatially *redistributed*, so use the per-sample
    # spatial variance instead, which is sensitive to that redistribution.
    torch.var(frame, dim=(-2, -1)).sum().backward()

    assert opd.grad is not None
    assert torch.isfinite(opd.grad).all()
    assert not torch.allclose(opd.grad, torch.zeros_like(opd.grad))


def test_gradient_flows_through_zernike_wfs(zernike_wfs, device):
    opd = (1e-7 * torch.randn(2, zernike_wfs.Nres, zernike_wfs.Nres, device=device)).requires_grad_()

    frame = zernike_wfs(opd)
    torch.var(frame, dim=(-2, -1)).sum().backward()

    assert opd.grad is not None
    assert torch.isfinite(opd.grad).all()
    assert not torch.allclose(opd.grad, torch.zeros_like(opd.grad))


def test_gradient_flows_through_opd_to_phase_conversion(pyramid_wfs, device):
    """Exercises the new WFS-internal OPD->phase conversion (opd * wavenumber)
    as its own gradient-flow path, distinct from the end-to-end WFS checks above.
    wavenumber is always a 4-D (1, Nwavelength, 1, 1) tensor (never a bare
    Python number), so the expected gradient is built by broadcasting rather
    than torch.full_like -- when opd's own wavelength axis is a broadcast
    singleton (as here), the gradient at each position is the *sum* over
    wavenumber's wavelength axis, not a single shared constant."""
    opd = (1e-7 * torch.randn(2, 1, pyramid_wfs.Nres, pyramid_wfs.Nres, device=device)).requires_grad_()

    phase = pyramid_wfs.wavenumber * opd
    phase.sum().backward()

    assert opd.grad is not None
    assert torch.isfinite(opd.grad).all()
    expected = pyramid_wfs.wavenumber.sum(dim=1, keepdim=True).expand_as(opd.grad)
    assert torch.allclose(opd.grad, expected)


def test_gradient_flows_through_multi_wavelength_propagation(pyramid_wfs, device):
    """Exercises gradient flow through the new batched-wavelength path: a
    wavelength tensor with requires_grad=True must still produce finite
    gradients through wfs(opd), since self.wavenumber = 2*pi/wavelength feeds
    directly into the phase computed inside Propagator."""
    wavelength = torch.tensor([635e-9, 750e-9], device=device).requires_grad_()
    pyramid_wfs.wavelength = wavelength

    opd = (1e-7 * torch.randn(2, pyramid_wfs.Nres, pyramid_wfs.Nres, device=device)).requires_grad_()

    frame = pyramid_wfs(opd)
    assert frame.shape == (2, pyramid_wfs.Npix, pyramid_wfs.Npix)

    torch.var(frame, dim=(-2, -1)).sum().backward()

    assert opd.grad is not None
    assert torch.isfinite(opd.grad).all()
    assert wavelength.grad is not None
    assert torch.isfinite(wavelength.grad).all()


def test_gradient_flows_through_polychromatic_get_psf(pyramid_wfs, device):
    """GetPSF's multi-wavelength path resamples via grid_sample using
    coordinates built from wavelength (scale = wavelength[-1]/wavelength), so
    gradients w.r.t. wavelength must still flow through that resampling, not
    just through the plain wavenumber*opd phase conversion."""
    wavelength = torch.tensor([635e-9, 750e-9], device=device).requires_grad_()
    pyramid_wfs.wavelength = wavelength

    opd = (1e-7 * torch.randn(2, pyramid_wfs.Nres, pyramid_wfs.Nres, device=device)).requires_grad_()

    psf = pyramid_wfs.GetPSF(opd)
    assert psf.shape == (2, pyramid_wfs.Npix, pyramid_wfs.Npix)

    psf.sum().backward()

    assert opd.grad is not None
    assert torch.isfinite(opd.grad).all()
    assert wavelength.grad is not None
    assert torch.isfinite(wavelength.grad).all()


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


def test_gradient_flows_end_to_end_opd_to_reconstructor_loss(
    pyramid_wfs, deformable_mirror, tiny_wfs_params, device
):
    frame_preprocessor = FramePreprocess(tiny_wfs_params(), pyramid_wfs, device)
    pyramid_wfs.BuildReferenceIntensity()
    frame_preprocessor.ProcessReference(pyramid_wfs.reference_intensity)

    total_act = int(deformable_mirror.totalAct.item())
    Nphases = 2
    opd = (1e-7 * torch.randn(Nphases, pyramid_wfs.Nres, pyramid_wfs.Nres, device=device)).requires_grad_()

    wfs_frame = pyramid_wfs(opd)
    preprocessed = frame_preprocessor.ProcessFrame(wfs_frame, add_pupil_noise=False)

    reconstructor = nn.Linear(preprocessed[0].numel(), total_act).to(device)
    z_output = reconstructor(preprocessed.flatten(start_dim=1))
    opd_reconstructed = deformable_mirror(z_output)  # mirrors the Trainer.train chain

    loss_fn = RMSELoss()
    Ze = torch.zeros_like(z_output)
    loss = loss_fn(Ze, z_output, pyramid_wfs.pupil, opd, opd_reconstructed, wfs_frame)
    loss.backward()

    assert opd.grad is not None
    assert torch.isfinite(opd.grad).all()
    assert not torch.allclose(opd.grad, torch.zeros_like(opd.grad))
    assert reconstructor.weight.grad is not None
    assert torch.isfinite(reconstructor.weight.grad).all()
