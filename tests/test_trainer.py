"""Tests for AI4AO.Trainer, isolating the two cheap/self-contained pieces:
the z_inv pseudo-inverse identity computed in __init__ (needs only a dm+M2C,
no WFS/dataset/reconstructor), and the checkpoint save/load round trip
(touches only phaseReconstructor/optimizer)."""
import torch
import torch.nn as nn

from AI4AO.Trainer import Trainer


class _StubWFS:
    """Trainer.__init__ only reads wfs.device; nothing else is touched
    unless .train()/.eval() or the forward pass is exercised."""
    def __init__(self, device):
        self.device = device


def _build_minimal_trainer(deformable_mirror, device, reconstructor=None, optimizer=None):
    total_act = int(deformable_mirror.totalAct.item())
    M2C = torch.eye(total_act, device=device)
    if reconstructor is None:
        reconstructor = nn.Linear(4, 4)
    if optimizer is None:
        optimizer = torch.optim.SGD(reconstructor.parameters(), lr=0.1)
    trainer = Trainer(
        wfs=_StubWFS(device),
        framePreprocessor=None,
        dm=deformable_mirror,
        M2C=M2C,
        phaseReconstructor=reconstructor,
        dataset=None,
        loss=None,
        optimizer=optimizer,
    )
    return trainer, M2C, total_act


def test_z_inv_is_pseudo_inverse_of_dm_basis(deformable_mirror, device):
    trainer, M2C, total_act = _build_minimal_trainer(deformable_mirror, device)

    IF_flat = deformable_mirror(M2C.T).flatten(start_dim=-2)
    identity_check = IF_flat @ trainer.z_inv

    assert trainer.z_inv.shape == (deformable_mirror.Nres * deformable_mirror.Nres, total_act)
    assert torch.allclose(identity_check, torch.eye(total_act, device=device), atol=1e-3)


def test_save_and_load_checkpoint_round_trip(deformable_mirror, device, tmp_path):
    reconstructor = nn.Linear(4, 4)
    optimizer = torch.optim.Adam(reconstructor.parameters(), lr=0.1)
    trainer, _, _ = _build_minimal_trainer(deformable_mirror, device, reconstructor, optimizer)

    x = torch.randn(2, 4)
    reconstructor(x).sum().backward()
    optimizer.step()  # populate Adam's per-parameter state

    path = tmp_path / "ckpt.pth"
    trainer.save_checkpoint(str(path), step=42)

    original_weight = reconstructor.weight.detach().clone()
    original_exp_avg = optimizer.state_dict()["state"][0]["exp_avg"].clone()

    with torch.no_grad():
        reconstructor.weight.add_(1.0)
    optimizer.state[reconstructor.weight]["exp_avg"].add_(1.0)

    result = trainer.load_checkpoint(str(path))

    # `load_checkpoint`'s docstring promises returning the full checkpoint
    # dict on success, but its success path has no `return` statement, so it
    # currently falls through and returns None even though the load worked
    # (weights/optimizer state below ARE correctly restored, since
    # load_state_dict runs before the function implicitly returns). This
    # documents today's actual behavior rather than the docstring's claim.
    assert result is None
    assert torch.allclose(reconstructor.weight, original_weight)
    assert torch.allclose(optimizer.state_dict()["state"][0]["exp_avg"], original_exp_avg)


def test_load_checkpoint_missing_path_returns_none(deformable_mirror, device, tmp_path):
    trainer, _, _ = _build_minimal_trainer(deformable_mirror, device)
    result = trainer.load_checkpoint(str(tmp_path / "does_not_exist.pth"))
    assert result is None
