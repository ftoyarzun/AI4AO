"""Tests for AI4AO.LossFunctions.

Physics_loss needs a callable WFS-like object, but since it only calls
`wfs(phase)` and reads `wfs.useNoise`, a minimal duck-typed stub is enough --
no full PyramidWFS/ZernikeWFS instance required.
"""
import copy

import pytest
import torch
import torch.nn as nn

from AI4AO.LossFunctions import (
    AOLoss,
    WeightedLossSum,
    Relative_Loss_Function,
    Physics_loss,
    WFSSignalLoss,
    LogResidualVarianceLoss,
    RMSELoss,
)


def _dummy_forward_args(batch=2, nmodes=3, nres=6):
    Ze = torch.randn(batch, nmodes)
    z_estimated = torch.randn(batch, nmodes, requires_grad=True)
    residual_phase = torch.randn(batch, nres, nres)
    corrected_residual_phase = torch.randn(batch, nres, nres)
    wfs_frames = torch.rand(batch, nres, nres)
    return Ze, z_estimated, residual_phase, corrected_residual_phase, wfs_frames


def test_relative_loss_function_scalar_and_gradient():
    loss_fn = Relative_Loss_Function()
    Ze, z_estimated, residual_phase, corrected_residual_phase, wfs_frames = _dummy_forward_args()

    loss = loss_fn(Ze, z_estimated, residual_phase, corrected_residual_phase, wfs_frames)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert z_estimated.grad is not None
    assert torch.isfinite(z_estimated.grad).all()


def test_relative_loss_is_zero_when_predictions_match_truth():
    loss_fn = Relative_Loss_Function()
    Ze = torch.randn(2, 3)
    loss = loss_fn(Ze, Ze.clone(), None, None, None)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_wfs_signal_loss_rewards_higher_batch_variance():
    loss_fn = WFSSignalLoss()

    low_variance_frames = torch.ones(4, 5, 5) + 0.001 * torch.randn(4, 5, 5)
    high_variance_frames = torch.randn(4, 5, 5) * 5

    low_var_loss = loss_fn(None, None, None, None, low_variance_frames)
    high_var_loss = loss_fn(None, None, None, None, high_variance_frames)

    # WFSSignalLoss = -mean(std(...)) * 1e6, so more variance -> more negative (lower) loss
    assert high_var_loss < low_var_loss


def test_rmse_loss_matches_manual_computation():
    loss_fn = RMSELoss()
    Ze = torch.randn(3, 4)
    z_estimated = torch.randn(3, 4)

    loss = loss_fn(Ze, z_estimated, None, None, None)
    expected = torch.sqrt(torch.mean((z_estimated - Ze) ** 2))

    assert torch.allclose(loss, expected, atol=1e-6)
    assert torch.allclose(loss_fn(Ze, Ze.clone(), None, None, None), torch.zeros_like(loss), atol=1e-6)


def test_log_residual_variance_loss_matches_manual_computation():
    pupil = torch.zeros(6, 6, dtype=torch.bool)
    pupil[1:5, 1:5] = True

    corrected_residual_phase = torch.randn(2, 6, 6)
    loss_fn = LogResidualVarianceLoss(pupil)

    loss = loss_fn(None, None, None, corrected_residual_phase, None)

    expected_var = corrected_residual_phase[..., pupil].var(dim=-1, keepdim=True)
    expected = torch.log(expected_var).mean()
    assert torch.allclose(loss, expected, atol=1e-5)


class _FakeWFS(nn.Module):
    """Minimal duck-typed stand-in for a WFS: Physics_loss only needs
    `.useNoise` and a callable forward(phase) -> frame."""
    def __init__(self):
        super().__init__()
        self.useNoise = True
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, phase):
        return self.scale * phase.abs()


def test_physics_loss_deep_copies_wfs_and_disables_noise():
    wfs = _FakeWFS()
    loss_fn = Physics_loss(wfs)

    assert loss_fn.wfs is not wfs  # deep-copied, not aliased
    assert loss_fn.wfs.useNoise is False
    assert wfs.useNoise is True  # original untouched


def test_physics_loss_matches_manual_computation():
    wfs = _FakeWFS()
    loss_fn = Physics_loss(wfs, degree=2)

    residual_phase = torch.randn(2, 4, 4)
    corrected_residual_phase = torch.randn(2, 4, 4)
    wfs_frames = torch.rand(2, 4, 4)

    loss = loss_fn(None, None, residual_phase, corrected_residual_phase, wfs_frames)

    reconstructed_phase = residual_phase - corrected_residual_phase
    expected = torch.mean(torch.abs(loss_fn.wfs(reconstructed_phase) - wfs_frames) ** 2) * 1e6
    assert torch.allclose(loss, expected, atol=1e-4)


def test_weighted_loss_sum_composition():
    class _ConstLoss(AOLoss):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def compute(self, Ze, z_estimated, residual_phase, corrected_residual_phase, wfs_frames):
            return torch.tensor(self.value)

    loss_a = _ConstLoss(2.0)
    loss_b = _ConstLoss(3.0)

    combined = loss_a * 0.5 + loss_b * 10.0
    assert isinstance(combined, WeightedLossSum)

    args = (None, None, None, None, None)
    result = combined(*args)
    assert torch.allclose(result, torch.tensor(2.0 * 0.5 + 3.0 * 10.0))

    # terms stay flat, not nested, under repeated combination
    loss_c = _ConstLoss(1.0)
    triple = combined + loss_c
    assert len(triple._terms()) == 3
