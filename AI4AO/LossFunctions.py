# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 09:52:19 2025

@author: franc
"""

import torch # type: ignore[import]
import torch.nn as nn # type: ignore[import]
import copy


class AOLoss(nn.Module):
    """
    Base class for losses pluggable into Trainer.loss. All subclasses share
    the same forward signature so Trainer can call any of them uniformly; a
    given loss uses only the arguments relevant to it and ignores the rest.
    Static, per-instrument context (e.g. a pupil mask) still belongs in
    __init__, the same way LogResidualVarianceLoss takes `pupil` there.

    forward args:
        Ze (Tensor): Ground-truth residual modal coefficients for this iteration.
        z_estimated (Tensor): Predicted modal coefficients for this iteration
            (Trainer's `z_output`, i.e. the reconstructor's raw prediction).
        residual_opd (Tensor): Residual wavefront (OPD, in meters) *before*
            this iteration's correction is applied -- what was actually fed
            into the WFS to produce `wfs_frames`.
        corrected_residual_opd (Tensor): Residual wavefront *after* this
            iteration's correction is applied -- the quantity you're actually
            trying to minimize.
        wfs_frames (Tensor): Raw WFS detector frame(s) for this iteration.
    """

    def forward(self, Ze, z_estimated, pupil, residual_opd, corrected_residual_opd, wfs_frames):
        return self.compute(Ze, z_estimated, pupil, residual_opd, corrected_residual_opd, wfs_frames)

    def compute(self, Ze, z_estimated, pupil, residual_opd, corrected_residual_opd, wfs_frames):
        raise NotImplementedError

    def _terms(self):
        """(loss, weight) pairs this loss is made of; overridden by WeightedLossSum."""
        return [(self, 1.0)]

    def __add__(self, other):
        if not isinstance(other, AOLoss):
            return NotImplemented
        return WeightedLossSum(self._terms() + other._terms())

    def __mul__(self, weight):
        if not isinstance(weight, (int, float)):
            return NotImplemented
        return WeightedLossSum([(loss, w * float(weight)) for loss, w in self._terms()])

    __rmul__ = __mul__


class WeightedLossSum(AOLoss):
    """
    An AOLoss that is itself a weighted sum of other AOLoss terms, built by
    combining losses with `+` and `*` (e.g. `loss_a * 0.5 + loss_b * 0.5`).
    Terms are stored flat (never nested), so repeated combination doesn't
    build up wrapper depth.
    """

    def __init__(self, terms):
        super().__init__()
        self.losses = nn.ModuleList([loss for loss, _ in terms])
        self.weights = [float(weight) for _, weight in terms]

    def _terms(self):
        return list(zip(self.losses, self.weights))

    def compute(self, Ze, z_estimated, pupil, residual_opd, corrected_residual_opd, wfs_frames):
        total = 0.0
        for loss, weight in zip(self.losses, self.weights):
            total = total + weight * loss(Ze, z_estimated, pupil, residual_opd, corrected_residual_opd, wfs_frames)
        return total


class Relative_Loss_Function(AOLoss):
    """
    Loss that normalizes the per-sample error between predicted and true
    coefficients by the magnitude of the true coefficients, so that the loss
    scale doesn't depend on the overall amplitude of Ze (e.g. residual
    Zernike coefficients across different turbulence strengths).

    Args:
        epsilon (float): Added to the denominator for numerical stability
            (avoids division by ~0 when Ze is small).
        degree (int): Power to which the error/coefficients are raised before
            averaging (e.g. 2 for an MSE-like norm).
        device (str): Unused; accepted for interface consistency with other losses.
    """

    def __init__(self, epsilon=0.005, degree=2, device=None):
        super().__init__()
        self.epsilon = epsilon
        self.degree = degree

    def compute(self, Ze, z_estimated, pupil, residual_opd, corrected_residual_opd, wfs_frames):
        """
        Computes the relative error between predicted and true modal
        coefficients, averaged over all non-batch dimensions and then over
        the batch.

        Returns:
            Tensor: Scalar loss value.
        """
        diff = z_estimated - Ze
        numerator = torch.mean(
            diff**self.degree, dim=tuple(range(1, Ze.ndim))
        ) ** (1 / self.degree)
        denominator = torch.mean(
            Ze**self.degree + self.epsilon, dim=tuple(range(1, Ze.ndim))
        ) ** (1 / self.degree)
        return torch.mean(numerator / denominator)


class Physics_loss(AOLoss):
    """
    Physics-consistency loss: reprojects the pre-correction residual OPD
    through a (noiseless) copy of the WFS forward model and penalizes the
    difference between the resulting simulated WFS frame and the observed
    (noisy) WFS frame.

    Args:
        wfs (nn.Module): WFS forward model to deep-copy; noise is disabled on
            the copy (`useNoise = False`) so the loss is deterministic.
        degree (int): Power to which the pixel-wise error is raised.
    """

    def __init__(self, wfs, degree=2):
        super().__init__()
        self.wfs = copy.deepcopy(wfs)
        self.wfs.useNoise = False
        self.degree = degree

    def compute(self, Ze, z_estimated, pupil, residual_opd, corrected_residual_opd, wfs_frames):
        """
        Computes the physics-based loss between the observed WFS frame and
        the frame simulated from residual_opd through the internal WFS copy.

        Returns:
            Tensor: Scalar loss value.
        """
        reconstructed_opd = residual_opd - corrected_residual_opd
        I_pred = self.wfs(reconstructed_opd, pupil)
        return (
            torch.mean(torch.abs(I_pred - wfs_frames) ** self.degree) * 1e6
        )



class WFSSignalLoss(AOLoss):
    """
    Loss function that rewards generating more signal (i.e. contrast) in the
    WFS frame, encouraging e.g. the WFS mask to spread flux across the frame.

    No constructor arguments.
    """

    def __init__(self):
        super().__init__()

    def compute(self, Ze, z_estimated, pupil, residual_opd, corrected_residual_opd, wfs_frames):
        """
        Computes the negative mean per-pixel standard deviation across the
        batch as a proxy for the amount of signal in the WFS frame.

        Returns:
            Tensor: Scalar loss value.
        """

        return -torch.mean(torch.std(wfs_frames, dim=(-2,-1))) * 1e6


class LogResidualVarianceLoss(AOLoss):
    """
    Loss function based on the log-variance of the post-correction residual
    phase (radians) over the pupil area; minimizing it drives down residual
    wavefront error while the log keeps the gradient well-scaled across a
    wide range of variances. Deliberately kept in phase units rather than
    OPD, since residual phase variance (Strehl-adjacent) is the physically
    meaningful quantity here -- `corrected_residual_opd` is converted to
    phase via this WFS's own sensing wavelength before computing the variance.

    Args:
        pupil (Tensor): Binary mask indicating the pupil region.
        wavelength (float): Sensing wavelength (meters) used to convert the
            incoming OPD (meters) to phase (radians).
        device (str): Unused; accepted for interface consistency with other losses.
    """

    def __init__(self, pupil, wavelength, device=None):
        super().__init__()
        self.pupil = torch.clone(pupil).bool()
        wavelength = torch.as_tensor(wavelength, device=self.pupil.device, dtype=torch.float32).reshape(-1)
        self.wavenumber = 2 * torch.pi / wavelength.mean()

    def compute(self, Ze, z_estimated, pupil, residual_opd, corrected_residual_opd, wfs_frames):
        """
        Computes the log of the post-correction residual phase variance over
        the pupil region.

        Returns:
            Tensor: Scalar loss value.
        """
        corrected_residual_phase = self.wavenumber * corrected_residual_opd
        residual_variance = torch.var(
            corrected_residual_phase[..., self.pupil], dim=-1, keepdim=True
        )
        return torch.mean(torch.log(residual_variance))


class RMSELoss(AOLoss):
    """Root Mean Square Error between predicted and true modal coefficients."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def compute(self, Ze, z_estimated, pupil, residual_opd, corrected_residual_opd, wfs_frames):
        return torch.sqrt(self.mse(z_estimated, Ze))
