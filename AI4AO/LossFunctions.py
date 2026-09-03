# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 09:52:19 2025

@author: franc
"""

import torch # type: ignore[import]
import torch.nn as nn # type: ignore[import]
import copy


class Relative_Loss_Function(nn.Module):
    """
    Loss that normalizes the per-sample error between predicted and true
    coefficients by the magnitude of the true coefficients, so that the loss
    scale doesn't depend on the overall amplitude of y_true (e.g. residual
    Zernike coefficients across different turbulence strengths).

    Args:
        epsilon (float): Added to the denominator for numerical stability
            (avoids division by ~0 when y_true is small).
        degree (int): Power to which the error/coefficients are raised before
            averaging (e.g. 2 for an MSE-like norm).
        device (str): Unused; accepted for interface consistency with other losses.
    """

    def __init__(self, epsilon=0.005, degree=2, device="cuda"):
        super().__init__()
        self.epsilon = epsilon
        self.degree = degree

    def forward(self, y_pred, y_true):
        """
        Computes the relative error between predicted and true coefficients,
        averaged over all non-batch dimensions and then over the batch.

        Args:
            y_pred (Tensor): Predicted coefficients, shape (batch, ...).
            y_true (Tensor): Ground truth coefficients, same shape as y_pred.

        Returns:
            Tensor: Scalar loss value.
        """
        diff = y_pred - y_true
        numerator = torch.mean(
            diff**self.degree, dim=tuple(range(1, y_true.ndim))
        ) ** (1 / self.degree)
        denominator = torch.mean(
            y_true**self.degree + self.epsilon, dim=tuple(range(1, y_true.ndim))
        ) ** (1 / self.degree)
        return torch.mean(numerator / denominator)


class Physics_loss(nn.Module):
    """
    Physics-consistency loss: reprojects an input phase through a (noiseless)
    copy of the WFS forward model and penalizes the difference between the
    resulting simulated WFS frame and an observed WFS frame.

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

    def forward(self, I_WFS, input_phase):
        """
        Computes the physics-based loss between an observed WFS frame and the
        frame simulated from input_phase through the internal WFS copy.

        Args:
            I_WFS (Tensor): Observed WFS image.
            input_phase (Tensor): Phase to propagate through the WFS forward
                model for comparison against I_WFS.

        Returns:
            Tensor: Scalar loss value.
        """
        I_pred = self.wfs(input_phase)
        return (
            torch.mean(torch.abs(I_pred - I_WFS) ** self.degree) * 1e6
        )



class WFSSignalLoss(nn.Module):
    """
    Loss function that rewards generating more signal (i.e. contrast) in the
    WFS frame, encouraging e.g. the WFS mask to spread flux across the frame.

    No constructor arguments.
    """

    def __init__(self):
        super().__init__()

    def forward(self, wfsFrame):
        """
        Computes the negative mean per-pixel standard deviation across the
        batch as a proxy for the amount of signal in the WFS frame.

        Args:
            wfsFrame (Tensor): Batch of WFS frames, shape (batch, ...).

        Returns:
            Tensor: Scalar loss value.
        """

        return -torch.mean(torch.std(wfsFrame, dim=0)) * 1e6


class LogResidualVarianceLoss(nn.Module):
    """
    Loss function based on the log-variance of the residual phase over the
    pupil area; minimizing it drives down residual wavefront error while the
    log keeps the gradient well-scaled across a wide range of variances.

    Args:
        pupil (Tensor): Binary mask indicating the pupil region.
        device (str): Unused; accepted for interface consistency with other losses.
    """

    def __init__(self, pupil, device="cuda"):
        super().__init__()
        self.pupil = torch.clone(pupil).bool().cpu()

    def forward(self, residual_phase):
        """
        Computes the log of the residual phase variance over the pupil region.

        Args:
            residual_phase (Tensor): Residual wavefront phase, shape
                (batch, Nres, Nres).

        Returns:
            Tensor: Scalar loss value.
        """
        residual_variance = torch.var(
            residual_phase[..., self.pupil], dim=-1, keepdim=True
        )
        return torch.mean(torch.log(residual_variance))


class RMSELoss(nn.Module):
    """Root Mean Square Error Loss"""

    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        return torch.sqrt(self.mse(predictions, targets))
