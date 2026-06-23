# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 09:52:19 2025

@author: franc
"""

import torch
import torch.nn as nn


class Custom_Loss_Function(nn.Module):
    """
    Custom loss function that penalizes the weighted difference between predicted
    and ground truth Zernike coefficients. The weights increase with Zernike index
    to emphasize higher-order modes.

    Args:
        epsilon (float): Small value for numerical stability (currently unused).
        degree (int): Power to which the error is raised (e.g., 2 for MSE).
        Nmodes (int): Number of modes considered in the loss.
        device (str): Device on which tensors will be allocated ('cuda' or 'cpu').
    """

    def __init__(self, epsilon=1e-2, degree=2, Nmodes=209, device="cuda"):
        super().__init__()
        self.epsilon = epsilon
        self.degree = degree
        self.Nmodes = Nmodes

        self.linspace = torch.sqrt(
            (torch.linspace(1, Nmodes, Nmodes, device=device))
        )
        # self.linspace = torch.linspace(1, Nmodes, Nmodes, device=device)

    def forward(self, y_pred, y_true, r0):
        """
        Computes the custom weighted loss between predicted and true Zernike coefficients.

        Args:
            y_pred (Tensor): Predicted Zernike coefficients.
            y_true (Tensor): Ground truth Zernike coefficients.
            r0 (Tensor): Fried parameter used for atmospheric scaling.

        Returns:
            Tensor: Scalar loss value.
        """
        diff = (y_pred - y_true)[..., : self.Nmodes] * self.linspace
        return torch.mean(torch.abs(diff) ** self.degree * r0)  # ** (5/3))


class Relative_Loss_Function(nn.Module):
    """
    Custom loss function that penalizes the weighted difference between predicted
    and ground truth Zernike coefficients. The weights increase with Zernike index
    to emphasize higher-order modes.

    Args:
        epsilon (float): Small value for numerical stability (currently unused).
        degree (int): Power to which the error is raised (e.g., 2 for MSE).
        Nmodes (int): Number of modes considered in the loss.
        device (str): Device on which tensors will be allocated ('cuda' or 'cpu').
    """

    def __init__(self, epsilon=0.005, degree=2, device="cuda"):
        super().__init__()
        self.epsilon = epsilon
        self.degree = degree

    def forward(self, y_pred, y_true):
        """
        Computes the custom weighted loss between predicted and true Zernike coefficients.

        Args:
            y_pred (Tensor): Predicted Zernike coefficients.
            y_true (Tensor): Ground truth Zernike coefficients.
            r0 (Tensor): Fried parameter used for atmospheric scaling.

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
    Physics-based loss that penalizes the difference between the observed and
    simulated wavefront sensor images based on reconstructed Zernike coefficients.

    Args:
        z_fullRes (Tensor): Matrix used to reconstruct the full phase from Zernike coefficients.
        phase_template (Tensor): Template phase used for reshaping during reconstruction.
        degree (int): Power to which the error is raised.
        device (str): Device on which computations are performed.
    """

    def __init__(self, z_fullRes, phase_template, degree=2, device="cuda"):
        super().__init__()
        self.z_fullRes = z_fullRes
        self.phase_shape = phase_template.shape
        self.degree = degree

    def forward(self, WFSModule, I_WFS, y_pred, r0):
        """
        Computes the physics-based loss between predicted and true wavefront sensor images.

        Args:
            WFSModule (nn.Module): Wavefront sensor forward model.
            I_WFS (Tensor): Observed WFS image.
            y_pred (Tensor): Predicted Zernike coefficients.
            r0 (Tensor): Fried parameter used for atmospheric scaling.

        Returns:
            Tensor: Scalar loss value.
        """
        I_pred = self.ComputeForwardImage(WFSModule, y_pred)
        return (
            torch.mean(torch.abs(I_pred - I_WFS) ** self.degree * r0 ** (5 / 3)) * 1e6
        )

    def ComputeForwardImage(self, WFSModule, y_pred):
        """
        Reconstructs the wavefront sensor image from Zernike coefficients.

        Args:
            WFSModule (nn.Module): Wavefront sensor forward model.
            y_pred (Tensor): Predicted Zernike coefficients.

        Returns:
            Tensor: Simulated WFS image.
        """
        reconstructed_phase = torch.matmul(y_pred, self.z_fullRes).view(
            *self.phase_shape
        )

        return WFSModule(reconstructed_phase)


class ResidualPhaseLoss(nn.Module):
    """
    Loss function based on the residual phase variance over the pupil area.

    Args:
        z_fullRes (Tensor): Matrix used to reconstruct the full phase from Zernike coefficients.
        pupil (Tensor): Binary mask indicating the pupil region.
        device (str): Device for computation.
    """

    def __init__(self, z_fullRes, pupil, device="cuda"):
        super().__init__()
        self.z_fullRes = z_fullRes
        self.pupil = pupil

    def forward(self, y_pred, phase, r0):
        """
        Computes the residual phase loss as the variance over the pupil region.

        Args:
            y_pred (Tensor): Predicted Zernike coefficients.
            phase (Tensor): True wavefront phase.
            r0 (Tensor): Fried parameter used for atmospheric scaling.

        Returns:
            Tensor: Scalar loss value.
        """
        reconstructed_phase = torch.matmul(y_pred, self.z_fullRes).view_as(phase)
        residual_phase = phase - reconstructed_phase
        normalization_factor = torch.var(
            phase[..., self.pupil.bool()], dim=-1, keepdim=True
        )
        return torch.mean(
            residual_phase[..., self.pupil.bool()] ** 2 / normalization_factor
        )


class WFSSignalLoss(nn.Module):
    """
    Loss function based on generating the most signal in the wfs frame.

    Args:
        z_fullRes (Tensor): Matrix used to reconstruct the full phase from Zernike coefficients.
        pupil (Tensor): Binary mask indicating the pupil region.
        device (str): Device for computation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, wfsFrame):
        """
        Computes the variance of the wfsFrame as a proxy for amount of signal

        Args:
            wfsFrame (Tensor): Frame fromt the wfs.

        Returns:
            Tensor: Scalar loss value.
        """

        return -torch.mean(torch.std(wfsFrame, dim=0)) * 1e6


class LogResidualVarianceLoss(nn.Module):
    """
    Loss function based on the residual phase variance over the pupil area.

    Args:
        z_fullRes (Tensor): Matrix used to reconstruct the full phase from Zernike coefficients.
        pupil (Tensor): Binary mask indicating the pupil region.
        device (str): Device for computation.
    """

    def __init__(self, pupil, device="cuda"):
        super().__init__()
        self.pupil = torch.clone(pupil).bool().cpu()

    def forward(self, residual_phase):
        """
        Computes the residual phase loss as the variance over the pupil region.

        Args:
            y_pred (Tensor): Predicted Zernike coefficients.
            phase (Tensor): True wavefront phase.
            r0 (Tensor): Fried parameter used for atmospheric scaling.

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


class TestLoss(nn.Module):
    """
    Loss function based on the residual phase variance over the pupil area.

    Args:
        z_fullRes (Tensor): Matrix used to reconstruct the full phase from Zernike coefficients.
        pupil (Tensor): Binary mask indicating the pupil region.
        device (str): Device for computation.
    """

    def __init__(self, z_fullRes, pupil, device="cuda"):
        super().__init__()
        self.z_fullRes = z_fullRes
        self.pupil = pupil

    def forward(self, y_pred, phase):
        """
        Computes the residual phase loss as the variance over the pupil region.

        Args:
            y_pred (Tensor): Predicted Zernike coefficients.
            phase (Tensor): True wavefront phase.
            r0 (Tensor): Fried parameter used for atmospheric scaling.

        Returns:
            Tensor: Scalar loss value.
        """
        reconstructed_phase = torch.matmul(y_pred, self.z_fullRes).view_as(phase)
        residual_phase = phase - reconstructed_phase
        residual_variance = torch.var(
            residual_phase[..., self.pupil.bool()], dim=-1, keepdim=True
        )
        residual_variance = torch.sqrt(residual_variance)
        return torch.mean((residual_variance))


class EstimatedSignalVarianceLoss(nn.Module):
    """
    Loss function based on the residual phase variance over the pupil area.

    """

    def __init__(self, alpha=1e-3):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred):
        return self.alpha / y_pred.var()
