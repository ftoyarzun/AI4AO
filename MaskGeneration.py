# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:37:06 2025

@author: franc
"""

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import time

from Constants import mask_types_list

from PhaseDataset import Zernike
from IPython.display import display, clear_output


class MaskManager(nn.Module):

    def __init__(self, ParamsDict, device, WFS):

        super().__init__()
        self.device = device
        self.maskType = ParamsDict["MaskType"]
        self.sampling = ParamsDict["sampling"]
        self.MTF_focal_upscale = ParamsDict["MTF_upscale"]
        self.use_MTF = ParamsDict["Use_MTF"]
        self.Nres = ParamsDict["Nres"]

        self.N = int(self.Nres * self.sampling)
        self.WFS = WFS  # Reference to actual WFS object

        # Build uv grid for freeform mask types
        self._build_uv_grid()
        self._build_xy_grid()

        # Initialize mask generators
        self.phaseMask = None
        self.transmisionMask = None
        self._init_mask_generators(ParamsDict)

        # Hold the position of the pupils
        self.pupil_centers = None

    def make_pupil(self, Rpx, nPx):
        """
        Generate a circular pupil mask.

        Parameters
        ----------
        Rpx : float
            Radius of the pupil in pixels.
        nPx : int
            Size of the square output array (number of pixels per side).

        Returns
        -------
        pupil : ndarray of shape (nPx, nPx)
            Binary circular mask with value 1 inside the pupil and 0 outside.
        """
        # Create grid of coordinates
        x = torch.arange(nPx, device=self.device, dtype=torch.float32) - (nPx - 1) / 2
        x, y = torch.meshgrid(x, x)

        # Circular mask
        pupil = (x**2 + y**2) <= Rpx**2
        return pupil

    def _build_uv_grid(self):
        u = torch.linspace(-1, 1 - 2 / self.N, self.N, device=self.device)
        U, V = torch.meshgrid(u, u, indexing="xy")
        self.UV = torch.stack([U.flatten(), V.flatten()], dim=1)  # (N², 2)
        self.circ_mask = (torch.sqrt(U**2 + V**2) < 0.9).flatten()  # (N²,)

    def _build_xy_grid(self):
        x_mask = torch.linspace(
            -self.N / 2, self.N / 2 - 1, self.N, dtype=torch.float32, device=self.device
        )
        [self.x_mask, self.y_mask] = torch.meshgrid(x_mask, x_mask)

        self.rho_mask = torch.sqrt(self.x_mask**2 + self.y_mask**2)
        self.abs_x_mask = torch.abs(self.x_mask)
        self.abs_y_mask = torch.abs(self.y_mask)

    def _init_mask_generators(self, ParamsDict):
        if self.maskType in ["FreePhase", "FreePhaseTransmision"]:
            self.phaseMaskGenerator = FreeMaskGenerator(isPhaseMask=True).to(
                self.device
            )
        if self.maskType in ["FreeTransmision", "FreePhaseTransmision"]:
            self.transmisionMaskGenerator = FreeMaskGenerator(isPhaseMask=False).to(
                self.device
            )
        if self.maskType in [
            "BiOEdge",
            "IBiOEdge",
            "DoublePyramid",
        ]:
            self.param = nn.Parameter(
                torch.tensor(ParamsDict["InitParam"], device=self.device)
            )
        if self.maskType == "ModalMask":
            self.phaseMaskGenerator = ModalMaskGeneration(self.WFS.pupil).to(
                self.device
            )
        if self.maskType in ["FullyFreePhase"]:
            sigma = 30.0
            self.phaseMask_FT = nn.Parameter(
                1 / sigma * torch.exp(-self.rho_mask**2 / (2 * sigma**2))
                + 0
                * torch.randn(self.N, self.N, device=self.device, dtype=torch.float32)
            )

        if self.maskType in ["FullyFreeTransmision"]:
            self.transmisionMaskGenerator = nn.Parameter(
                0.00001
                * torch.randn(
                    2, self.N, self.N, device=self.device, dtype=torch.float32
                )
            )

        if self.maskType in ["Papyrus"]:
            self.mainSlope = nn.Parameter(
                torch.tensor(torch.pi / 2, device=self.device, dtype=torch.float32)
            )
            self.maskShifts = nn.Parameter(
                torch.ones(4, 2, device=self.device, dtype=torch.float32)
            )
            self.rooftop = nn.Parameter(
                torch.tensor(-2.2999, device=self.device, dtype=torch.float32)
            )
            self.coordinatesRotation = nn.Parameter(
                torch.tensor(0.0, device=self.device, dtype=torch.float32)
            )

        if self.maskType in ["Pyramid"]:
            self.mainSlope = nn.Parameter(
                torch.tensor(torch.pi / 2, device=self.device, dtype=torch.float32)
            )
            self.maskShifts = nn.Parameter(
                torch.ones(4, 2, device=self.device, dtype=torch.float32)
            )
            self.rooftop = nn.Parameter(
                torch.tensor(0, device=self.device, dtype=torch.float32)
            )
            self.coordinatesRotation = nn.Parameter(
                torch.tensor(0.0, device=self.device, dtype=torch.float32)
            )

        if self.maskType in ["DoubleZernike"]:
            self.depths = nn.Parameter(
                torch.tensor(
                    [torch.pi * 0.76, -torch.pi * 0.33],
                    # [torch.pi * 0.5, -torch.pi * 1],
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.diameters = nn.Parameter(
                torch.tensor([2.0, 2.0], device=self.device, dtype=torch.float32)
            )
            self.positions = nn.Parameter(
                torch.tensor(
                    [[-torch.pi * 0.37, 0], [torch.pi * 0.37, 0.0]],
                    device=self.device,
                    dtype=torch.float32,
                )
            )

        if self.maskType in ["Zernike"]:
            self.depths = nn.Parameter(
                torch.tensor(
                    [-torch.pi * 0.33],
                    # [torch.pi * 0.5, -torch.pi * 1],
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.diameters = nn.Parameter(
                torch.tensor([2.14], device=self.device, dtype=torch.float32)
            )
            self.positions = nn.Parameter(
                torch.tensor(
                    [[0.0, 0.0]],
                    device=self.device,
                    dtype=torch.float32,
                )
            )

        if self.maskType not in mask_types_list:
            raise ValueError(f"Unsupported mask type: {self.maskType}")

    def update_masks(self):
        if self.maskType == "Pyramid":
            self.phaseMask = self.BuildPyramidMask()

        if self.maskType == "Zernike" and self.use_MTF is False:
            self.phaseMask = self.BuildZernikeMask()
        if self.maskType == "Zernike" and self.use_MTF is True:
            self.phaseMask, self.transmisionMask = self.BuildZernikeMaskMTF()

        if self.maskType == "BiOEdge":
            self.transmisionMask = self.BuildBiOEdgeMask()
            self.phaseMask = self.PupilDisplacementMask()

        if self.maskType == "IBiOEdge":
            [self.transmisionMask, self.phaseMask] = self.BuildIBiOEdgeMask()
            self.phaseMask += self.PupilDisplacementMask()

        if self.maskType == "DoublePyramid":
            [self.transmisionMask, self.phaseMask] = self.DoublePyramidMask()

        if self.maskType == "Papyrus":
            self.phaseMask = self.BuildPapyrusPyramidMask()

        if self.maskType == "DoubleZernike" and self.use_MTF is False:
            self.phaseMask = self.DoubleZernikeMask()

        if self.maskType == "DoubleZernike" and self.use_MTF is True:
            self.phaseMask, self.transmisionMask = self.DoubleZernikeMaskMTF()

        if self.maskType == "FullyFreeTransmision":
            self.transmisionMask = self.DoubleTransmisionMask(
                torch.sigmoid(self.transmisionMaskGenerator[0]),
                torch.sigmoid(self.transmisionMaskGenerator[1]),
            )

        if self.maskType == "FullyFreePhase":
            # Msym = 0.5 * (self.phaseMask_FT + torch.conj(torch.flip(self.phaseMask_FT, [0,1])))
            self.phaseMask = torch.fft.fftshift(
                torch.fft.ifft2(torch.fft.fftshift(self.phaseMask_FT), norm="ortho")
            ).real

        if self.maskType in ["FreePhase", "FreePhaseTransmision"]:
            self.phaseMask = self.phaseMaskGenerator(self.UV)
            self.phaseMask = self._remove_tip_tilt(self.phaseMask).view(self.N, self.N)

        if self.maskType in ["FreeTransmision", "FreePhaseTransmision"]:
            self.transmisionMask = self.transmisionMaskGenerator(self.UV).view(
                self.N, self.N
            )

        if self.maskType == "ModalMask":
            self.phaseMask = self.phaseMaskGenerator()

        if self.maskType not in mask_types_list:
            raise ValueError(f"Unsupported mask type: {self.maskType}")

        self.WFS.SetMask(phaseMask=self.phaseMask, transmisionMask=self.transmisionMask)

    def _remove_tip_tilt(self, mask):
        # Least-squares fit to remove linear plane (tip/tilt)
        coeffs = torch.linalg.lstsq(
            self.UV[self.circ_mask], mask[self.circ_mask]
        ).solution
        tilt_plane = self.UV @ coeffs
        return mask - tilt_plane

    def GetPupilCenter(self):
        sign_tensor = -torch.tensor(
            [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], device=self.device
        )
        frame_center = torch.ones(4, 2, device=self.device) * self.N / 2
        pupil_center = frame_center + sign_tensor * self.maskShifts * self.N / 4
        pupil_center = torch.round(pupil_center).to(dtype=torch.int)
        return pupil_center

    def BuildZernikeMask(self):
        diameter_in_pixels = self.param[0] * self.sampling

        # this line is not differentiable I use a tanh function to model the mask
        # zernike_mask = self.param[1] * (rho < diameter_in_pixels / 2.)

        if self.param.shape[0] == 2:
            slope = 10.0
        else:
            slope = self.param[2]

        ring_mask = torch.tanh(slope * (diameter_in_pixels / 2.0 - self.rho_mask)) / 2
        annular = ring_mask + 0.5

        zernike_mask = self.param[1] * annular

        return zernike_mask

    def BuildPyramidMask(self):

        rooftop_in_pixels = self.rooftop * self.sampling / np.sqrt(2)

        P1 = (self.x_mask + rooftop_in_pixels / 2) * self.maskShifts[0, 0] + (
            self.y_mask + rooftop_in_pixels / 2
        ) * self.maskShifts[0, 1]
        P2 = -self.x_mask * self.maskShifts[1, 0] + self.y_mask * self.maskShifts[1, 1]
        P3 = (
            -(self.x_mask - rooftop_in_pixels / 2) * self.maskShifts[2, 0]
            - (self.y_mask - rooftop_in_pixels / 2) * self.maskShifts[2, 1]
        )
        P4 = self.x_mask * self.maskShifts[3, 0] - self.y_mask * self.maskShifts[3, 1]

        stacked = torch.stack([P1, P2, P3, P4])  # shape: (4, H, W)

        F = torch.max(stacked * self.mainSlope, dim=0).values  # shape (H, W)

        self.pupil_centers = self.GetPupilCenter()

        return F

    def BuildPapyrusPyramidMask(self):

        rooftop_in_pixels = self.rooftop * self.sampling / np.sqrt(2)

        P1 = (self.x_mask + rooftop_in_pixels / 2) * self.maskShifts[0, 0] + (
            self.y_mask + rooftop_in_pixels / 2
        ) * self.maskShifts[0, 1]
        P2 = -self.x_mask * self.maskShifts[1, 0] + self.y_mask * self.maskShifts[1, 1]
        P3 = (
            -(self.x_mask - rooftop_in_pixels / 2) * self.maskShifts[2, 0]
            - (self.y_mask - rooftop_in_pixels / 2) * self.maskShifts[2, 1]
        )
        P4 = self.x_mask * self.maskShifts[3, 0] - self.y_mask * self.maskShifts[3, 1]

        stacked = torch.stack([P1, P2, P3, P4])  # shape: (4, H, W)

        F = torch.max(stacked * self.mainSlope, dim=0).values  # shape (H, W)

        self.pupil_centers = self.GetPupilCenter()

        return F

    def BuildBiOEdgeMask(self):

        return self.DoubleTransmisionMask(
            self.linear_ramp(self.x_mask, self.param[0]),
            self.linear_ramp(self.y_mask, self.param[0]),
        )

    def BuildIBiOEdgeMask(self):
        x_mask = 1.0 / (self.abs_x_mask + 1)
        y_mask = 1.0 / (self.abs_y_mask + 1)
        transmision = self.DoubleTransmisionMask(x_mask, y_mask)

        value = torch.tensor([1.0], device=self.device, dtype=torch.float32)
        phase = torch.zeros(
            1, 4, self.N, self.N, device=self.device, dtype=torch.float32
        )
        phase[0, 0] = (torch.heaviside(self.x_mask, value)) * torch.pi / 2
        phase[0, 1] = (torch.heaviside(self.x_mask, value)) * -torch.pi / 2
        phase[0, 2] = (torch.heaviside(self.y_mask, value)) * torch.pi / 2
        phase[0, 3] = (torch.heaviside(self.y_mask, value)) * -torch.pi / 2
        return transmision, phase * 0

    def DoubleTransmisionMask(self, mask_x, mask_y):
        mask = torch.zeros(
            1, 4, self.N, self.N, device=self.device, dtype=torch.float32
        )

        m0 = mask_x
        m1 = 1 - m0

        m2 = mask_y
        m3 = 1 - m2

        mask[0, 0] = m0
        mask[0, 1] = m1
        mask[0, 2] = m2
        mask[0, 3] = m3

        return torch.sqrt(mask)

    def linear_ramp(self, x, delta):
        """
        x: input tensor
        a: start of linear ramp
        b: end of linear ramp
        """
        delta = delta / 2.0 * self.sampling
        return torch.clamp((x + delta) / (2 * delta), min=0.0, max=1.0)

    def PupilDisplacementMask(self):
        mask = torch.zeros(
            1, 4, self.N, self.N, device=self.device, dtype=torch.float32
        )

        mask[0, 0] = np.pi / 2 * (-self.x_mask - self.y_mask)
        mask[0, 1] = np.pi / 2 * (-self.x_mask + self.y_mask)
        mask[0, 2] = np.pi / 2 * (self.x_mask - self.y_mask)
        mask[0, 3] = np.pi / 2 * (self.x_mask + self.y_mask)

        sign_tensor = -torch.tensor(
            [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], device=self.device
        )
        frame_center = torch.ones(4, 2, device=self.device) * self.N / 2
        pupil_center = frame_center + sign_tensor * self.N / 4
        self.pupil_centers = torch.round(pupil_center).to(dtype=torch.int).cpu().numpy()

        return mask

    def DoublePyramidMask(self):
        phaseMask = torch.zeros(
            1, 2, self.N, self.N, device=self.device, dtype=torch.float32
        )
        phaseMask[0, 0] = self.BuildPyramidMask() / 1.57 * 0.80
        phaseMask[0, 1] = self.BuildPyramidMask() * 1.2

        transmisionMask = torch.zeros(
            1, 2, self.N, self.N, device=self.device, dtype=torch.float32
        )
        transmisionMask[0, 0] = self.BuildPyramidMask() / self.param[0]
        transmisionMask[0, 0] /= self.sampling * 8.0
        transmisionMask[0, 0] = torch.clamp(transmisionMask[0, 0], 0, 1)
        transmisionMask[0, 1] = 1.0 - transmisionMask[0, 0]

        return torch.sqrt(transmisionMask), phaseMask

    def DoubleZernikeMask(self):
        phaseMask = torch.zeros(
            1, 2, self.N, self.N, device=self.device, dtype=torch.float32
        )
        phaseMask[0, 0] = (
            self.positions[0, 0] * self.x_mask + self.positions[0, 1] * self.y_mask
        )
        phaseMask[0, 1] = (
            self.positions[1, 0] * self.x_mask + self.positions[1, 1] * self.y_mask
        )

        frame_center = torch.ones(2, 2, device=self.device) * self.N / 2
        pupil_center = frame_center + self.positions / 2 / torch.pi * self.N
        self.pupil_centers = torch.round(pupil_center).to(dtype=torch.int).cpu().numpy()

        slope = 10
        diameters_in_pixels = self.diameters * self.sampling

        ring_mask = (
            torch.tanh(
                slope
                * (
                    diameters_in_pixels.unsqueeze(1).unsqueeze(1) / 2.0
                    - self.rho_mask.unsqueeze(0)
                )
            )
            / 2
        )
        annular = ring_mask + 0.5

        zernike_mask = self.depths.unsqueeze(1).unsqueeze(1) * annular

        phaseMask[0] = phaseMask[0] + zernike_mask

        return phaseMask

    def BuildZernikeMaskMTF(self):
        N = int(self.sampling * self.MTF_focal_upscale * self.diameters[0])
        transmisionMask = torch.zeros(
            1, 1, N, N, device=self.device, dtype=torch.float32
        )
        phaseMask = torch.ones(1, 1, 1, 1, device=self.device, dtype=torch.float32)
        transmisionMask[0, 0] = self.make_pupil(
            self.sampling * self.MTF_focal_upscale * self.diameters[0] / 2, N
        )

        phaseMask[0, 0] = phaseMask[0, 0] * self.depths[0]

        self.WFS.MakeMTFMatrices(self.diameters[0])

        frame_center = np.ones((1, 1)) * self.N // 2
        pupil_center = (
            frame_center + self.positions.detach().cpu().numpy() / 2 / np.pi * self.N
        )
        self.pupil_centers = np.round(pupil_center).astype(np.int32)

        self.WFS.pupil_shifts = (self.pupil_centers - frame_center)[:, 0].astype(
            np.int32
        )

        return phaseMask, transmisionMask

    def DoubleZernikeMaskMTF(self):
        N = int(self.sampling * self.MTF_focal_upscale * self.diameters[0])
        transmisionMask = torch.zeros(
            1, 2, N, N, device=self.device, dtype=torch.float32
        )
        phaseMask = torch.ones(1, 2, 1, 1, device=self.device, dtype=torch.float32)
        transmisionMask[0, 0] = self.make_pupil(
            self.sampling * self.MTF_focal_upscale * self.diameters[0] / 2, N
        )
        transmisionMask[0, 1] = self.make_pupil(
            self.sampling * self.MTF_focal_upscale * self.diameters[1] / 2, N
        )
        phaseMask[0, 0] = phaseMask[0, 0] * self.depths[0]
        phaseMask[0, 1] = phaseMask[0, 1] * self.depths[1]

        self.WFS.MakeMTFMatrices(self.diameters[0])

        frame_center = np.ones((2, 2)) * self.N // 2
        pupil_center = (
            frame_center + self.positions.detach().cpu().numpy() / 2 / np.pi * self.N
        )
        self.pupil_centers = np.round(pupil_center).astype(np.int32)

        self.WFS.pupil_shifts = (self.pupil_centers - frame_center)[:, 0].astype(
            np.int32
        )

        return phaseMask, transmisionMask


class ScaledTanh(nn.Module):
    def forward(self, x):
        return 0.5 * (torch.tanh(x) + 1)


class FreeMaskGenerator(nn.Module):
    """
    Neural network module that generates a phase or transmission mask
    based on 2D input coordinates (u, v).

    Args:
        hidden_size (int): Number of hidden units in the fully connected layers.
        isPhaseMask (bool): Whether the mask should represent a phase mask (True)
                            or a transmission mask (False).
    """

    def __init__(self, hidden_size=128, isPhaseMask=True):
        super().__init__()

        self.isPhaseMask = isPhaseMask

        # self.initialMap = nn.Parameter(torch.randn((1,1,240,240)))

        self.net = nn.Sequential(
            nn.Linear(16, hidden_size),  # Input: (u, v)
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),  # Output: Mask value
        )
        # self.net = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
        #     nn.GELU(),
        #     nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
        #     nn.GELU(),
        #     nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2),
        #     nn.Flatten()
        #     )

        # Apply custom weight initialization
        self.apply(self._init_weights)

        # If it is a transmision mask constrain to 0-1
        self.transmisionSigmoid = nn.Sigmoid()
        self.transmissionTanh = ScaledTanh()

    def _init_weights(self, module):
        """
        Applies custom initialization to the network weights.
        Weights are drawn from a normal distribution and biases are set to a constant.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.1)  # Normal distribution
            nn.init.constant_(module.bias, 0.1)  # Set bias to zero

    def forward(self, uv_coords):
        """
        Evaluates the mask for given UV coordinates.

        Args:
            uv_coords (Tensor): Tensor of shape (N, 2) containing normalized coordinates.

        Returns:
            Tensor: Output mask values (phase or transmission).
        """
        features = []
        L = 4
        for i in range(L):
            f = i**2 * torch.pi
            features.append(torch.sin(f * uv_coords))
            features.append(torch.cos(f * uv_coords))
            # features.append(uv_coords ** i)
        test = torch.stack(features, dim=1).view(-1, 4 * L)

        x = self.net(test)
        # x = self.net(uv_coords)
        # N = np.sqrt(uv_coords.shape[0]).astype(np.int32)
        # x = self.net(0*uv_coords.T.view(1,2,N, N)).squeeze()
        # x = self.net(self.initialMap).squeeze()

        if self.isPhaseMask:
            return x

        # x = self.transmisionSigmoid(x)
        x = self.transmissionTanh(x)
        return x


class ModalMaskGeneration(nn.Module):

    def __init__(self, pupil, device, NumberOfModes=30):
        super().__init__()

        # self.coefs = nn.Parameter(torch.randn(size=(NumberOfModes, 1, 1), device = device) / 100., requires_grad=True)
        self.coefs = nn.Parameter(torch.zeros(NumberOfModes, 1, 1, device=device))
        self.coefs.data[0] = -6.6173
        self.coefs.data[7] = 0.7833
        self.coefs.data[10] = 1.0787

        _, self.modes = Zernike(pupil, j = NumberOfModes)
        self.modes = self.modes.permute(1,0).view(NumberOfModes, *pupil.shape[-2:])

    def forward(self):
        return torch.sum(self.modes * self.coefs, dim=0)


class MaskVisualizator:
    """
    Utility class for visualizing dynamically generated phase and/or transmission masks
    from an End-to-End Wavefront Sensor model.

    Args:
        E2E_WFS: The model containing the mask information and type.
    """

    def __init__(self, E2E_WFS, loss, loss_ideal):
        self.E2E_WFS = E2E_WFS
        self.loss = loss
        self.loss_ideal = loss_ideal

    def SetCanvas(self):
        """
        Initializes the matplotlib canvas and image for mask visualization
        based on the type of mask being used.
        """
        if self.E2E_WFS.maskType in ["FreePhase", "ModalMask", "FullyFreePhase"]:
            self.fig, self.ax = plt.subplots(1, 3, figsize=(21, 5))
            self.img = self.ax[2].imshow(
                self.E2E_WFS.maskManager.phaseMask.cpu().detach().numpy()
            )
            self.fig.colorbar(self.img)
            self.ax[2].set_title("Phase mask")

        elif self.E2E_WFS.maskType == "FreeTransmision":
            self.fig, self.ax = plt.subplots(1, 3, figsize=(21, 5))
            self.img = self.ax[2].imshow(
                self.E2E_WFS.maskManager.transmisionMask.cpu().detach().numpy()
            )
            self.fig.colorbar(self.img)
            self.ax[2].set_title("Transmision mask")

        elif self.E2E_WFS.maskType == "FreePhaseTransmision":
            self.fig, self.ax = plt.subplots(1, 5, figsize=(21, 4))
            self.img1 = self.ax[2].imshow(
                self.E2E_WFS.maskManager.phaseMask.cpu().detach().numpy()
            )
            self.fig.colorbar(self.img1)
            self.img2 = self.ax[3].imshow(
                self.E2E_WFS.maskManager.transmisionMask.cpu().detach().numpy()
            )
            self.fig.colorbar(self.img2)
            self.img2.set_clim(0, 1)

            self.ax[2].set_title("Phase mask")
            self.ax[3].set_title("Transmision mask")

        else:
            self.fig, self.ax = plt.subplots(1, 2, figsize=(14, 5))

        plt.pause(0.3)

        (self.lossPlot,) = self.ax[0].plot(self.loss.cpu().detach())
        (self.lossIdealPlot,) = self.ax[0].plot(self.loss_ideal.cpu().detach())
        (self.reconstructionPlotTheoretical,) = self.ax[1].plot(
            self.loss.cpu().detach()
        )
        (self.reconstructionPlotEstimated,) = self.ax[1].plot(self.loss.cpu().detach())
        self.ax[0].set_title("Loss Evolution")
        self.ax[1].set_title("Sample Reconstruction")

    def show(self):
        """
        Updates and redraws the mask image(s) based on current model parameters.
        Called regularly during training to reflect mask updates.
        """
        if self.E2E_WFS.maskType in ["FreePhase", "ModalMask", "FullyFreePhase"]:
            self.img.set_data(self.E2E_WFS.maskManager.phaseMask.cpu().detach().numpy())
            self.img.set_clim(
                vmin=np.min(self.img.get_array()), vmax=np.max(self.img.get_array())
            )

        elif self.E2E_WFS.maskType == "FreeTransmision":
            self.img.set_data(
                self.E2E_WFS.maskManager.transmisionMask.cpu().detach().numpy()
            )
            self.img.set_clim(
                vmin=np.min(self.img.get_array()), vmax=np.max(self.img.get_array())
            )

        elif self.E2E_WFS.maskType == "FreePhaseTransmision":
            self.img1.set_data(
                self.E2E_WFS.maskManager.phaseMask.cpu().detach().numpy()
            )
            self.img1.set_clim(
                vmin=np.min(self.img1.get_array()), vmax=np.max(self.img1.get_array())
            )

            self.img2.set_data(
                self.E2E_WFS.maskManager.transmisionMask.cpu().detach().numpy()
            )

        plt.pause(0.1)

    def update_plots(self, modeTeo, modeEst):
        clear_output(wait=True)
        smooth_loss = np.convolve(
            self.loss.cpu().detach().numpy(), np.ones(100) / 100, "valid"
        )
        smooth_loss_ideal = np.convolve(
            self.loss_ideal.cpu().detach().numpy(), np.ones(100) / 100, "valid"
        )
        self.lossPlot.set_xdata(np.arange(len(smooth_loss)))
        self.lossPlot.set_ydata(smooth_loss)
        self.lossIdealPlot.set_xdata(np.arange(len(smooth_loss_ideal)))
        self.lossIdealPlot.set_ydata(smooth_loss_ideal)
        self.ax[0].relim()
        self.ax[0].autoscale_view()

        self.reconstructionPlotTheoretical.set_xdata(np.arange(len(modeTeo[0])))
        self.reconstructionPlotTheoretical.set_ydata(modeTeo[0].cpu().detach())
        self.reconstructionPlotEstimated.set_xdata(np.arange(len(modeEst[0])))
        self.reconstructionPlotEstimated.set_ydata(modeEst[0].cpu().detach())
        self.ax[1].relim()
        self.ax[1].autoscale_view()
        display(self.fig, clear=True)


def trainMask(
    maskGenerator, uv_coords, mask, loss, TrainRunNb, optimizer, device="cuda"
):
    """
    Trains the MaskGenerator neural network to match a target mask pattern.
    Generates live plots and saves intermediate mask states into a GIF.

    Args:
        maskGenerator (nn.Module): Neural network to generate the mask.
        uv_coords (Tensor): 2D coordinates input to the mask generator.
        mask (Tensor): Target mask used for loss computation.
        loss (function): Loss function to optimize.
        TrainRunNb (int): Number of training iterations.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        device (str): Device for computation.

    Returns:
        None
    """
    final_train_loss = 0

    fig, ax = plt.subplots(figsize=(10, 10))

    img = ax.imshow(maskGenerator(uv_coords).view(N, N).cpu().detach().numpy())
    fig.colorbar(img)
    plt.show()

    # writer = imageio.get_writer(gif_path, mode="i", fps = 10)

    maskGenerator.train()

    for u in range(0, TrainRunNb):

        optimizer.zero_grad()

        output = maskGenerator(uv_coords).view(N, N)

        l = loss(output, mask.to(output.dtype))

        l.backward()

        optimizer.step()

        # parameters values should change during the loop
        # print(" Run n°  {}, train loss : {:.7f} param_opt 1 :   {:.7f} param_opt 2 :   {:.7f}  param_proc  : {:.7f}\n".format(u, l,Trained_End2EndWFS.WFSmodule.WFS.param[0].item(),Trained_End2EndWFS.WFSmodule.WFS.param[1].item(),Trained_End2EndWFS.PhaseEstimator.param[0,0].item()), end="")
        if u % 100 == 0:
            # print(" Run n°  {}, train loss : {:.7f} param_opt 1 :   {:.7f} param_opt 2 :   {:.7f}  \n".format(u, l,Trained_End2EndWFS.WFSmodule.WFS.param[0].item(),Trained_End2EndWFS.WFSmodule.WFS.param[1].item()), end="")
            print(f" Run n°  {u}, train loss : {l.item():.5f}")
            img.set_data(maskGenerator(uv_coords).view(N, N).cpu().detach().numpy())
            img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))

            fig.canvas.draw()
            image = np.array(fig.canvas.buffer_rgba())
            # writer.append_data(image)

            plt.pause(0.1)
            # loss_tracker.append(l.item())
            # param_tracker.append(Trained_End2EndWFS.WFSmodule.WFS.param.tolist())

        final_train_loss = l + final_train_loss
        # writer.close()
    return


if __name__ == "__main__":

    device = "cuda"  # set to "cpu" if Cuda is not available

    paramfile = "params_exp.py"  # file of experimental parameters

    gif_path = "test_mask_animation.gif"

    mask_path = "Pyramid_mask.pth"
    # Config extraction

    N = 140

    # Setting the loss function
    loss = torch.nn.MSELoss()

    # Initialisation of the system
    maskGenerator = FreeMaskGenerator().to(device)

    u = torch.linspace(-N // 2, N // 2 - 1, N) / (N / 2)  # Normalized frequency range
    U, V = torch.meshgrid(u, u, indexing="xy")  # Create the full grid

    pyr_mask = (np.pi / 2 * (torch.abs(U) + torch.abs(V)) * N / 2).to(device)
    zernike_mask = np.pi / 2 * (torch.sqrt(U**2 + V**2) < 4 / N).to(device)
    random_mask = torch.randn_like(zernike_mask)

    uv_coords = torch.stack([U.flatten(), V.flatten()], dim=1).to(device)

    # Flatten and stack into (N^2, 2) shape

    # Optimization parameters (learning rate lr and nb of runs)

    optimizer = torch.optim.Adam(maskGenerator.parameters(), 0.001)

    a = time.time()
    train_loss = trainMask(
        maskGenerator, uv_coords, pyr_mask, loss, 3000, optimizer, device
    )
    b = time.time() - a

    torch.save(
        {
            "Phase_Mask_state_dict": maskGenerator.state_dict(),
            "optimizer_o_state_dict": optimizer.state_dict(),
        },
        mask_path,
    )
