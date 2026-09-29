# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:37:06 2025

@author: franc
"""

import torch # type: ignore[import]
import torch.nn as nn # type: ignore[import]

import numpy as np
import matplotlib.pyplot as plt
import time

from AI4AO.PhaseDataset import Zernike
from AI4AO.Utils import MakePupil


class MaskManager(nn.Module):

    def __init__(self, ParamsDict, device):

        super().__init__()
        self.device = device
        self.sampling = ParamsDict["sampling"]
        self.Nres = ParamsDict["Nres"]
        self.N = int(self.Nres * self.sampling)

        # Build uv grid for freeform mask types
        self._build_uv_grid()
        self._build_xy_grid()

        # Initialize mask generators
        self.phaseMask = None
        self.transmisionMask = None


    def _build_uv_grid(self):
        u = torch.linspace(-1, 1 - 2 / self.N, self.N, device=self.device)
        U, V = torch.meshgrid(u, u, indexing="xy")
        self.UV = torch.stack([U.flatten(), V.flatten()], dim=1)  # (N², 2)
        self.circ_mask = (torch.sqrt(U**2 + V**2) < 0.9).flatten()  # (N²,)

    def _build_xy_grid(self):
        x_mask = torch.linspace(-self.N / 2, self.N / 2 - 1, self.N, dtype=torch.float32, device=self.device)
        [self.x_mask, self.y_mask] = torch.meshgrid(x_mask, x_mask)

        self.rho_mask = torch.sqrt(self.x_mask**2 + self.y_mask**2)
        self.abs_x_mask = torch.abs(self.x_mask)
        self.abs_y_mask = torch.abs(self.y_mask)

    def make_mask(self):
        raise NotImplementedError

    def forward(self, wfs):
        self.make_mask()
        wfs.SetMask(phaseMask=self.phaseMask, transmisionMask=self.transmisionMask)


    def BuildBiOEdgeMask(self):

        return self.DoubleTransmisionMask(
            self.linear_ramp(self.x_mask, self.param[0]),
            self.linear_ramp(self.y_mask, self.param[0]),
        )


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


class FreeMaskGenerator(MaskManager):
    
    def __init__(self, ParamsDict, number_of_masks=1, device = 'None'):
        super().__init__(ParamsDict, device)
        self.basePhaseMask = nn.Parameter(0.01*torch.randn((number_of_masks, self.N, self.N), device=self.device))
        if number_of_masks == 1:
            self.push = 1
        else:
            steps = torch.linspace(0, 2*torch.pi * (1 - 1/number_of_masks), number_of_masks, device = self.device)[:,None, None]
            self.push = torch.pi/2 * (self.x_mask * torch.cos(steps) + self.y_mask * torch.sin(steps))
    def make_mask(self):
        self.phaseMask = self.push + self.basePhaseMask

class FeatureMaskGenerator(MaskManager):

    def __init__(self, ParamsDict, features = 4, hidden_size=128, device = 'None'):
        super().__init__(ParamsDict, device)

        fourierModes = [self.UV]
        for i in range(features):
            f = i*2 * torch.pi
            fourierModes.append(torch.sin(f * self.UV))
            fourierModes.append(torch.cos(f * self.UV))
        self.fourierModes = torch.stack(fourierModes, dim=1).view(-1, 4 * features + 2)

        self.net = nn.Sequential(
            nn.Linear(self.fourierModes.shape[-1], hidden_size),  # Input: (u, v)
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),  # Output: Mask value
        )

        # Apply custom weight initialization
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """
        Applies custom initialization to the network weights.
        Weights are drawn from a normal distribution and biases are set to a constant.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.1)  # Normal distribution
            nn.init.constant_(module.bias, 0.1)  # Set bias to zero

    def make_mask(self):
        self.phaseMask = self.net(self.fourierModes).view(self.N, self.N)


class ModalMaskGeneration(MaskManager):

    def __init__(self, ParamsDict, NumberOfModes=30, device = 'None'):
        super().__init__(ParamsDict, device)

        # self.coefs = nn.Parameter(torch.randn(size=(NumberOfModes, 1, 1), device = device) / 100., requires_grad=True)
        self.coefs = nn.Parameter(torch.zeros(NumberOfModes, 1, 1, device=self.device))
        self.coefs.data[0] = -6.6173
        self.coefs.data[7] = 0.7833
        self.coefs.data[10] = 1.0787

        pupil = MakePupil(self.N, device=self.device)
        _, self.modes = Zernike(pupil, j = NumberOfModes+2)
        self.modes = self.modes[2:]

    def make_mask(self):
        self.phaseMask = torch.sum(self.modes * self.coefs, dim=0)

        