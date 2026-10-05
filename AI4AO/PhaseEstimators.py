# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 09:41:35 2025

@author: foyarzun
"""

from AI4AO.TorchPropagator import WFS
from AI4AO.PhaseDataset import Zernike

import numpy as np
import torch.nn as nn # type: ignore[import]
import torch # type: ignore[import]

import torch.nn.functional as F  # type: ignore[import]

class SimpleNet(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()

        Nmodes = wfsParams["Nmodes"]

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11, padding=7),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=7, padding=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=2, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.1),
        )

        self.outputlayer = nn.Sequential(
            nn.Linear(512, Nmodes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder(x[:, 0].unsqueeze(1))
        x = self.outputlayer(x)
        return x


class Papyrus1stStage(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()

        Nmodes = wfsParams["Nmodes"]
        self.binning = wfsParams["Bin_factor"]
        self.bin_step = nn.AvgPool2d(self.binning)

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=11, padding=7),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=7, padding=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=2, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.1),
        )

        self.outputlayer = nn.Sequential(
            nn.Linear(512, Nmodes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.binning > 1:
            x = self.bin_step(x)
        x = self.encoder(x)
        x = self.outputlayer(x)
        return x


class Rama(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()

        Nmodes = wfsParams["Nmodes"]

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=7, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=7, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(0.1),
        )

        self.outputlayer = nn.Sequential(
            nn.Linear(128*4*4, Nmodes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.outputlayer(x)
        return x


class Papyrus2ndStage(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()

        Nmodes = wfsParams["Nmodes"]
        self.binning = wfsParams["Bin_factor"]
        self.bin_step = nn.AvgPool2d(self.binning)

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=11, padding=7),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=7, padding=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=2, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.1),
        )

        self.outputlayer = nn.Sequential(
            nn.Linear(512, Nmodes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.binning > 1:
            x = self.bin_step(x)
        x = self.encoder(x)
        x = self.outputlayer(x)
        return x


def pyramid_combinations(pupil_centers):
    """(4, 4) orthonormal map from the four Pyramid pupil images to total flux, x-slope,
    y-slope and the cross term, built from the pupil layout (`pupil_centers` rows are
    [y, x], e.g. `wfs.pupil_centers`). Raises if the pupils are not on a 2x2 grid."""
    c = torch.as_tensor(np.asarray(pupil_centers), dtype=torch.float32)
    if c.shape != (4, 2):
        raise ValueError(f"pyramid_combinations needs 4 pupil centers, got shape {tuple(c.shape)}")
    sy = torch.sign(c[:, 0] - c[:, 0].mean())
    sx = torch.sign(c[:, 1] - c[:, 1].mean())
    H = 0.5 * torch.stack([torch.ones_like(sx), sx, sy, sx * sy])
    if not torch.allclose(H @ H.T, torch.eye(4)):
        raise ValueError("the four pupil centers are not on a 2x2 grid")
    return H


class StemEncoderCNN(nn.Module):
    """
    Pyramid-WFS reconstructor: preprocessed pupils (B, n_channels, Nout, Nout) -> modal
    coefficients (B, Nmodes).

    A grouped-conv stem processes each pupil image independently (11x11 then 7x7, one
    group per pupil), then VGG-style stages (two 3x3 convs + 2x max-pool, channels
    doubling) mix the pupils, and a linear head maps the features to the modes.

    The defaults are `BestCNN` (`Tutorials/Advanced/Comparisons/BestCNN.ipynb`: base 12,
    ELU, all switches off, 2.79M parameters for 4 pupils of 42x42 and 190 modes); with
    them the layer layout and state_dict keys match that notebook, so
    `Data/BestCNN/BestCNN.pth` loads directly. The switches come from
    `Ideas/18-SearchingForABetterCNN.md`:

    cross_pupil : learnable 1x1 conv in front, initialized to `pyramid_combinations`
        (flux, x-slope, y-slope, cross term). Needs `pupil_centers` and 4 pupils.
    ceil_pool : MaxPool2d(2, ceil_mode=True), so odd-sized maps keep their last row/column.
    spatial_head : drop the last stage's pool and the global average pool; a 1x1 conv
        squeezes the last map to `head_channels`, flattened into the linear head (keeps
        where features are instead of averaging them away).
    max_channels : stop doubling the stage width at this value (None: no cap).

    The compact configuration from that study, 1,005,170 parameters, about 1.8 nm RMS
    behind BestCNN in closed loop under the same training recipe:
    `StemEncoderCNN(4, 190, 42, pupil_centers=wfs.pupil_centers, cross_pupil=True,
    ceil_pool=True, spatial_head=True, head_channels=32, max_channels=128)`.

    The stem is `self.stem`, so it can get its own optimizer parameter group: the study's
    recipe uses 5x the base learning rate there (see `Ideas/19-training-good-practices.md`).
    """

    def __init__(self, n_channels, Nmodes, Nout, base_channels=12, activation=nn.ELU, pupil_centers=None,
                 cross_pupil=False, ceil_pool=False, spatial_head=False, head_channels=32, max_channels=None):
        super().__init__()

        def pool():
            return nn.MaxPool2d(2, ceil_mode=ceil_pool)

        self.front = nn.Identity()
        if cross_pupil:
            if pupil_centers is None:
                raise ValueError("cross_pupil=True needs pupil_centers (e.g. wfs.pupil_centers)")
            if n_channels != 4:
                raise ValueError(f"cross_pupil=True needs 4 Pyramid pupils, got n_channels={n_channels}")
            self.front = nn.Conv2d(n_channels, n_channels, kernel_size=1)
            with torch.no_grad():
                self.front.weight.copy_(pyramid_combinations(pupil_centers)[:, :, None, None])
                self.front.bias.zero_()

        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, base_channels * n_channels, kernel_size=11, padding=5, groups=n_channels),
            activation(),
            nn.Conv2d(base_channels * n_channels, 2 * base_channels * n_channels, kernel_size=7, padding=3,
                      groups=n_channels),
            activation(),
            pool(),
        )

        # One stage per halving of the stem's output map until it would drop below 2 pixels.
        n_stages, size = 0, Nout // 2
        while size // 2 >= 2:
            size //= 2
            n_stages += 1
        stem_out_channels = 2 * base_channels * n_channels
        channels = [stem_out_channels] + [stem_out_channels * 2 ** i for i in range(n_stages)]
        if max_channels is not None:
            channels = [min(c, max_channels) for c in channels]

        layers = []
        for i in range(n_stages):
            layers += [
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                activation(),
                nn.Conv2d(channels[i + 1], channels[i + 1], kernel_size=3, padding=1),
                activation(),
            ]
            if not (spatial_head and i == n_stages - 1):
                layers.append(pool())
        if spatial_head:
            layers += [nn.Conv2d(channels[-1], head_channels, kernel_size=1), activation()]
        else:
            layers.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*layers)

        with torch.no_grad():
            probe = torch.zeros(1, n_channels, Nout, Nout)
            n_features = self.encoder(self.stem(self.front(probe))).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_features, Nmodes))

    def forward(self, x):
        return self.head(self.encoder(self.stem(self.front(x))))


class LinearEstimator(nn.Module):

    def __init__(self, WFS: WFS):

        super().__init__()
        self.WFS = WFS

    def forward(self, image):

        ## Build the reconstruction matrix for each forward (because it depends on the optimized parameters)
        # self.WFS.BuildReconstructionMatrix(self.z_FullRes, self.WFS.mask)
        if self.training:
            self.WFS.BuildReferenceIntensity()
            self.WFS.BuildReconstructionMatrix(self.z_FullRes)

        return self.WFS.GetReconstructedOPD(image)


if __name__ == "__main__":
    print("Nothing to do here")
