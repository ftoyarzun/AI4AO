# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 09:41:35 2025

@author: foyarzun
"""

from AI4AO.TorchPropagator import WFS
from AI4AO.PhaseDataset import Zernike

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

        return self.WFS.GetReconstructedPhase(image)


if __name__ == "__main__":
    print("Nothing to do here")
