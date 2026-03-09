#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:25:13 2025

@author: ptrouve


"""
import numpy as np

# %% Set general parameter


## WFS and Telescope parameters
WFSParams = dict(
    {
        "Nres": 78,  # 40,                                                              # Number of pixels in the aperture of the telescope
        "sampling": 240
        / 78,  # 6,                                                        # Zero-padding factor (2 is Shannon)
        "D": 1.52,  # Telescope diameter (m)
        "Modulation": 0,  # Modulation in \lambda/D
        "Nphotons": [4.5, 6],  # Log range of number of photons in measurement
        "RON": [1, 2],  # Read-out noise in photons per pixel per frame
        "Nzernike": 150,  # Number of modes to reconstruct
        "Nactuator": 17,  # Number of actuators across the diameter of the DM
        "useNoise": True,  # use photon and read-out noise
        "InitParam": [1.57, 1.57],  # wavefront sensor specific parameters
        "MaskType": "Papyrus",  # Type of wfs. The full list can be found in Constants.py in mask_types_list
        "Reconstruction": "Papyrus",  # Reconstruction algorithm. The full list can be found in Constants.py in reconstruction_types_list
        "beamSplitProportionForWFSDetector": 1.0,  # Add the effects of a beam splitter to assign the correct amount of light to the wfs frame and psf frame
        "ModalBasis": "Papyrus_KL",  # Basis used for the reconstruction. The full list can be found in Constants.py in basis_list
        "Substract_Reference": True,  # Substract or not the reference intensity frame
        "Extract_pupils": True,
        "Bin_factor": 1,
        "Center_noise": 2,
        "Use_MTF": True,
        "MTF_upscale": 10,
    }
)


## Atmosphere parameters
AtmosParams = dict(
    {
        "r0": [0.05, 0.2],  # Fired parameter range (m)
        "L0": [20, 30.0000],  # Outter scale range (m)
        "Nphases": 16,  # Number of phases in the batch
        "Layers": [3, 5],  # Number of layers in phase range
        "f_slope": 11.0 / 6.0,  # Slope of the spectrum (11/6) is the default
    }
)


## Loop parameters
LoopParams = dict(
    {
        "loopFrequency": 1000,
        "delayFrames": 1,
        "windSpeedVector": [-10, 10],
        "levelOfCorrection": [0.0, 1.0],
    }
)

## Train parameters
TrainParams = dict(
    {
        "lro": 2e-4,
        "lrn": 1e-4,
        "TrainRunNb": 5000,
        "TestRunNb": 10,
        "OptimizeMask": False,
    }
)
