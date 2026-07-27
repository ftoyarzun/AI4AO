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
        "Nres":36,
        "sampling":96/36,
        "D": 0.6,
        "useNoise": True,
        "Modulation": 0,
        "Wavelength": 635e-9,
        "Nphotons": [4.5, 6],  # Log range of number of photons in measurement
        "RON": [1, 3],  # Read-out noise in photons per pixel per frame
        "Substract_Reference": True,  # Substract or not the reference intensity frame
        "Extract_pupils_pad": 6,
        "Center_noise": 2,
        "Bin_factor": 1
    }
)


## Atmosphere parameters
AtmosParams = dict(
    {
        "r0": [0.05, 0.2],  # Fired parameter range (m)
        "L0": [20, 30.0000],  # Outter scale range (m)
        "Nphases": 16,  # Number of phases in the batch
        "Layers": [5, 10],  # Number of layers in phase range
        "f_slope": 11.0 / 6.0,  # Slope of the spectrum (11/6) is the default
        "Scintillation": False # Use or not scinitillation
    }
)


## Loop parameters
LoopParams = dict(
    {
        "loopFrequency": 500,
        "delayFrames": 1,
        "windSpeedVector": [1, 10],
        "levelOfCorrection": [0.0, 1.0],
        "loopGain": [0.2, 0.5],
        "loopLeak": [0.9,1.] 
    }
)

## Atmosphere parameters
DMParams = dict(
    {
        "Nactuator": 11,
        "Nmodes": 50,
        "moffatParam": 2,
        "signedAmplitude": -5e-6,
        "MechCoupling": 0.36,
        "FlipLeftRight": False,
        "FlipTopBottom": False
    }
)

TrainParams = dict(
    {
        "lro": 2e-4,
        "lrn": 1e-4,
        "TrainRunNb": 5000,
        "TestRunNb": 10,
        "OptimizeMask": False,
    }
)
