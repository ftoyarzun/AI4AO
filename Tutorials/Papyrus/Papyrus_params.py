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
        "Nres":76,
        "sampling":240/76,
        "D": 1.52,
        "useNoise": False,
        "Modulation": 3,
        "Wavelength": 635e-9
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
        "Nactuator": 17,
        "moffatParam": 2,
        "signedAmplitude": -5e-6,
        "MechCoupling": 0.36,
        "FlipLeftRight": False,
        "FlipTopBottom": False
    }
)
