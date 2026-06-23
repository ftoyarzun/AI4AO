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
        "Nres": 80,  # Number of pixels in the aperture of the telescope
        "sampling": 240 / 80,  # Zero-padding factor (2 is Shannon)
        "D": 0.6,  # Telescope diameter (m)
        "Nphotons": [4.5, 6],  # Log range of number of photons in measurement
        "RON": [1, 2],  # Read-out noise in photons per pixel per frame
        "Nmodes": 150,  # Number of modes to reconstruct
        "Nactuator": 17,  # Number of actuators across the diameter of the DM
        "ModalBasis": "Zernike",  # Basis used for the reconstruction. The full list can be found in Constants.py in basis_list
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
        "Wavelength": 550e-9, # Sensing wavelength (m) 
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
