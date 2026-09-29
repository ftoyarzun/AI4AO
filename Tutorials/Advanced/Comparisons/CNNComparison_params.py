#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% Set general parameters

## WFS and telescope parameters
WFSParams = dict(
    {
        "Nres": 40,
        "sampling": 3,
        "D": 1.8,
        "centralObstruction": 0.3,
        "useNoise": True,
        "Modulation": 0,
        "Wavelength": 700e-9,
        "Nphotons": [4.5, 6],  # Log range of number of photons in measurement
        "RON": [1, 3],  # Read-out noise in photons per pixel per frame
        "Substract_Reference": False,  # Substract or not the reference intensity frame
        "Extract_pupils_pad": 2,
        "Center_noise": 0,
        "Pupil_size_noise": 0.00,  # +/-3% pupil size jitter
        "Bin_factor": 1,
    }
)

## Atmosphere parameters
AtmosParams = dict(
    {
        "r0": [0.02, 0.1],  # Fried parameter range (m), at the 500nm PSD reference wavelength
        "L0": [10., 30.],  # Outter scale range (m)
        "Nphases": 16,  # Number of phases in the batch
        "Layers": [5, 10],  # Number of layers in phase range
        "f_slope": 11.0 / 6.0,  # Slope of the spectrum (11/6) is the default
        "Scintillation": False,  # Use or not scintillation
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
        "loopLeak": [0.9, 1.0],
    }
)

## DM driven by the Pyramid -- see ../Reconstructors/FrameDenoiser_params.py's own
## comment for why "Nmodes" is a fixed integer here.
DMParams = dict(
    {
        "Nactuator": 17,
        "Nmodes": 190,
        "moffatParam": 2,
        "signedAmplitude": 5e-6,
        "MechCoupling": 0.36,
        "FlipLeftRight": False,
        "FlipTopBottom": False,
    }
)

TrainParams = dict(
    {
        "lrn": 1e-4,
        "TrainRunNb": 10000,  # optimizer steps for EACH of the 4 trained sizes
        "ClosedLoopIterations": 1,  # BPTT window per optimizer step, matching 05_TrainingAReconstructor.ipynb
        "TestRunNb": 200,  # closed-loop rollout length for the seeded nm-RMS comparison
        "Seed": 1234,  # reset before each size's closed-loop rollout, so all four see identical draws
    }
)
