#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared parameter file for the CNN architecture comparison notebook
(CNNArchitectureComparison.ipynb): a single modulated Pyramid WFS driving one
DM. This is a fictional demo instrument (no real bench behind it), built
purely to compare five reconstructor architectures under identical
atmosphere/noise/DM/loss/training-budget conditions -- see
Ideas/07-cnn-architecture-comparison.md for the full motivation.

Unlike DualSensorFusion_params.py/TwoStageAO_params.py, there is only ever
one WFS and one DM here, so WFSParams carries the sensing wavelength and
modulation directly rather than being copied/extended per sensor.
"""

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
        "Extract_pupils_pad": 6,
        "Center_noise": 2,
        "Pupil_size_noise": 0.03,  # +/-3% pupil size jitter
        "Bin_factor": 1,
    }
)

## Atmosphere parameters
AtmosParams = dict(
    {
        "r0": [0.05, 0.2],  # Fried parameter range (m), at the 500nm PSD reference wavelength
        "L0": [20, 30.0000],  # Outter scale range (m)
        "Nphases": 16,  # Number of phases in the batch
        "Layers": [5, 10],  # Number of layers in phase range
        "f_slope": 11.0 / 6.0,  # Slope of the spectrum (11/6) is the default
        "Scintillation": False,  # Use or not scintillation
    }
)

## Loop parameters
LoopParams = dict(
    {
        "loopFrequency": 1000,
        "delayFrames": 1,
        "windSpeedVector": [1, 10],
        "levelOfCorrection": [0.0, 1.0],
        "loopGain": [0.2, 0.5],
        "loopLeak": [0.9, 1.0],
    }
)

## DM driven by the Pyramid. "Nmodes" is a fixed integer here (rather than the
## None-placeholder pattern DualSensorFusion_params.py uses), matching
## Tutorials/basics/wfs_params_exp.py's convention: DeformableMirror.MakeZernikeM2C()
## defaults to nModes = DMParams["Nmodes"], and every architecture's Nmodes-sized
## head below reads this same key directly.
DMParams = dict(
    {
        "Nactuator": 11,
        "Nmodes": 64,
        "moffatParam": 2,
        "signedAmplitude": -5e-6,
        "MechCoupling": 0.36,
        "FlipLeftRight": False,
        "FlipTopBottom": False,
    }
)

TrainParams = dict(
    {
        "lrn": 1e-4,
        "TrainRunNb": 2000,  # optimizer steps for EACH of the 5 trained reconstructors
        "ClosedLoopIterations": 1,  # BPTT window per optimizer step, matching 05_TrainingAReconstructor.ipynb
        "TestRunNb": 200,  # closed-loop rollout length for the seeded nm-RMS comparison
        "Seed": 1234,  # reset before each architecture's closed-loop rollout, so all five see identical draws
    }
)
