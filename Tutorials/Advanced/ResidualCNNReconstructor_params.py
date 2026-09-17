#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared parameter file for the residual-CNN reconstructor notebook
(ResidualCNNReconstructor.ipynb): a single modulated Pyramid WFS driving one
DM. This is a fictional demo instrument (no real bench behind it), built to
demonstrate a residual CNN that complements -- rather than replaces -- a
frozen, pre-calibrated linear reconstructor. See
Ideas/10-residual-cnn-linear-complement.md for the full motivation.

Deliberately identical to CNNArchitectureComparison_params.py's instrument
(same nominal Pyramid/DM geometry), since this notebook directly reuses that
notebook's CNN-only condition as one arm of its own three-way ablation
(Linear-only / CNN-only / Linear+Residual-CNN).
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

## DM driven by the Pyramid. "Nmodes" is a fixed integer, matching
## CNNArchitectureComparison_params.py's convention: DeformableMirror.MakeZernikeM2C()
## reads it directly, and the same M2C is reused both as the DM command basis and
## (via dm(M2C.T)) as the linear reconstructor's calibration basis -- see the notebook.
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
        "TrainRunNb": 2000,  # optimizer steps for EACH of the 2 trained conditions (CNN-only, Linear+Residual-CNN)
        "ClosedLoopIterations": 1,  # BPTT window per optimizer step, matching 05_TrainingAReconstructor.ipynb
        "TestRunNb": 200,  # closed-loop rollout length for the seeded nm-RMS comparison
        "Seed": 1234,  # reset before each condition's closed-loop rollout, so all three see identical draws
    }
)
