#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared parameter file for the CNN size comparison notebook
(CNNSizeComparison.ipynb): the same fictional-demo modulated Pyramid WFS + DM
as CNNArchitectureComparison_params.py, but here the architecture is held
fixed (ClassicCNN) and only its capacity (base_channels, hence trainable
parameter count) varies -- see CNNArchitectureComparison.ipynb/
Ideas/07-cnn-architecture-comparison.md for the shared instrument/training
setup this notebook reuses, and Tutorials/Advanced/CNNSizeComparison.ipynb
for the size-specific comparison itself.

Kept as its own params file (rather than importing CNNArchitectureComparison's)
to stay self-contained and independently editable, following this folder's
existing one-params-file-per-notebook convention (e.g. DualSensorFusion_params.py
and TwoStageAO_params.py duplicate a similar amount of shared structure).
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

## DM driven by the Pyramid -- see CNNArchitectureComparison_params.py's own
## comment for why "Nmodes" is a fixed integer here.
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
        "TrainRunNb": 2000,  # optimizer steps for EACH of the 4 trained sizes
        "ClosedLoopIterations": 1,  # BPTT window per optimizer step, matching 05_TrainingAReconstructor.ipynb
        "TestRunNb": 200,  # closed-loop rollout length for the seeded nm-RMS comparison
        "Seed": 1234,  # reset before each size's closed-loop rollout, so all four see identical draws
        # ClassicCNN's base_channels for each of the 4 compared sizes, chosen (by an
        # actual parameter-count sweep, see CNNSizeComparison.ipynb) to roughly
        # log-span ~100k to ~3M trainable parameters at this instrument's Nout=46:
        # 10 -> ~120k, 18 -> ~381k, 30 -> ~1.05M, 50 -> ~2.89M.
        "BaseChannelsList": [8, 16, 32, 64],
    }
)
