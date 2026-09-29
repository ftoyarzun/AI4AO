#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter file for the self-shaping-mask notebook (SelfShapingMask.ipynb):
a fully free-form, learned WFS phase mask trained jointly with a CNN
reconstructor (condition A), compared against a classic modulated Pyramid
WFS run with both its own trained CNN reconstructor (condition B) and its
frozen linear reconstructor (condition C). This is a fictional demo
instrument (no real bench behind it).

WFSParams/AtmosParams/LoopParams/DMParams are copied verbatim from
ResidualCNNReconstructor_params.py so results are directly comparable to
that notebook's own Pyramid conditions, rather than introducing yet another
incomparable fictional instrument. See Ideas/01-self-shaping-mask.md for the
full design discussion.
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

## DM shared by every condition (self-shaping-mask and both Pyramid
## conditions) -- the physical mirror is identical across the comparison.
## "Nmodes" is a fixed integer: DeformableMirror.MakeZernikeM2C() reads it
## directly, and the same M2C is reused as the DM command basis everywhere.
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

## Training parameters. lr_mask/lr_cnn_A/lr_cnn_B are three separate
## optimizers -- the raw, unscaled mask phase, condition A's CNN weights, and
## condition B's CNN weights all sit at different scales. MaskUpdatePeriod
## ("k" in Ideas/01) unifies joint and alternating training: k=1 makes every
## step a true joint mask+CNN update from one shared backward pass; k>1 holds
## the mask fixed for k-1 steps between joint updates (genuine alternating).
TrainParams = dict(
    {
        "lr_mask": 1e-3,
        "lr_cnn_A": 1e-4,
        "lr_cnn_B": 1e-4,
        "MaskUpdatePeriod": 1,
        "TrainRunNb": 2000,  # optimizer steps for the combined A+B training loop
        "ClosedLoopIterations": 1,  # BPTT window per optimizer step
        "TestRunNb": 200,  # closed-loop rollout length for the seeded nm-RMS comparison
        "Seed": 1234,  # reset before each condition's closed-loop rollout, so all three see identical draws
    }
)
