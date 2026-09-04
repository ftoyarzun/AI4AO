#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared parameter file for the two-stage AO simulation (TwoStageAO.ipynb):
a modulated Pyramid WFS + "woofer" DM as stage 1, feeding a vZWFS + "tweeter"
DM as stage 2. This is a fictional demo instrument (no real bench behind it),
built purely to demonstrate the two-stage architecture, not a calibrated twin.

WFSParams holds only what is genuinely shared by both stages (the one shared
telescope, detector-noise, and frame-preprocessing settings) -- the notebook
copies it once per stage and adds the WFS-specific keys itself
(Wavelength/Modulation for the Pyramid, Wavelength/MaskType/Use_MTF/MTF_upscale
for the vZWFS).

LoopParams sets the *fast* tick rate (matching stage 2 / the vZWFS+DM2 loop);
PhaseDataset uses loopFrequency purely to convert wind speed into how far the
phase screen translates from one dataset[idx] call to the next. Stage 1 (the
Pyramid+DM1 loop) runs N times slower than this, so the notebook derives
LoopParams1 = copy of LoopParams with loopFrequency divided by
TrainParams['SpeedRatioN'].
"""
import numpy as np

# %% Set general parameter


## WFS and telescope parameters shared by both stages
WFSParams = dict(
    {
        "Nres": 40,
        "sampling": 3,
        "D": 1.8,
        "centralObstruction": 0.3,
        "useNoise": True,
        "Nphotons": [4.5, 6],  # Log range of number of photons in measurement
        "RON": [1, 3],  # Read-out noise in photons per pixel per frame
        "Substract_Reference": False,  # Substract or not the reference intensity frame
        "Extract_pupils_pad": 6,
        "Center_noise": 2,
        "Pupil_size_noise": 0.03,  # +/-5% pupil size jitter
        "Bin_factor": 1,
    }
)


## Atmosphere parameters (one shared atmosphere ahead of both stages)
AtmosParams = dict(
    {
        "r0": [0.05, 0.2],  # Fried parameter range (m)
        "L0": [20, 30.0000],  # Outter scale range (m)
        "Nphases": 16,  # Number of phases in the batch
        "Layers": [5, 10],  # Number of layers in phase range
        "f_slope": 11.0 / 6.0,  # Slope of the spectrum (11/6) is the default
        "Scintillation": False,  # Use or not scintillation
    }
)


## Loop parameters at the *fast* (stage 2) tick rate -- see module docstring
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

## DM1: the "woofer" stage, driven by the modulated Pyramid at the slow rate
DMParams1 = dict(
    {
        "Nactuator": 15,
        "moffatParam": 2,
        "signedAmplitude": -5e-6,
        "MechCoupling": 0.36,
        "FlipLeftRight": False,
        "FlipTopBottom": False,
    }
)

## DM2: the "tweeter" stage, driven by the vZWFS at the fast rate
DMParams2 = dict(
    {
        "Nactuator": 7,
        "moffatParam": 2,
        "signedAmplitude": -5e-6,
        "MechCoupling": 0.36,
        "FlipLeftRight": False,
        "FlipTopBottom": False,
    }
)

TrainParams = dict(
    {
        "lro": 2e-4,
        "lrn": 1e-4,
        "TrainRunNb1": 5000,  # stage 1 (Pyramid/DM1) optimizer steps
        "TrainRunNb2": 5000,  # stage 2 (vZWFS/DM2) optimizer steps
        "ClosedLoopIterations1": 1,  # BPTT window for stage 1 training
        "ClosedLoopIterations2": 1,  # BPTT window for stage 2 training (>= SpeedRatioN to see a stage-1 update)
        "SpeedRatioN": 2,  # stage 2 runs this many times faster than stage 1
        "TestRunNb": 10,
        "OptimizeMask": False,
    }
)
