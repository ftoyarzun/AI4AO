#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared parameter file for the concurrent dual-sensor fusion AO simulation
(ConcurrentDualSensorFusion.ipynb): a modulated Pyramid WFS (700 nm) and a
ZWFS (1100 nm) observe the *same* residual OPD at the *same* tick rate and
drive one shared DM through one fusion reconstructor. This is a fictional
demo instrument (no real bench behind it), built purely to demonstrate the
concurrent dual-sensor-fusion architecture -- not a calibrated twin.

Unlike TwoStageAO_params.py (two sequential stages at different wavelengths
and different tick rates, needing a per-stage r0 rescale and two LoopParams),
this notebook only needs ONE AtmosParams/LoopParams/DMParams: both sensors
see the identical physical OPD screen concurrently, so there is nothing to
rescale between them and only one shared DM/M2C to fit.

WFSParams holds only what is genuinely shared by both sensors (the one
shared telescope, detector-noise, and frame-preprocessing settings) -- the
notebook copies it once per sensor and adds the sensor-specific keys itself
(Wavelength/Modulation for the Pyramid, Wavelength/MaskType/Use_MTF/
MTF_upscale for the ZWFS), exactly as TwoStageAO_params.py does.
"""

# %% Set general parameters

## WFS and telescope parameters shared by both sensors
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
        "Pupil_size_noise": 0.03,  # +/-3% pupil size jitter
        "Bin_factor": 1,
    }
)

## Atmosphere parameters (one shared atmosphere seen by both sensors concurrently)
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

## Loop parameters -- a single shared tick rate: both sensors and the one
## shared DM run synchronously, unlike TwoStageAO's two independent rates.
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

## The one shared DM, driven jointly by both sensors through the fusion
## reconstructor. "Nmodes" is left as a placeholder -- the notebook reads
## dm.totalAct back after construction and uses it as the Zernike modal M2C
## size (dm.MakeZernikeM2C(nModes=...)), exactly like Tutorials/basics/03's
## pattern, rather than guessing a mode count ahead of time.
DMParams = dict(
    {
        "Nactuator": 15,
        "moffatParam": 2,
        "signedAmplitude": -5e-6,
        "MechCoupling": 0.36,
        "FlipLeftRight": False,
        "FlipTopBottom": False,
        "Nmodes": None,  # placeholder, resolved in the notebook from dm.totalAct
    }
)

TrainParams = dict(
    {
        "lrn": 1e-4,
        "TrainRunNb": 5000,  # optimizer steps for each of the 3 trained reconstructors (baseline A, baseline B, fused)
        "ClosedLoopIterations": 1,  # BPTT window per optimizer step
    }
)
