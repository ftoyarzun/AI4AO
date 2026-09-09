#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter file for the temporal-predictive reconstructor notebook
(TemporalPredictiveReconstructor.ipynb): a single modulated Pyramid WFS whose
last N frames are used to predict the modal DM correction M frames ahead, to
compensate for the AO loop's own structural latency (see
Ideas/05-temporal-predictive-reconstructor.md). This is a fictional demo
instrument (no real bench behind it), following the same "no real bench,
nominal geometry" pattern as TwoStageAO_params.py/DualSensorFusion_params.py.

Unlike DualSensorFusion_params.py, there is only one WFS here, so there is no
second sensor-specific dict to split off -- WFSParams already carries
Wavelength/Modulation directly.

N and M in TrainParams are the user's suggested starting point for this
fictional instrument's loop dynamics (window length / frames-ahead horizon),
not values derived from any real measured latency -- see the idea doc's
"Known risks" section for why these should be treated as swept
hyperparameters, not tuned defaults.
"""

# %% Set general parameters

## WFS and telescope parameters (single Pyramid WFS, no bench calibration)
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
        "Wavelength": 700e-9,
        "Modulation": 0,
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

## Loop parameters -- only used for the closed-loop leaky-integrator EVALUATION
## rollout (gain/leak draws); training itself is open-loop supervised and does
## not touch these. delayFrames stays at the repo-wide default (1) -- it only
## shapes a synthetic closed-loop-residual PSD when generateClosedLoop=True
## (never during this notebook's training), and is NOT the latency this
## notebook targets -- see the idea doc's "Known risks" section.
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

## The one shared DM. "Nmodes" is left as a placeholder -- the notebook reads
## dm.totalAct back after construction and uses it as the Zernike modal M2C
## size (dm.MakeZernikeM2C(nModes=...)).
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
        "lrn": 2e-4,
        "TrainRunNb": 5000,  # optimizer steps for each of the 3 main reconstructors (N=1 baseline, channel-stack, temporal)
        "N": 10,  # temporal window length (frames)
        "M": 1,  # frames-ahead prediction horizon
        "MSweepValues": [1, 2, 3],  # small M sweep for the channel-stack architecture (see idea doc's "Known risks")
        "MSweepRunNb": 2000,  # shorter training budget for the exploratory M sweep
        "EvalSteps": 100,  # closed-loop rollout length for evaluation/ablation
        "EvalSeed": 1234,  # fixed seed for fair, identical-atmosphere ablation comparisons
    }
)
