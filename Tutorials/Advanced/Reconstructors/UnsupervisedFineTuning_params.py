#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter file for the unsupervised physics-loss fine-tuning demo
(UnsupervisedFineTuning.ipynb, see Ideas/08-unsupervised-physics-finetuning.md).
Fictional demo instrument (no real bench behind it) -- a single modulated
Pyramid WFS + DM twin, built fresh in the notebook rather than loaded from a
saved calibration, since the notebook itself demonstrates a from-scratch
supervised warm start followed by an unsupervised physics-loss fine-tune.

TrainParams carries two separate training-recipe pairs:
- `lrn_warmstart`/`WarmStartSteps`: Phase A, ordinary supervised training
  (`LogResidualVarianceLoss + Physics_loss` against known ground truth),
  deliberately stopped short of convergence so Phase B has headroom to show
  a measurable improvement.
- `lrn_finetune`/`FineTuneSteps`: Phase B, the unsupervised fine-tune (a bare
  `Physics_loss`, no ground-truth-dependent term) -- see the notebook for why
  swapping `trainer.loss` alone is enough to make the training signal
  unsupervised, with WFS frames still generated on the fly from `PhaseDataset`
  exactly as in Phase A.

`closed_loop_iterations` is shared by both phases (kept at 1: open-loop,
single-frame estimation). `TestRunNb` sets the rollout length for the final
before/after `trainer.evaluate()` comparison.
"""
import numpy as np

# %% Set general parameters

## WFS and telescope parameters
WFSParams = dict(
    {
        "Nres": 40,
        "sampling": 3,
        "D": 1.5,
        "centralObstruction": 0.3,
        "useNoise": True,
        "Modulation": 0,
        "Wavelength": 635e-9,
        "Nphotons": [4.5, 6],  # Log range of number of photons in measurement
        "RON": [1, 3],  # Read-out noise in photons per pixel per frame
        "Substract_Reference": True,  # Subtract or not the reference intensity frame
        "Extract_pupils_pad": 6,
        "Center_noise": 2,
        "Pupil_size_noise": 0.02,  # +/-5% pupil size jitter
        "Bin_factor": 1,
    }
)

## Atmosphere parameters
AtmosParams = dict(
    {
        "r0": [0.05, 0.15],  # Fried parameter range (m)
        "L0": [20, 30.0000],  # Outer scale range (m)
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

## DM parameters
DMParams = dict(
    {
        "Nactuator": 11,
        "Nmodes": 64,
        "moffatParam": 2,
        "signedAmplitude": 1e-5,
        "MechCoupling": 0.36,
        "FlipLeftRight": False,
        "FlipTopBottom": False,
    }
)

## Training parameters -- see module docstring for the Phase A / Phase B split
TrainParams = dict(
    {
        "lrn_warmstart": 1e-4,  # Phase A: supervised warm start
        "WarmStartSteps": 3000,  # deliberately short of convergence -- see Ideas/08
        "lrn_finetune": 3e-5,  # Phase B: unsupervised physics-loss fine-tune
        "FineTuneSteps": 1500,
        "closed_loop_iterations": 1,  # open-loop, single-frame estimation
        "TestRunNb": 200,  # evaluate() rollout length for the before/after comparison
    }
)
