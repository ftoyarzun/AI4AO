#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared parameter file for the NCPA-estimation notebook (NCPAEstimation.ipynb):
calibration of the DM misregistration on the science camera and NCPA
estimation from PSFs. See Ideas/21-psf-dm-misregistration.md. This is a
fictional demo instrument (no real bench behind it), built purely to
demonstrate the technique -- the same "no real bench" framing
../SystemDesign/TwoStageAO.ipynb and DualSensorFusion.ipynb already use for their own demo twins.

Unlike the other Tutorials/Advanced params files, this notebook never touches
PhaseDataset or Trainer -- NCPA is measured on a static internal source with
zero atmosphere, so there is no AtmosParams/LoopParams/TrainParams here, only
the WFSParams/DMParams needed to build the DM+PSF forward model, plus an
NCPAParams dict for the probes and the fit.
"""

# %% Set general parameters

## Telescope/aperture parameters. The WFS's own sensing path (mask, noise) is
## never exercised by this notebook -- only wfs.pupil/D/Nres (aperture
## geometry) and wfs.wavelength (the default GetPSF wavelength, overridable
## via NCPAParams["ScienceWavelength"]) matter here.
WFSParams = dict(
    {
        "Nres": 48,
        "sampling": 3,
        "D": 1.8,
        "centralObstruction": 0.3,
        "useNoise": False,  # internal calibration source; GetPSF itself never adds noise anyway
        "Wavelength": 700e-9,
    }
)

## The DM as calibrated on the WFS path; the notebook copies it and calibrates
## the copy's relay geometry on the science camera. "Nmodes" is a placeholder
## resolved in the notebook from dm.totalAct, exactly like
## ../SystemDesign/DualSensorFusion_params.py.
DMParams = dict(
    {
        "Nactuator": 11,
        "moffatParam": 2,
        "signedAmplitude": -5e-6,
        "MechCoupling": 0.36,
        "FlipLeftRight": False,
        "FlipTopBottom": False,
        "Nmodes": None,  # placeholder, resolved in the notebook from dm.totalAct
    }
)

## Probes and fit settings. The "demo only" keys describe the synthetic camera
## data, chosen to make this demo's self-consistency check well-posed -- not
## measured from any real instrument (see CLAUDE.md's "never invent real
## hardware parameters" rule). On a bench, the plate scale comes from
## estimate_psf_sampling and the frames from the camera.
NCPAParams = dict(
    {
        "ScienceWavelength": 635e-9,  # GetPSF's wl= override; the science camera and the WFS
                                       # sensing path need not share a wavelength.
        "TrueSampling": 130 / 48,     # demo only: plate scale (pixels per lambda/D) of the synthetic
                                       # images; a value GetPSF can produce exactly (Npix / Nres)
        "PSFWindow": 96,              # camera window in pixels (GetPSF's fov_pix)
        "Photons": 1e7,               # demo only: photons per frame
        "BackgroundCounts": 1.0,      # demo only: background per pixel, in photons
        # Probes: M2C columns (Noll ordering, AI4AO/PhaseDataset.py:Zernike): 0-1 are tip/tilt
        # (never used as probes: each frame has its own free tip-tilt), 2 is focus, 3-4
        # astigmatism, 5-6 coma, ...
        "LowOrderProbes": [2, 3, 4, 5, 6, 7, 8, 9],
        "HighOrderProbes": [10, 13, 16, 19, 22, 25, 28],
        "ProbeAmplitude": 1.0,        # dimensionless MakeZernikeM2C coefficient -- NOT meters; its
                                       # z carries a 1/wavenumber scaling, so 1.0 is a large
                                       # (~1 rad) poke
        "NDiversity": 10,             # random diversity commands (Ideas/06 style)
        "DiversityAmplitude": 0.3,
        "NCPAModes": 30,              # NCPA basis: the first NCPAModes columns of the Zernike M2C
        "NIterations": 500,
    }
)
