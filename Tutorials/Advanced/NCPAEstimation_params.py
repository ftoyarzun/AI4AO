#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared parameter file for the NCPA-estimation notebook (NCPAEstimation.ipynb).
See Ideas/06-ncpa-estimation.md for the full plan. This is a fictional demo
instrument (no real bench behind it), built purely to demonstrate the
diversity-command / PSF-fitting technique -- the same "no real bench" framing
TwoStageAO.ipynb/DualSensorFusion.ipynb already use for their own demo twins.

Unlike the other Tutorials/Advanced params files, this notebook never touches
PhaseDataset or Trainer -- NCPA is measured on a static internal source with
zero atmosphere, so there is no AtmosParams/LoopParams/TrainParams here, only
the WFSParams/DMParams needed to build the DM+PSF forward model, plus an
NCPAParams dict for the fit itself.
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

## The DM whose "previously calibrated" misregistration this notebook mostly
## treats as fixed (see Ideas/06-ncpa-estimation.md step 2/8). "Nmodes" is a
## placeholder resolved in the notebook from dm.totalAct, exactly like
## DualSensorFusion_params.py.
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

## NCPA-fit-specific settings. Diversity amplitudes/ground-truth injected values
## are chosen only to make this demo's synthetic self-consistency check well-posed
## (see DiversityAmplitudes below) -- not measured from any real instrument
## (see CLAUDE.md's "never invent real hardware parameters" rule).
NCPAParams = dict(
    {
        "ScienceWavelength": 635e-9,  # GetPSF's wl= override; equal to WFSParams["Wavelength"] here,
                                       # but kept as its own key since the science camera and the WFS
                                       # sensing path need not share a wavelength in general.
        "PSFSampling": 4,             # GetPSF's `sampling` kwarg (oversampling factor -> plate scale)
        "PSFFov": 20,                 # GetPSF's `fov` kwarg, in lambda/D
        "DiversityModeColumns": [2, 3, 4, 5, 6],  # M2C columns 0-1 are tip/tilt (skipped -- see
                                                    # Ideas/06-ncpa-estimation.md step 7); 2 is focus,
                                                    # 3-4 are astigmatism, 5-6 are coma (Noll ordering,
                                                    # AI4AO/PhaseDataset.py:Zernike).
        "DiversityAmplitudes": [0.3, 0.7, 1.1],  # dimensionless MakeZernikeM2C coefficients -- NOT
                                             # meters. MakeZernikeM2C's z already carries a
                                             # 1/wavenumber OPD-per-unit-coefficient scaling
                                             # internally (DeformableMirror.py:230), so a
                                             # coefficient of 1.0 corresponds to a physically
                                             # large OPD; these values were chosen to land the
                                             # resulting diversity pokes at an NCPA-scale ~30-80nm
                                             # OPD RMS for this demo instrument, verified directly
                                             # against dm_model(C).
        "NIterations": 1000,
        "LearningRateNCPA": 1e-2,      # applied to c_ncpa's raw, 1e-6-reparametrized optimizer
                                        # variable (see NCPA_COMMAND_SCALE in the notebook's
                                        # fit_ncpa) -- NOT to the physical meters-valued command,
                                        # matching DeformableMirror.sign's own scaling convention
        "LearningRateGeometry": 1e-3,
        "RegularizationWeight": 0.0,  # Tikhonov weight on ||c_ncpa_raw||^2 (the dimensionless,
                                       # 1e-6-reparametrized optimizer variable, not the physical
                                       # command); keep at 0 unless the synthetic recovery check
                                       # shows a noisy/non-physical fit
        "FitDMGeometry": True,       # optional step 8 toggle -- also fit rotationAngle/
                                       # radialScaling/tangentialScaling alongside c_ncpa
    }
)
