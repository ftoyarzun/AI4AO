"""
Shared fixtures for the AI4AO test suite.

Running the suite requires `torch` to be installed separately (per the
project README/CLAUDE.md convention -- it is intentionally not listed in
pyproject.toml's `dependencies`). Install the test extras with:

    pip install -e ".[test]"

then run the fast suite with `pytest tests/ -m "not slow"`, or the full
suite (including the slow end-to-end smoke tests) with `pytest tests/`.
"""
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import torch


@pytest.fixture
def device():
    """Defaults to CPU for speed/portability; override with AI4AO_TEST_DEVICE=cuda."""
    return torch.device(os.environ.get("AI4AO_TEST_DEVICE", "cpu"))


@pytest.fixture(autouse=True)
def seed_rng():
    """PhaseDataset and the noise paths in TorchPropagator/FramePreprocess draw from
    the *global* NumPy/torch RNGs (not per-instance generators), so both must be
    seeded for reproducibility."""
    np.random.seed(0)
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Params-dict factories, mirroring the plain-dict convention used by the
# Tutorials/*_params.py files, shrunk for fast tests.
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_wfs_params():
    """Pyramid-flavored WFSParams factory. Nres=12, sampling=3.0 keep Npix=36
    (an exact multiple, so the two independent pad-size computations in
    TorchPropagator agree) while staying fast."""
    def _make(modulation=0.0):
        return {
            "Nres": 12,
            "sampling": 3.0,
            "D": 1.0,
            "centralObstruction": 0.3,
            "useNoise": False,
            "Wavelength": 635e-9,
            "Nphotons": [5.0, 5.0],
            "RON": [1.0, 1.0],
            "Modulation": modulation,
            # FramePreprocess keys
            "Substract_Reference": True,
            "Bin_factor": 1,
            "Center_noise": 0.0,
            "Extract_pupils_pad": 4,
            "Pupil_size_noise": 0.0,
        }
    return _make


@pytest.fixture
def tiny_zernike_wfs_params(tiny_wfs_params):
    def _make(mask_type="Zernike", use_mtf=False):
        params = tiny_wfs_params(modulation=0.0)
        params.update({
            "MTF_upscale": 4,
            "Use_MTF": use_mtf,
            "MaskType": mask_type,
        })
        return params
    return _make


@pytest.fixture
def tiny_atmos_params():
    return {
        "L0": [25.0, 25.0],
        "r0": [0.05, 0.05],
        "Nphases": 4,
        "Layers": [2, 3],
        "f_slope": 11.0 / 6.0,
        "Scintillation": False,
    }


@pytest.fixture
def tiny_loop_params():
    return {
        "levelOfCorrection": [0.0, 1.0],
        "loopFrequency": 500.0,
        "delayFrames": 1,
        "loopGain": [0.2, 0.5],
        "loopLeak": [0.9, 1.0],
        "windSpeedVector": [1.0, 10.0],
    }


@pytest.fixture
def tiny_dm_params():
    return {
        "Nactuator": 5,
        "Nmodes": 8,
        "moffatParam": 2.0,
        "signedAmplitude": 1e-5,
        "MechCoupling": 0.36,
        "FlipLeftRight": False,
        "FlipTopBottom": False,
    }


# ---------------------------------------------------------------------------
# Constructed-object fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pyramid_wfs(tiny_wfs_params, device):
    from AI4AO.PyramidWFS import PyramidWFS
    return PyramidWFS(tiny_wfs_params(modulation=0.0), device)


@pytest.fixture
def zernike_wfs(tiny_zernike_wfs_params, device):
    from AI4AO.ZernikeWFS import ZernikeWFS
    return ZernikeWFS(tiny_zernike_wfs_params(mask_type="Zernike", use_mtf=False), device)


@pytest.fixture
def phase_dataset(tiny_wfs_params, tiny_atmos_params, tiny_loop_params, tiny_dm_params, device):
    from AI4AO.PhaseDataset import PhaseDataset
    return PhaseDataset(tiny_wfs_params(), tiny_atmos_params, tiny_loop_params, tiny_dm_params, device)


@pytest.fixture
def deformable_mirror(tiny_wfs_params, tiny_dm_params, device):
    from AI4AO.DeformableMirror import DeformableMirror
    return DeformableMirror(tiny_wfs_params(), tiny_dm_params, device)
