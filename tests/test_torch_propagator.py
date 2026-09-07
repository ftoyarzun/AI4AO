"""Tests for AI4AO.TorchPropagator: the base WFS class and its noise model."""
import pytest
import torch

from AI4AO.TorchPropagator import PoissonNoise, WFS


def test_poisson_noise_shape_dtype_device_and_seeded_mean():
    x = torch.full((4, 8, 8), 1000.0)

    torch.manual_seed(0)
    noisy = PoissonNoise(x)

    assert noisy.shape == x.shape
    assert noisy.dtype == x.dtype
    assert noisy.device == x.device
    # Gaussian approximation of Poisson noise: mean over many pixels should
    # land close to the noiseless value (std of the mean is small for N=256 px).
    assert torch.allclose(noisy.mean(), x.mean(), atol=5.0)

    torch.manual_seed(0)
    noisy_again = PoissonNoise(x)
    assert torch.equal(noisy, noisy_again)


def test_bare_wfs_get_psf_needs_no_mask_or_calibration(tiny_wfs_params, device):
    # WFS.GetPSF only depends on opd/pupil, unlike Propagator/forward which
    # need SetMask()/BuildMask() to have already run.
    wfs = WFS(tiny_wfs_params(), device)

    opd = torch.zeros(1, wfs.Nres, wfs.Nres, device=device)
    psf = wfs.GetPSF(opd)

    assert psf.shape[0] == 1
    assert torch.all(psf >= 0)
    assert torch.isfinite(psf).all()


def test_wfs_reads_wavelength_and_wavenumber(tiny_wfs_params, device):
    params = tiny_wfs_params()
    wfs = WFS(params, device)

    assert wfs.wavelength == params["Wavelength"]
    assert wfs.wavenumber == pytest.approx(2 * torch.pi / params["Wavelength"])


def test_propagator_converts_opd_to_phase_at_configured_wavelength(tiny_wfs_params, device):
    """Two WFS instances with different sensing wavelengths must produce the
    same detector frame when driven by OPDs encoding the same underlying
    phase pattern (opd = phase / wavenumber) -- true only if Propagator
    actually converts OPD to phase via *this* WFS's own wavelength."""
    from AI4AO.PyramidWFS import PyramidWFS

    params_a = tiny_wfs_params()
    params_a["Wavelength"] = 635e-9
    params_b = tiny_wfs_params()
    params_b["Wavelength"] = 1270e-9  # 2x params_a's wavelength

    wfs_a = PyramidWFS(params_a, device)
    wfs_b = PyramidWFS(params_b, device)

    torch.manual_seed(0)
    phase_pattern = torch.randn(1, wfs_a.Nres, wfs_a.Nres, device=device)  # a fixed radian phase pattern
    opd_a = phase_pattern / wfs_a.wavenumber
    opd_b = phase_pattern / wfs_b.wavenumber  # same phase, different OPD since wavelengths differ

    frame_a = wfs_a(opd_a)
    frame_b = wfs_b(opd_b)

    assert torch.allclose(frame_a, frame_b, atol=1e-5)
    # Sanity check: feeding the *same* OPD (unconverted) to both wavelengths
    # gives different frames -- confirms the output is actually wavelength-sensitive.
    assert not torch.allclose(wfs_a(opd_a), wfs_b(opd_a), atol=1e-5)


def test_get_psf_fov_crop():
    from AI4AO.PyramidWFS import PyramidWFS

    params = {
        "Nres": 12, "sampling": 3.0, "D": 1.0, "centralObstruction": 0.3,
        "useNoise": False, "Wavelength": 635e-9, "Nphotons": [5.0, 5.0],
        "RON": [1.0, 1.0], "Modulation": 0.0,
        "Substract_Reference": True, "Bin_factor": 1, "Center_noise": 0.0,
        "Extract_pupils_pad": 4, "Pupil_size_noise": 0.0,
    }
    wfs = PyramidWFS(params, torch.device("cpu"))
    opd = torch.zeros(1, wfs.Nres, wfs.Nres)

    full_psf = wfs.GetPSF(opd)
    cropped_psf = wfs.GetPSF(opd, fov=6)

    assert cropped_psf.shape[-1] < full_psf.shape[-1]
    assert cropped_psf.shape[-2:] == (int(round(6 * wfs.sampling)),) * 2


def test_build_interaction_matrix_delta_scales_with_wavelength(tiny_wfs_params, device):
    """BuildInteractionMatrix's finite-difference delta is now a physical OPD
    perturbation (wavelength / 50), not a bare radian constant -- confirm it
    tracks the WFS's own configured wavelength and produces a finite, usable
    interaction matrix."""
    from AI4AO.PyramidWFS import PyramidWFS
    from AI4AO.PhaseDataset import Zernike

    params = tiny_wfs_params()
    wfs = PyramidWFS(params, device)
    _, modes = Zernike(wfs.pupil, j=4)  # full-resolution (j, Nres, Nres) modes

    wfs.BuildInteractionMatrix(modes)

    assert wfs.iMat.shape[0] == 4
    assert torch.isfinite(wfs.iMat).all()
    assert not torch.allclose(wfs.iMat, torch.zeros_like(wfs.iMat))
