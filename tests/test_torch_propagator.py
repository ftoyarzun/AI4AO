"""Tests for AI4AO.TorchPropagator: the base WFS class and its noise model."""
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
    # WFS.GetPSF only depends on phase/pupil, unlike Propagator/forward which
    # need SetMask()/BuildMask() to have already run.
    wfs = WFS(tiny_wfs_params(), device)

    phase = torch.zeros(1, wfs.Nres, wfs.Nres, device=device)
    psf = wfs.GetPSF(phase)

    assert psf.shape[0] == 1
    assert torch.all(psf >= 0)
    assert torch.isfinite(psf).all()


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
    phase = torch.zeros(1, wfs.Nres, wfs.Nres)

    full_psf = wfs.GetPSF(phase)
    cropped_psf = wfs.GetPSF(phase, fov=6)

    assert cropped_psf.shape[-1] < full_psf.shape[-1]
    assert cropped_psf.shape[-2:] == (int(round(6 * wfs.sampling)),) * 2
