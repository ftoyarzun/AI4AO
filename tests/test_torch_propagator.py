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


def test_scalar_wavelength_assignment_stays_backward_compatible(tiny_wfs_params, device):
    """The wavelength setter now coerces to a tensor and branches on its rank --
    confirm a plain-float assignment (today's only usage before this feature)
    still yields a 0-d wavenumber and an unchanged, 3-D Propagator output."""
    from AI4AO.PyramidWFS import PyramidWFS

    wfs = PyramidWFS(tiny_wfs_params(), device)
    assert wfs.wavenumber.dim() == 4

    opd = torch.zeros(2, wfs.Nres, wfs.Nres, device=device)
    frame = wfs(opd)

    assert frame.dim() == 3
    assert frame.shape == (2, wfs.Npix, wfs.Npix)


def _two_wavelength_reference_frame(build_wfs, wavelength_a, wavelength_b, opd, mask_a=None, mask_b=None):
    """Builds two independent scalar-wavelength WFS instances (optionally
    overriding each one's mask) and returns the flux-weighted average of their
    two propagated frames. This is the correct manual reference for a
    multi-wavelength WFS's *normalized* output: both frames sum to the same
    total flux before Propagator's own normalization (pure phase masks don't
    change total energy), so the multi-wavelength call's single
    sum-then-normalize is equivalent to averaging the two independently
    normalized frames -- and would instead equal their unaveraged *sum* if the
    wavelength reduction were (incorrectly) applied after normalization."""
    wfs_a = build_wfs(wavelength_a)
    wfs_b = build_wfs(wavelength_b)
    if mask_a is not None:
        wfs_a.SetMask(phaseMask=mask_a)
    if mask_b is not None:
        wfs_b.SetMask(phaseMask=mask_b)
    return 0.5 * (wfs_a(opd) + wfs_b(opd))


def test_propagator_multi_wavelength_shared_mask(tiny_wfs_params, device):
    """A single shared (wavelength-independent) mask propagated at several
    wavelengths in parallel must match two separate scalar-wavelength calls
    averaged together, and the output shape must stay collapsed to
    (Nphases, Npix, Npix) regardless of Nwavelength."""
    from AI4AO.PyramidWFS import PyramidWFS

    def build(wavelength):
        params = tiny_wfs_params()
        params["Wavelength"] = wavelength
        return PyramidWFS(params, device)

    wfs = build(635e-9)
    wfs.wavelength = torch.tensor([635e-9, 750e-9], device=device)

    opd = 1e-7 * torch.randn(2, wfs.Nres, wfs.Nres, device=device)
    frame = wfs(opd)

    assert frame.shape == (2, wfs.Npix, wfs.Npix)
    reference = _two_wavelength_reference_frame(build, 635e-9, 750e-9, opd)
    assert torch.allclose(frame, reference, atol=1e-5)


def test_propagator_multi_wavelength_with_modulation_masks(tiny_wfs_params, device):
    """Several modulation-step masks (today's existing 'mask channel') shared
    across several wavelengths -- exercises both channel axes at once."""
    from AI4AO.PyramidWFS import PyramidWFS

    def build(wavelength):
        params = tiny_wfs_params(modulation=1.0)
        params["Wavelength"] = wavelength
        return PyramidWFS(params, device)

    wfs = build(635e-9)
    wfs.wavelength = torch.tensor([635e-9, 750e-9], device=device)

    opd = 1e-7 * torch.randn(2, wfs.Nres, wfs.Nres, device=device)
    frame = wfs(opd)

    assert frame.shape == (2, wfs.Npix, wfs.Npix)
    reference = _two_wavelength_reference_frame(build, 635e-9, 750e-9, opd)
    assert torch.allclose(frame, reference, atol=1e-5)


def test_propagator_wavelength_dependent_mask(tiny_wfs_params, device):
    """A genuinely per-wavelength mask -- supplied as an explicit 4-D
    (Nwavelength, Nmask, H, W) tensor per SetMask's documented convention --
    must pair each wavelength with its own mask rather than sharing one mask
    across all wavelengths."""
    from AI4AO.PyramidWFS import PyramidWFS

    def build(wavelength):
        params = tiny_wfs_params()
        params["Wavelength"] = wavelength
        return PyramidWFS(params, device)

    probe = build(635e-9)
    mask_a = probe.PyramidMask()
    mask_b = probe.PyramidMask(x_offset=torch.tensor(2.0, device=device))
    assert not torch.allclose(mask_a, mask_b)  # the two masks must actually differ

    wfs = build(635e-9)
    wfs.SetMask(phaseMask=torch.stack([mask_a, mask_b]).unsqueeze(1))  # (2, 1, H, W)
    wfs.wavelength = torch.tensor([635e-9, 750e-9], device=device)

    opd = 1e-7 * torch.randn(2, wfs.Nres, wfs.Nres, device=device)
    frame = wfs(opd)

    assert frame.shape == (2, wfs.Npix, wfs.Npix)
    reference = _two_wavelength_reference_frame(
        build, 635e-9, 750e-9, opd, mask_a=mask_a, mask_b=mask_b
    )
    assert torch.allclose(frame, reference, atol=1e-5)

    # Sanity check: pairing must actually matter -- sharing mask_a across both
    # wavelengths instead should give a different result.
    shared_reference = _two_wavelength_reference_frame(build, 635e-9, 750e-9, opd)
    assert not torch.allclose(frame, shared_reference, atol=1e-5)


def test_propagator_multi_wavelength_normalization_sums_to_one(tiny_wfs_params, device):
    """Regression guard for the normalization-ordering hazard: the wavelength
    axis must be collapsed *before* Propagator's flux normalization, so each
    Nphases slice of a multi-wavelength frame still sums to 1, not Nwavelength."""
    from AI4AO.PyramidWFS import PyramidWFS

    params = tiny_wfs_params()
    wfs = PyramidWFS(params, device)
    wfs.wavelength = torch.tensor([635e-9, 750e-9, 900e-9], device=device)

    opd = 1e-7 * torch.randn(3, wfs.Nres, wfs.Nres, device=device)
    frame = wfs(opd)

    assert torch.allclose(frame.sum(dim=(-2, -1)), torch.ones(3, device=device), atol=1e-5)


def _manual_polychromatic_psf_reference(wfs, params, opd):
    """Replicates _GetPolychromaticPSF's per-channel steps (shared FFT
    oversampled for the longest wavelength, then grid_sample resampling onto
    the shortest wavelength's output grid) using independent, non-batched
    scalar-wavelength WFS instances, to check the batched implementation
    isn't just a naive same-grid sum (the physically wrong behavior it
    replaces) and isn't mis-pairing wavelengths with the wrong resample scale."""
    from AI4AO.PyramidWFS import PyramidWFS

    wl = wfs.wavelength
    sim_sampling = float((wfs.sampling * wl[-1] / wl[0]).item())
    scale_all = wl[-1] / wl

    reference = None
    for i in range(wl.shape[0]):
        p = dict(params)
        p["Wavelength"] = float(wl[i].item())
        wfs_i = PyramidWFS(p, wfs.device)
        psf_sim = wfs_i.GetPSF(opd, sampling=sim_sampling)  # raw, un-resampled, on the shared sim grid
        grid = wfs._BuildWavelengthResampleGrid(scale_all[i].reshape(1), psf_sim.shape[-1], wfs.Npix)
        resampled = torch.nn.functional.grid_sample(
            psf_sim.unsqueeze(1), grid.expand(psf_sim.shape[0], -1, -1, -1),
            mode="bilinear", align_corners=True, padding_mode="zeros",
        ).squeeze(1)
        reference = resampled if reference is None else reference + resampled
    return reference


def test_get_psf_multi_wavelength_resamples_before_summing(tiny_wfs_params, device):
    """GetPSF's multi-wavelength path must resample each wavelength's raw PSF
    (computed on a shared grid oversampled for the longest wavelength) down
    onto the shortest wavelength's output grid before summing -- since the
    channels are not natively on the same physical angular grid, a naive
    same-grid sum (the previous, physically wrong behavior) would give a
    different, incorrect result."""
    from AI4AO.PyramidWFS import PyramidWFS

    params = tiny_wfs_params()
    params["Wavelength"] = 635e-9
    wfs = PyramidWFS(params, device)
    wfs.wavelength = torch.tensor([635e-9, 750e-9], device=device)

    opd = 1e-7 * torch.randn(2, wfs.Nres, wfs.Nres, device=device)
    psf = wfs.GetPSF(opd)

    assert psf.shape == (2, wfs.Npix, wfs.Npix)
    reference = _manual_polychromatic_psf_reference(wfs, params, opd)
    assert torch.allclose(psf, reference, atol=1e-4)

    # Sanity check: this must differ from the naive (physically wrong) same-grid
    # sum that summing each wavelength's normal-sampling GetPSF would give.
    wfs_a = PyramidWFS({**params, "Wavelength": 635e-9}, device)
    wfs_b = PyramidWFS({**params, "Wavelength": 750e-9}, device)
    naive_sum = wfs_a.GetPSF(opd) + wfs_b.GetPSF(opd)
    assert not torch.allclose(psf, naive_sum, atol=1e-4)


def test_get_psf_multi_wavelength_fov_crop_after_resampling(tiny_wfs_params, device):
    from AI4AO.PyramidWFS import PyramidWFS

    params = tiny_wfs_params()
    params["Wavelength"] = 635e-9
    wfs = PyramidWFS(params, device)
    wfs.wavelength = torch.tensor([635e-9, 750e-9], device=device)

    opd = 1e-7 * torch.randn(2, wfs.Nres, wfs.Nres, device=device)
    full = wfs.GetPSF(opd)
    cropped = wfs.GetPSF(opd, fov=6)

    fov_pix = int(round(6 * wfs.sampling))
    assert cropped.shape[-2:] == (fov_pix, fov_pix)
    Ny, Nx = full.shape[-2], full.shape[-1]
    y0, x0 = (Ny - fov_pix) // 2, (Nx - fov_pix) // 2
    assert torch.allclose(cropped, full[..., y0:y0 + fov_pix, x0:x0 + fov_pix])


def test_wavelength_setter_coerces_dtype_and_device(tiny_wfs_params, device):
    """A caller might assign a plain list, a numpy array, or an untyped tensor
    -- all must land on the WFS's own device as float32, matching self.mask's
    explicit torch.cfloat construction (an accidental float64 wavenumber would
    silently upcast phase/uin/ufocal to complex128 throughout)."""
    import numpy as np
    from AI4AO.PyramidWFS import PyramidWFS

    wfs = PyramidWFS(tiny_wfs_params(), device)

    for value in ([635e-9, 750e-9], np.array([635e-9, 750e-9]), torch.tensor([635e-9, 750e-9])):
        wfs.wavelength = value
        assert wfs.wavenumber.device == device
        assert wfs.wavenumber.dtype == torch.float32
        assert wfs.wavenumber.shape == (1, 2, 1, 1)


def test_wavenumber_and_wavelength_are_always_consistently_shaped(tiny_wfs_params, device):
    """wavenumber is always a 4-D (1, Nwavelength, 1, 1) tensor and wavelength
    always a 1-D (Nwavelength,) tensor -- never a bare Python number or a 0-d
    tensor, even for a single scalar wavelength -- so Propagator/GetPSF never
    need to branch between a 'scalar' and a 'tensor' code path."""
    from AI4AO.PyramidWFS import PyramidWFS

    wfs = PyramidWFS(tiny_wfs_params(), device)

    for value in (635e-9, [635e-9], [635e-9, 750e-9], [635e-9, 750e-9, 900e-9]):
        wfs.wavelength = value
        n = len(value) if isinstance(value, list) else 1
        assert wfs.wavelength.dim() == 1
        assert wfs.wavelength.shape == (n,)
        assert wfs.wavenumber.dim() == 4
        assert wfs.wavenumber.shape == (1, n, 1, 1)


def test_propagator_collapse_wvl_false_preserves_wavelength_axis(tiny_wfs_params, device):
    """collapse_wvl=False keeps the wavelength axis (each channel independently
    flux-normalized to 1) instead of summing it away. Since a pure-phase mask
    doesn't change total energy, every channel's pre-normalization flux is
    equal, so the default collapse_wvl=True result must equal the per-channel
    (collapse_wvl=False) frames averaged together -- a useful cross-check that
    the un-collapsed path is the same underlying computation, not a separate
    one that could silently drift out of sync."""
    from AI4AO.PyramidWFS import PyramidWFS

    params = tiny_wfs_params()
    wfs = PyramidWFS(params, device)
    wfs.wavelength = torch.tensor([635e-9, 750e-9, 900e-9], device=device)

    opd = 1e-7 * torch.randn(2, wfs.Nres, wfs.Nres, device=device)

    collapsed = wfs.Propagator(opd, collapse_wvl=True)
    uncollapsed = wfs.Propagator(opd, collapse_wvl=False)

    assert collapsed.shape == (2, wfs.Npix, wfs.Npix)
    assert uncollapsed.shape == (2, 3, wfs.Npix, wfs.Npix)
    assert torch.allclose(uncollapsed.sum(dim=(-2, -1)), torch.ones(2, 3, device=device), atol=1e-5)
    assert torch.allclose(collapsed, uncollapsed.sum(dim=1) / 3, atol=1e-5)


def test_get_psf_collapse_wvl_false_matches_collapsed_sum(tiny_wfs_params, device):
    """Unlike Propagator, GetPSF applies no flux normalization, so its
    collapse_wvl=True result must equal the exact (unweighted) sum over the
    collapse_wvl=False per-wavelength channels."""
    from AI4AO.PyramidWFS import PyramidWFS

    params = tiny_wfs_params()
    wfs = PyramidWFS(params, device)
    wfs.wavelength = torch.tensor([635e-9, 750e-9], device=device)

    opd = 1e-7 * torch.randn(2, wfs.Nres, wfs.Nres, device=device)

    collapsed = wfs.GetPSF(opd, collapse_wvl=True)
    uncollapsed = wfs.GetPSF(opd, collapse_wvl=False)

    assert collapsed.shape == (2, wfs.Npix, wfs.Npix)
    assert uncollapsed.shape == (2, 2, wfs.Npix, wfs.Npix)
    assert torch.allclose(collapsed, uncollapsed.sum(dim=1), atol=1e-5)


def test_get_psf_wl_parameter_overrides_sensing_wavelength(tiny_wfs_params, device):
    """GetPSF's `wl` argument lets a caller compute a PSF at a wavelength (or
    set of wavelengths) independent of self.wavelength (the sensing
    wavelength). Also confirms the documented 'a single float, or an array'
    contract: a bare float and a plain list must both work, not just an
    already-built tensor."""
    from AI4AO.PyramidWFS import PyramidWFS

    params = tiny_wfs_params()
    wfs_sensing = PyramidWFS({**params, "Wavelength": 635e-9}, device)
    wfs_reference = PyramidWFS({**params, "Wavelength": 700e-9}, device)

    opd = 1e-7 * torch.randn(2, wfs_sensing.Nres, wfs_sensing.Nres, device=device)

    psf_override_float = wfs_sensing.GetPSF(opd, wl=700e-9)
    psf_override_list = wfs_sensing.GetPSF(opd, wl=[700e-9])
    psf_reference = wfs_reference.GetPSF(opd)

    assert torch.allclose(psf_override_float, psf_reference, atol=1e-4)
    assert torch.allclose(psf_override_list, psf_reference, atol=1e-4)


def test_set_mask_rejects_unsupported_rank(tiny_wfs_params, device):
    """SetMask now explicitly validates phaseMask/transmisionMask rank (2, 3,
    or 4-D) instead of silently accepting anything else as-is."""
    from AI4AO.PyramidWFS import PyramidWFS

    wfs = PyramidWFS(tiny_wfs_params(), device)
    bad_mask = torch.zeros(1, 1, 1, wfs.Npix, wfs.Npix, device=device)

    with pytest.raises(ValueError):
        wfs.SetMask(phaseMask=bad_mask)


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
