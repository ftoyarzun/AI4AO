"""Small CPU regressions for Zernike pairs and signal/full detector frames.

Run with the repository's dependencies: python -m unittest discover -s tests.
No shards, datasets, or checkpoints are written outside a temporary directory.
"""
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import poppy
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PhaseDataset import PhaseDataset
from ShackHartmann import ShackHartmann
import generate_webdataset as exporter


def small_parameters():
    wfs = dict(
        Nres=32, D=7.92223, c_obs=2.59767, Nzernike=5, Nactuator=9,
        Nphotons=[4., 4.], RON=[0., 0.], ModalBasis='Zernike',
        useNoise=False, beamSplitProportionForWFSDetector=1., QE=1.,
        FWC=None, gain=1., bits=None, sensor='CCD', darkCurrent=0.,
        applyDigitalization=False, exposureTime=.001, backgroundRange=[0., 0.],
        wavelength=658e-9, nSubap=4, lightRatio=.1, binning_factor=1,
        pixel_scale=None, shannon_sampling=True, n_pixel_per_subap=None,
        half_pixel_shift=False,
    )
    atmosphere = dict(r0=[.118, .118], L0=[20., 20.], Nphases=1,
                      Layers=[1, 2], f_slope=11/6)
    loop = dict(levelOfCorrection=[0., 1.], loopFrequency=1000.,
                delayFrames=1, windSpeedVector=[0., 0.])
    return wfs, atmosphere, loop


class CircularProjectionTests(unittest.TestCase):
    def setUp(self):
        self.wfs, self.atmosphere, self.loop = small_parameters()
        self.dataset = PhaseDataset(self.wfs, self.atmosphere, self.loop, 'cpu')

    def test_fit_uses_full_circle_and_is_independent_of_obscuration(self):
        expected = poppy.CircularAperture(radius=self.wfs['D']/2).sample(
            npix=32, grid_size=self.wfs['D'], what='amplitude') > 0
        np.testing.assert_array_equal(self.dataset.projection_pupil.numpy(), expected)
        hidden = self.dataset.projection_pupil & (self.dataset.pupil == 0)
        self.assertTrue(hidden.any())
        self.assertTrue(self.dataset.z_FullRes[hidden].abs().max() > 0)

        other_wfs = dict(self.wfs, c_obs=0.)
        other = PhaseDataset(other_wfs, self.atmosphere, self.loop, 'cpu')
        self.assertFalse(torch.equal(self.dataset.pupil, other.pupil))
        torch.testing.assert_close(self.dataset.z_FullRes, other.z_FullRes, rtol=0, atol=0)
        torch.testing.assert_close(self.dataset.invZ, other.invZ, rtol=0, atol=0)

        torch.manual_seed(20261013)
        phase, coefficients, *_ = self.dataset[0]
        torch.manual_seed(20261013)
        other_phase, other_coefficients, *_ = other[0]
        torch.testing.assert_close(phase, other_phase, rtol=0, atol=0)
        torch.testing.assert_close(coefficients, other_coefficients, rtol=0, atol=0)
        self.assertTrue(phase[:, hidden].abs().max() > 0)
        self.assertLess(float(phase[:, self.dataset.projection_pupil].mean().abs()), 2e-6)

    def test_analytic_tilt_and_defocus_cancel_on_the_full_circle(self):
        row, col = torch.meshgrid(torch.arange(32, dtype=torch.float32)-15.5,
                                  torch.arange(32, dtype=torch.float32)-15.5,
                                  indexing='ij')
        support = self.dataset.projection_pupil
        polynomial = .02*row + .001*(row.square()+col.square())
        phase = ((polynomial-polynomial[support].mean())*support).unsqueeze(0)
        basis = self.dataset.z_FullRes.flatten(0, 1)
        coefficients = phase.flatten(1) @ self.dataset.invZ
        residual = phase - (coefficients @ basis.T).view_as(phase)
        self.assertLess(float(residual.abs().max()), 2e-6)

        reference, *_ = np.linalg.lstsq(basis.double().numpy(), phase.flatten().double().numpy(), rcond=None)
        np.testing.assert_allclose(coefficients.numpy()[0], reference, atol=2e-7, rtol=2e-6)

    def test_moving_phase_retains_hidden_samples_and_removes_circular_piston(self):
        # An independent Fourier realization whose cropped phase is a known ramp.
        full_phase = torch.arange(64, dtype=torch.float32).view(1, 1, 64, 1).expand(1, 1, 64, 64)
        self.dataset.movingWavefrontGenerator = torch.fft.ifft2(full_phase, norm='ortho')
        self.dataset.fractionalr0 = torch.ones(1, 1, 1, 1)
        compressed = self.dataset.CompressAtmosphere()
        expected = full_phase[0, :, :32, :32]*self.dataset.projection_pupil
        torch.testing.assert_close(compressed, expected, rtol=1e-6, atol=5e-6)
        phase = self.dataset.RemovePiston(compressed)
        support = self.dataset.projection_pupil
        self.assertLess(float(phase[:, support].mean().abs()), 2e-6)
        hidden = support & (self.dataset.pupil == 0)
        self.assertTrue(phase[:, hidden].abs().max() > 0)


class DetectorFrameTests(unittest.TestCase):
    def test_star_only_signal_is_independent_of_sky_brightness(self):
        parameters, *_ = small_parameters()
        parameters.update(QE=.7, gain=3., backgroundRange=[2e6, 2e6],
                          bits=12, applyDigitalization=False)
        sensor = ShackHartmann(parameters, 'cpu')
        signal = sensor.Propagator(torch.zeros(1, 32, 32))
        expected_background = (2e6*sensor.exposureTime*sensor.pixel_area_arcsec2
                               *sensor.bg_pupil.repeat_interleave(sensor.detector_pitch, 0)
                               .repeat_interleave(sensor.detector_pitch, 1)).unsqueeze(0)
        torch.testing.assert_close(sensor.background_photons_no_noise, expected_background)
        self.assertGreater(float(expected_background.sum()), 0.)
        torch.testing.assert_close(sensor.frame_photons_no_noise,
                                   sensor.source_photons_no_noise+expected_background)
        torch.testing.assert_close(signal, sensor.source_photons_no_noise*.7*3.)
        torch.testing.assert_close(signal, sensor.frame_no_noise)
        sensor.backgroundRange = [4e6, 4e6]
        brighter_sky_signal = sensor.Propagator(torch.zeros(1, 32, 32))
        torch.testing.assert_close(brighter_sky_signal, signal, rtol=0, atol=0)
        torch.testing.assert_close(sensor.background_photons_no_noise, 2*expected_background)

    def test_no_star_gives_zero_signal_but_nonzero_full_sky_frame(self):
        parameters, *_ = small_parameters()
        parameters.update(useNoise=True, backgroundRange=[2e6, 2e6])
        sensor = ShackHartmann(parameters, 'cpu')
        sensor.Nphotons = 0.
        torch.manual_seed(20261019)
        full = sensor.Propagator(torch.zeros(1, 32, 32))
        self.assertEqual(float(sensor.frame_no_noise.abs().max()), 0.)
        self.assertGreater(float(full.sum()), 0.)
        torch.testing.assert_close(sensor.frame_photons_no_noise, sensor.background_photons_no_noise)

    def test_signal_and_full_have_matching_gain_and_optional_digitization(self):
        for sensor_type in ['CCD', 'EMCCD']:
            for digitization in [False, True]:
                with self.subTest(sensor=sensor_type, digitization=digitization):
                    parameters, *_ = small_parameters()
                    parameters.update(useNoise=True, sensor=sensor_type, QE=.7, gain=2.,
                                      FWC=1e4, bits=12, applyDigitalization=digitization,
                                      backgroundRange=[2e6, 2e6])
                    sensor = ShackHartmann(parameters, 'cpu')
                    # With random draws disabled, compare each output to its own
                    # astronomical input: star only, or star plus sky.
                    with patch('ShackHartmann.PoissonNoise', side_effect=lambda mean: mean):
                        full = sensor.Propagator(torch.zeros(1, 32, 32))
                    for actual, photons in [(full, sensor.frame_photons_no_noise),
                                            (sensor.frame_no_noise, sensor.source_photons_no_noise)]:
                        expected = photons.numpy()*.7*2.
                        if digitization:
                            expected = np.clip(expected/1e4*(2**12-1), 0, 2**12-1).astype(np.uint16)
                            np.testing.assert_array_equal(actual.numpy(), expected)
                        else:
                            np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12, rtol=1e-12)
                    self.assertFalse(np.array_equal(full.numpy(), sensor.frame_no_noise.numpy()))

    def test_full_noise_mean_and_variance_and_signal_preservation(self):
        parameters, *_ = small_parameters()
        parameters.update(useNoise=True, gain=2., RON=[3., 3.])
        sensor = ShackHartmann(parameters, 'cpu')
        samples = 32768
        source_photons, sky_photons = 80., 20.
        mean_photons = source_photons + sky_photons
        sensor.frame_photons_no_noise = torch.full((samples, 1, 1), mean_photons, dtype=torch.float64)
        sensor.frame_no_noise = torch.full_like(sensor.frame_photons_no_noise, source_photons*2.)
        signal_before = sensor.frame_no_noise.clone()
        torch.manual_seed(20261019)
        sensor._AddNoiseToFrame()
        # In this unsaturated QE=1 regime, the existing photon-noise approximation
        # has variance source+sky, followed by independent readout noise in electrons.
        expected_mean = mean_photons*2.
        expected_variance = (mean_photons+3.**2)*2.**2
        mean_tolerance = 6*(expected_variance/samples)**.5
        variance_tolerance = 6*expected_variance*(2/(samples-1))**.5
        self.assertLess(abs(float(sensor.frame_with_noise.mean())-expected_mean), mean_tolerance)
        self.assertLess(abs(float(sensor.frame_with_noise.var())-expected_variance), variance_tolerance)
        torch.testing.assert_close(sensor.frame_no_noise, signal_before, rtol=0, atol=0)

    def test_noise_can_be_disabled_without_stale_full_frames(self):
        parameters, *_ = small_parameters()
        parameters.update(useNoise=True, RON=[3., 3.], backgroundRange=[2e6, 2e6])
        sensor = ShackHartmann(parameters, 'cpu')
        phase = torch.zeros(1, 32, 32)
        torch.manual_seed(20261019)
        sensor.Propagator(phase)
        signal_with_noise_enabled = sensor.frame_no_noise.clone()
        self.assertIsNotNone(sensor.frame_with_noise)
        sensor.useNoise = False
        signal = sensor.Propagator(phase)
        torch.testing.assert_close(signal, signal_with_noise_enabled, rtol=0, atol=0)
        self.assertIsNone(sensor.frame_with_noise)
        self.assertIsNone(sensor.frame_photons_with_noise)
        self.assertIsNone(sensor.frame_electrons_with_noise)


class ExportedPairTests(unittest.TestCase):
    def test_saved_phase_and_both_frames_use_the_same_residual(self):
        for correction_enabled in [False, True]:
            for higher_order_phase in [False, True]:
                with self.subTest(correction_enabled=correction_enabled,
                                  higher_order_phase=higher_order_phase):
                    self.check_export(correction_enabled, higher_order_phase)

    def test_export_with_source_and_sky_supports_signal_and_full_datasets(self):
        for use_noise in [False, True]:
            with self.subTest(use_noise=use_noise):
                self.check_export(True, True, use_noise=use_noise, sky_background=2e6)

    def check_export(self, correction_enabled, higher_order_phase, use_noise=False, sky_background=0.):
        wfs_params, atmosphere, loop = small_parameters()
        wfs_params.update(useNoise=use_noise, backgroundRange=[sky_background, sky_background])
        dataset = PhaseDataset(wfs_params, atmosphere, loop, 'cpu')
        row = torch.arange(32, dtype=torch.float32).view(1, 32, 1)-15.5
        original_phase = (.04*row.expand(1, 32, 32))*dataset.projection_pupil
        if higher_order_phase:
            # Cubic phase cannot be completely represented by these five modes.
            original_phase = original_phase + .2*(row/15.5).pow(3)*dataset.projection_pupil
        coefficients = original_phase.flatten(1) @ dataset.invZ
        basis = dataset.z_FullRes.flatten(0, 1)
        reference_coefficients, *_ = np.linalg.lstsq(
            basis.double().numpy(), original_phase.flatten().double().numpy(), rcond=None)
        expected_phase = original_phase.numpy()
        if correction_enabled:
            expected_phase = expected_phase - (basis.double().numpy() @ reference_coefficients).reshape(1, 32, 32)

        class KnownPhaseDataset:
            def __init__(self, *args):
                self.z_FullRes = dataset.z_FullRes

            def __getitem__(self, index):
                return (original_phase.clone(), coefficients.clone(),
                        torch.full((1, 1, 1), 1e4), torch.zeros(1, 1, 1),
                        torch.full((1, 1, 1), .118))

        captured_sensors = []

        class RecordingSensor(ShackHartmann):
            def __init__(self, params, device):
                super().__init__(params, device)
                captured_sensors.append(self)
                self.propagated_phases = []
                self.propagated_frames = []
                self.propagated_signals = []

            def Propagator(self, phase):
                self.propagated_phases.append(phase.clone())
                frame = super().Propagator(phase)
                self.propagated_frames.append(frame.clone())
                self.propagated_signals.append(self.frame_no_noise.clone())
                return frame

        captured_samples = []

        class MemoryWriter:
            def __init__(self, *args, **kwargs):
                pass

            def write(self, sample):
                captured_samples.append(sample)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            argv = ['generate_webdataset.py', '--n', '2', '--device', 'cpu',
                    '--out', str(Path(temporary)/'pairs-%02d.tar')]
            if correction_enabled:
                argv.append('--correct-zernike')
            parameters = SimpleNamespace(WFSParams=wfs_params, AtmosParams=atmosphere, LoopParams=loop)
            torch.manual_seed(20261019)
            with patch.object(sys, 'argv', argv), patch.object(exporter, 'params', parameters), \
                    patch.object(exporter, 'PhaseDataset', KnownPhaseDataset), \
                    patch.object(exporter, 'ShackHartmann', RecordingSensor), \
                    patch.object(exporter.wds, 'ShardWriter', MemoryWriter):
                exporter.main()

        self.assertEqual(len(captured_samples), 2)
        sensor = captured_sensors[0]
        for index, sample in enumerate(captured_samples):
            saved_phase = np.load(io.BytesIO(sample['phase.npy']))
            np.testing.assert_allclose(saved_phase, expected_phase, rtol=0, atol=2e-6)
            np.testing.assert_array_equal(saved_phase, sensor.propagated_phases[index].numpy())
            np.testing.assert_array_equal(np.load(io.BytesIO(sample['frame.npy'])),
                                          sensor.propagated_frames[index].squeeze().numpy())
            np.testing.assert_array_equal(np.load(io.BytesIO(sample['signal_frame.npy'])),
                                          sensor.propagated_signals[index].squeeze().numpy())
        if use_noise:
            self.assertFalse(torch.equal(sensor.propagated_frames[0], sensor.propagated_signals[0]))
            self.assertFalse(torch.equal(sensor.propagated_frames[0], sensor.propagated_frames[1]))
            torch.testing.assert_close(sensor.propagated_signals[0], sensor.propagated_signals[1], rtol=0, atol=0)
        else:
            torch.testing.assert_close(sensor.propagated_frames[0], sensor.propagated_signals[0], rtol=0, atol=0)

        reference_phase = torch.from_numpy(expected_phase).to(dtype=original_phase.dtype)
        reference_frame = sensor.Propagator(reference_phase)
        torch.testing.assert_close(sensor.propagated_signals[0], sensor.frame_no_noise, rtol=2e-6, atol=2e-5)
        torch.testing.assert_close(sensor.propagated_signals[0], sensor.source_photons_no_noise,
                                   rtol=2e-6, atol=2e-5)  # These export cases use QE=gain=1, no ADC.
        if not use_noise:
            torch.testing.assert_close(sensor.propagated_frames[0], reference_frame, rtol=2e-6, atol=2e-5)
        flat_frame = sensor.Propagator(torch.zeros_like(original_phase))
        if correction_enabled and not higher_order_phase:
            torch.testing.assert_close(sensor.propagated_frames[0], flat_frame, rtol=2e-6, atol=2e-5)
        if not correction_enabled:
            self.assertGreater(float((sensor.propagated_frames[0]-flat_frame).abs().max()), 1.)
        if correction_enabled and higher_order_phase:
            self.assertGreater(float(np.abs(expected_phase).max()), .01)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
