# Params reference

An instrument is configured with plain Python dicts, usually defined together in one
params file per instrument (`Tutorials/<Instrument>/<Instrument>_params.py`). There is
no YAML/JSON layer: the constructors read the keys they need directly.

```{code-block} python
from mmengine import Config

p = Config.fromfile("Tutorials/basics/wfs_params_exp.py")
dataset = PhaseDataset(p.WFSParams, p.AtmosParams, p.LoopParams, p.DMParams, device)
wfs = PyramidWFS(p.WFSParams, device)
dm = DeformableMirror(p.WFSParams, p.DMParams, device)
```

`Tutorials/basics/wfs_params_exp.py` is the demo instrument used throughout this
documentation. Copy it as a starting point, but take the physical values (diameter,
actuator count, wavelength, …) from your own instrument.

**Ranges.** A value written `[min, max]` is a range: every sample in a batch draws its
own value uniformly in that range (except where stated otherwise). Use `[x, x]` for a
fixed value.

**Read by.** The last column lists the constructors that read each key. A key a
constructor doesn't use can be left out of the dict it receives.

## `WFSParams`: telescope, sensor and detector

| Key | Type / unit | Meaning | Read by |
|---|---|---|---|
| `Nres` | int, pixels | Pupil diameter on the simulation grid. Sets the size of every OPD screen. | WFS, PhaseDataset, DM, FramePreprocess |
| `sampling` | float | Zero-padding factor of the WFS propagation: detector frames are `Nres * sampling` pixels across. Also the default PSF sampling, in pixels per λ/D. | WFS, PSFDataset |
| `D` | m | Telescope diameter | WFS, PhaseDataset, DM |
| `centralObstruction` | fraction | Central obstruction, as a fraction of the pupil radius (0 = none) | WFS |
| `Wavelength` | m | Sensing wavelength. OPD becomes phase at this wavelength, and the DM's modal basis is scaled with it (1 unit = 1 rad RMS). | WFS, DM, PhaseDataset |
| `useNoise` | bool | Add Poisson and read-out noise to the detector frames | WFS |
| `Nphotons` | `[min, max]`, log10 | Range of **log10** of the photons per frame | PhaseDataset, PSFDataset |
| `RON` | `[min, max]` | Read-out noise per pixel per frame | PhaseDataset, PSFDataset |
| `Modulation` | λ/D | Pyramid modulation radius (0 = unmodulated). Can be changed later with `wfs.modulation`. | PyramidWFS |
| `MaskType` | str | `"zwfs"` (or `"zernike"`) for a Zernike WFS, `"vzwfs"` for a vector Zernike WFS | ZernikeWFS |
| `Use_MTF` | bool | Propagate with a zoomed matrix Fourier transform instead of an FFT. Optional for `WFS` (default `False`), required by `ZernikeWFS`. | WFS, ZernikeWFS |
| `MTF_upscale` | int | Focal-plane zoom factor of the MTF path | ZernikeWFS |
| `pupilShiftX`, `pupilShiftY` | pixels, optional | Pupil decentering (default 0) | WFS |
| `pupilUpscale` | int, optional | Edge supersampling: 1 (default) gives a hard-edged pupil, more gives an anti-aliased float pupil | WFS |
| `Substract_Reference` | bool | Subtract the reference frame before normalizing (otherwise only the mean) | FramePreprocess |
| `Bin_factor` | int | Binning of the extracted pupil images | FramePreprocess, PSFDataset (optional) |
| `Extract_pupils_pad` | pixels | Margin added to `Nres` when cropping each pupil image | FramePreprocess |
| `Center_noise` | pixels | Training augmentation: uniform ± jitter of each pupil's crop center | FramePreprocess |
| `Pupil_size_noise` | fraction | Training augmentation: uniform ± jitter of the crop size (0.05 = ±5 %) | FramePreprocess |

## `AtmosParams`: turbulence

| Key | Type / unit | Meaning | Read by |
|---|---|---|---|
| `r0` | `[min, max]`, m | Fried parameter **at 500 nm**, whatever the sensing wavelength | PhaseDataset, PSFDataset |
| `L0` | `[min, max]`, m | Outer scale | PhaseDataset, PSFDataset |
| `Nphases` | int | Batch size: screens per draw | PhaseDataset, PSFDataset |
| `Layers` | `[min, max)` | Number of turbulent layers. One integer per batch, drawn with `max` **excluded**. | PhaseDataset, PSFDataset |
| `f_slope` | float | Exponent of the von Kármán spectrum (11/6 for Kolmogorov) | PhaseDataset, PSFDataset |
| `Scintillation` | bool | Fresnel-propagate the layers to the pupil (amplitude fluctuations). Ignored, with a warning, by `PSFDataset`. | PhaseDataset |

## `LoopParams`: AO loop

These keys shape the temporal evolution and, in closed-loop mode
(`dataset.generateClosedLoop = True`), the residual left by the loop.

| Key | Type / unit | Meaning | Read by |
|---|---|---|---|
| `loopFrequency` | Hz | Loop rate, which is also the time step between frames | PhaseDataset, PSFDataset |
| `delayFrames` | frames | Loop delay used in the servo-lag (temporal error) model | PhaseDataset, PSFDataset |
| `windSpeedVector` | `[min, max]`, m/s | Wind speed per layer. Directions are uniform in 0–2π. | PhaseDataset, PSFDataset |
| `levelOfCorrection` | `[min, max]` | Fraction of the correctable band that the loop corrects (0 = open loop, 1 = full correction) | PhaseDataset, PSFDataset |
| `loopGain` | `[min, max]` | Integrator gain | PhaseDataset, PSFDataset |
| `loopLeak` | `[min, max]` | Integrator leak (1 = no leak) | PhaseDataset, PSFDataset |

## `DMParams`: deformable mirror

| Key | Type / unit | Meaning | Read by |
|---|---|---|---|
| `Nactuator` | int | Actuators across the diameter (an `Nactuator × Nactuator` grid clipped to the pupil). Also sets the correctable band of the closed-loop residual. | DM, PhaseDataset, PSFDataset |
| `Nmodes` | int | Default number of modes of `MakeZernikeM2C`, and so the reconstructor's output size | DM |
| `moffatParam` | float | Moffat exponent of the influence functions | DM |
| `signedAmplitude` | m | Stroke per unit command. A negative sign flips the DM response. | DM |
| `MechCoupling` | fraction | Mechanical coupling: an actuator's influence at its neighbors | DM |
| `FlipLeftRight`, `FlipTopBottom` | bool | Mirror the actuator grid | DM |

The DM geometry (rotation, shifts, anamorphosis) is not a params key. It is given to the
constructor as `misreg=dict(...)` or fitted by
{class}`~AI4AO.TwinCalibrator.TwinCalibrator` (see {doc}`guide/deformable_mirror`).

## `PSFParams`: science camera, for `PSFModel` / `PSFDataset`

| Key | Type / unit | Meaning |
|---|---|---|
| `Wavelength` | m | Science wavelength (default: the sensing wavelength) |
| `sampling` | float or `[min, max]` | PSF pixels per λ/D (default `WFSParams["sampling"]`). A range draws one value per batch. |
| `fov` | λ/D, float or `[min, max]`, optional | Field of view (default: the full grid). A range draws one value per batch. |
| `NCPA` | `[min, max]`, m | RMS of the random static aberration (default `[0, 0]`) |
| `NCPASlope` | float | Power-law slope of the static aberration spectrum (default 2.2) |
| `Jitter` | `[min, max]`, mas | Residual jitter RMS per axis (default `[0, 0]`) |
| `NoisePhotonFactor`, `NoiseRONFactor` | float | WFS sensitivity constants of the noise term. Required unless `"noise"` is removed from `terms`. |

## `TrainParams`

`TrainParams` (learning rates, number of training steps, …) is not read by the package.
The tutorial notebooks read it to configure their own training cells. With
{meth}`Trainer.train_many <AI4AO.Trainer.Trainer.train_many>`, the training settings are
given as {class}`~AI4AO.Trainer.Stage` and {class}`~AI4AO.Trainer.Recipe` objects
instead (see {doc}`guide/training`).
