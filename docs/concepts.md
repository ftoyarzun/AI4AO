# Concepts

## The simulation pipeline

Each instrument "twin" is a fixed chain of modules:

| Stage | Module | Role |
|---|---|---|
| Turbulence | {doc}`api/PhaseDataset`, {doc}`api/InfinitePhaseScreen` | Synthesize atmospheric OPD screens from a von Kármán PSD (multi-layer, wind, optional scintillation). |
| Sensing | {doc}`api/TorchPropagator` | The optical propagator (`WFS`): mask, modulation, detector noise, interaction-matrix calibration. |
| WFS masks | {doc}`api/PyramidWFS`, {doc}`api/ZernikeWFS` | WFS-specific mask implementations, subclassing `WFS`. |
| Preprocessing | {doc}`api/FramePreprocess` | Crop, bin and normalize pupil images from the detector frame. |
| Correction | {doc}`api/DeformableMirror` | Turn command coefficients into a DM surface, with learnable misregistration. |
| Training | {doc}`api/Trainer` | Closed-loop training and evaluation of reconstructors. |
| Calibration | {doc}`api/TwinCalibrator` | Fit a constructed WFS/DM twin to bench data. |

{doc}`api/PSFDataset` is an alternative to turbulent screens: it maps the AO conditions to
a residual-OPD power spectrum and from there to long- or short-exposure PSFs.

Every OPD screen is generated on the fly from the PSD, so training sees an infinite,
non-repeating stream of samples. This is why there is no train/validation split anywhere
in the code.

## Units: OPD in meters

The fundamental wavefront quantity is the **optical path difference (OPD) in meters**,
carried end-to-end. Atmosphere generation, DM output and residual-wavefront arithmetic
all work in OPD. Conversion to phase in radians,

$$\phi = \frac{2\pi}{\lambda}\,\mathrm{OPD},$$

happens only at the sensing boundary: inside `WFS.Propagator`/`WFS.GetPSF`, in the
scintillation (Fresnel propagation) step and in the PSF model's OTF. Variables holding
OPD are named `opd`/`*_opd`.

## Tensor conventions

- OPD screens and pupils have shape `(Nphases, Nres, Nres)`. `pupil` is a boolean
  circular mask, and reductions over the aperture use `opd[..., pupil.bool()]`.
- `WFS.wavelength` can be a scalar or a 1-D tensor of wavelengths. A tensor turns on
  polychromatic propagation: all wavelengths are computed in parallel and summed, so the
  returned frame keeps the shape `(Nphases, Nres, Nres)`.
- Everything from OPD generation to the loss is differentiable with respect to the OPD,
  the DM parameters and, optionally, the WFS mask.

## Instrument configuration

Each instrument has a folder under `Tutorials/<Instrument>/` with a params file that
defines plain Python dicts (`WFSParams`, `AtmosParams`, `LoopParams`, `DMParams`,
`TrainParams`). These are passed to the constructors above. There is no separate
YAML/JSON configuration layer.
