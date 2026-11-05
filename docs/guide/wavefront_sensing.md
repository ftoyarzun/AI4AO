# Wavefront sensing

A wavefront sensor (WFS) turns an OPD screen into a detector frame. In AI4AO the base
class {class}`~AI4AO.TorchPropagator.WFS` owns the optics shared by all sensors: the
pupil, the Fourier propagation, detector noise, PSFs and the classical interaction
matrix. Each sensor type subclasses it and provides its focal-plane mask:

| Class | Sensor |
|---|---|
| {class}`~AI4AO.PyramidWFS.PyramidWFS` | Pyramid WFS, with optional circular modulation |
| {class}`~AI4AO.ZernikeWFS.ZernikeWFS` | Zernike WFS (`MaskType="zwfs"`) and vector Zernike WFS (`"vzwfs"`) |

The WFS is also the place where OPD (meters) becomes phase (radians), at its sensing
wavelength `WFSParams["Wavelength"]`. Everything upstream stays in meters.

All examples run from the repository root on CPU. The params come from
`Tutorials/basics/wfs_params_exp.py` (see {doc}`../params`).

## Setup

```python
import torch
from mmengine import Config

from AI4AO import PhaseDataset, PyramidWFS, ZernikeWFS, FramePreprocess, DeformableMirror

device = "cpu"
p = Config.fromfile("Tutorials/basics/wfs_params_exp.py")
WFSParams, AtmosParams = p.WFSParams, p.AtmosParams
LoopParams, DMParams = p.LoopParams, p.DMParams

dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, DMParams, device)
batch = dataset[0]
```

## A pyramid WFS frame

`.eval()` builds the mask and the reference (flat-wavefront) intensity, and freezes the
sensor's learnable parameters. Keep the sensor in train mode only when you optimize the
sensor itself, for example during calibration.

```python
wfs = PyramidWFS(WFSParams, device).eval()

# Detector noise follows the conditions drawn with the screens
wfs.SetPhotonsAndRON(batch["nphotons"], batch["ron"])

with torch.no_grad():
    frame = wfs(batch["opd"], batch["pupil"])   # pupil is optional for a flat aperture

print(frame.shape)                       # (16, 120, 120): Nres * sampling pixels
print(wfs.reference_intensity.shape)     # (120, 120)
```

The frame is normalized to unit sum. With `useNoise=True`, Poisson noise (with
`nphotons` photons per frame) and Gaussian read-out noise are added before
normalization.

### Modulation

`modulation` is the radius of the circular modulation, in λ/D. Setting it rebuilds the
mask.

```python
wfs.modulation = 3
with torch.no_grad():
    modulated = wfs(batch["opd"])
wfs.modulation = 0
```

## A Zernike WFS

The Zernike sensor needs three extra keys. `Use_MTF=True` switches to a matrix Fourier
transform that zooms on the focal-plane dot, with `MTF_upscale` as the zoom factor.

```python
zernike_params = dict(WFSParams, MaskType="zwfs", Use_MTF=False, MTF_upscale=4)
zwfs = ZernikeWFS(zernike_params, device).eval()
with torch.no_grad():
    zframe = zwfs(batch["opd"])
print(zframe.shape)
```

## PSFs

`GetPSF` uses the same pupil and optics. `sampling` is in pixels per λ/D and `fov` in
λ/D. A Strehl ratio is the PSF peak divided by the peak of the aberration-free PSF.

```python
psf = wfs.GetPSF(batch["opd"], sampling=4, fov=20)
psf_perfect = wfs.GetPSF(torch.zeros_like(batch["opd"][:1]), sampling=4, fov=20)
strehl = psf.amax(dim=(-2, -1)) / psf_perfect.max()
print(psf.shape, strehl[:3])     # (16, 80, 80)
```

## Polychromatic sensing

Assign a 1-D tensor of wavelengths to sense a band. All wavelengths are propagated in
parallel and summed, so the frame keeps its shape. Pass `collapse_wvl=False` to keep
one frame per wavelength.

```python
wfs.wavelength = torch.linspace(500e-9, 800e-9, 4)
with torch.no_grad():
    broadband = wfs.Propagator(batch["opd"])                        # detector frame, with noise
    per_wavelength = wfs.Propagator(batch["opd"], collapse_wvl=False)   # noiseless, per wavelength
print(broadband.shape, per_wavelength.shape)   # (16, 120, 120) and (16, 4, 120, 120)

wfs.wavelength = WFSParams["Wavelength"]       # back to monochromatic
```

The per-wavelength frames are a diagnostic view of the optics. Detector noise applies
to the frame the detector records, which is their sum, so `collapse_wvl=False` frames
never get noise, whatever `useNoise` says.

## Preparing frames for a network

{class}`~AI4AO.FramePreprocess.FramePreprocess` crops each pupil image out of the detector
frame, optionally bins it, subtracts the reference and normalizes it. The output is the
input of a reconstructor network: one channel per pupil image (4 for a pyramid).

```python
preprocess = FramePreprocess(WFSParams, wfs, device)
preprocess.ProcessReference(wfs.reference_intensity)   # call again if the reference changes

with torch.no_grad():
    frame = wfs(batch["opd"])
    x = preprocess.ProcessFrame(frame)                          # with pupil-position/size jitter
    x_clean = preprocess.ProcessFrame(frame, add_pupil_noise=False)
print(x.shape)    # (16, 4, 46, 46): Nres + Extract_pupils_pad pixels per side
```

The jitter (`Center_noise`, `Pupil_size_noise`) is a training-time augmentation that
makes the network robust to small pupil misregistrations. Disable it for evaluation.

## The classical linear reconstructor

The WFS also calibrates the classical reconstructor. It pushes each mode of a basis
through the sensor to build the interaction matrix, then inverts it.

```python
dm = DeformableMirror(WFSParams, DMParams, device).eval()
M2C = dm.MakeZernikeM2C()           # modes-to-commands matrix, (totalAct, Nmodes)
modes_opd = dm(M2C.T)               # the modes as DM surfaces, (Nmodes, Nres, Nres)

with torch.no_grad():
    wfs.BuildReconstructionMatrix(modes_opd)
    z = wfs.GetReconstructedOPD(wfs(batch["opd"]))   # modal coefficients
print(wfs.iMat.shape, z.shape)      # (64, 120, 120) and (16, 64)
```

Higher-level code rarely calls these methods directly:
{meth}`Trainer.linear_reconstructor <AI4AO.Trainer.Trainer.linear_reconstructor>` wraps
them (see {doc}`training`).
