# PSFs from the residual PSD

To produce many AO point-spread functions (PSFs) quickly, for example to train a PSF
estimator, simulating turbulent screens is often unnecessary.
{class}`~AI4AO.PSFDataset.PSFModel` computes the **long-exposure PSF analytically** from
the residual-OPD power spectrum of the AO system. The spectrum sums fitting, servo-lag,
aliasing and WFS-noise terms, plus static aberrations (NCPA) and jitter.
{class}`~AI4AO.PSFDataset.PSFDataset` draws random conditions and returns batches of
these PSFs.

The model is differentiable with respect to its parameters (r0, gain, …). It reuses the
grid and normalization of `WFS.GetPSF`: with every term switched off, it reproduces
`GetPSF` exactly.

All examples run from the repository root on CPU (see {doc}`../params`).

## Setup

The PSF side has its own params dict. The values below are illustrative, for the demo
instrument only.

```python
import torch
from mmengine import Config

from AI4AO import WFS, PSFDataset, PSFModel

device = "cpu"
p = Config.fromfile("Tutorials/basics/wfs_params_exp.py")
WFSParams, AtmosParams = p.WFSParams, p.AtmosParams
LoopParams, DMParams = p.LoopParams, p.DMParams

PSFParams = {
    "Wavelength": 1.2e-6,       # science wavelength (m)
    "sampling": 3,              # PSF pixels per lambda/D
    "fov": 24,                  # field of view (lambda/D)
    "NCPA": [10e-9, 40e-9],     # static aberration rms range (m)
    "Jitter": [1.0, 5.0],       # residual jitter rms per axis (mas)
    "NoisePhotonFactor": 1 / 4,
    "NoiseRONFactor": 1.0,
}
```

## A batch of PSFs

```python
wfs = WFS(WFSParams, device)
psf_dataset = PSFDataset(wfs, WFSParams, AtmosParams, LoopParams, DMParams, PSFParams, device)

batch = psf_dataset[0]
print(batch["psf"].shape)           # (16, 72, 72): fov * sampling pixels
print(batch["strehl"][:3])          # Strehl ratio at the science wavelength
print(batch["r0"][:3])              # the conditions behind each PSF are in the batch
```

## Using the model directly

`psf_dataset.model` is the `PSFModel`. Draw a parameter dict, or build one yourself,
then ask for the PSD, the OTF, the PSF or the Strehl ratio:

```python
model = psf_dataset.model
params = psf_dataset.DrawRandomParameters()

psd, terms = model.ResidualPSD(params, return_terms=True)
for name, term in terms.items():
    rms_nm = torch.sqrt(term[0].sum() * model.dF**2) * 1e9
    print(f"{name:8s} {rms_nm:6.1f} nm rms")

long_exposure = model.PSF(params)                  # (B, H, W)
short_exposure = model.ShortExposurePSF(params)    # one random instantaneous PSF per sample
strehl = model.Strehl(model.OTF(params))
```

## Fitting conditions to a PSF

Because the model is differentiable, conditions can be fitted to an observed PSF by
gradient descent. Here r0 is recovered from a PSF made with a known value:

```python
LAYER_KEYS = ("fractional_r0", "layer_heights", "wind_speed", "wind")

def select_sample(params, i):
    """Sample i of a params dict, kept as a batch of one."""
    out = {}
    for key, value in params.items():
        if not torch.is_tensor(value):       # sampling and fov are shared by the batch
            out[key] = value
        elif key in LAYER_KEYS:               # layer quantities: batch on the last axis
            out[key] = value[..., i:i + 1]
        else:                                 # everything else: batch on the first axis
            out[key] = value[i:i + 1]
    return out

truth = select_sample(psf_dataset.DrawRandomParameters(), 0)
truth["r0"] = torch.tensor([0.08])
target = model.PSF(truth)

log_r0 = torch.tensor([0.0], requires_grad=True)       # start at r0 = 1 m
optimizer = torch.optim.Adam([log_r0], lr=0.1)
for step in range(100):
    fit = dict(truth, r0=log_r0.exp())
    loss = ((model.PSF(fit) - target) ** 2).mean() / (target ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"fitted r0 = {log_r0.exp().item():.3f} m (true 0.080)")
```

Note the layout of a params dict: layer quantities (`fractional_r0`, `layer_heights`,
`wind_speed`, `wind`) carry the batch on their **last** axis, everything else on the
first.

## Variable PSF sizes

`sampling` and `fov` can also be `[min, max]` ranges. Each batch then draws one shared
sampling and field of view, so the PSF size changes from batch to batch:

```python
ranged = PSFDataset(wfs, WFSParams, AtmosParams, LoopParams, DMParams,
                    dict(PSFParams, sampling=[2, 4], fov=[12, 24]), device)
for _ in range(3):
    b = ranged[0]
    print(round(b["sampling"], 2), round(b["fov"], 1), tuple(b["psf"].shape[-2:]))
```
