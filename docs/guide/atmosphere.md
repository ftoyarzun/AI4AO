# Atmospheric turbulence

AI4AO synthesizes atmospheric **OPD screens** (optical path difference, in meters) on the
fly from a von Kármán power spectrum. No dataset is stored: every call draws fresh
screens, so training sees an unlimited, non-repeating stream of samples. Two
interchangeable generators exist:

| Class | Use it when |
|---|---|
| {class}`~AI4AO.PhaseDataset.PhaseDataset` | The default. Fast, periodic screens (a torus that repeats after $2D/v$). |
| {class}`~AI4AO.InfinitePhaseScreen.InfinitePhaseScreen` | Long sequences that must never repeat, or the full tip/tilt power. About 5× slower per frame. |

Every example on this page runs from the repository root on CPU and uses the demo
instrument defined in `Tutorials/basics/wfs_params_exp.py` (see {doc}`../params`).

## Setup

```python
import torch
from mmengine import Config

from AI4AO import PhaseDataset, InfinitePhaseScreen

device = "cpu"  # or "cuda"
p = Config.fromfile("Tutorials/basics/wfs_params_exp.py")
WFSParams, AtmosParams = p.WFSParams, p.AtmosParams
LoopParams, DMParams = p.LoopParams, p.DMParams
```

## Drawing a batch of screens

`dataset[i]` returns a dict. Index `0` draws new conditions (r0, L0, wind, number of
layers, photons, read noise, …) for a whole batch of `AtmosParams["Nphases"]`
independent screens. Indices `1, 2, …` advance the **same** screens in time by one loop
period each (frozen flow, every layer moving with its own wind).

```python
dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, DMParams, device)

batch = dataset[0]
opd = batch["opd"]          # (Nphases, Nres, Nres) = (16, 40, 40), meters
pupil = batch["pupil"]      # (16, 40, 40) bool, the telescope aperture

print(opd.shape, opd.dtype)
print(f"r0 of the first screen: {batch['r0'][0]:.3f} m (at 500 nm)")

# RMS wavefront error over the aperture, in nm, for the first three screens
rms_nm = opd[:, dataset.pupil].std(dim=-1) * 1e9
print(rms_nm[:3])

next_frame = dataset[1]["opd"]   # the same screens, one loop period later
```

The batch dict also carries the conditions that were drawn, so that the WFS noise and
the loop can be configured to match:

| Key | Shape | Meaning |
|---|---|---|
| `opd` | `(B, Nres, Nres)` | OPD screen, meters, piston removed |
| `pupil` | `(B, Nres, Nres)` | Aperture (bool), or the scintillated amplitude when `Scintillation=True` |
| `nphotons`, `ron` | `(B, 1, 1)` | Photons per frame and read-out noise, for `WFS.SetPhotonsAndRON` |
| `r0`, `L0` | `(B,)` | Fried parameter at 500 nm and outer scale, meters |
| `wind` | `(2, L, B)` | Wind vector per layer, m/s |
| `fractional_r0` | `(L, B)` | Relative $C_n^2$ weight of each layer |
| `loop_gain`, `loop_leak`, `level_of_correction` | `(B, 1)` | Loop conditions (used in closed-loop mode) |

## Open-loop vs. closed-loop residuals

By default the screens are the full, uncorrected atmosphere. With
`generateClosedLoop = True`, the generator instead produces the *residual* that an AO
loop with the drawn gain, leak, delay and level of correction would leave. Use this to
train a reconstructor on realistic closed-loop residuals without simulating the loop
history.

```python
dataset.generateClosedLoop = True
residual = dataset[0]["opd"]
print(residual[:, dataset.pupil].std(dim=-1).mean() * 1e9, "nm RMS on average")
dataset.generateClosedLoop = False
```

## Reproducibility

The generators draw from the **global** torch and NumPy RNGs. Seed both to reproduce a
draw:

```python
from AI4AO.Utils import seed_everything

seed_everything(0)
a = dataset[0]["opd"]
seed_everything(0)
b = dataset[0]["opd"]
assert torch.equal(a, b)
```

## Scintillation

With `AtmosParams["Scintillation"] = True`, each layer is Fresnel-propagated to the
pupil, and `batch["pupil"]` becomes the resulting (float) amplitude instead of a boolean
mask. Pass it to the WFS so the sensor sees the intensity fluctuations.

```python
scint_params = dict(AtmosParams, Scintillation=True)
scint_dataset = PhaseDataset(WFSParams, scint_params, LoopParams, DMParams, device)
amplitude = scint_dataset[0]["pupil"]
print(amplitude.dtype, float(amplitude[0][scint_dataset.pupil].std()))
```

## Non-repeating screens: `InfinitePhaseScreen`

`InfinitePhaseScreen` is a drop-in replacement with the same constructor, batch dict and
public attributes. Its screens never repeat, and it can add "boiling" (AR(1) evolution
of the screens). Indices must not go backwards within a sequence: `[0]` starts a new one.

```python
infinite = InfinitePhaseScreen(WFSParams, AtmosParams, LoopParams, DMParams, device,
                               boiling=True, boilingTime=0.05)
first = infinite[0]
for i in range(1, 5):
    frame = infinite[i]
assert set(first) == set(batch)       # same keys as PhaseDataset
print(frame["opd"].shape)
```
