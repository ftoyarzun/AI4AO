# Calibrating a twin to a bench

A simulated twin is only useful if it behaves like the real instrument.
{class}`~AI4AO.TwinCalibrator.TwinCalibrator` fits an already-constructed WFS and DM to
data acquired on the bench, by gradient descent through the simulation:

- the WFS **reference frame** (flat wavefront) constrains the pupil geometry and the
  mask parameters;
- the bench **interaction matrix** (the WFS response to each DM mode) constrains the DM
  misregistration, the influence functions and static aberrations.

`TwinCalibrator` is instrument-agnostic. Loading the bench files and converting them to
the simulator's frame format is up to the caller (see `Tutorials/<Instrument>/` for
real examples).

This page fits a twin to a **synthetic bench** with a known misregistration, so the
result can be checked against the truth. It is a shortened version of the
`04_TwinCalibrationGroundTruth` tutorial. All examples run from the repository root on
CPU (see {doc}`../params` for the params).

## Setup: a synthetic "bench"

```python
import tempfile

import matplotlib.pyplot as plt
import torch
from mmengine import Config

from AI4AO import PyramidWFS, DeformableMirror, TwinCalibrator, imshow_multiple

device = "cpu"
p = Config.fromfile("Tutorials/basics/wfs_params_exp.py")
WFSParams = dict(p.WFSParams, useNoise=False, centralObstruction=0.0, Modulation=3.0)
DMParams = dict(p.DMParams)

# The "bench": a WFS and a DM that is rotated and shifted
true_misreg = dict(rotationAngle=5.0, shiftX=0.01, shiftY=-0.01,
                   anamorphosisAngle=0.0, radialScaling=0.0, tangentialScaling=0.0)
wfs_bench = PyramidWFS(WFSParams, device).eval()
dm_bench = DeformableMirror(WFSParams, DMParams, device, misreg=true_misreg).eval()

M2C = DeformableMirror(WFSParams, DMParams, device).MakeZernikeM2C()
with torch.no_grad():
    wfs_bench.BuildInteractionMatrix(dm_bench(M2C.T))
bench_iMat = wfs_bench.iMat.clone()            # (Nmodes, H, W): what the bench measured
bench_reference = wfs_bench.reference_intensity.clone()
```

On a real bench, `bench_iMat` and `bench_reference` come from files, and `M2C` is the
modal basis that was applied to the DM.

## Fitting the twin

Start from a nominal twin built from the same params:

```python
wfs = PyramidWFS(WFSParams, device)
dm = DeformableMirror(WFSParams, DMParams, device)
calibrator = TwinCalibrator(wfs, dm, device)
```

The usual sequence has three steps:

1. **WFS pupil and mask** against the reference frame. Here the twin already matches,
   so the fit has nothing to do. On a real bench it recovers the pupil positions,
   mask slopes and shifts.
2. **Rough DM orientation**: a discrete search over rotations, flips and sign.
3. **Joint fine fit** of the DM geometry, the WFS mask and the static offsets against
   the interaction matrix.

```python
calibrator.fit_pupil_to_reference(bench_reference, wfs.parameters(), n_iter=10,
                                  live_plot=False)
calibrator.rough_calibrate_dm(bench_iMat, M2C)

mode_index = list(range(M2C.shape[1]))
final_loss, _, _ = calibrator.fit_dm_and_offsets(bench_iMat, M2C, mode_index,
                                                 n_iter=30, live_plot=False)

recovered, _ = dm.GetMisreg()
for key in ("rotationAngle", "shiftX", "shiftY"):
    print(f"{key:14s} true {true_misreg[key]:+.4f}   fitted {recovered[key]:+.4f}")
```

30 iterations keep this page fast. Use a few hundred in practice, and keep
`live_plot=True` in a notebook to watch the fit.

## Checking and using the result

```python
modes = calibrator.rebuild_reconstruction_matrix(M2C)    # rebuild the twin's iMat
cross_talk = calibrator.crosstalk_diagnostic(bench_iMat)  # twin vs. bench, per mode

# The response to one mode: bench, twin and their difference, on the bench's scale
k = 10
imshow_multiple([
    {"tensor": bench_iMat[k], "title": f"bench, mode {k}"},
    {"tensor": wfs.iMat[k], "title": "twin"},
    {"tensor": bench_iMat[k] - wfs.iMat[k], "title": "bench - twin"},
], scale_reference=bench_iMat[k], colorbar=True)
plt.show()
```

`save` and `load` store the calibrated WFS and DM under `DATA_DIR / <instrument>`, or
under `data_dir` when given. The training notebooks then load them with
`wfs.LoadCalibration` / `dm.LoadCalibration`.

```python
wfs_path, dm_path = calibrator.save("Demo", data_dir=tempfile.mkdtemp())
```

## Calibrating from science-camera PSFs

When no WFS is available, `rough_calibrate_dm_to_psfs`, `fit_dm_to_psfs` and
`estimate_ncpa` fit the DM geometry and the non-common-path aberrations from focal-plane
images taken while poking the DM. See their API documentation in
{mod}`AI4AO.TwinCalibrator`.
