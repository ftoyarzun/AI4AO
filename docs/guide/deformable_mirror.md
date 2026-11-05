# Deformable mirror and the closed loop

{class}`~AI4AO.DeformableMirror.DeformableMirror` turns actuator commands into a mirror
surface, as an OPD in meters. Each actuator has a Moffat-shaped influence function on a
square grid clipped to the pupil. The geometry of the DM relative to the WFS pupil
(rotation, shift, anamorphic scaling, flips, sign) and the influence-function shape are
learnable parameters, so the DM can be calibrated by gradient descent (see
{doc}`calibration`).

All examples run from the repository root on CPU. The params come from
`Tutorials/basics/wfs_params_exp.py` (see {doc}`../params`).

## Setup

```python
import torch
from mmengine import Config

from AI4AO import PhaseDataset, PyramidWFS, DeformableMirror

device = "cpu"
p = Config.fromfile("Tutorials/basics/wfs_params_exp.py")
WFSParams, AtmosParams = p.WFSParams, p.AtmosParams
LoopParams, DMParams = p.LoopParams, p.DMParams
```

## Commands, modes and the M2C matrix

The DM is driven with **zonal** commands, one value per actuator. Training and
reconstruction work in a **modal** basis instead. `MakeZernikeM2C` builds the
modes-to-commands matrix `M2C` that fits Zernike modes with the DM's influence functions.

```python
dm = DeformableMirror(WFSParams, DMParams, device).eval()
print(dm.Nact, int(dm.totalAct))     # 11 x 11 grid, 101 actuators inside the pupil

M2C = dm.MakeZernikeM2C()            # (totalAct, Nmodes) = (101, 64)

# Surfaces for a batch of modal coefficients z (B, Nmodes)
z = torch.zeros(2, M2C.shape[1])
z[0, 3] = 1.0                        # one unit of the 4th Zernike mode
surface = dm(z @ M2C.T)              # (2, Nres, Nres), meters
```

The modes are scaled so that a coefficient of 1 is **1 radian RMS at the sensing
wavelength**: an OPD of λ/2π, about 101 nm for the 635 nm demo sensor.

```python
modes = dm(M2C.T)                    # every mode as a surface, (Nmodes, Nres, Nres)
print(modes[:4][:, dm.pupil].std(dim=-1) * 1e9)    # ~101 nm each
```

To project an OPD onto the modes (the "oracle" measurement), use the pseudo-inverse of
the mode surfaces:

```python
z_inv = torch.linalg.pinv(modes.flatten(start_dim=-2))     # (Nres*Nres, Nmodes)
z_back = surface.flatten(start_dim=-2) @ z_inv
print(z_back[0, :5])                 # ~[0, 0, 0, 1, 0]
```

## Misregistration

Pass a `misreg` dict to place the DM off-nominal, and read the current values back with
`GetMisreg()`. Rotations and the anamorphosis angle are in degrees, shifts in meters, and
scalings in percent of the diameter.

```python
misreg = dict(rotationAngle=10.0, shiftX=0.01, shiftY=0.0,
              anamorphosisAngle=0.0, radialScaling=1.0, tangentialScaling=0.0)
dm_rotated = DeformableMirror(WFSParams, DMParams, device, misreg=misreg).eval()

geometry, dm_config = dm_rotated.GetMisreg()
print(round(geometry["rotationAngle"], 3), dm_config["MechCoupling"])
```

Parameters with a property setter, for example `dm.flip_lr`, `dm.rotationAngle` or
`dm.offset_to_fit_number_of_actuators`, rebuild the actuator grid when assigned.

## A closed loop with a perfect sensor

The pieces combine into an AO loop. Here the "measurement" is the exact modal projection
of the residual, so the loop shows the best correction this DM can reach. A real loop
replaces that line with a reconstructor applied to the WFS frame. The
{class}`~AI4AO.Trainer.Trainer` runs these loops for you (see {doc}`training`).

```python
dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, DMParams, device)
batch = dataset[0]
gain, leak = batch["loop_gain"], batch["loop_leak"]       # (B, 1) each

z_command = torch.zeros(batch["opd"].shape[0], M2C.shape[1])
correction = torch.zeros_like(batch["opd"])

with torch.no_grad():
    for i in range(30):
        opd = batch["opd"] if i == 0 else dataset[i]["opd"]
        residual = opd - correction
        z_measured = residual.flatten(start_dim=-2) @ z_inv     # oracle measurement
        z_command = leak * z_command + gain * z_measured          # leaky integrator
        correction = dm(z_command @ M2C.T)

open_loop = opd[:, dm.pupil].std(dim=-1).mean() * 1e9
closed_loop = residual[:, dm.pupil].std(dim=-1).mean() * 1e9
print(f"{open_loop:.0f} nm -> {closed_loop:.0f} nm RMS")
```

## Saving and loading a calibrated DM

`SaveCalibration` writes the learnable state and the configuration needed to rebuild the
geometry. `LoadCalibration` restores them into a DM built from the same params.

```python
import tempfile
from pathlib import Path

path = Path(tempfile.mkdtemp()) / "DemoDM.pth"
dm_rotated.SaveCalibration(path)

restored = DeformableMirror(WFSParams, DMParams, device)
restored.LoadCalibration(path)
restored.eval()
print(round(restored.GetMisreg()[0]["rotationAngle"], 3))
```
