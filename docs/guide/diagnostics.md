# Diagnosing a reconstructor CNN

When a reconstructor trains badly, {class}`~AI4AO.CNNDiagnostics.CNNDiagnostics` shows
what happens inside it: feature maps, weight and activation gradients, input saliency,
per-mode Jacobians and a per-layer history of scalar statistics. It works with hooks,
so it records the **real** end-to-end gradients of your own training step, backpropagated
through the WFS, the DM and the loss.

All examples run from the repository root on CPU (see {doc}`../params`).

## Setup

A small network and a batch of real preprocessed WFS frames to feed it:

```python
from contextlib import nullcontext

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from mmengine import Config

from AI4AO import PhaseDataset, PyramidWFS, FramePreprocess, CNNDiagnostics, imshow

device = "cpu"
p = Config.fromfile("Tutorials/basics/wfs_params_exp.py")
dataset = PhaseDataset(p.WFSParams, p.AtmosParams, p.LoopParams, p.DMParams, device)
wfs = PyramidWFS(p.WFSParams, device).eval()
preprocess = FramePreprocess(p.WFSParams, wfs, device)
preprocess.ProcessReference(wfs.reference_intensity)

with torch.no_grad():
    x = preprocess.ProcessFrame(wfs(dataset[0]["opd"]), add_pupil_noise=False)

model = nn.Sequential(
    nn.Conv2d(4, 16, 5, padding=2), nn.GELU(), nn.MaxPool2d(4),
    nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1),
    nn.Flatten(), nn.Linear(32, 64),
)

imshow(x[:2])     # the network input: four pupil images for each of two samples
plt.show()
```

## A quick look: `run`

`run` does one forward and backward pass on a batch (the default loss is the sum of
squared outputs) and records everything. `summary()` lists the hooked layers. Layers are
addressed by their `named_modules()` path (`"0"`, `"3"`, … for a `Sequential`) or by an
integer index.

```python
diag = CNNDiagnostics(model)
output = diag.run(x, input_grad=True)
print(diag.summary())

diag.plot_feature_maps("0", max_channels=8)
diag.plot_gradient_flow()
diag.plot_saliency()          # which input pixels the output is most sensitive to
plt.show()
```

## Per-mode Jacobians

`jacobian(x, modes)` gives the derivative of selected output modes with respect to the
input frames. It shows which pupil pixels each mode is read from.

```python
J = diag.jacobian(x[:2], modes=[0, 1, 2])     # (n_modes, batch, 4, 46, 46)
print(J.shape)
diag.plot_jacobian(J, x[:2], modes=[0, 1, 2])
plt.show()
```

## Inside a training loop

Wrap the steps you want to inspect in `capture()`. Everything computed inside the block
(forward, `backward`, `optimizer.step`) is recorded, so the gradients are those of your
real loss. `record(step)` appends one row of per-layer scalars to the history.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
target = torch.zeros(x.shape[0], 64)

for step in range(20):
    log = step % 5 == 0
    with (diag.capture(input_grad=True) if log else nullcontext()):
        loss = (model(x) - target).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if log:
        diag.record(step)

diag.plot_history("weight_grad_norm")
diag.plot_history("update_ratio")
plt.show()
```

{class}`~AI4AO.Trainer.TrainingRun` also takes a `diagnostics=` argument, so
`Trainer.train_many` records diagnostics every `log_every` steps without a hand-written
loop.

## Derivative maps of the twin

{func}`~AI4AO.Utils.jacobian_column` gives the derivative of a whole output (a WFS frame, a
PSF, an iMat) with respect to one scalar parameter, as a map shaped like the output. It
uses two backward passes, so it also works through the property setters of the twin.
Here, how a Pyramid frame changes with the DM rotation, per degree:

```python
from AI4AO import DeformableMirror
from AI4AO.Utils import jacobian_column

wfs_d = PyramidWFS(dict(p.WFSParams, useNoise=False), device).eval()
dm_d = DeformableMirror(p.WFSParams, p.DMParams, device).train()
M2C_d = dm_d.MakeZernikeM2C()
command = M2C_d[:, 6][None]                    # one mode, 1 rad RMS

frame = wfs_d(dm_d(command))
# rotationAngle is stored as rotation / 180, so divide by 180 for a derivative per degree
d_frame_d_rotation = jacobian_column(frame, dm_d._rotationAngle, 0) / 180
imshow(d_frame_d_rotation[0], colorbar=True)
plt.show()
```

## Saving an animation

{func}`~AI4AO.Utils.save_gif` writes frames rendered with
{func}`~AI4AO.Utils.figure_to_rgb` to a GIF. It is much lighter than embedding
`FuncAnimation.to_jshtml()` in a notebook. Pass the colormaps used in the frames, so
the palette keeps every colormap level.

```python
import os
import tempfile

from AI4AO.Utils import figure_to_rgb, save_gif

fig, ax = plt.subplots(figsize=(3, 3))
im = ax.imshow(x[0, 0], cmap="RdBu_r")
frames = []
for k in range(4):
    im.set_data(x[k % x.shape[0], k % x.shape[1]])
    frames.append(figure_to_rgb(fig))
plt.close(fig)

path = os.path.join(tempfile.mkdtemp(), "pupils.gif")
size_mb = save_gif(frames, path, fps=4, cmaps=["RdBu_r"])
print(f"{size_mb:.3f} MB")
```
