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
