# Training and evaluating reconstructors

{class}`~AI4AO.Trainer.Trainer` trains a reconstructor network **inside the closed loop**.
At every step it draws turbulence, propagates the residual through the WFS, feeds the
preprocessed frame to the network, applies the predicted correction with the DM, and
backpropagates a loss through the whole chain. The WFS and DM stay frozen (eval mode)
while the network trains.

All examples run from the repository root on CPU, with the demo instrument of
`Tutorials/basics/wfs_params_exp.py` (see {doc}`../params`). The step counts are tiny so
that the page runs in seconds. Real training uses thousands of steps on a GPU.

## Setup: the twin

```python
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from mmengine import Config

from AI4AO import PhaseDataset, PyramidWFS, FramePreprocess, DeformableMirror, Trainer
from AI4AO.Trainer import Stage, Recipe, TrainingRun, load_reconstructor
from AI4AO.LossFunctions import LogResidualVarianceLoss, Physics_loss, RMSELoss

device = "cpu"
p = Config.fromfile("Tutorials/basics/wfs_params_exp.py")
WFSParams, AtmosParams = p.WFSParams, p.AtmosParams
LoopParams, DMParams = p.LoopParams, p.DMParams

dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, DMParams, device)
dataset.generateClosedLoop = True        # train on closed-loop residuals

wfs = PyramidWFS(WFSParams, device).eval()
dm = DeformableMirror(WFSParams, DMParams, device).eval()
preprocess = FramePreprocess(WFSParams, wfs, device)
preprocess.ProcessReference(wfs.reference_intensity)
M2C = dm.MakeZernikeM2C()
```

## The reconstructor

Any `nn.Module` works if it maps the preprocessed frames `(B, 4, 46, 46)` to modal
coefficients `(B, Nmodes)`. This one is deliberately tiny:

```python
def small_cnn():
    return nn.Sequential(
        nn.Conv2d(4, 16, 5, padding=2), nn.GELU(), nn.MaxPool2d(4),
        nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(32, M2C.shape[1]),
    )
```

## Losses

Losses live in {mod}`AI4AO.LossFunctions`. They all share one call signature, so the
`Trainer` can use any of them, and they combine with `+` and `*`:

```python
loss = LogResidualVarianceLoss(dataset.pupil, wfs.wavelength) + 0.1 * Physics_loss(wfs)
```

| Loss | What it minimizes |
|---|---|
| `LogResidualVarianceLoss` | log of the residual phase variance after correction. Strehl-like, and the usual choice. |
| `Physics_loss` | mismatch between the observed WFS frame and the frame predicted from the reconstruction |
| `RMSELoss`, `ModalMAELoss`, `Relative_Loss_Function` | error on the modal coefficients themselves |

Each loss is called as
`loss(Ze, z_estimated, pupil, residual_opd, corrected_residual_opd, wfs_frames)`, where
`Ze` holds the true modal coefficients of the residual. To write your own, subclass
{class}`~AI4AO.LossFunctions.AOLoss` and implement `compute` with that signature.

## Training several networks: `train_many`

`train_many` trains a list of {class}`~AI4AO.Trainer.TrainingRun` on a shared timeline of
{class}`~AI4AO.Trainer.Stage`. All runs see the same atmosphere at every step, so their
losses compare directly. Each stage counts simulated frames and the number of
closed-loop iterations backpropagated per optimizer step. Each run gets a fresh AdamW
per stage, from its {class}`~AI4AO.Trainer.Recipe` (lr, warm-up + cosine schedule,
optional per-submodule lr factors).

```python
out_dir = Path(tempfile.mkdtemp())       # use DATA_DIR / "<Instrument>" for real runs

trainer = Trainer(wfs, preprocess, dm, M2C, dataset=dataset, loss=loss)

runs = [
    TrainingRun("baseline", small_cnn(), output=out_dir / "baseline.pth"),
    TrainingRun("high_lr", small_cnn(), recipes=Recipe(lr=1e-3), output=out_dir / "high_lr.pth"),
]
stages = [Stage(frames=20, closed_loop_iterations=1),     # 20 optimizer steps
          Stage(frames=20, closed_loop_iterations=5)]     # 4 steps through 5 loop iterations

report = trainer.train_many(runs, stages, seed=0)
print(list(report.runs), [len(l) for l in report.runs["baseline"].loss])
```

`report.plot_losses()` draws one panel per stage. The final weights go to each run's
`output`, and an existing file there is first renamed to `<name>.prev.pth`. Pass
`init=path` to a `TrainingRun` to start from saved weights, and
`skip_trained=True` to `train_many` to load runs that are already trained instead of
retraining them.

## Evaluating: `evaluate_many`

`evaluate_many` runs closed-loop rollouts of every reconstructor with the same seeds, so
they see identical atmosphere and noise, and reports the steady-state residual (nm RMS)
and Strehl ratios. `references` adds the classical linear reconstructor and the oracle
(exact modal projection: the floor any reconstructor can reach).

```python
cnn = small_cnn()
load_reconstructor(cnn, out_dir / "baseline.pth")

evaluation = trainer.evaluate_many({"cnn": cnn}, seeds=(1, 2), n_steps=30,
                                   references=("linear", "oracle"), baseline="linear")
print(evaluation.table())
mean_delta, standard_error, per_seed = evaluation.paired_delta("cnn")
```

With such short training the network is still far behind the linear reconstructor. The
table format is what matters here. For real comparisons, use the default five seeds and
100 or more steps.

## Single-network wrappers

For a single network, `train()` and `evaluate()` are thin wrappers over the same loops.
The trainer then holds the network and its optimizer.

```python
net = small_cnn()
optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)
single = Trainer(wfs, preprocess, dm, M2C, phaseReconstructor=net, dataset=dataset,
                 loss=loss, optimizer=optimizer)

loss_history, ideal_history = single.train(training_steps=10, closed_loop_iterations=1)
result = single.evaluate(n_steps=20)
print(result.opd.shape, round(result.metrics["nm_rms"]))   # trajectories are (n_steps, B, ...)

single.save_checkpoint(out_dir / "single.pth")
```
