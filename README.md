# AI4AO

AI4AO ("Artificial Intelligence for Adaptive Optics") is a PyTorch-based,
fully differentiable end-to-end simulation framework for
adaptive optics (AO): atmospheric turbulence, wavefront sensor (WFS)
propagation, phase reconstruction, and deformable mirror (DM) response, all
as a chain of `nn.Module`s. Because gradients flow through the whole chain,
the framework can be used not just to simulate an AO system but to calibrate
one via backpropagation, training a neural-network (NN) reconstructor, and also
fitting the WFS mask shape and DM misregistration parameters.

 This code is inspired from OOPAO: https://github.com/cheritier/OOPAO developped by C.T. Heritier.
The project is under active research development — conventions and module
APIs still shift, so treat in-repo docstrings/code as more authoritative than
this README where they disagree. It was initially built for personal use and
is now intended for outside collaborators too.

## Pipeline

Each instrument "twin" is built from these modules, in order:

- **`PhaseDataset.py`** — synthesizes atmospheric phase screens on the fly
  from a von Kármán PSD (multi-layer, with optional scintillation).
- **`TorchPropagator.py` (`WFS`)** — base optical propagator, with optional modulation and noise; also does classical
  interaction-/reconstruction-matrix calibration.
- **`PyramidWFS.py` / `ZernikeWFS.py`** — the current WFS-specific mask
  implementations, subclassing `WFS`.
- **`FramePreprocess.py`** — crops/bins/normalizes pupil images from the raw
  detector frame for the NN reconstructor.
- **`DeformableMirror.py`** — turns command coefficients into a DM phase
  surface; misregistration is modeled as learnable parameters.
- **`Trainer.py`** — the closed-loop training loop for a reconstructor, plus
  checkpointing.
- **`TwinCalibrator.py`** — fits a constructed WFS/DM twin to reference data
  (bench or simulated) and saves/loads a calibrated twin's state.

`MaskGeneration.py` (`MaskManager`) and `PhaseEstimators.py` are older,
not-currently-wired modules (a generic pre-`PyramidWFS`/`ZernikeWFS` mask
dispatcher, and a collection of reconstructor architectures) — not where new
work should start.

## Instrument "twin" configuration

Each instrument (Ekarus, Oziriis, Papyrus, Rama, ...) has a params file under
`Tutorials/` defining five plain Python dicts — `WFSParams`, `AtmosParams`,
`LoopParams`, `DMParams`, `TrainParams`, used by the
pipeline constructors.

Bench interaction matrices and the calibrated/trained artifacts the notebooks
produce live under `Data/<Instrument>/`, resolved via `AI4AO.paths`
(`<repo root>/Data` by default, or `$AI4AO_DATA_DIR`). The bench files you need
to supply are listed in [`Data/README.md`](Data/README.md); the `basics/`
series needs none of them.

## Getting started

[`Tutorials/`](Tutorials/README.md) is the primary way this codebase is
exercised and learned. Start with the `Tutorials/basics/` series, in order:

1. `01_Dataset.ipynb` — generating turbulence with `PhaseDataset`.
2. `02_WFSAndPreprocessing.ipynb` — the optical propagator and preprocessing.
3. `03_DeformableMirrorAndClosedLoop.ipynb` — the DM and closed-loop
   feedback, using a perfect reconstructor.
4. `04_TwinCalibrationGroundTruth.ipynb` — Calibrating a WFS and DM to a ground-truth bench
5. `05_TrainingAReconstructor.ipynb` — using the calibrated WFS and DM, and replacing the perfect reconstructor with a trained
   network via `Trainer`.

Notebooks 1–3 run against a synthetic, uncalibrated instrument; notebook 4
then calibrates that same instrument against a synthetic "fake bench" with
deliberately known parameter values, so the fit can be checked against ground
truth — something no real per-instrument calibration can do — and notebook 5
trains a reconstructor on the resulting calibrated twin. No real bench data is
needed anywhere in this series. See
[`Tutorials/README.md`](Tutorials/README.md) for the per-instrument
calibration/training notebooks that follow.

## Installation

Python >= 3.10 is required.

### Create a virtual environment (recommended)

```bash
python -m venv venv

# Unix
source ./venv/bin/activate

# Windows PowerShell
.\venv\Scripts\activate
```

### Install dependencies

Upgrade packaging tools first:

```bash
python -m pip install --upgrade pip setuptools wheel typing-extensions
```

Install PyTorch. For a specific CUDA build, install it first following the
instructions for your platform at
<https://pytorch.org/get-started/locally/> (CUDA is strongly recommended):

```bash
# example -- pick the command for your platform/CUDA from the link above
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then clone and install AI4AO:

```bash
git clone https://github.com/ftoyarzun/AI4AO.git
python -m pip install -e AI4AO
```

`torch` is not a hard dependency (so the step above stays in your control). If
you don't need a specific CUDA build, skip it and let pip pull a default wheel:

```bash
python -m pip install -e "AI4AO[torch]"
```

The TensorRT deployment path (`AI4AO/DAO_Utils.py`) needs extra packages and a
matching CUDA/TensorRT toolchain on the machine:

```bash
python -m pip install -e "AI4AO[trt]"   # pycuda, tensorrt
```

### Updating an existing checkout

If you already had AI4AO installed from an earlier version:

```bash
git pull                       # or: git checkout <branch>
python -m pip install -e .     # re-run: picks up new modules (AI4AO.paths),
                               # clears any stale editable registration
```

What changed that you may need to act on:

- **Data directory.** Bench files and saved twins/checkpoints are now found
  through `AI4AO.paths` — `<repo root>/Data` by default (the same place the
  notebooks' old `../../Data` resolved to, so nothing to do if you kept data
  there). If yours lives elsewhere, `export AI4AO_DATA_DIR=/path/to/data`
  instead of moving it. See [`Data/README.md`](Data/README.md).
- **`TwinCalibrator.save()` / `load()`** now default `data_dir` to that data
  directory (was `"../Data"`, relative to the working directory). Pass
  `data_dir=...` explicitly if you relied on the old default.
- **CPU.** The simulation, calibration and training now run without a GPU with
  no code edits — `device` auto-selects, and CUDA-only optimizer paths are
  gated. (`AI4AO/DAO_Utils.py`'s TensorRT inference is still GPU-only.)
- **Notebooks.** The tutorial notebooks' `device = ...` and `PATH = ...` cells
  were rewritten; if you have local edits to them, expect merge conflicts
  there.

## Testing

`tests/` contains a `pytest` suite covering the core modules, differentiability
regression checks, and an end-to-end training smoke test. Install the test
extras and run it from the repo root:

```bash
python -m pip install -e "AI4AO[test]"
pytest tests/
```

The `test` extra pulls in `pytest` and a default `torch` wheel. If you already
installed a specific CUDA build of torch, that one is kept.

Tests marked `slow` exercise `Trainer.train`/`evaluate` or
`TwinCalibrator.fit_*` loops end-to-end; skip them for a faster run:

```bash
pytest tests/ -m "not slow"
```

By default tests run on CPU; set `AI4AO_TEST_DEVICE=cuda` to run on GPU
instead.

## License

MIT — see [`LICENSE`](LICENSE).
