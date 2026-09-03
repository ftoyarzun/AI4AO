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
## Getting started

[`Tutorials/`](Tutorials/README.md) is the primary way this codebase is
exercised and learned. Start with the `Tutorials/basics/` series, in order:

1. `01_Dataset.ipynb` — generating turbulence with `PhaseDataset`.
2. `02_WFSAndPreprocessing.ipynb` — the optical propagator and preprocessing.
3. `03_DeformableMirrorAndClosedLoop.ipynb` — the DM and closed-loop
   feedback, using a perfect reconstructor.
4. `04_TrainingAReconstructor.ipynb` — replacing the perfect reconstructor with a trained
   network via `Trainer`.

These run against a synthetic, uncalibrated instrument and need no real bench
data. See [`Tutorials/README.md`](Tutorials/README.md) for the per-instrument
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

Install PyTorch separately, following the instructions for your platform at
<https://pytorch.org/get-started/locally/> (CUDA is strongly recommended).

Then clone and install AI4AO:

```bash
git clone https://github.com/ftoyarzun/AI4AO.git
python -m pip install -e AI4AO
```

## Testing

`tests/` contains a `pytest` suite covering the core modules, differentiability
regression checks, and an end-to-end training smoke test. Install the test
extras and run it from the repo root:

```bash
python -m pip install -e "AI4AO[test]"
pytest tests/
```

Tests marked `slow` exercise `Trainer.train`/`evaluate` or
`TwinCalibrator.fit_*` loops end-to-end; skip them for a faster run:

```bash
pytest tests/ -m "not slow"
```

By default tests run on CPU; set `AI4AO_TEST_DEVICE=cuda` to run on GPU
instead.

## License

MIT — see [`LICENSE`](LICENSE).
