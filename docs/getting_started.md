# Getting started

## Installation

AI4AO needs Python 3.10 or later. A virtual environment is recommended:

```bash
python -m venv venv
source ./venv/bin/activate        # Unix
.\venv\Scripts\activate           # Windows PowerShell

python -m pip install --upgrade pip setuptools wheel typing-extensions
```

**Install PyTorch first.** It is deliberately not a hard dependency, so that you can
choose the build that matches your CUDA setup. Follow
<https://pytorch.org/get-started/locally/> (CUDA is strongly recommended), for example:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then clone and install AI4AO in editable mode:

```bash
git clone https://github.com/ftoyarzun/AI4AO.git
python -m pip install -e AI4AO
```

If you don't need a specific CUDA build, let pip pull a default torch wheel instead:

```bash
python -m pip install -e "AI4AO[torch]"
```

The TensorRT deployment path ({doc}`api/DAO_Utils`) needs extra packages and a matching
CUDA/TensorRT toolchain:

```bash
python -m pip install -e "AI4AO[trt]"   # pycuda, tensorrt
```

## Data directory

Bench interaction matrices, calibrated twins and trained checkpoints live under
`Data/<Instrument>/`, resolved by {doc}`api/paths`. The default location is
`<repo root>/Data`. To keep the data elsewhere, set the `AI4AO_DATA_DIR` environment
variable. The `Tutorials/basics/` series needs no bench data.

## First steps

Work through the {doc}`tutorials <tutorials/01_Dataset>` in order. Notebooks 1–3 build a
synthetic, uncalibrated instrument. Notebook 4 calibrates it against a synthetic "fake
bench" with known parameters, and notebook 5 trains a reconstructor on the calibrated
twin. Notebook 6 shows the PSD-based PSF model.

```{tip}
The tutorial pages on this site show code and explanations only. The notebooks are
committed without outputs and are not executed when the docs are built, because they
need a GPU. Run them locally from `Tutorials/basics/` to see the figures.
```

## Running the tests

```bash
python -m pip install -e "AI4AO[test]"
pytest tests/                 # full suite
pytest tests/ -m "not slow"   # skip the end-to-end training / calibration loops
```

Tests run on CPU by default. Set `AI4AO_TEST_DEVICE=cuda` to run them on the GPU.
