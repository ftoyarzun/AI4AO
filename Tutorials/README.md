# Tutorials

Jupyter notebooks demonstrating AI4AO's end-to-end, fully differentiable adaptive optics (AO) simulation pipeline — atmospheric turbulence generation, WFS propagation, phase reconstruction, and deformable mirror (DM) response, chained together so that reconstruction networks (and the WFS mask itself) can be trained via backpropagation through the physics. These notebooks are the primary way changes to `AI4AO/` get exercised and validated in this repo.

## Layout

### `basics/`

A five-part series that builds the simulation pipeline one stage at a time, meant to be read in order:

1. **`01_Dataset.ipynb`** — `AI4AO.PhaseDataset`: generating atmospheric turbulence from a von Kármán PSD (open-loop and closed-loop/AO-residual), multi-layer `Cn²` profiles, and scintillation.
2. **`02_WFSAndPreprocessing.ipynb`** — the optical propagator (`AI4AO.PyramidWFS`, subclassing `AI4AO.TorchPropagator.WFS`) turning phase + pupil into a detector frame, and `FramePreprocess` cropping/normalizing pupil images.
3. **`03_DeformableMirrorAndClosedLoop.ipynb`** — `AI4AO.DeformableMirror` and the closed-loop feedback mechanic (measure residual → reconstruct → command the DM), using an oracle reconstructor.
4. **`04_TwinCalibrationGroundTruth.ipynb`** — calibrates a synthetic WFS/DM twin against a synthetic "fake bench" interaction matrix with `AI4AO.TwinCalibrator`, and — unlike the real per-instrument calibration notebooks below — checks the fit against known ground truth, since the "bench" here was generated from deliberately injected parameter values.
5. **`05_TrainingAReconstructor.ipynb`** — loads the twin notebook 4 calibrated and saved, and replaces the oracle from notebook 3 with a trained network via `AI4AO.Trainer`.

Notebooks 1–3 configure the same synthetic, uncalibrated instrument from `basics/wfs_params_exp.py`. Notebook 4 calibrates a synthetic twin and saves it to `Data/Tutorials/`; notebook 5 loads that calibrated twin to train on (falling back to the uncalibrated defaults if notebook 4 hasn't been run yet). There's no real optical bench behind any of it, which makes this series a good place to learn the pipeline — calibration included — without needing bench data, but notebooks 4 and 5 do need to be run in that order for notebook 5 to train against the calibrated twin.

### Per-instrument folders (`Ekarus/`, `Oziriis/`, `Papyrus/`, `Rama/`)

Each instrument folder configures the pipeline for a specific real or simulated instrument twin:

- **`<Instrument>_params.py`** — that instrument's params file (see below).
- **`CalibrateExample<Instrument>Twin.ipynb`** — calibrates the instrument twin (e.g. interaction matrix, DM misregistration) and writes the resulting artifacts.
- **`TrainExample<Instrument>.ipynb`** — trains a reconstructor for that instrument using its calibrated twin. Present for `Ekarus`, `Oziriis`, and `Rama`; `Papyrus` currently only has the calibration notebook.

## Params-file convention

Each params file (`wfs_params_exp.py`, `<Instrument>_params.py`) defines five plain Python dicts — `WFSParams`, `AtmosParams`, `LoopParams`, `DMParams`, `TrainParams` — consumed positionally by the pipeline constructors (`PhaseDataset`, `WFS`/`PyramidWFS`/`ZernikeWFS`, `DeformableMirror`, `Trainer`, ...). This is the only configuration mechanism in AI4AO; there is no YAML/JSON config layer.

## Recommended run order for a new user

1. Work through `basics/01_Dataset.ipynb` → `05_TrainingAReconstructor.ipynb` in order to learn the pipeline stage by stage.
2. Pick an instrument folder and run its `CalibrateExample<Instrument>Twin.ipynb` to build/load that instrument's twin.
3. Run the corresponding `TrainExample<Instrument>.ipynb` (where available) to train a reconstructor against the calibrated twin.

## Trained/calibrated artifacts

Interaction matrices, M2C bases, and DM/WFS/CNN checkpoints produced by the calibration and training notebooks are read from and written to `Data/<Instrument>/` — the `basics/` series uses `Data/Tutorials/` the same way, `"Tutorials"` standing in for an instrument name.
