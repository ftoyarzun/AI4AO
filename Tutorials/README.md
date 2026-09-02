# Tutorials

Jupyter notebooks demonstrating AI4AO's end-to-end, fully differentiable adaptive optics (AO) simulation pipeline — atmospheric turbulence generation, WFS propagation, phase reconstruction, and deformable mirror (DM) response, chained together so that reconstruction networks (and the WFS mask itself) can be trained via backpropagation through the physics. These notebooks are the primary way changes to `AI4AO/` get exercised and validated in this repo.

## Layout

### `basics/`

A four-part series that builds the simulation pipeline one stage at a time, meant to be read in order:

1. **`01_Dataset.ipynb`** — `AI4AO.PhaseDataset`: generating atmospheric turbulence from a von Kármán PSD (open-loop and closed-loop/AO-residual), multi-layer `Cn²` profiles, and scintillation.
2. **`02_WFSAndPreprocessing.ipynb`** — the optical propagator (`AI4AO.PyramidWFS`, subclassing `AI4AO.TorchPropagator.WFS`) turning phase + pupil into a detector frame, and `FramePreprocess` cropping/normalizing pupil images.
3. **`03_DeformableMirrorAndClosedLoop.ipynb`** — `AI4AO.DeformableMirror` and the closed-loop feedback mechanic (measure residual → reconstruct → command the DM), using an oracle reconstructor.
4. **`04_TrainingAReconstructor.ipynb`** — replacing the oracle with a trained network via `AI4AO.Trainer`.

All four notebooks configure the same synthetic, uncalibrated instrument from `basics/wfs_params_exp.py` — there's no real optical bench behind it, which makes this series a good place to learn the pipeline without needing bench data.

### Per-instrument folders (`Ekarus/`, `Oziriis/`, `Papyrus/`, `Rama/`)

Each instrument folder configures the pipeline for a specific real or simulated instrument twin:

- **`<Instrument>_params.py`** — that instrument's params file (see below).
- **`CalibrateExample<Instrument>Twin.ipynb`** — calibrates the instrument twin (e.g. interaction matrix, DM misregistration) and writes the resulting artifacts.
- **`TrainExample<Instrument>.ipynb`** — trains a reconstructor for that instrument using its calibrated twin. Present for `Ekarus`, `Oziriis`, and `Rama`; `Papyrus` currently only has the calibration notebook.

## Params-file convention

Each params file (`wfs_params_exp.py`, `<Instrument>_params.py`) defines five plain Python dicts — `WFSParams`, `AtmosParams`, `LoopParams`, `DMParams`, `TrainParams` — consumed positionally by the pipeline constructors (`PhaseDataset`, `WFS`/`PyramidWFS`/`ZernikeWFS`, `DeformableMirror`, `Trainer`, ...). This is the only configuration mechanism in AI4AO; there is no YAML/JSON config layer.

## Recommended run order for a new user

1. Work through `basics/01_Dataset.ipynb` → `04_TrainingAReconstructor.ipynb` in order to learn the pipeline stage by stage.
2. Pick an instrument folder and run its `CalibrateExample<Instrument>Twin.ipynb` to build/load that instrument's twin.
3. Run the corresponding `TrainExample<Instrument>.ipynb` (where available) to train a reconstructor against the calibrated twin.

## Trained/calibrated artifacts

Interaction matrices, M2C bases, and DM/WFS/CNN checkpoints produced by the calibration and training notebooks are read from and written to `Data/<Instrument>/`.
