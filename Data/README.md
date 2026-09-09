# `Data/` — bench data and calibration artifacts

The tutorial notebooks read bench measurements from, and write calibrated
twins / trained reconstructors to, this directory. Its contents are
**git-ignored** (only this file is tracked) — the raw bench files are large and
instrument-specific, so you obtain them out of band and drop them in here.

## Location

Notebooks resolve this directory through `AI4AO.paths`:

```python
from AI4AO.paths import DATA_DIR      # <repo root>/Data by default
```

Override with an environment variable if the data lives elsewhere on your
machine (set it *before* importing AI4AO):

```bash
export AI4AO_DATA_DIR=/scratch/$USER/AI4AO_data
```

Layout is one sub-folder per instrument:

```
Data/
├── Rama/
├── Ekarus/
├── Oziriis/
├── Papyrus/
├── Tutorials/      # synthetic — produced by basics/04, no external input
└── TwoStageAO/     # synthetic — produced by Advanced/TwoStageAO
```

## Inputs you must provide

These are **not** produced by any notebook — they come from the bench / from
whoever calibrated the instrument (Rama: the RTC machine's
`~/rama-dev/foyarzun/CNN/Data/`).

| File | Used by | What it is |
|------|---------|------------|
| `Rama/Rama_imats2.npz` | `Rama/CalibrateExampleRamaTwin.ipynb` | Bench interaction matrices. Keys: `iMat_modal`, `iMat_zonal` (flat bench-pixel vectors, one row per poke), `pupils` (full-frame valid-pixel map for the 4 pyramid pupils). |
| `Rama/M2C.npy` | `Rama/` calibrate + train | Modes-to-commands matrix (modal basis → DM actuator commands). |
| `Ekarus/IM.npy` | `Ekarus/` calibrate + train | Real bench interaction matrix (KL modes, same basis as `M2C_KL_OOPAO.npy`). |
| `Ekarus/M2C_KL_OOPAO.npy` | `Ekarus/` calibrate + train | KL modal basis computed in OOPAO from the same calibration as `IM.npy`. |
| `Ekarus/valid_pix_map.npy` | `Ekarus/` calibrate + train | Boolean detector-pixel mask `IM.npy` was measured over. |
| `Oziriis/IM_fullframe.npy` | `Oziriis/CalibrateExampleOziriisTwin.ipynb` | Full-frame bench interaction matrix, shape `(n_modes, W, H)`. |
| `Oziriis/M2C_KL.npy` | `Oziriis/` calibrate + train | KL modal basis. |
| `Papyrus/Papyrus_iMat.npy` | `Papyrus/CalibrateExamplePapyrusTwin.ipynb` | Bench interaction matrix. |
| `Papyrus/M2C.npy` | `Papyrus/` calibrate | Modes-to-commands matrix. |
| `Papyrus/Reference_frame.npy` | `Papyrus/` calibrate | Reference (no-aberration) detector frame, reshaped to `240×240`. |

`Tutorials/` and `TwoStageAO/` need no inputs — those notebooks build a
synthetic "fake bench" from scratch.

## Artifacts the notebooks write here

Created automatically (parent folders are made on save):

| File pattern | Written by | Read by |
|--------------|-----------|---------|
| `<Instrument>/<Instrument>WFS.pth` | `Calibrate*Twin.ipynb` (`TwinCalibrator.save`) | `TrainExample*.ipynb` (`WFS.LoadCalibration`) |
| `<Instrument>/<Instrument>DM.pth` | `Calibrate*Twin.ipynb` | `TrainExample*.ipynb` (`DeformableMirror.LoadCalibration`) |
| `<Instrument>/<Instrument>CNN*.pth` | `TrainExample*.ipynb` (`Trainer.save_checkpoint`) | resumed by the same notebook |
| `Tutorials/TutorialsWFS.pth`, `TutorialsDM.pth` | `basics/04_TwinCalibrationGroundTruth.ipynb` | `basics/05_TrainingAReconstructor.ipynb` |
| `Tutorials/ReconstructorCNN.pth` | `basics/05` | `basics/05` |
| `TwoStageAO/Stage{1,2}CNN.pth` | `Advanced/TwoStageAO.ipynb` | same notebook |
