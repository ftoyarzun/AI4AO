# Advanced tutorials

Research-style notebooks that go beyond the `basics/` series: ablation studies, alternative reconstructor designs, multi-sensor/learned-optics systems, and PSF-domain techniques. None of them needs bench data — each builds its own fictional, nominal-geometry demo instrument from a params file sitting next to the notebook. Most have a design write-up in `Ideas/`.

Run each notebook from its own folder (its params file is loaded by relative path). Checkpoints go to `Data/<NotebookName>/` via `AI4AO.paths.DATA_DIR`, independent of where the notebook lives.

## `Comparisons/` — CNN reconstructor ablations

One fixed instrument (`CNNComparison_params.py`, a modulated Pyramid WFS + 17-actuator DM), one training recipe, one knob varied per notebook. Suggested order:

| Notebook | What varies |
|---|---|
| `CNNArchitectureComparison.ipynb` | Six CNN architectures (start here — defines `ClassicCNN` and the shared recipe) |
| `CNNSizeComparison.ipynb` | `ClassicCNN` capacity (`base_channels`) |
| `WidthDepthComparison.ipynb` | Width vs. depth (plain and residual) at three fixed parameter budgets |
| `ActivationComparison.ipynb` | Activation function |
| `NormalizationComparison.ipynb` | Normalization layers |
| `PoolingComparison.ipynb` | Downsampling strategy |
| `LossFunctionComparison.ipynb` | Training loss |
| `AdvancedArchitectureComparison.ipynb` | ViT and ConvNeXt V2 with actuator-aligned patches |
| `BestCNN.ipynb` | Combines the winners; loads the other notebooks' checkpoints as baselines, so run it last |

## `Reconstructors/` — alternative reconstruction approaches

| Notebook | Idea |
|---|---|
| `ResidualCNNReconstructor.ipynb` | A CNN that complements, rather than replaces, a frozen linear reconstructor |
| `TemporalPredictiveReconstructor.ipynb` | Uses the last N frames to predict the correction ahead of loop latency |
| `UnsupervisedFineTuning.ipynb` | Fine-tunes a pretrained reconstructor from WFS frames alone (physics loss, no ground-truth OPD) |
| `FrameDenoiser.ipynb` | Denoises raw WFS frames (with optional frame history) before a linear reconstructor |

## `SystemDesign/` — multi-sensor and learned-optics systems

| Notebook | Idea |
|---|---|
| `TwoStageAO.ipynb` | Two cascaded AO stages: modulated Pyramid "woofer" + vZWFS "tweeter" |
| `DualSensorFusion.ipynb` | Pyramid + ZWFS sensing the same residual concurrently, fused into one reconstructor driving one DM |
| `SelfShapingMask.ipynb` | Learns a free-form WFS phase mask jointly with its CNN reconstructor, vs. a Pyramid baseline |

## `PSF/` — science-PSF techniques

| Notebook | Idea |
|---|---|
| `NCPAEstimation.ipynb` | Fits a static DM command cancelling non-common-path aberrations from recorded PSFs |
| `Deconvolution.ipynb` | Trains a UNet to deblur images convolved with PSFs drawn from `PSFDataset` |
