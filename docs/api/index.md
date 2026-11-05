# API reference

Generated from the docstrings in the `AI4AO` package. The most-used classes are also
importable from the top-level package, for example `from AI4AO import WFS, PhaseDataset,
DeformableMirror, Trainer`.

```{toctree}
:maxdepth: 1
:caption: Atmosphere

PhaseDataset
InfinitePhaseScreen
PSD
```

```{toctree}
:maxdepth: 1
:caption: Wavefront sensing

TorchPropagator
PyramidWFS
ZernikeWFS
FramePreprocess
```

```{toctree}
:maxdepth: 1
:caption: Correction and PSFs

DeformableMirror
PSFDataset
```

```{toctree}
:maxdepth: 1
:caption: Training and calibration

Trainer
TwinCalibrator
LossFunctions
CNNDiagnostics
```

```{toctree}
:maxdepth: 1
:caption: Utilities

Utils
paths
```

```{toctree}
:maxdepth: 1
:caption: Bench hardware

BenchCalibrationUtils
DAO_Utils
```

```{toctree}
:maxdepth: 1
:caption: Legacy

MaskGeneration
PhaseEstimators
```

The bench-hardware modules need `pycuda`/`tensorrt` and a matching GPU stack to run.
The legacy modules are not part of the current pipeline. They are kept for experimental
WFS types and for reference architectures.
