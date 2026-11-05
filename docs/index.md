# AI4AO

**AI4AO** ("Artificial Intelligence for Adaptive Optics") is a PyTorch-based, fully
differentiable end-to-end simulation framework for astronomical adaptive optics (AO).
It models the whole optical and control chain as `torch.nn.Module`s: atmospheric
turbulence, wavefront-sensor (WFS) propagation, phase reconstruction and
deformable-mirror (DM) response.

Gradients flow through the whole chain, so the framework does more than simulate an AO
system. You can use it to:

- train a neural-network reconstructor in closed loop, through the physical simulation;
- calibrate a simulated "twin" against a real bench, by fitting DM misregistration and
  WFS parameters with backpropagation;
- optimize the WFS optical mask itself.

AI4AO is inspired by [OOPAO](https://github.com/cheritier/OOPAO), developed by
C. T. Heritier.

```{note}
AI4AO is under active research development. Module APIs still change, and when this
documentation and the code disagree, the docstrings in the code are the reference.
```

```{toctree}
:maxdepth: 2
:caption: User guide

getting_started
concepts
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/01_Dataset
tutorials/02_WFSAndPreprocessing
tutorials/03_DeformableMirrorAndClosedLoop
tutorials/04_TwinCalibrationGroundTruth
tutorials/05_TrainingAReconstructor
tutorials/06_PSFDataset
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```
