"""End-to-end smoke test mirroring Tutorials/basics/04_TrainingAReconstructor.ipynb
(the tutorial notebook that needs no external bench data), shrunk to a tiny
config. Exercises the full dataset -> WFS -> DM -> FramePreprocess ->
reconstructor -> loss -> backprop -> optimizer step chain via Trainer, plus
Trainer.evaluate(). Marked slow since it's an integration test, not a unit test
(opt out with `pytest -m "not slow"`), even though it runs in well under a
second at this tiny scale."""
import pytest
import torch
import torch.nn as nn

from AI4AO.FramePreprocess import FramePreprocess
from AI4AO.LossFunctions import LogResidualVarianceLoss
from AI4AO.Trainer import Trainer


class _TinyReconstructor(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.net(x.flatten(start_dim=1))


pytestmark = pytest.mark.slow


def test_trainer_train_and_evaluate_end_to_end(pyramid_wfs, deformable_mirror, phase_dataset, tiny_wfs_params, device):
    pyramid_wfs.eval()
    deformable_mirror.eval()

    frame_preprocessor = FramePreprocess(tiny_wfs_params(), pyramid_wfs, device)
    pyramid_wfs.BuildReferenceIntensity()
    frame_preprocessor.ProcessReference(pyramid_wfs.reference_intensity)

    total_act = int(deformable_mirror.totalAct.item())
    M2C = torch.eye(total_act, device=device)

    in_features = 4 * frame_preprocessor.Nout * frame_preprocessor.Nout  # 4 pupil images
    reconstructor = _TinyReconstructor(in_features, total_act).to(device)
    initial_weight = reconstructor.net.weight.detach().clone()

    loss = LogResidualVarianceLoss(phase_dataset.pupil)
    optimizer = torch.optim.Adam(reconstructor.parameters(), lr=1e-2)

    trainer = Trainer(
        wfs=pyramid_wfs,
        framePreprocessor=frame_preprocessor,
        dm=deformable_mirror,
        M2C=M2C,
        phaseReconstructor=reconstructor,
        dataset=phase_dataset,
        loss=loss,
        optimizer=optimizer,
    )

    loss_tracker, loss_tracker_ideal = trainer.train(training_steps=2, closed_loop_iterations=1)

    assert loss_tracker.shape == (2,)
    assert loss_tracker_ideal.shape == (2,)
    assert torch.isfinite(loss_tracker).all()
    assert torch.isfinite(loss_tracker_ideal).all()
    assert not torch.allclose(reconstructor.net.weight, initial_weight)

    result = trainer.evaluate(n_steps=3, dataset=phase_dataset)

    n_steps = 3
    assert result.phase.shape[0] == n_steps
    assert result.pupil.shape[0] == n_steps
    assert result.phase_reconstructed.shape[0] == n_steps
    assert result.residual_phase.shape[0] == n_steps
    assert result.wfs_frames.shape[0] == n_steps
    assert result.psfs.shape[0] == n_steps
    assert result.z_output.shape[0] == n_steps
    assert torch.isfinite(result.residual_phase).all()
