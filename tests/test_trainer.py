"""Tests for AI4AO.Trainer, isolating the two cheap/self-contained pieces:
the z_inv pseudo-inverse identity computed in __init__ (needs only a dm+M2C,
no WFS/dataset/reconstructor), and the checkpoint save/load round trip
(touches only phaseReconstructor/optimizer)."""
import copy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from AI4AO.Trainer import (
    Recipe,
    Stage,
    Trainer,
    TrainingRun,
    load_reconstructor,
    make_optimizer,
    make_scheduler,
)
from AI4AO.Utils import seed_everything


class _StubWFS:
    """Trainer.__init__ only reads wfs.device; nothing else is touched
    unless .train()/.eval() or the forward pass is exercised."""
    def __init__(self, device):
        self.device = device


def _build_minimal_trainer(deformable_mirror, device, reconstructor=None, optimizer=None):
    total_act = int(deformable_mirror.totalAct.item())
    M2C = torch.eye(total_act, device=device)
    if reconstructor is None:
        reconstructor = nn.Linear(4, 4)
    if optimizer is None:
        optimizer = torch.optim.SGD(reconstructor.parameters(), lr=0.1)
    trainer = Trainer(
        wfs=_StubWFS(device),
        framePreprocessor=None,
        dm=deformable_mirror,
        M2C=M2C,
        phaseReconstructor=reconstructor,
        dataset=object(),  # never touched by these tests; Trainer only requires one
        loss=None,
        optimizer=optimizer,
    )
    return trainer, M2C, total_act


def test_z_inv_is_pseudo_inverse_of_dm_basis(deformable_mirror, device):
    trainer, M2C, total_act = _build_minimal_trainer(deformable_mirror, device)

    IF_flat = deformable_mirror(M2C.T).flatten(start_dim=-2)
    identity_check = IF_flat @ trainer.z_inv

    assert trainer.z_inv.shape == (deformable_mirror.Nres * deformable_mirror.Nres, total_act)
    assert torch.allclose(identity_check, torch.eye(total_act, device=device), atol=1e-3)


def test_save_and_load_checkpoint_round_trip(deformable_mirror, device, tmp_path):
    reconstructor = nn.Linear(4, 4)
    optimizer = torch.optim.Adam(reconstructor.parameters(), lr=0.1)
    trainer, _, _ = _build_minimal_trainer(deformable_mirror, device, reconstructor, optimizer)

    x = torch.randn(2, 4)
    reconstructor(x).sum().backward()
    optimizer.step()  # populate Adam's per-parameter state

    path = tmp_path / "ckpt.pth"
    trainer.save_checkpoint(str(path), step=42)

    original_weight = reconstructor.weight.detach().clone()
    original_exp_avg = optimizer.state_dict()["state"][0]["exp_avg"].clone()

    with torch.no_grad():
        reconstructor.weight.add_(1.0)
    optimizer.state[reconstructor.weight]["exp_avg"].add_(1.0)

    result = trainer.load_checkpoint(str(path))

    # `load_checkpoint`'s docstring promises returning the full checkpoint
    # dict on success, but its success path has no `return` statement, so it
    # currently falls through and returns None even though the load worked
    # (weights/optimizer state below ARE correctly restored, since
    # load_state_dict runs before the function implicitly returns). This
    # documents today's actual behavior rather than the docstring's claim.
    assert result is None
    assert torch.allclose(reconstructor.weight, original_weight)
    assert torch.allclose(optimizer.state_dict()["state"][0]["exp_avg"], original_exp_avg)


def test_load_checkpoint_missing_path_returns_none(deformable_mirror, device, tmp_path):
    trainer, _, _ = _build_minimal_trainer(deformable_mirror, device)
    result = trainer.load_checkpoint(str(tmp_path / "does_not_exist.pth"))
    assert result is None


# ---------------------------------------------------------------------------
# A real (tiny) twin with every noise source on, so RNG-order changes show up
# ---------------------------------------------------------------------------

class _TinyReconstructor(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.stem = nn.Linear(in_features, 16)
        self.head = nn.Linear(16, out_features)

    def forward(self, x):
        return self.head(torch.tanh(self.stem(x.flatten(start_dim=1))))


@pytest.fixture
def noisy_twin(tiny_wfs_params, tiny_atmos_params, tiny_loop_params, tiny_dm_params, device):
    from AI4AO.DeformableMirror import DeformableMirror
    from AI4AO.FramePreprocess import FramePreprocess
    from AI4AO.LossFunctions import LogResidualVarianceLoss
    from AI4AO.PhaseDataset import PhaseDataset
    from AI4AO.PyramidWFS import PyramidWFS

    params = tiny_wfs_params()
    params.update(useNoise=True, Center_noise=0.3, Pupil_size_noise=0.02)
    wfs = PyramidWFS(params, device)
    wfs.eval()
    dm = DeformableMirror(params, tiny_dm_params, device)
    dm.eval()
    fp = FramePreprocess(params, wfs, device)
    fp.ProcessReference(wfs.reference_intensity)
    dataset = PhaseDataset(params, tiny_atmos_params, tiny_loop_params, tiny_dm_params, device)
    dataset.generateClosedLoop = True
    total_act = int(dm.totalAct.item())
    M2C = torch.eye(total_act, device=device)
    n_in = wfs.pupil_centers.shape[0] * fp.Nout * fp.Nout

    def make_model(seed=0):
        torch.manual_seed(seed)
        return _TinyReconstructor(n_in, total_act).to(device)

    loss = LogResidualVarianceLoss(dataset.pupil, wfs.wavelength)
    return SimpleNamespace(wfs=wfs, dm=dm, fp=fp, dataset=dataset, M2C=M2C, loss=loss,
                           make_model=make_model, device=device)


def _trainer(twin, model=None, optimizer=None, loss="default"):
    return Trainer(twin.wfs, twin.fp, twin.dm, twin.M2C, model, twin.dataset,
                   twin.loss if loss == "default" else loss, optimizer)


# ---------------------------------------------------------------------------
# Frozen copies of Trainer.train / Trainer.evaluate as they were before the
# train_many/evaluate_many refactor (Ideas/19). Do not edit: they are the
# bit-identity reference. Only the progress bar was removed.
# ---------------------------------------------------------------------------

def _reference_train(self, training_steps=1000, closed_loop_iterations=1):
    if self.dm.training:
        self.dm.eval()
    if self.wfs.training:
        self.wfs.eval()

    loss_tracker = torch.zeros(training_steps // closed_loop_iterations, device=self.device)
    loss_tracker_ideal = torch.zeros(training_steps // closed_loop_iterations, device=self.device)

    self.phaseReconstructor.train()

    M2C_T = self.M2C.T

    for u in range(training_steps // closed_loop_iterations):
        with torch.no_grad():
            batch = self.dataset[0]
            opd_gt = batch["opd"]
            pupilGT = batch["pupil"]
            gain = batch["loop_gain"]
            leak = batch["loop_leak"]
            photons = batch["nphotons"]
            ron = batch["ron"]

            modes = torch.matmul(opd_gt.flatten(start_dim = -2), self.z_inv)

            self.wfs.SetPhotonsAndRON(photons, ron)

            z_estimated = torch.zeros_like(modes)
            z_buffer = torch.zeros_like(modes)
            z_output = torch.zeros_like(modes)
            opd_reconstructed = torch.zeros_like(opd_gt)

            total_loss = 0
            ideal_loss = 0

        for i in range(closed_loop_iterations):
            with torch.no_grad():
                if i > 0:
                    batch = self.dataset[i]
                    opd_gt = batch["opd"]
                    pupilGT = batch["pupil"]

                residual_opd = opd_gt - opd_reconstructed

                modes = torch.matmul(opd_gt.flatten(start_dim = -2), self.z_inv)
                Ze = torch.matmul(residual_opd.flatten(start_dim = -2), self.z_inv)

                z_estimated = z_estimated * leak + gain * z_buffer
                z_buffer = torch.clone(z_output)

                wfs_frames = self.wfs(residual_opd, pupilGT)
                preprocessed_frames = self.framePreprocessor.ProcessFrame(wfs_frames)
            z_output = self.phaseReconstructor(preprocessed_frames)

            opd_reconstructed = self.dm(z_estimated @ M2C_T)
            opd_reconstructed_iter = self.dm(z_output @ M2C_T)
            opd_reconstructed_iter_ideal = self.dm(Ze @ M2C_T)

            corrected_residual_opd = residual_opd - opd_reconstructed_iter
            total_loss += self.loss(Ze, z_output, pupilGT, residual_opd, corrected_residual_opd, wfs_frames) / closed_loop_iterations

            with torch.no_grad():
                ideal_corrected_residual_opd = residual_opd - opd_reconstructed_iter_ideal
                ideal_loss += self.loss(Ze, Ze, pupilGT, residual_opd, ideal_corrected_residual_opd, wfs_frames) / closed_loop_iterations

        self.optimizer.zero_grad(set_to_none = True)
        total_loss.backward()
        self.optimizer.step()

        loss_tracker[u] = total_loss.detach()
        loss_tracker_ideal[u] = ideal_loss.detach()

    return loss_tracker, loss_tracker_ideal


@torch.no_grad()
def _reference_evaluate(self, n_steps = 100, dataset = None, gain = None, leak = None, psf_sampling = 4, psf_fov = 20):
    dataset = dataset if dataset is not None else self.dataset

    if self.dm.training:
        self.dm.eval()
    if self.wfs.training:
        self.wfs.eval()

    M2C_T = self.M2C.T
    Nmodes = self.z_inv.shape[-1]

    self.phaseReconstructor.eval()

    batch = dataset[0]
    opd_gt = batch["opd"]
    pupilGT = batch["pupil"]
    gain = gain if gain is not None else batch["loop_gain"]
    leak = leak if leak is not None else batch["loop_leak"]
    photons = batch["nphotons"]
    ron = batch["ron"]

    self.wfs.SetPhotonsAndRON(photons, ron)

    z_estimated = torch.zeros(opd_gt.shape[0], Nmodes, device=self.device)
    z_buffer = torch.zeros_like(z_estimated)
    z_output = torch.zeros_like(z_estimated)
    opd_reconstructed = torch.zeros_like(opd_gt)

    opds, pupils, reconstructed, residuals, frames, psfs, outputs = [], [], [], [], [], [], []

    for i in range(n_steps):
        if i > 0:
            batch = dataset[i]
            opd_gt = batch["opd"]
            pupilGT = batch["pupil"]

        residual_opd = opd_gt - opd_reconstructed

        wfs_frames = self.wfs(residual_opd, pupilGT)
        psf = self.wfs.GetPSF(residual_opd, pupilGT, psf_sampling, psf_fov)
        preprocessed_frames = self.framePreprocessor.ProcessFrame(wfs_frames, False)
        z_output = self.phaseReconstructor(preprocessed_frames)

        z_buffer = torch.clone(z_output)

        if i > n_steps * 0.3:
            z_estimated = z_estimated * leak + gain * z_buffer

        opd_reconstructed = self.dm(z_estimated @ M2C_T)

        opds.append(opd_gt)
        pupils.append(pupilGT)
        reconstructed.append(opd_reconstructed)
        residuals.append(residual_opd)
        frames.append(wfs_frames)
        psfs.append(psf)
        outputs.append(z_output)

    return dict(
        opd = torch.stack(opds),
        pupil = torch.stack(pupils),
        opd_reconstructed = torch.stack(reconstructed),
        residual_opd = torch.stack(residuals),
        wfs_frames = torch.stack(frames),
        psfs = torch.stack(psfs),
        z_output = torch.stack(outputs),
    )


@pytest.mark.slow
@pytest.mark.parametrize("closed_loop_iterations", [1, 3])
def test_train_is_bit_identical_to_frozen_reference(noisy_twin, closed_loop_iterations):
    steps = 4 * closed_loop_iterations
    model_ref = noisy_twin.make_model()
    model_new = copy.deepcopy(model_ref)
    trainer_ref = _trainer(noisy_twin, model_ref, torch.optim.Adam(model_ref.parameters(), lr=1e-3))
    trainer_new = _trainer(noisy_twin, model_new, torch.optim.Adam(model_new.parameters(), lr=1e-3))

    seed_everything(5)
    ref_loss, ref_ideal = _reference_train(trainer_ref, steps, closed_loop_iterations)
    seed_everything(5)
    new_loss, new_ideal = trainer_new.train(steps, closed_loop_iterations)

    assert torch.isfinite(ref_loss).all()
    assert torch.equal(new_loss, ref_loss)
    assert torch.equal(new_ideal, ref_ideal)
    for p, q in zip(model_ref.parameters(), model_new.parameters()):
        assert torch.equal(p, q)


def test_train_with_fewer_frames_than_iterations_takes_no_step(noisy_twin):
    model = noisy_twin.make_model()
    before = [p.detach().clone() for p in model.parameters()]
    loss, ideal = _trainer(noisy_twin, model, torch.optim.Adam(model.parameters())).train(2, 3)
    assert loss.shape == ideal.shape == (0,)
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))


def test_evaluate_with_old_window_is_bit_identical_to_frozen_reference(noisy_twin):
    model = noisy_twin.make_model()
    trainer = _trainer(noisy_twin, model, torch.optim.Adam(model.parameters()))

    seed_everything(9)
    ref = _reference_evaluate(trainer, n_steps=10)
    seed_everything(9)
    new = trainer.evaluate(n_steps=10, loop_start=0.3)

    for key, value in ref.items():
        assert torch.equal(getattr(new, key), value), key


# ---------------------------------------------------------------------------
# train_many: shared draws, validation, optional constructor arguments
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_train_many_copies_with_same_recipe_see_identical_draws(noisy_twin):
    model = noisy_twin.make_model()
    runs = [TrainingRun("a", model), TrainingRun("b", copy.deepcopy(model))]
    report = _trainer(noisy_twin).train_many(runs, [Stage(5, 1)], default_recipe=Recipe(lr=1e-3), seed=1)

    loss_a, loss_b = report.runs["a"].loss[0], report.runs["b"].loss[0]
    assert torch.isfinite(loss_a).all()
    assert torch.equal(loss_a, loss_b)
    for p, q in zip(runs[0].model.parameters(), runs[1].model.parameters()):
        assert torch.equal(p, q)


@pytest.mark.slow
def test_one_run_train_many_equals_train_with_recipe(noisy_twin):
    recipe = Recipe(lr=1e-3, warmup_frac=0.4, group_lr_factors={"stem": 5.0})
    model = noisy_twin.make_model()
    twin_copy = copy.deepcopy(model)

    trainer = _trainer(noisy_twin, model)
    seed_everything(3)
    loss, ideal = trainer.train(6, 1, recipe=recipe)

    report = _trainer(noisy_twin).train_many([TrainingRun("x", twin_copy, recipes=recipe)],
                                             [Stage(6, 1)], seed=3)
    assert torch.equal(loss, report.runs["x"].loss[0])
    assert torch.isfinite(ideal).all()
    # train(recipe=...) replaces the trainer's optimizer with the fresh one
    assert trainer.optimizer is trainer.last_report.runs["model"].optimizer
    assert len(trainer.optimizer.param_groups) == 2
    # lr history: the lr each step used, one column per group (base, then stem at 5x)
    lr = report.runs["x"].lr[0]
    assert lr.shape == (6, 2)
    assert torch.allclose(lr[:, 1], 5 * lr[:, 0])
    assert lr[0, 0] == pytest.approx(1e-3 / 2)       # warm-up: 2 steps
    assert lr[1, 0] == pytest.approx(1e-3)           # peak at the end of warm-up


def test_train_many_validation_errors(noisy_twin, tmp_path):
    trainer = _trainer(noisy_twin)
    stages = [Stage(2, 1), Stage(2, 2)]
    m = noisy_twin.make_model()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)

    cases = [
        ([TrainingRun("a", m, recipes=Recipe(), optimizer=opt)], "not both"),
        ([TrainingRun("a", m, recipes=[Recipe()])], "1 recipes for 2 stages"),
        ([TrainingRun("a", m, recipes=Recipe(group_lr_factors={"nope": 2.0}))], "'nope'"),
        ([TrainingRun("a", m), TrainingRun("a", copy.deepcopy(m))], "duplicate run names"),
        ([TrainingRun("a", m, checkpoint=tmp_path / "x.pth"),
          TrainingRun("b", copy.deepcopy(m), checkpoint=str(tmp_path / "x.pth"))], "share a checkpoint"),
    ]
    for runs, message in cases:
        with pytest.raises(ValueError, match=message):
            trainer.train_many(runs, stages)
    # an unknown group in the default recipe is caught too
    with pytest.raises(ValueError, match="'nope'"):
        trainer.train_many([TrainingRun("a", m)], stages,
                           default_recipe=Recipe(group_lr_factors={"nope": 2.0}))
    # a run with no loss on a trainer with no loss
    with pytest.raises(ValueError, match="no loss"):
        _trainer(noisy_twin, loss=None).train_many([TrainingRun("a", m)], stages)


def test_optional_constructor_arguments(noisy_twin):
    trainer = Trainer(noisy_twin.wfs, noisy_twin.fp, noisy_twin.dm, noisy_twin.M2C,
                      dataset=noisy_twin.dataset, loss=noisy_twin.loss)
    assert trainer.phaseReconstructor is None and trainer.optimizer is None
    for call in (lambda: trainer.train(2, 1), lambda: trainer.evaluate(n_steps=2),
                 lambda: trainer.save_checkpoint("unused.pth"),
                 lambda: trainer.load_checkpoint("unused.pth")):
        with pytest.raises(ValueError, match="phaseReconstructor"):
            call()
    # evaluate_many works on a trainer with no reconstructor of its own
    report = trainer.evaluate_many({"oracle": trainer.oracle_reconstructor()}, seeds=[0], n_steps=4)
    assert np.isfinite(report.nm_rms["oracle"][0])

    # positional construction is unchanged
    model = noisy_twin.make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    positional = Trainer(noisy_twin.wfs, noisy_twin.fp, noisy_twin.dm, noisy_twin.M2C, model,
                         noisy_twin.dataset, noisy_twin.loss, opt)
    assert (positional.phaseReconstructor, positional.loss, positional.optimizer) == (model, noisy_twin.loss, opt)

    with pytest.raises(TypeError, match="dataset"):
        Trainer(noisy_twin.wfs, noisy_twin.fp, noisy_twin.dm, noisy_twin.M2C, model)
    # train() needs either an optimizer or a recipe
    with pytest.raises(ValueError, match="optimizer"):
        _trainer(noisy_twin, model).train(2, 1)


# ---------------------------------------------------------------------------
# Optimizer and scheduler
# ---------------------------------------------------------------------------

def test_make_optimizer_group_learning_rates():
    model = _TinyReconstructor(8, 3)
    opt = make_optimizer(model, Recipe(lr=2e-4, group_lr_factors={"stem": 5.0}, weight_decay=0.0))
    base, stem = opt.param_groups
    assert base["lr"] == pytest.approx(2e-4)
    assert stem["lr"] == pytest.approx(1e-3)
    assert {id(p) for p in stem["params"]} == {id(p) for p in model.stem.parameters()}
    assert {id(p) for p in base["params"]} == {id(p) for p in model.head.parameters()}
    assert base["weight_decay"] == 0.0
    with pytest.raises(AttributeError):
        make_optimizer(model, Recipe(group_lr_factors={"nope": 2.0}))


def test_make_scheduler_warmup_peak_and_floor():
    model = _TinyReconstructor(8, 3)
    recipe = Recipe(lr=1.0, warmup_frac=0.1, final_factor=0.2, group_lr_factors={"stem": 5.0})
    opt = make_optimizer(model, recipe)
    sched = make_scheduler(opt, recipe, total_steps=50)
    lrs = []
    for _ in range(50):
        lrs.append([g["lr"] for g in opt.param_groups])
        opt.step()
        sched.step()
    lrs = np.array(lrs)
    base = lrs[:, 0]
    assert np.allclose(lrs[:, 1], 5 * base)               # every group follows the schedule
    assert base[0] == pytest.approx(0.2)                    # warm-up: 5 steps
    assert base.argmax() == 4 and base[4] == pytest.approx(1.0)   # peak at the end of warm-up
    assert np.all(np.diff(base[4:]) <= 1e-12)              # then decays
    assert base[-1] > 0.2 and base[-1] == pytest.approx(0.2, abs=1e-2)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.2)  # floor reached at the end of the stage

    assert make_scheduler(opt, Recipe(schedule="constant"), 50) is None
    with pytest.raises(ValueError, match="schedule"):
        make_scheduler(opt, Recipe(schedule="step"), 50)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_rollout_window_and_metrics_match_hand_computation(noisy_twin):
    trainer = _trainer(noisy_twin)
    ds = noisy_twin.dataset
    n = 10
    seed_everything(1)
    metrics, res = trainer._rollout(trainer.oracle_reconstructor(), ds, n, loop_start=0.2,
                                    record_start=0.3, keep_trajectories=True)
    assert res.metrics == metrics

    # the integrator stays at zero for i <= 0.2 n, then closes
    assert torch.all(res.opd_reconstructed[:3] == 0)
    assert res.opd_reconstructed[3].abs().sum() > 0

    pupil = ds.pupil.bool()
    v = res.residual_opd[3:][..., pupil].double()           # frames int(0.3 n) .. n-1
    assert metrics["nm_rms"] == pytest.approx(float(v.pow(2).mean().sqrt()) * 1e9, rel=1e-9)
    peak0 = noisy_twin.wfs.GetPSF(torch.zeros_like(res.opd[0][:1]), res.pupil[0][:1], 4, 20).amax()
    strehl = (res.psfs[3:].flatten(2).amax(2) / peak0).mean()
    assert metrics["strehl"] == pytest.approx(float(strehl), rel=1e-6)
    k = 2 * np.pi / float(torch.as_tensor(noisy_twin.wfs.wavelength).double().reshape(-1).mean())
    marechal = torch.exp(-torch.var(k * v, dim=-1).mean())
    assert metrics["strehl_marechal"] == pytest.approx(float(marechal), rel=1e-9)
    assert 0 < metrics["strehl_marechal"] <= 1


def test_rollout_residual_grows_with_loop_delay(noisy_twin):
    trainer = _trainer(noisy_twin)
    oracle = trainer.oracle_reconstructor()
    nm = []
    for delay in (0, 1, 2):
        report = trainer.evaluate_many({"oracle": oracle}, seeds=[4], n_steps=40, delay=delay,
                                       gain=0.4, leak=1.0, loop_start=0.0, record_start=0.5)
        nm.append(report.nm_rms["oracle"][0])
    assert nm[0] < nm[1] < nm[2]


def test_evaluate_many_pairs_models_and_reports(noisy_twin):
    trainer = _trainer(noisy_twin)
    model = noisy_twin.make_model()
    recs = {"cnn": model, "cnn_again": model, "oracle": trainer.oracle_reconstructor(),
            "linear": trainer.linear_reconstructor()}
    report = trainer.evaluate_many(recs, seeds=[0, 1], n_steps=6, baseline="cnn",
                                   keep_trajectories=True)

    assert report.nm_rms["cnn"] == report.nm_rms["cnn_again"]
    assert report.strehl["cnn"] == report.strehl["cnn_again"]
    mean, sem, per_seed = report.paired_delta("cnn_again")
    assert mean == 0 and sem == 0 and np.all(per_seed == 0)
    assert report.nm_rms["cnn"][0] != report.nm_rms["cnn"][1]           # seeds differ
    assert report.paired_delta("oracle")[0] < 0                           # the floor
    assert all(np.isfinite(report.nm_rms[name]).all() for name in recs)

    result = report.trajectories["linear"][1]
    B = noisy_twin.dataset.Nphases
    Nres = noisy_twin.wfs.Nres
    assert result.opd.shape == (6, B, Nres, Nres)
    assert result.z_output.shape == (6, B, noisy_twin.M2C.shape[0])
    assert result.psfs.shape[:2] == (6, B)
    assert "baseline" in report.table() and "Δ vs cnn" in report.table()


def test_evaluate_many_references(noisy_twin):
    trainer = _trainer(noisy_twin)
    model = noisy_twin.make_model()
    report = trainer.evaluate_many({"cnn": model}, seeds=[0, 1], n_steps=6, baseline="cnn",
                                   references=("linear", "oracle"))
    assert list(report.nm_rms) == ["cnn", "linear", "oracle"]

    # the references are the same rollouts as passing the reconstructors by hand
    by_hand = trainer.evaluate_many({"cnn": model, "linear": trainer.linear_reconstructor(),
                                     "oracle": trainer.oracle_reconstructor()},
                                    seeds=[0, 1], n_steps=6)
    assert report.nm_rms == by_hand.nm_rms
    assert report.nm_rms["oracle"][0] < report.nm_rms["linear"][0]
    assert "linear" in report.table() and "oracle" in report.table()

    only_oracle = trainer.evaluate_many({"cnn": model}, seeds=[0], n_steps=4, references="oracle",
                                        baseline="oracle")
    assert list(only_oracle.nm_rms) == ["cnn", "oracle"]
    with pytest.raises(ValueError, match="unknown references"):
        trainer.evaluate_many({"cnn": model}, seeds=[0], n_steps=2, references=("lmmse",))
    with pytest.raises(ValueError, match="clash"):
        trainer.evaluate_many({"oracle": model}, seeds=[0], n_steps=2, references=("oracle",))


def test_evaluate_many_single_seed_has_no_error_bar(noisy_twin):
    trainer = _trainer(noisy_twin)
    model = noisy_twin.make_model()
    report = trainer.evaluate_many({"a": model, "oracle": trainer.oracle_reconstructor()},
                                   seeds=[0], n_steps=4, baseline="a")
    assert np.isnan(report.paired_delta("oracle")[1])
    assert report.trajectories is None
    table = report.table()
    assert "–" in table
    assert "nan" not in table.lower()
    with pytest.raises(ValueError, match="baseline"):
        trainer.evaluate_many({"a": model}, seeds=[0], n_steps=2, baseline="b")


# ---------------------------------------------------------------------------
# Diagnostics, checkpoints, plots
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_diagnostics_recorded_every_log_every_steps_across_stages(noisy_twin):
    from AI4AO.CNNDiagnostics import CNNDiagnostics

    model = noisy_twin.make_model()
    diag = CNNDiagnostics(model)
    run = TrainingRun("a", model, recipes=Recipe(lr=1e-3), diagnostics=diag, log_every=3)
    _trainer(noisy_twin).train_many([run], [Stage(5, 1), Stage(6, 2)], seed=0)   # 5 + 3 steps
    assert diag.history_steps == [0, 3, 6]
    assert all(np.isfinite(v).all() for v in diag.history["update_ratio"].values())


@pytest.mark.slow
def test_checkpoint_final_only_resume_and_load_reconstructor(noisy_twin, tmp_path):
    path = tmp_path / "sub" / "a.pth"
    stages = [Stage(3, 1), Stage(4, 2)]
    recipe = Recipe(lr=1e-3, schedule="constant")
    seen = []

    def loss(*args):  # records whether the checkpoint already exists while training
        seen.append(path.exists())
        return noisy_twin.loss(*args)

    trainer = _trainer(noisy_twin, loss=None)
    model = noisy_twin.make_model()
    trainer.train_many([TrainingRun("a", model, recipes=recipe, loss=loss, checkpoint=path)],
                       stages, seed=0)
    assert seen and not any(seen)
    assert sorted(p.name for p in path.parent.iterdir()) == ["a.pth"]   # no temporary file left
    weights = {k: v.clone() for k, v in model.state_dict().items()}

    # resume=True: loaded, not trained
    fresh = noisy_twin.make_model(seed=1)
    seen.clear()
    report = trainer.train_many([TrainingRun("a", fresh, recipes=recipe, loss=loss, checkpoint=path,
                                             track_ideal=True)], stages, resume=True)
    assert not seen
    h = report.runs["a"]
    assert h.resumed and h.optimizer is None
    assert [len(t) for t in h.loss] == [3, 2]
    assert all(torch.isnan(t).all() for t in h.loss + h.loss_ideal)
    for k, v in fresh.state_dict().items():
        assert torch.equal(v, weights[k])

    # without resume, an existing checkpoint raises
    with pytest.raises(FileExistsError):
        trainer.train_many([TrainingRun("a", fresh, recipes=recipe, loss=loss, checkpoint=path)], stages)
    # a checkpoint that doesn't fit the model raises, also with resume=True
    with pytest.raises(RuntimeError):
        trainer.train_many([TrainingRun("a", nn.Linear(3, 3), recipes=recipe, loss=loss,
                                        checkpoint=path)], stages, resume=True)

    # load_reconstructor round-trips the weights and returns what was stored
    other = noisy_twin.make_model(seed=2)
    checkpoint = load_reconstructor(other, path)
    for k, v in other.state_dict().items():
        assert torch.equal(v, weights[k])
    assert checkpoint["recipes"] == [asdict(recipe)] * 2
    assert checkpoint["stages"] == [asdict(s) for s in stages]
    assert (checkpoint["delay"], checkpoint["seed"]) == (2, 0)
    assert [len(t) for t in checkpoint["loss_history"]] == [3, 2]
    assert checkpoint["optimizer_state_dict"] is not None
    with pytest.raises(FileNotFoundError):
        load_reconstructor(other, tmp_path / "missing.pth")
    # Trainer.load_checkpoint reads train_many's files too
    t = _trainer(noisy_twin, noisy_twin.make_model(seed=3), None)
    t.load_checkpoint(path)
    for k, v in t.phaseReconstructor.state_dict().items():
        assert torch.equal(v, weights[k])


@pytest.mark.slow
def test_plot_losses_one_panel_per_stage(noisy_twin, tmp_path):
    import matplotlib.pyplot as plt

    path = tmp_path / "done.pth"
    trainer = _trainer(noisy_twin)
    stages = [Stage(4, 1), Stage(4, 2)]
    trainer.train_many([TrainingRun("done", noisy_twin.make_model(), checkpoint=path)], stages)
    runs = [TrainingRun("done", noisy_twin.make_model(), checkpoint=path),
            TrainingRun("new", noisy_twin.make_model(), track_ideal=True)]
    report = trainer.train_many(runs, stages, resume=True)

    fig, axes = report.plot_losses(smoothing_window=3)
    assert len(axes) == 2
    for ax in axes:
        assert [line.get_label() for line in ax.lines] == ["new", "new (ideal)"]
    assert len(axes[1].lines[0].get_xdata()) == 1    # window shrunk to the 2-step stage
    fig2, axes2 = plt.subplots(1, 2)
    report.plot_losses(runs=["new"], ideal=False, fig=fig2, axes=axes2)
    assert [len(ax.lines) for ax in axes2] == [1, 1]
    plt.close("all")
