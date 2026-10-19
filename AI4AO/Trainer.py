"""Closed-loop training and evaluation of reconstructors on a frozen AO twin.

Two loops, both over several networks on shared conditions (Ideas/19):

- `Trainer.train_many(runs, stages, ...)`: trains every `TrainingRun` on the same
  draws, stage by stage. The atmosphere and the first closed-loop iteration's frames
  are drawn once per optimizer step and shared by every run.
- `Trainer.evaluate_many(reconstructors, seeds, ...)`: closed-loop rollouts of every
  reconstructor with `seed_everything(seed)` before each one, so they all see the
  same atmosphere and noise; returns steady-state nm RMS and Strehl per seed, with
  paired differences against a baseline.

`Trainer.train()` and `Trainer.evaluate()` are single-network wrappers of these two
loops. `train()` is bit-identical to the version before the refactor; `evaluate()`
closes the loop at 0.2 n instead of 0.3 n (`loop_start=0.3` gives the old rollout).
"""
from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch  # type: ignore[import]
import numpy as np
import torch.nn as nn # type: ignore[import]
import matplotlib.pyplot as plt
from tqdm import tqdm

from .paths import ensure_parent
from .Utils import DelayLine, fused_optimizer_supported, seed_everything

if TYPE_CHECKING:
    from .CNNDiagnostics import CNNDiagnostics


# =====================================================================
# Training timeline and recipes
# =====================================================================

@dataclass(frozen=True)
class Stage:
    """One segment of the training timeline shared by every run of a train_many call.
    `frames` counts simulated closed-loop frames, as Trainer.train's training_steps."""
    frames: int
    closed_loop_iterations: int = 1

    @property
    def steps(self):
        """Optimizer steps in this stage."""
        return self.frames // self.closed_loop_iterations


@dataclass(frozen=True)
class Recipe:
    """How one run is optimized during one stage. A fresh AdamW and scheduler are built
    from it at the start of every stage."""
    lr: float = 3e-4
    schedule: str = "cosine"                 # "cosine": linear warm-up then cosine decay | "constant"
    warmup_frac: float = 0.02
    final_factor: float = 0.0                # cosine floor, as a fraction of lr
    group_lr_factors: dict | None = None     # submodule name -> lr multiplier, e.g. {"stem": 5.0}
    weight_decay: float = 0.01


TWO_STAGE = [Stage(10_000, 1), Stage(10_000, 10)]          # BestCNN.ipynb's timeline
BESTCNN_RECIPE = Recipe(lr=1e-4, schedule="constant")      # old recipe, kept reproducible
RECOMMENDED_RECIPE = Recipe(lr=3e-4, schedule="cosine", warmup_frac=0.02,
                            group_lr_factors={"stem": 5.0})
EVAL_SEEDS = (1234, 1235, 1236, 1237, 1238)


def make_optimizer(model, recipe, fused="auto"):
    """AdamW: the parameters not named in recipe.group_lr_factors in the base group
    (recipe.lr), then one group per entry of group_lr_factors (lr x factor). An unknown
    submodule name raises; it is never ignored."""
    groups, used = [], set()
    for name, factor in (recipe.group_lr_factors or {}).items():
        params = list(model.get_submodule(name).parameters())
        groups.append({"params": params, "lr": recipe.lr * factor})
        used |= {id(p) for p in params}
    rest = [p for p in model.parameters() if id(p) not in used]
    if fused == "auto":
        fused = fused_optimizer_supported(next(model.parameters()).device)
    return torch.optim.AdamW([{"params": rest, "lr": recipe.lr}, *groups],
                             weight_decay=recipe.weight_decay, fused=fused)


def make_scheduler(optimizer, recipe, total_steps):
    """None for a constant lr; otherwise a LambdaLR: linear warm-up over
    recipe.warmup_frac of the stage (peak lr on the last warm-up step), then cosine
    decay to final_factor x lr, reached at the end of the stage, one step after the
    last one (the schedule of prototyping/SearchingForABetterCNN/common.py, which
    produced the Ideas/18 numbers). Applied to every parameter group."""
    if recipe.schedule == "constant":
        return None
    if recipe.schedule != "cosine":
        raise ValueError(f"unknown schedule {recipe.schedule!r}; use 'cosine' or 'constant'")
    warm = max(1, round(recipe.warmup_frac * total_steps))

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        t = min(1.0, (step - warm) / max(1, total_steps - warm))
        return recipe.final_factor + (1 - recipe.final_factor) * 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def previous_version_path(path):
    """Where train_many keeps the file it replaces at `path`: RamaCNN.pth -> RamaCNN.prev.pth."""
    path = Path(path)
    return path.with_name(f"{path.stem}.prev{path.suffix}")


def load_reconstructor(model, path, map_location=None):
    """Load a checkpoint's reconstructor weights into `model`, with no Trainer needed.
    Strict: a checkpoint that doesn't fit the model raises, and a missing file raises
    FileNotFoundError. Reads Trainer.save_checkpoint's and train_many's format. Returns
    the checkpoint dict (recipes, loss history, ...) for inspection."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["phase_reconstructor_state_dict"])
    return checkpoint


# =====================================================================
# Runs, histories, reports
# =====================================================================

@dataclass(eq=False)
class TrainingRun:
    """One network to train with train_many, and how to train it."""
    name: str
    model: nn.Module
    recipes: Recipe | list[Recipe] | None = None
        # None   -> train_many's default_recipe in every stage
        # Recipe -> that recipe in every stage (still a fresh optimizer per stage)
        # list   -> one recipe per stage, len(recipes) == len(stages)
    loss: Callable | None = None             # None -> trainer.loss
    optimizer: torch.optim.Optimizer | None = None
        # Legacy mode: use this optimizer as is in every stage (no fresh optimizer, no
        # scheduler). Cannot be combined with recipes. train() uses it to stay bit-identical.
    diagnostics: CNNDiagnostics | None = None
    log_every: int = 250                     # diagnostics period, in optimizer steps
    init: str | Path | None = None
        # None -> train from the model's current weights (from scratch for a fresh model);
        # a path -> load these weights first (strict). init == output keeps training in place.
    output: str | Path | None = None
        # final weights, written once after the last stage; an existing file is first
        # renamed to <stem>.prev<suffix> (one previous version is kept)
    track_ideal: bool = False                # also track the loss of the oracle correction

    def recipe_for(self, k, default):
        if self.recipes is None:
            return default
        return self.recipes if isinstance(self.recipes, Recipe) else self.recipes[k]


@dataclass
class RunHistory:
    """What train_many recorded for one run. Tensors stay on the trainer's device."""
    loss: list = field(default_factory=list)          # one (steps,) tensor per stage; NaN if skipped
    loss_ideal: list | None = None                    # same, for track_ideal runs
    lr: list = field(default_factory=list)            # one (steps, n_param_groups) tensor per stage
    optimizer: torch.optim.Optimizer | None = None    # the last stage's optimizer (None if skipped)
    skipped: bool = False                             # skip_trained: loaded from run.output, not trained
    seconds: float | None = None                      # this run's compute time; only with time_runs=True

    @classmethod
    def skipped_run(cls, stages, track_ideal, device):
        nan = [torch.full((s.steps,), float("nan"), device=device) for s in stages]
        return cls(loss=nan, loss_ideal=[t.clone() for t in nan] if track_ideal else None,
                   lr=[torch.full((s.steps, 0), float("nan")) for s in stages], skipped=True)


@dataclass
class TrainingReport:
    """What a train_many call returns: one RunHistory per run name, plus timing."""
    runs: dict[str, RunHistory]
    stages: list[Stage]
    wall_seconds: float
    draw_seconds: float                      # synchronized with CUDA only when time_runs=True
    peak_memory_gib: float                   # whole call (the runs interleave); NaN on CPU

    def plot_losses(self, smoothing_window=100, runs=None, ideal=True, log_y=False,
                    ylim=None, fig=None, axes=None):
        """One panel per stage (their lengths differ: stage 2 has 1/10 of the steps), each
        run's training loss smoothed within the stage (never across a boundary), plus a
        dashed black curve per run that tracked the ideal loss. Skipped runs (loaded with
        skip_trained, not trained) are left out.
        The window shrinks to the stage length if needed. `runs` selects run names (all by
        default); pass `fig`/`axes` (one axis per stage) to draw into existing axes.
        Returns (fig, axes)."""
        names = [n for n in (runs if runs is not None else self.runs) if not self.runs[n].skipped]
        if axes is None:
            fig, axes = plt.subplots(1, len(self.stages), figsize=(5.5 * len(self.stages), 4),
                                     squeeze=False)
        axes = np.asarray(axes).ravel()
        if len(axes) != len(self.stages):
            raise ValueError(f"need one axis per stage ({len(self.stages)}), got {len(axes)}")

        for k, (stage, ax) in enumerate(zip(self.stages, axes)):
            for name in names:
                h = self.runs[name]
                curves = [(h.loss[k], {"label": name})]
                if ideal and h.loss_ideal is not None:
                    curves.append((h.loss_ideal[k], {"label": f"{name} (ideal)", "color": "black",
                                                     "linestyle": "--"}))
                for values, style in curves:
                    y = values.detach().cpu().numpy()
                    if len(y) == 0:
                        continue
                    w = min(smoothing_window, len(y))
                    smoothed = np.convolve(y, np.ones(w) / w, "valid")
                    ax.plot(np.arange(w - 1, len(y)), smoothed, marker="o" if len(smoothed) == 1 else None,
                            **style)
            ax.set_title(f"Stage {k + 1}: {stage.frames} frames, "
                         f"{stage.closed_loop_iterations} closed-loop iteration(s)")
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel("Training loss")
            if log_y:
                ax.set_yscale("log")
            if ylim is not None:
                ax.set_ylim(*ylim)
            if names:
                ax.legend()
        if fig is None:
            fig = axes[0].figure
        return fig, axes


@dataclass
class EvaluationResult:
    """Trajectories from a closed-loop evaluation rollout, each shaped (n_steps, ...),
    plus the rollout's steady-state metrics (see Trainer._rollout)."""
    opd: torch.Tensor
    pupil: torch.Tensor
    opd_reconstructed: torch.Tensor
    residual_opd: torch.Tensor
    wfs_frames: torch.Tensor
    psfs: torch.Tensor
    z_output: torch.Tensor
    metrics: dict | None = None


@dataclass
class EvaluationReport:
    """What evaluate_many returns: per-seed metrics for every reconstructor name."""
    seeds: list[int]
    nm_rms: dict[str, list[float]]           # steady-state residual OPD, nm RMS over the pupil, per seed
    strehl: dict[str, list[float]]           # mean PSF-peak Strehl over the same frames, per seed
    strehl_marechal: dict[str, list[float]]  # exp(-sigma^2), sigma^2 = mean piston-free residual phase variance
    baseline: str | None = None
    trajectories: dict[str, list[EvaluationResult]] | None = None   # per seed, keep_trajectories only

    def paired_delta(self, name, baseline=None):
        """Per-seed nm RMS difference to the baseline: (mean, standard error, per-seed
        array). The standard error is NaN with a single seed."""
        baseline = baseline or self.baseline
        if baseline is None:
            raise ValueError("no baseline: pass baseline= here or to evaluate_many")
        d = np.array(self.nm_rms[name]) - np.array(self.nm_rms[baseline])
        sem = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        return float(d.mean()), float(sem), d

    def table(self):
        """name | nm RMS mean +- sd | delta vs baseline +- s.e. | Strehl | Marechal Strehl,
        as printable text. Undefined spreads (a single seed) print as "–"."""
        def pm(mean, spread, sign=""):
            return f"{mean:{sign}7.1f} ± " + (f"{spread:5.1f}" if np.isfinite(spread) else "    –")

        n = len(self.seeds)
        name_w = max([len("model")] + [len(k) for k in self.nm_rms])
        delta_head = (f"Δ vs {self.baseline}" if self.baseline else "Δ") + " (± s.e.)"
        delta_w = max(15, len(delta_head))
        lines = [f"{'model':<{name_w}}  {'nm RMS (± sd)':>15}  {delta_head:>{delta_w}}  {'Strehl':>6}  {'Maréchal':>8}"]
        for name, values in self.nm_rms.items():
            v = np.array(values)
            if self.baseline is None:
                delta = "–"
            elif name == self.baseline:
                delta = "baseline"
            else:
                mean, sem, _ = self.paired_delta(name)
                delta = pm(mean, sem, "+")
            lines.append(f"{name:<{name_w}}  {pm(v.mean(), v.std(ddof=1) if n > 1 else math.nan):>15}  "
                         f"{delta:>{delta_w}}  {np.mean(self.strehl[name]):6.3f}  "
                         f"{np.mean(self.strehl_marechal[name]):8.3f}")
        lines.append(f"({n} evaluation seed{'s' if n != 1 else ''}: {', '.join(map(str, self.seeds))})")
        return "\n".join(lines)


@dataclass
class StepDraw:
    """Everything one optimizer step needs that is model-independent."""
    batches: list[dict]          # dataset[0..cl-1]
    frames0_raw: torch.Tensor    # first iteration's WFS frame: depends on the atmosphere only
    frames0: torch.Tensor        # ... preprocessed (with pupil noise)


# =====================================================================
# Trainer
# =====================================================================

class Trainer:
    def __init__(self, wfs, framePreprocessor, dm, M2C, phaseReconstructor=None, dataset=None,
                 loss=None, optimizer=None):
        """`phaseReconstructor`, `loss` and `optimizer` are optional, so a comparison can build
        one trainer for the twin alone and pass its networks to train_many/evaluate_many.
        `dataset` is required (its default only lets the arguments before it be optional)."""
        if dataset is None:
            raise TypeError("Trainer needs a dataset")
        self.wfs = wfs
        self.framePreprocessor = framePreprocessor
        self.dm = dm
        self.M2C = M2C
        self.phaseReconstructor = phaseReconstructor
        self.dataset = dataset
        self.loss = loss
        self.optimizer = optimizer
        self.device = wfs.device
        self.last_report = None
        self._linear_ready = False

        self.z_inv = torch.linalg.pinv(dm(M2C.T).flatten(start_dim = -2))

    def _require_reconstructor(self, what):
        if self.phaseReconstructor is None:
            raise ValueError(f"Trainer.{what} needs a phaseReconstructor, and this trainer was built "
                             f"without one. Pass phaseReconstructor=..., or use train_many/evaluate_many.")

    def _freeze_twin(self):
        # eval() rebuilds the WFS mask/reference and the DM influence functions, so only
        # call it when needed (as the loops always did).
        if self.dm.training:
            self.dm.eval()
        if self.wfs.training:
            self.wfs.eval()

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    @torch.no_grad()
    def draw_step(self, closed_loop_iterations):
        """Everything one optimizer step needs that is model-independent: the atmosphere
        for every closed-loop iteration and the first iteration's WFS frames (no correction
        is applied yet, so they don't depend on the model). Shared by every run."""
        batches = [self.dataset[i] for i in range(closed_loop_iterations)]
        self.wfs.SetPhotonsAndRON(batches[0]["nphotons"], batches[0]["ron"])
        frames0_raw = self.wfs(batches[0]["opd"], batches[0]["pupil"])
        return StepDraw(batches, frames0_raw, self.framePreprocessor.ProcessFrame(frames0_raw))

    def closed_loop_loss(self, model, loss_fn, draw, delay=2, track_ideal=False):
        """One model's closed loop on a pre-drawn step. Returns (loss, ideal_loss), both
        averaged over the closed-loop iterations; ideal_loss is None unless track_ideal.
        The loop delay `delay` (frames, may be fractional) follows DelayLine's convention;
        2 is what train() always used."""
        M2C_T = self.M2C.T
        gain, leak = draw.batches[0]["loop_gain"], draw.batches[0]["loop_leak"]
        n = len(draw.batches)
        with torch.no_grad():
            opd0 = draw.batches[0]["opd"]
            z_estimated = torch.zeros(opd0.shape[0], self.z_inv.shape[-1], device=self.device)
            opd_reconstructed = torch.zeros_like(opd0)
            line = DelayLine(delay)
        total, ideal = 0, (0 if track_ideal else None)

        for i, batch in enumerate(draw.batches):
            with torch.no_grad():
                opd_gt, pupil = batch["opd"], batch["pupil"]
                residual_opd = opd_gt - opd_reconstructed
                Ze = residual_opd.flatten(start_dim=-2) @ self.z_inv
                if i == 0:  # no correction yet: the shared frames are this model's frames
                    wfs_frames, frames = draw.frames0_raw, draw.frames0
                else:
                    wfs_frames = self.wfs(residual_opd, pupil)
                    frames = self.framePreprocessor.ProcessFrame(wfs_frames)

            z_output = model(frames)
            corrected = residual_opd - self.dm(z_output @ M2C_T)
            total = total + loss_fn(Ze, z_output, pupil, residual_opd, corrected, wfs_frames) / n

            with torch.no_grad():
                if track_ideal:
                    ideal_corrected = residual_opd - self.dm(Ze @ M2C_T)
                    ideal = ideal + loss_fn(Ze, Ze, pupil, residual_opd, ideal_corrected, wfs_frames) / n
                line.push(z_output.detach())
                z_estimated = z_estimated * leak + gain * line.read()
                opd_reconstructed = self.dm(z_estimated @ M2C_T)
        return total, ideal

    def _train_step(self, run, optimizer, scheduler, draw, delay, global_step):
        log = run.diagnostics is not None and global_step % run.log_every == 0
        # CNNDiagnostics contract: backward() and step() inside the capture block
        with (run.diagnostics.capture(activations=False) if log else nullcontext()):
            loss_fn = run.loss if run.loss is not None else self.loss
            loss, ideal = self.closed_loop_loss(run.model, loss_fn, draw, delay, run.track_ideal)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if log:
            run.diagnostics.record(global_step)  # the step index runs on across stages
        if scheduler is not None:
            scheduler.step()
        return loss.detach(), ideal

    def _validate(self, runs, stages, default_recipe):
        if not runs:
            raise ValueError("train_many needs at least one TrainingRun")
        if not stages:
            raise ValueError("train_many needs at least one Stage")
        for s in stages:
            if s.closed_loop_iterations < 1:
                raise ValueError(f"{s}: closed_loop_iterations must be >= 1")
        names = [r.name for r in runs]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate run names: {duplicates}")
        paths = [Path(r.output).resolve() for r in runs if r.output is not None]
        duplicates = sorted({str(p) for p in paths if paths.count(p) > 1})
        if duplicates:
            raise ValueError(f"several runs share an output path: {duplicates}")
        for r in runs:
            if r.init is not None and not Path(r.init).exists():
                raise FileNotFoundError(f"run {r.name!r}: init {r.init} does not exist")
            if r.loss is None and self.loss is None:
                raise ValueError(f"run {r.name!r} has no loss, and the trainer has none either")
            if r.optimizer is not None:
                if r.recipes is not None:
                    raise ValueError(f"run {r.name!r}: pass recipes or an optimizer, not both")
                continue
            if isinstance(r.recipes, (list, tuple)) and len(r.recipes) != len(stages):
                raise ValueError(f"run {r.name!r}: {len(r.recipes)} recipes for {len(stages)} stages")
            for k in range(len(stages)):
                for group in (r.recipe_for(k, default_recipe).group_lr_factors or {}):
                    try:
                        r.model.get_submodule(group)
                    except AttributeError:
                        raise ValueError(f"run {r.name!r}: recipe group {group!r} is not a "
                                         f"submodule of the model") from None

    def train_many(self, runs, stages, *, default_recipe=Recipe(), delay=2, seed=None,
                   skip_trained=False, print_every=None, time_runs=False):
        """Train every run on the same draws, stage by stage, one graph alive at a time.

        dm and wfs are put in eval mode: the twin is frozen while the networks train.
        Shared between runs: the atmosphere and the first closed-loop iteration's frames.
        Not shared: WFS/preprocessing noise in iterations i > 0, drawn per model (same
        statistics, different draws). Every run gets a fresh AdamW (and scheduler) per
        stage from its recipe, except legacy runs given an `optimizer`.

        What happens to each run:
        1. Skip: with skip_trained=True, a run with no `init` whose `output` exists is
           loaded from it and not trained. If that file doesn't fit the model, this raises
           (delete it, or pass skip_trained=False to retrain).
        2. Start: a run with `init` loads those weights first (strict: a missing file or
           weights that don't fit raise). Without `init`, it trains from the model's
           current weights. A run with `init` is never skipped.
        3. Save: after the last stage, the final weights go to `output`. An existing file
           there is first renamed to <stem>.prev<suffix>, so one previous version is kept.

        `seed` (optional) is applied once with seed_everything before the first draw.
        With `time_runs`, CUDA is synchronized around every step to time each run.
        Returns a TrainingReport.
        """
        runs, stages = list(runs), list(stages)
        self._validate(runs, stages, default_recipe)
        t_start = time.perf_counter()
        self._freeze_twin()
        cuda = torch.device(self.device).type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

        def sync():
            if time_runs and cuda:
                torch.cuda.synchronize(self.device)

        todo, history = [], {}
        for r in runs:
            output = Path(r.output) if r.output is not None else None
            if skip_trained and r.init is None and output is not None and output.exists():
                try:
                    load_reconstructor(r.model, output, self.device)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"run {r.name!r}: {output} exists but doesn't fit the model. Delete it, or "
                        f"pass skip_trained=False to retrain (the old file is kept as .prev).") from e
                history[r.name] = RunHistory.skipped_run(stages, r.track_ideal, self.device)
                print(f"{r.name}: already trained, loaded from {output}")
                continue
            if r.init is not None:
                try:
                    load_reconstructor(r.model, r.init, self.device)
                except RuntimeError as e:
                    raise RuntimeError(f"run {r.name!r}: the weights in init {r.init} don't fit "
                                       f"the model") from e
            todo.append(r)
            history[r.name] = RunHistory(loss_ideal=[] if r.track_ideal else None,
                                         seconds=0.0 if time_runs else None)
            print(f"{r.name}: training " + (f"from {r.init}" if r.init is not None else "from scratch")
                  + (f", final weights to {output}" if output is not None else ""))

        draw_seconds = 0.0
        if todo and seed is not None:
            seed_everything(seed)

        global_step = 0
        for k, stage in enumerate(stages if todo else []):
            opt, sched = {}, {}
            for r in todo:
                r.model.train()
                if r.optimizer is not None:  # legacy mode
                    opt[r.name], sched[r.name] = r.optimizer, None
                else:
                    recipe = r.recipe_for(k, default_recipe)
                    opt[r.name] = make_optimizer(r.model, recipe)
                    sched[r.name] = make_scheduler(opt[r.name], recipe, stage.steps)
            loss = {r.name: torch.zeros(stage.steps, device=self.device) for r in todo}
            ideal = {r.name: torch.zeros(stage.steps, device=self.device) for r in todo if r.track_ideal}
            lr = {r.name: [] for r in todo}

            bar = tqdm(range(stage.steps), desc=f"stage {k + 1}/{len(stages)}")
            every = print_every or max(1, stage.steps // 100)
            for step in bar:
                sync(); t0 = time.perf_counter()
                draw = self.draw_step(stage.closed_loop_iterations)
                sync(); draw_seconds += time.perf_counter() - t0
                for r in todo:
                    lr[r.name].append([g["lr"] for g in opt[r.name].param_groups])  # lr used by this step
                    t0 = time.perf_counter()
                    loss[r.name][step], ideal_loss = self._train_step(
                        r, opt[r.name], sched[r.name], draw, delay, global_step)
                    if r.track_ideal:
                        ideal[r.name][step] = ideal_loss.detach()
                    if time_runs:
                        sync(); history[r.name].seconds += time.perf_counter() - t0
                global_step += 1
                if step % every == 0 or step == stage.steps - 1:
                    window = slice(max(0, step - every + 1), step + 1)
                    bar.set_postfix({n: float(l[window].mean()) for n, l in loss.items()})
            for r in todo:
                h = history[r.name]
                h.loss.append(loss[r.name])
                h.lr.append(torch.tensor(lr[r.name], dtype=torch.float64)
                            .reshape(stage.steps, len(opt[r.name].param_groups)))
                if r.track_ideal:
                    h.loss_ideal.append(ideal[r.name])
                h.optimizer = opt[r.name]

        for r in todo:
            if r.output is not None:
                self._save_final(r, history[r.name], stages, default_recipe, delay, seed)

        peak = torch.cuda.max_memory_allocated(self.device) / 2 ** 30 if cuda else float("nan")
        return TrainingReport(history, stages, time.perf_counter() - t_start, draw_seconds, peak)

    def _save_final(self, run, history, stages, default_recipe, delay, seed):
        """train_many's checkpoint: Trainer.save_checkpoint's keys plus the timeline, the
        recipes, the starting weights (`init`) and the loss history. Written to a temporary
        file first, so an interrupted save never leaves a half-written file that
        skip_trained would take as finished; an existing file becomes <stem>.prev<suffix>."""
        recipes = ("legacy optimizer" if run.optimizer is not None
                   else [asdict(run.recipe_for(k, default_recipe)) for k in range(len(stages))])
        checkpoint = {
            "phase_reconstructor_state_dict": run.model.state_dict(),
            "optimizer_state_dict": history.optimizer.state_dict() if history.optimizer is not None else None,
            "stages": [asdict(s) for s in stages],
            "recipes": recipes,
            "delay": delay,
            "seed": seed,
            "init": str(run.init) if run.init is not None else None,
            "loss_history": [t.detach().cpu() for t in history.loss],
            "loss_ideal_history": ([t.detach().cpu() for t in history.loss_ideal]
                                   if history.loss_ideal is not None else None),
            "lr_history": history.lr,
        }
        path = Path(run.output)
        ensure_parent(path)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, tmp)
        if path.exists():
            previous = previous_version_path(path)
            os.replace(path, previous)
            print(f"{run.name}: previous {path.name} kept as {previous.name}")
        os.replace(tmp, path)

    def train(self, training_steps = 1000, closed_loop_iterations = 1, *, delay=2, recipe=None,
              diagnostics=None, log_every=250, track_ideal=True):
        """Train self.phaseReconstructor for training_steps // closed_loop_iterations
        optimizer steps (training_steps counts simulated frames). Single-network wrapper of
        train_many; dm and wfs are put in eval mode (the twin is frozen).

        recipe=None keeps self.optimizer as is (the behaviour of before train_many); a
        Recipe builds a fresh optimizer and scheduler, which then replaces self.optimizer
        so save_checkpoint saves it. Saves nothing itself. Returns (loss_tracker,
        loss_tracker_ideal), the latter None unless track_ideal. The full TrainingReport
        is kept in self.last_report."""
        self._require_reconstructor("train")
        if recipe is None and self.optimizer is None:
            raise ValueError("Trainer.train needs an optimizer: pass recipe=Recipe(...), "
                             "or set trainer.optimizer")
        run = TrainingRun("model", self.phaseReconstructor, recipes=recipe, loss=self.loss,
                          optimizer=None if recipe is not None else self.optimizer,
                          diagnostics=diagnostics, log_every=log_every, track_ideal=track_ideal)
        report = self.train_many([run], [Stage(training_steps, closed_loop_iterations)], delay=delay)
        h = report.runs["model"]
        self.optimizer = h.optimizer
        self.last_report = report
        return h.loss[0], (h.loss_ideal[0] if track_ideal else None)

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------

    def as_reconstructor(self, r):
        """nn.Module -> put in eval mode and fed preprocessed frames (no pupil noise);
        any other callable (wfs_frames, residual_opd) -> modal coefficients is used as is."""
        if isinstance(r, nn.Module):
            r.eval()
            return lambda wfs_frames, residual_opd: r(self.framePreprocessor.ProcessFrame(wfs_frames, False))
        return r

    def linear_reconstructor(self):
        """The classical reconstructor; builds the WFS's reconstruction matrix once."""
        if not self._linear_ready:
            self._freeze_twin()
            with torch.no_grad():
                self.wfs.BuildReconstructionMatrix(self.dm(self.M2C.T))
            self._linear_ready = True
        return lambda wfs_frames, residual_opd: self.wfs.GetReconstructedOPD(wfs_frames)

    def oracle_reconstructor(self):
        """Exact modal projection of the residual: the floor any reconstructor can reach."""
        return lambda wfs_frames, residual_opd: residual_opd.flatten(start_dim=-2) @ self.z_inv

    @torch.no_grad()
    def _rollout(self, reconstruct, dataset, n_steps, delay=0, gain=None, leak=None,
                 loop_start=0.2, record_start=0.3, keep_trajectories=False,
                 psf_sampling=4, psf_fov=20):
        """Closed-loop rollout of `reconstruct(wfs_frames, residual_opd) -> modes`.

        The integrator runs for i > loop_start * n_steps. Metrics are accumulated on the
        fly over frames i >= int(record_start * n_steps), over dataset.pupil:
        nm_rms (residual OPD, nm RMS), strehl (PSF peak / diffraction-limited peak, mean)
        and strehl_marechal (exp(-mean per-frame, per-sample piston-free residual phase
        variance), at the WFS wavelength, the band mean if polychromatic). With
        record_start <= loop_start the first recorded frames are uncorrected (the old
        evaluate window, loop_start=0.3). Returns (metrics, EvaluationResult or None);
        the trajectories are stacked only when keep_trajectories."""
        M2C_T = self.M2C.T
        batch = dataset[0]
        opd_gt, pupil_gt = batch["opd"], batch["pupil"]
        gain = gain if gain is not None else batch["loop_gain"]
        leak = leak if leak is not None else batch["loop_leak"]
        self.wfs.SetPhotonsAndRON(batch["nphotons"], batch["ron"])
        z_estimated = torch.zeros(opd_gt.shape[0], self.z_inv.shape[-1], device=self.device)
        opd_reconstructed = torch.zeros_like(opd_gt)
        line = DelayLine(delay)

        record_from = int(n_steps * record_start)
        pupil = dataset.pupil.bool()
        peak0 = self.wfs.GetPSF(torch.zeros_like(opd_gt[:1]), pupil_gt[:1], psf_sampling, psf_fov)
        peak0 = peak0.flatten(1).amax(1)[0]
        wavelength = torch.as_tensor(self.wfs.wavelength, dtype=torch.float64).reshape(-1).mean()
        k = 2 * math.pi / float(wavelength)
        sum_sq = torch.zeros((), dtype=torch.float64, device=self.device)
        sum_strehl = torch.zeros((), dtype=torch.float64, device=self.device)
        sum_var = torch.zeros((), dtype=torch.float64, device=self.device)
        n_values, n_samples = 0, 0
        keep = {key: [] for key in ("opd", "pupil", "opd_reconstructed", "residual_opd",
                                    "wfs_frames", "psfs", "z_output")}

        for i in range(n_steps):
            if i > 0:
                batch = dataset[i]
                opd_gt, pupil_gt = batch["opd"], batch["pupil"]
            residual_opd = opd_gt - opd_reconstructed
            wfs_frames = self.wfs(residual_opd, pupil_gt)
            psf = self.wfs.GetPSF(residual_opd, pupil_gt, psf_sampling, psf_fov)
            z_output = reconstruct(wfs_frames, residual_opd)
            line.push(z_output)              # pushed every frame, also before the loop closes
            if i > n_steps * loop_start:
                z_estimated = z_estimated * leak + gain * line.read()
            opd_reconstructed = self.dm(z_estimated @ M2C_T)

            if i >= record_from:
                v = residual_opd[..., pupil].double()                  # (B, Npupil)
                sum_sq += (v ** 2).sum()
                n_values += v.numel()
                sum_strehl += (psf.flatten(1).amax(1) / peak0).double().sum()
                sum_var += torch.var(k * v, dim=-1).sum()
                n_samples += v.shape[0]
            if keep_trajectories:
                for key, value in zip(keep, (opd_gt, pupil_gt, opd_reconstructed, residual_opd,
                                             wfs_frames, psf, z_output)):
                    keep[key].append(value)

        if n_samples:
            metrics = {"nm_rms": math.sqrt(float(sum_sq) / n_values) * 1e9,
                       "strehl": float(sum_strehl) / n_samples,
                       "strehl_marechal": math.exp(-float(sum_var) / n_samples)}
        else:
            metrics = {"nm_rms": float("nan"), "strehl": float("nan"), "strehl_marechal": float("nan")}
        result = None
        if keep_trajectories:
            result = EvaluationResult(**{key: torch.stack(v) for key, v in keep.items()}, metrics=metrics)
        return metrics, result

    def evaluate_many(self, reconstructors, seeds=EVAL_SEEDS, n_steps=100, *, dataset=None,
                      baseline=None, references=(), delay=0, gain=None, leak=None, loop_start=0.2,
                      record_start=0.3, keep_trajectories=False, psf_sampling=4, psf_fov=20):
        """Closed-loop rollouts of every reconstructor (name -> nn.Module or callable
        (wfs_frames, residual_opd) -> modes) under identical conditions:
        seed_everything(seed) before each model's rollout, so the atmosphere and the noise
        are the same for all of them (reconstructors must not draw random numbers).
        `references` adds the reference reconstructors, after the networks and under
        their own names: "linear" (linear_reconstructor()) and/or "oracle"
        (oracle_reconstructor(), the floor), e.g. references=("linear", "oracle").
        They can also serve as the baseline. dm and wfs are put in eval mode. See
        _rollout for the window and the metrics. Returns an EvaluationReport;
        print(report.table()) for the summary."""
        builders = {"linear": self.linear_reconstructor, "oracle": self.oracle_reconstructor}
        if isinstance(references, str):
            references = (references,)
        unknown = [r for r in references if r not in builders]
        if unknown:
            raise ValueError(f"unknown references {unknown}; use {list(builders)}")
        clashes = [r for r in references if r in reconstructors]
        if clashes:
            raise ValueError(f"reconstructor names {clashes} clash with the references of the same name")
        reconstructors = {**reconstructors, **{r: builders[r]() for r in dict.fromkeys(references)}}
        if baseline is not None and baseline not in reconstructors:
            raise ValueError(f"baseline {baseline!r} is not one of {list(reconstructors)}")
        dataset = dataset if dataset is not None else self.dataset
        self._freeze_twin()
        recs = {name: self.as_reconstructor(r) for name, r in reconstructors.items()}
        nm_rms = {name: [] for name in recs}
        strehl = {name: [] for name in recs}
        strehl_marechal = {name: [] for name in recs}
        trajectories = {name: [] for name in recs} if keep_trajectories else None
        for seed in seeds:
            for name, rec in recs.items():
                seed_everything(seed)
                metrics, result = self._rollout(rec, dataset, n_steps, delay, gain, leak,
                                                loop_start, record_start, keep_trajectories,
                                                psf_sampling, psf_fov)
                nm_rms[name].append(metrics["nm_rms"])
                strehl[name].append(metrics["strehl"])
                strehl_marechal[name].append(metrics["strehl_marechal"])
                if keep_trajectories:
                    trajectories[name].append(result)
            print(f"seed {seed}: " + "  ".join(f"{n}={nm_rms[n][-1]:.1f}" for n in recs))
        return EvaluationReport(list(seeds), nm_rms, strehl, strehl_marechal, baseline, trajectories)

    @torch.no_grad()
    def evaluate(self, n_steps = 100, dataset = None, gain = None, leak = None, psf_sampling = 4,
                 psf_fov = 20, delay = 0, loop_start = 0.2, record_start = 0.3):
        """Closed-loop rollout of self.phaseReconstructor with no backprop and no pupil
        noise, for inspection/plotting. Does not reseed. dm and wfs are put in eval mode.
        Returns an EvaluationResult with every frame, and the metrics of frames
        i >= int(record_start * n_steps) in result.metrics (see _rollout).
        The loop closes for i > loop_start * n_steps; loop_start=0.3 reproduces the
        evaluate() of before train_many/evaluate_many exactly."""
        self._require_reconstructor("evaluate")
        self._freeze_twin()
        _, result = self._rollout(self.as_reconstructor(self.phaseReconstructor),
                                  dataset if dataset is not None else self.dataset, n_steps,
                                  delay, gain, leak, loop_start=loop_start, record_start=record_start,
                                  keep_trajectories=True, psf_sampling=psf_sampling, psf_fov=psf_fov)
        return result

    # -----------------------------------------------------------------
    # Plotting and checkpoints
    # -----------------------------------------------------------------

    def plot_losses(self, loss_tracker, loss_tracker_ideal = None, smoothing_window = 100, ylim = None, log_x = False, show = True):
        losses = loss_tracker.detach().cpu().numpy()
        smoothed = np.convolve(losses, np.ones(smoothing_window) / smoothing_window, "valid")

        fig, ax = plt.subplots()
        plot_fn = ax.semilogx if log_x else ax.plot
        plot_fn(smoothed, label = "Training loss")

        if loss_tracker_ideal is not None:
            ideal = loss_tracker_ideal.detach().cpu().numpy()
            smoothed_ideal = np.convolve(ideal, np.ones(smoothing_window) / smoothing_window, "valid")
            plot_fn(smoothed_ideal, label = "Ideal loss")

        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Loss = $\ln(\mathrm{var}(\phi - \phi_{est}))$")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend()

        if show:
            plt.show()

        return fig, ax

    def save_checkpoint(self, path, **extra_state):
        """Save the reconstructor + optimizer state. Extra keyword args (e.g. step, loss_tracker,
        TrainParams) are stored alongside for reference and returned as-is by load_checkpoint."""
        self._require_reconstructor("save_checkpoint")
        checkpoint = {
            "phase_reconstructor_state_dict": self.phaseReconstructor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
        }
        checkpoint.update(extra_state)
        ensure_parent(path)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, load_optimizer = True):
        """Load the reconstructor (+ optimizer) state from path (strict, via
        load_reconstructor). A missing file prints a message and returns None."""
        self._require_reconstructor("load_checkpoint")
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}, starting from scratch")
            return None

        checkpoint = load_reconstructor(self.phaseReconstructor, path, self.device)

        if load_optimizer and self.optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
