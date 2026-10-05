"""
Diagnostics for reconstructor CNNs: feature maps, gradients and sensitivities.

`CNNDiagnostics` wraps a model and, inside a `capture()` block, records what flows
through its Conv2d/Linear layers during *your own* forward/backward pass (so the
gradients are the real end-to-end ones, e.g. through the WFS -> DM -> loss chain of a
closed-loop training step). It also offers a self-contained `run()` for quick
single-batch checks, per-mode input Jacobians, a cheap per-layer scalar history, and
plotting methods built on `Utils.imshow`/`imshow_multiple`.

Design and rationale: `Ideas/17-cnn-diagnostics.md`.

Typical use in a hand-rolled training loop::

    diag = CNNDiagnostics(phaseReconstructor)
    fig = axes = None
    for step in range(n_steps):
        log = step % 50 == 0
        with (diag.capture(input_grad=True) if log else nullcontext()):
            ...closed loop...
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
        if log:
            diag.record(step)
            fig, axes = diag.plot_gradient_flow(fig=fig, axes=axes)
"""

import math
import re
import warnings
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]

from AI4AO.Utils import imshow, imshow_multiple


METRICS = ("weight_grad_norm", "act_grad_rms", "update_ratio", "act_mean", "act_std", "dead_frac")
_LOG_METRICS = {"weight_grad_norm", "act_grad_rms", "update_ratio", "act_std"}
_INPUT = "__input__"


def _to_numpy(t):
    if torch.is_tensor(t):
        return t.detach().float().cpu().numpy()
    return np.asarray(t)


def _mosaic(arr, gap=1, block_gap=4):
    """Tile a [B, C, H, W] array into one 2-D image. Each of the B entries becomes
    a near-square block of its C tiles; the blocks are themselves laid out on a
    near-square grid. Gaps are NaN, so they render as the colormap's "bad" colour."""
    B, C, H, W = arr.shape
    cc = math.ceil(math.sqrt(C))
    cr = math.ceil(C / cc)
    bc = math.ceil(math.sqrt(B))
    br = math.ceil(B / bc)
    block_h = cr * (H + gap) - gap + block_gap
    block_w = cc * (W + gap) - gap + block_gap
    out = np.full((br * block_h - block_gap, bc * block_w - block_gap), np.nan, dtype=np.float32)
    for b in range(B):
        by, bx = divmod(b, bc)
        for c in range(C):
            ty, tx = divmod(c, cc)
            y = by * block_h + ty * (H + gap)
            x = bx * block_w + tx * (W + gap)
            out[y:y + H, x:x + W] = arr[b, c]
    return out


def _grid_figsize(B, C, panel=1.4):
    """Figure size matching imshow()'s [B, C] layout (groups and channels each laid
    out on their "middle divisor" grid), so the panels are not padded with blank
    space."""
    def divisor(n):
        divs = [d for d in range(1, n + 1) if n % d == 0]
        return divs[len(divs) // 2]
    bcols, ncols = divisor(B), divisor(C)
    brows, nrows = math.ceil(B / bcols), math.ceil(C / ncols)
    return (max(3.0, panel * bcols * ncols * 1.1), max(2.0, panel * brows * nrows * 1.1))


def _mosaic_figsize(shape, max_inches=12.0):
    """Figure size following a mosaic's aspect ratio, longest side capped."""
    h, w = shape
    scale = min(max_inches / max(h, w), 0.15)
    return (max(3.0, w * scale), max(2.0, h * scale))


def _depth_colors(n):
    """One sequential hue, light (shallow) -> dark (deep); the lightest end is
    trimmed so the first layer stays visible on a white background."""
    cmap = plt.get_cmap("Blues")
    return [cmap(v) for v in np.linspace(0.35, 1.0, max(n, 1))]


class CNNDiagnostics:
    """
    Observe a CNN's activations, gradients and input sensitivities.

    Parameters
    ----------
    model : nn.Module
        The network to diagnose (e.g. a `PhaseEstimators` reconstructor).
    layer_types : tuple of nn.Module types
        Modules to hook. Defaults to the weight-carrying Conv2d and Linear layers;
        add e.g. `nn.ReLU` or `nn.BatchNorm2d` to also see post-activation/norm maps.
    include, exclude : str, optional
        Regexes matched (`re.search`) against `named_modules()` paths to narrow the
        hooked layers further.
    max_samples : int
        Number of batch samples whose tensors are stored (scalar statistics always use
        the full batch).
    store_device : str or torch.device
        Where stored tensors are kept. CPU by default, to spare GPU memory.

    Notes
    -----
    Layers are addressed by their `named_modules()` path (`"encoder.0"`) or by an
    integer index into `self.layers`; every method accepts either.

    With the default Conv2d/Linear hooks, feature maps (and `dead_frac`) are measured
    *before* the following nonlinearity. `dead_frac` is therefore exact for a ReLU
    and only indicative for smooth activations (GELU/SiLU). It is NaN for the layer
    whose output is the model's output, since no nonlinearity follows it.
    """

    def __init__(
        self,
        model,
        layer_types=(nn.Conv2d, nn.Linear),
        include=None,
        exclude=None,
        max_samples=4,
        store_device="cpu",
    ):
        self.model = model
        self.max_samples = max_samples
        self.store_device = torch.device(store_device)

        self.layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, tuple(layer_types))
            and (include is None or re.search(include, name))
            and (exclude is None or not re.search(exclude, name))
        }
        if not self.layers:
            raise ValueError(
                f"No module of type {[t.__name__ for t in layer_types]} matched "
                f"include={include!r}, exclude={exclude!r}."
            )

        self.forward_order = []
        self._capturing = False
        self._token = 0
        self._pending_stats = None
        self._weight_snapshot = {}
        self._output_shapes = {}
        self._output_layers = set()  # layers whose output is the model's output
        self._clear_results()
        self.reset_history()

    # =====================================================================
    # Layer registry
    # =====================================================================

    def layer(self, key):
        """Resolve a layer name or integer index into `(name, module)`."""
        if isinstance(key, int):
            names = list(self.layers)
            if not -len(names) <= key < len(names):
                raise IndexError(f"Layer index {key} out of range: {len(names)} hooked layers.")
            name = names[key]
            return name, self.layers[name]

        if key not in self.layers:
            raise KeyError(f"Unknown layer {key!r}. Hooked layers: {list(self.layers)}")
        return key, self.layers[key]

    def weights(self, key):
        """Current weight tensor of a layer (read live, detached, on `store_device`)."""
        name, module = self.layer(key)
        weight = getattr(module, "weight", None)
        if weight is None:
            raise ValueError(f"Layer {name!r} has no weight.")
        return weight.detach().to(self.store_device)

    # =====================================================================
    # Capture
    # =====================================================================

    def _clear_results(self):
        self.input = None
        self.input_grad = None
        self.activations = {}
        self.activation_grads = {}
        self.param_grads = {}

    def _store(self, t):
        return t[: self.max_samples].detach().to(self.store_device)

    @contextmanager
    def capture(self, activations=True, activation_grads=True, input_grad=False,
                calls="last", track_updates=True):
        """
        Record activations and gradients during the forward/backward pass run inside
        the block. Hooks are removed on exit, also when the block raises.

        Parameters
        ----------
        activations : bool
            Store each hooked layer's output (first `max_samples` samples).
        activation_grads : bool
            Store dLoss/d(layer output), via tensor hooks on the outputs.
        input_grad : bool
            Store the model input and dLoss/d(input) (saliency). An input that does
            not require grad is replaced by a detached leaf that does (numerically
            identical); an input that already carries a graph is left untouched, so
            gradients keep flowing upstream.
        calls : {"last", "all"}
            The model may run several times per backward (closed-loop rollout).
            "last" keeps only the last call per layer; "all" stores a list with one
            entry per call (`None` where no gradient arrived).
        track_updates : bool
            Snapshot the hooked weights on entry, so `record()` can report the
            per-layer update ratio after `optimizer.step()`.

        Notes
        -----
        Parameter gradients are read from `p.grad` on exit. Run `backward()` inside
        the block and do not `zero_grad()` after it before the block ends.
        Scalar statistics for `record()` are accumulated over every call and the full
        batch, independently of `calls` and `max_samples`.
        """
        if calls not in ("last", "all"):
            raise ValueError(f"calls must be 'last' or 'all', got {calls!r}.")
        if self._capturing:
            raise RuntimeError("capture() blocks cannot be nested.")

        self._begin_capture(calls, track_updates)
        handles = self._register_hooks(activations, activation_grads, input_grad, calls)
        completed = False
        try:
            yield self
            completed = True
        finally:
            for handle in handles + self._tensor_handles:
                handle.remove()
            self._last_outputs = {}  # drop references into the autograd graph
            self._capturing = False

        if completed:
            self._finalize(calls, activation_grads, param_grads=None)

    def _begin_capture(self, calls, track_updates):
        self._capturing = True
        self._token += 1
        self._clear_results()
        self._tensor_handles = []
        self._last_outputs = {}
        self._last_input = None
        self._order = []
        self._ncalls = {}
        self._raw_acts = {}
        self._raw_grads = {}
        self._any_output_requires_grad = False
        self._acc = {}

        self._weight_snapshot = {}
        if track_updates:
            for name, module in self.layers.items():
                weight = getattr(module, "weight", None)
                if weight is not None:
                    self._weight_snapshot[name] = weight.detach().clone()

    def _acc_for(self, name, device):
        if name not in self._acc:
            zero = torch.zeros((), device=device)
            self._acc[name] = {
                "s1": zero.clone(), "s2": zero.clone(), "n": 0,
                "dead": zero.clone(), "ndead": 0,
                "g2": zero.clone(), "ng": 0,
            }
        return self._acc[name]

    def _register_hooks(self, store_acts, store_grads, input_grad, calls):
        token = self._token

        def keep(name, store, idx, t):
            """Store tensor `t` for call `idx`; in "last" mode only the newest call
            survives, so memory does not grow with the rollout length."""
            if calls == "last":
                if idx < self._ncalls.get(name, 0) - 1:
                    return
                store[name] = {i: v for i, v in store.get(name, {}).items() if i >= idx}
            store.setdefault(name, {})[idx] = t

        def grad_hook(name, idx, with_stats):
            def hook(grad):
                if not self._capturing or token != self._token:
                    return
                if with_stats:
                    acc = self._acc_for(name, grad.device)
                    g = grad.detach().float()
                    acc["g2"] = acc["g2"] + (g * g).sum()
                    acc["ng"] += g.numel()
                if name == _INPUT or store_grads:
                    keep(name, self._raw_grads, idx, self._store(grad))
            return hook

        def forward_hook(name):
            def hook(module, inputs, output):
                if not torch.is_tensor(output):
                    return
                idx = self._ncalls.get(name, 0)
                self._ncalls[name] = idx + 1
                if name not in self._order:
                    self._order.append(name)
                self._output_shapes[name] = tuple(output.shape)

                o = output.detach().float()
                acc = self._acc_for(name, o.device)
                acc["s1"] = acc["s1"] + o.sum()
                acc["s2"] = acc["s2"] + (o * o).sum()
                acc["n"] += o.numel()
                if o.ndim >= 2:
                    per_unit_max = o.transpose(0, 1).flatten(1).amax(dim=1)
                    acc["dead"] = acc["dead"] + (per_unit_max <= 0).float().mean()
                    acc["ndead"] += 1

                self._last_outputs[name] = output
                if store_acts:
                    keep(name, self._raw_acts, idx, self._store(output))
                if output.requires_grad:
                    self._any_output_requires_grad = True
                    self._tensor_handles.append(output.register_hook(grad_hook(name, idx, True)))
            return hook

        def input_pre_hook(module, args):
            if not args or not torch.is_tensor(args[0]):
                warnings.warn("input_grad=True but the model's first argument is not a tensor; skipped.")
                return None
            x = args[0]
            if not x.requires_grad:
                if not x.is_floating_point():
                    warnings.warn("input_grad=True but the input is not floating point; skipped.")
                    return None
                x = x.detach().requires_grad_()
            idx = self._ncalls.get(_INPUT, 0)
            self._ncalls[_INPUT] = idx + 1
            keep(_INPUT, self._raw_acts, idx, self._store(x))
            self._last_input = x
            self._tensor_handles.append(x.register_hook(grad_hook(_INPUT, idx, False)))
            return (x,) + tuple(args[1:])

        def model_output_hook(module, inputs, output):
            # A layer whose output *is* the model output feeds no nonlinearity, so
            # its dead_frac would be meaningless; remember it to skip that metric.
            for name, out in self._last_outputs.items():
                if out is output:
                    self._output_layers.add(name)
            self._last_outputs.clear()

        handles = [m.register_forward_hook(forward_hook(n)) for n, m in self.layers.items()]
        handles.append(self.model.register_forward_hook(model_output_hook))
        if input_grad:
            handles.append(self.model.register_forward_pre_hook(input_pre_hook))
        return handles

    def _collapse(self, raw, name, calls):
        entries = raw.get(name)
        if calls == "all":
            return [
                (entries or {}).get(i) for i in range(self._ncalls.get(name, 0))
            ]
        last = self._ncalls.get(name, 0) - 1
        return (entries or {}).get(last)

    def _finalize(self, calls, activation_grads_requested, param_grads):
        if self._order:
            self.forward_order = list(self._order)

        for name in self.layers:
            if name not in self._ncalls:
                continue
            act = self._collapse(self._raw_acts, name, calls)
            grad = self._collapse(self._raw_grads, name, calls)
            if act is not None and (calls == "last" or any(a is not None for a in act)):
                self.activations[name] = act
            if grad is not None and (calls == "last" or any(g is not None for g in grad)):
                self.activation_grads[name] = grad

        if _INPUT in self._ncalls:
            self.input = self._collapse(self._raw_acts, _INPUT, calls)
            self.input_grad = self._collapse(self._raw_grads, _INPUT, calls)

        # Parameter grads: from p.grad (capture) or from autograd.grad results (run).
        weight_grad_norm = {}
        any_param_requires_grad = False
        for name, module in self.layers.items():
            for pname, p in module.named_parameters(recurse=False):
                any_param_requires_grad |= p.requires_grad
                g = p.grad if param_grads is None else param_grads.get((name, pname))
                if g is None:
                    continue
                key = name if pname == "weight" else f"{name}.{pname}"
                self.param_grads[key] = g.detach().to(self.store_device).clone()
                if pname == "weight":
                    weight_grad_norm[name] = g.detach().float().norm()

        got_act_grads = any(acc["ng"] > 0 for acc in self._acc.values())
        if activation_grads_requested and self._ncalls and not got_act_grads:
            if not self._any_output_requires_grad:
                warnings.warn(
                    "capture(): no hooked layer output required grad -- the forward pass ran "
                    "under torch.no_grad() (e.g. Trainer.evaluate) or nothing requires grad. "
                    "Pass activation_grads=False for a forward-only capture."
                )
            else:
                warnings.warn("capture(): no gradient arrived -- call backward() inside the block.")
        if param_grads is None and got_act_grads and any_param_requires_grad and not self.param_grads:
            warnings.warn(
                "capture(): all parameter grads were None at exit -- was zero_grad() called "
                "after backward() inside the block?"
            )

        self._pending_stats = self._compute_stats(weight_grad_norm)

    def _compute_stats(self, weight_grad_norm):
        nan = torch.tensor(float("nan"))
        stats = {metric: {} for metric in METRICS if metric != "update_ratio"}
        for name in self.layers:
            acc = self._acc.get(name)
            stats["weight_grad_norm"][name] = weight_grad_norm.get(name, nan)
            if acc is None or acc["n"] == 0:
                for metric in ("act_mean", "act_std", "dead_frac", "act_grad_rms"):
                    stats[metric][name] = nan
                continue
            mean = acc["s1"] / acc["n"]
            stats["act_mean"][name] = mean
            stats["act_std"][name] = (acc["s2"] / acc["n"] - mean * mean).clamp_min(0).sqrt()
            dead_defined = acc["ndead"] and name not in self._output_layers
            stats["dead_frac"][name] = acc["dead"] / acc["ndead"] if dead_defined else nan
            stats["act_grad_rms"][name] = (acc["g2"] / acc["ng"]).sqrt() if acc["ng"] else nan
        return stats

    # =====================================================================
    # Self-contained passes
    # =====================================================================

    def run(self, x, loss_fn=None, **capture_kwargs):
        """
        Forward `x`, compute `loss_fn(output)` (default: `output.pow(2).sum()`) and
        capture everything, including the input gradient. Gradients are computed with
        `torch.autograd.grad`, so the model's `.grad` buffers are never touched and
        calling this mid-training cannot pollute the next optimizer step.

        This sees the CNN-level loss only (no closed loop); use `capture()` around a
        real training step to see the end-to-end signal. Returns the detached output.
        """
        capture_kwargs.setdefault("track_updates", False)
        capture_kwargs["input_grad"] = True
        calls = capture_kwargs.get("calls", "last")
        if calls not in ("last", "all"):
            raise ValueError(f"calls must be 'last' or 'all', got {calls!r}.")
        if self._capturing:
            raise RuntimeError("run() cannot be called inside a capture() block.")

        keys, params = [], []
        for name, module in self.layers.items():
            for pname, p in module.named_parameters(recurse=False):
                if p.requires_grad:
                    keys.append((name, pname))
                    params.append(p)

        self._begin_capture(calls, capture_kwargs["track_updates"])
        handles = self._register_hooks(
            capture_kwargs.get("activations", True),
            capture_kwargs.get("activation_grads", True),
            True,
            calls,
        )
        try:
            with torch.enable_grad():
                out = self.model(x)
                loss = out.pow(2).sum() if loss_fn is None else loss_fn(out)
                inputs = params + ([self._last_input] if self._last_input is not None else [])
                grads = torch.autograd.grad(loss, inputs, allow_unused=True)
        finally:
            for handle in handles + self._tensor_handles:
                handle.remove()
            self._last_outputs = {}  # drop references into the autograd graph
            self._capturing = False

        self._finalize(calls, capture_kwargs.get("activation_grads", True),
                       param_grads=dict(zip(keys, grads)))
        if self._last_input is not None:
            # The leaf's own gradient is exact even if its tensor hook did not fire.
            input_grad = self._store(grads[-1]) if grads[-1] is not None else None
            if calls == "all":
                self.input_grad = [input_grad]
            else:
                self.input_grad = input_grad
        self._last_input = None
        return out.detach()

    def jacobian(self, x, modes):
        """
        Per-mode input Jacobian `J[k] = d z_{modes[k]} / d x`, of shape
        `(len(modes), B, *x.shape[1:])`, on `store_device`.

        Runs one forward pass and one backward pass per mode, so keep `modes` to a
        handful (tip, tilt, focus, one high-order mode). The model is switched to
        eval mode for the computation -- in train mode BatchNorm couples the samples
        of a batch and the per-sample rows would mix -- and restored afterwards, also
        on error.
        """
        modes = list(modes)
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.enable_grad():
                x_leaf = x.detach().requires_grad_()
                z = self.model(x_leaf)
                rows = []
                for i, k in enumerate(modes):
                    (g,) = torch.autograd.grad(
                        z[:, k].sum(), x_leaf, retain_graph=i < len(modes) - 1
                    )
                    rows.append(g.to(self.store_device))
        finally:
            self.model.train(was_training)
        return torch.stack(rows)

    # =====================================================================
    # Scalar history
    # =====================================================================

    def reset_history(self):
        self.history = {metric: {name: [] for name in self.layers} for metric in METRICS}
        self.history_steps = []

    def record(self, step):
        """
        Append one row of per-layer scalars (see `METRICS`) from the last capture to
        `self.history`. Call after `optimizer.step()` so `update_ratio`
        (||W_now - W_at_capture|| / ||W_at_capture||) reflects the step.
        """
        if self._pending_stats is None:
            raise RuntimeError("record() needs a completed capture() or run() first.")

        nan = torch.tensor(float("nan"))
        update_ratio = {}
        for name in self.layers:
            snap = self._weight_snapshot.get(name)
            if snap is None:
                update_ratio[name] = nan
                continue
            w = self.layers[name].weight.detach()
            update_ratio[name] = (w - snap).float().norm() / snap.float().norm().clamp_min(1e-30)

        stats = dict(self._pending_stats, update_ratio=update_ratio)
        names = list(self.layers)
        # Gather on the model's device, then one device -> host transfer for the row.
        device = self._stats_device()
        table = torch.stack([
            torch.stack([stats[metric][name].detach().float().to(device) for name in names])
            for metric in METRICS
        ]).cpu().numpy()

        for i, metric in enumerate(METRICS):
            for j, name in enumerate(names):
                self.history[metric][name].append(float(table[i, j]))
        self.history_steps.append(step)

    def _stats_device(self):
        for module in self.layers.values():
            for p in module.parameters(recurse=False):
                return p.device
        return torch.device("cpu")

    # =====================================================================
    # Plots -- each returns (fig, axes) and accepts them back to update in place
    # =====================================================================

    def _select_call(self, store, name, call, what):
        if name not in store:
            raise KeyError(f"No {what} stored for layer {name!r}; run capture()/run() first.")
        value = store[name]
        if isinstance(value, list):
            value = value[call]
        if value is None:
            raise KeyError(f"No {what} stored for layer {name!r}, call {call}.")
        return value

    @staticmethod
    def _new_container(figsize):
        """Fresh figure with one container axes. imshow() embeds into the current
        axes when one exists, so this keeps every plot in its own figure."""
        fig = plt.figure(figsize=figsize)
        fig.add_subplot(111)
        return fig

    def _show_bcwh(self, arr, max_b, max_c, mosaic, symmetric, fig, axes, figsize, **kwargs):
        arr = _to_numpy(arr)[:max_b, :max_c]
        if symmetric:
            a = float(np.nanmax(np.abs(arr))) or 1.0
            kwargs.setdefault("vmin", -a)
            kwargs.setdefault("vmax", a)
            kwargs.setdefault("cmap", "RdBu_r")
        if mosaic:
            arr = _mosaic(arr)
            # A NaN-separated mosaic needs an explicit scale (imshow's per-image
            # min/max would be NaN in update mode), and gray separators so they
            # are not mistaken for zeros on a white-centred diverging map.
            kwargs.setdefault("vmin", float(np.nanmin(arr)))
            kwargs.setdefault("vmax", float(np.nanmax(arr)))
            kwargs["cmap"] = plt.get_cmap(kwargs.get("cmap", "viridis")).with_extremes(bad="0.6")
        if fig is None or axes is None:
            if figsize is None:
                figsize = _mosaic_figsize(arr.shape) if mosaic else _grid_figsize(*arr.shape[:2])
            self._new_container(figsize)
        return imshow(arr, fig=fig, axes=axes, **kwargs)

    def plot_feature_maps(self, layer, call=-1, max_samples=None, max_channels=16, mosaic=False,
                          fig=None, axes=None, figsize=None, **kwargs):
        """Feature maps of one layer: a [B, C, H, W] grid (one boxed group per sample),
        or one line per sample for a Linear layer's [B, F] output."""
        name, _ = self.layer(layer)
        act = _to_numpy(self._select_call(self.activations, name, call, "activations"))
        max_samples = max_samples or self.max_samples

        if act.ndim == 2:
            act = act[:max_samples]
            if fig is None or axes is None:
                fig, ax = plt.subplots(figsize=figsize or (6, 3))
                for b, color in zip(range(act.shape[0]), _depth_colors(act.shape[0])):
                    ax.plot(act[b], lw=1.5, color=color, label=f"sample {b}")
                ax.set_xlabel("unit")
                ax.set_title(name)
                ax.grid(alpha=0.3)
                ax.legend(fontsize="small")
                return fig, [ax]
            ax = axes[0]
            for line, row in zip(ax.lines, act):
                line.set_ydata(row)
            ax.relim()
            ax.autoscale_view()
            return fig, axes

        if act.ndim != 4:
            raise ValueError(f"Cannot display activations of shape {act.shape} for {name!r}.")
        return self._show_bcwh(act, max_samples, max_channels, mosaic, False,
                               fig, axes, figsize, **kwargs)

    def plot_kernels(self, layer, grad=False, max_out=16, max_in=4, mosaic=False,
                     fig=None, axes=None, figsize=None, **kwargs):
        """Conv weights (or their grads) [Cout, Cin, k, k] as Cout boxed groups of Cin
        kernels, on a symmetric diverging colour scale. With `mosaic=True` they are
        tiled into one image: one block per input channel, holding that channel's
        kernel in every output filter."""
        name, module = self.layer(layer)
        t = self._select_call(self.param_grads, name, -1, "weight grads") if grad else self.weights(name)
        if t.ndim != 4:
            raise ValueError(f"plot_kernels needs a Conv2d weight; {name!r} has shape {tuple(t.shape)}.")
        t = t[:max_out, :max_in]
        if mosaic:
            t = t.transpose(0, 1)  # one block per input channel, holding every filter
        return self._show_bcwh(t, None, None, mosaic, True, fig, axes, figsize, **kwargs)

    def _ordered_names(self):
        order = [n for n in self.forward_order if n in self.layers]
        return order + [n for n in self.layers if n not in order]

    def plot_gradient_flow(self, source="weight", fig=None, axes=None, figsize=None):
        """
        Mean and max |grad| per layer in forward order, log scale -- the classic
        vanishing/exploding-gradient check. `source="weight"` uses parameter grads,
        `source="activation"` the stored dLoss/d(layer output) (first `max_samples`
        samples, all stored calls pooled).
        """
        if source not in ("weight", "activation"):
            raise ValueError(f"source must be 'weight' or 'activation', got {source!r}.")
        names = self._ordered_names()
        mean, peak = [], []
        for name in names:
            if source == "weight":
                g = self.param_grads.get(name)
            else:
                g = self.activation_grads.get(name)
                if isinstance(g, list):
                    parts = [v.flatten() for v in g if v is not None]
                    g = torch.cat(parts) if parts else None
            if g is None:
                mean.append(np.nan)
                peak.append(np.nan)
            else:
                a = g.detach().abs().float()
                mean.append(float(a.mean()))
                peak.append(float(a.max()))

        if fig is None or axes is None:
            fig, ax = plt.subplots(figsize=figsize or (max(6, 0.6 * len(names)), 4))
            x = np.arange(len(names))
            ax.bar(x, peak, width=0.8, color=plt.get_cmap("Blues")(0.45), label="max |grad|")
            ax.bar(x, mean, width=0.5, color=plt.get_cmap("Blues")(0.9), label="mean |grad|")
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=60, ha="right", fontsize="small")
            ax.set_ylabel(f"|grad| ({source})")
            ax.set_title("Gradient flow (input -> output)")
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            return fig, [ax]

        ax = axes[0]
        containers = ax.containers
        for bar, h in zip(containers[0], peak):
            bar.set_height(h)
        for bar, h in zip(containers[1], mean):
            bar.set_height(h)
        ax.relim()
        ax.autoscale_view()
        return fig, axes

    def plot_histograms(self, kind="activation", layers=None, bins=50, call=-1,
                        fig=None, axes=None, figsize=None):
        """Per-layer histograms (small multiples) of `kind` in {"activation",
        "activation_grad", "weight", "weight_grad"}. Update mode redraws into the
        same axes."""
        sources = {
            "activation": lambda n: self._select_call(self.activations, n, call, "activations"),
            "activation_grad": lambda n: self._select_call(self.activation_grads, n, call, "activation grads"),
            "weight": self.weights,
            "weight_grad": lambda n: self._select_call(self.param_grads, n, -1, "weight grads"),
        }
        if kind not in sources:
            raise ValueError(f"kind must be one of {list(sources)}, got {kind!r}.")
        if layers is not None:
            names = [self.layer(k)[0] for k in layers]
        elif kind == "weight":
            names = [n for n in self._ordered_names() if getattr(self.layers[n], "weight", None) is not None]
        else:
            stored = {"activation": self.activations, "activation_grad": self.activation_grads,
                      "weight_grad": self.param_grads}[kind]
            names = [n for n in self._ordered_names() if n in stored]
        if not names:
            raise KeyError(f"No {kind} data stored; run capture()/run() first.")

        if fig is None or axes is None:
            ncols = min(4, len(names))
            nrows = math.ceil(len(names) / ncols)
            fig, grid = plt.subplots(nrows, ncols, figsize=figsize or (3.2 * ncols, 2.4 * nrows),
                                     squeeze=False)
            axes = list(grid.flat)
            for ax in axes[len(names):]:
                ax.axis("off")
            axes = axes[: len(names)]
        elif len(axes) != len(names):
            raise ValueError(f"{len(axes)} axes given for {len(names)} layers.")

        for ax, name, color in zip(axes, names, _depth_colors(len(names))):
            values = _to_numpy(sources[kind](name)).ravel()
            ax.clear()
            ax.hist(values, bins=bins, color=color)
            if values.size > 1000:  # log counts only help when there are tails to see
                ax.set_yscale("log")
            ax.set_title(name, fontsize="small")
            ax.grid(alpha=0.3)
        fig.suptitle(kind.replace("_", " "))
        fig.tight_layout()
        return fig, axes

    def plot_saliency(self, samples=None, call=-1, percentile=99.5,
                      fig=None, axes=None, figsize=None):
        """Input frames next to dLoss/d(input) on a symmetric diverging scale, one
        boxed group per sample (needs `capture(input_grad=True)` or `run()`). The
        scale is the `percentile` of |grad| rather than its max, so a few extreme
        pixels do not wash out the rest (use 100 for the plain max)."""
        x = self.input[call] if isinstance(self.input, list) else self.input
        g = self.input_grad[call] if isinstance(self.input_grad, list) else self.input_grad
        if x is None or g is None:
            raise KeyError("No input gradient stored; use capture(input_grad=True) + backward, or run().")
        x, g = _to_numpy(x), _to_numpy(g)
        if x.ndim != 4:
            raise ValueError(f"plot_saliency expects [B, C, H, W] inputs, got {x.shape}.")
        if samples is not None:
            x, g = x[samples], g[samples]
        a = float(np.percentile(np.abs(g), percentile)) or 1.0
        items = [
            {"tensor": x, "title": "input"},
            {"tensor": g, "title": "dL/d(input)", "cmap": "RdBu_r", "vmin": -a, "vmax": a},
        ]
        if figsize is None:
            w, h = _grid_figsize(*x.shape[:2], panel=1.6)
            figsize = (2 * w, h + 0.4)
        return imshow_multiple(items, fig=fig, axes=axes, figsize=figsize)

    def plot_jacobian(self, J, x=None, modes=None, sample=0, signed=True, percentile=99.5,
                      fig=None, axes=None, figsize=None):
        """One panel per mode showing `J[k, sample]` (from `jacobian()`), preceded by
        the input frame when `x` is given. Signed maps use a symmetric diverging
        scale per mode, set by the `percentile` of |J|; `signed=False` shows |J|."""
        J = _to_numpy(J)
        modes = list(range(J.shape[0])) if modes is None else list(modes)
        items = []
        if x is not None:
            items.append({"tensor": _to_numpy(x)[sample], "title": "input"})
        for k, mode in enumerate(modes):
            m = J[k, sample]
            if signed:
                a = float(np.percentile(np.abs(m), percentile)) or 1.0
                items.append({"tensor": m, "title": f"dz[{mode}]/dx", "cmap": "RdBu_r",
                              "vmin": -a, "vmax": a})
            else:
                items.append({"tensor": np.abs(m), "title": f"|dz[{mode}]/dx|"})
        return imshow_multiple(items, fig=fig, axes=axes, figsize=figsize)

    def plot_history(self, metric, layers=None, fig=None, axes=None, figsize=None):
        """One line per layer against `history_steps`; layers are shaded light
        (shallow) to dark (deep). Log scale for norm/ratio metrics."""
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}.")
        names = [self.layer(k)[0] for k in layers] if layers is not None else self._ordered_names()
        steps = np.asarray(self.history_steps)

        if fig is None or axes is None:
            fig, ax = plt.subplots(figsize=figsize or (8, 4))
            for name, color in zip(names, _depth_colors(len(names))):
                ax.plot(steps, self.history[metric][name], lw=2, color=color, label=name)
            if metric in _LOG_METRICS:
                ax.set_yscale("log")
            ax.set_xlabel("step")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.3)
            # Outside the axes, so it never hides the lines.
            ax.legend(fontsize="small", loc="center left", bbox_to_anchor=(1.01, 0.5),
                      ncol=max(1, math.ceil(len(names) / 12)))
            fig.tight_layout()
            return fig, [ax]

        ax = axes[0]
        for line, name in zip(ax.lines, names):
            line.set_data(steps, self.history[metric][name])
        ax.relim()
        ax.autoscale_view()
        return fig, axes

    # =====================================================================
    # Text
    # =====================================================================

    def summary(self):
        """Table of hooked layers: type, last output shape, #params and the latest
        recorded scalars (if any)."""
        cols = ("weight_grad_norm", "act_grad_rms", "update_ratio", "dead_frac")
        header = f"{'layer':<28}{'type':<10}{'output':<22}{'params':>10}" + "".join(f"{c:>18}" for c in cols)
        lines = [header, "-" * len(header)]
        for name in self._ordered_names():
            module = self.layers[name]
            nparams = sum(p.numel() for p in module.parameters(recurse=False))
            shape = str(self._output_shapes.get(name, "-"))
            row = f"{name:<28}{type(module).__name__:<10}{shape:<22}{nparams:>10}"
            for c in cols:
                values = self.history[c][name]
                row += f"{values[-1]:>18.3e}" if values else f"{'-':>18}"
            lines.append(row)
        return "\n".join(lines)

    @staticmethod
    def print_graph(fn, indent=0, visited=None):
        """Print the autograd graph below `fn` (e.g. `loss.grad_fn`), one node per
        line, each node once."""
        if visited is None:
            visited = set()
        if fn is None or fn in visited:
            return
        visited.add(fn)
        print(" " * indent + type(fn).__name__)
        for next_fn, _ in fn.next_functions:
            CNNDiagnostics.print_graph(next_fn, indent + 2, visited)
