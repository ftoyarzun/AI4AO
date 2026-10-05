"""Tests for AI4AO.CNNDiagnostics. Generic PyTorch/matplotlib code with no AO-domain
dependency beyond one PhaseEstimators CNN used for realism -- runs on CPU with the Agg
backend set in conftest.py."""
import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn

from AI4AO.CNNDiagnostics import CNNDiagnostics, METRICS, _mosaic
from AI4AO.PhaseEstimators import SimpleNet


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class _ToyConvNet(nn.Module):
    def __init__(self, batchnorm=False):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 2, 3, padding=1)
        self.norm = nn.BatchNorm2d(2) if batchnorm else nn.Identity()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(2, 3, 3, padding=1)
        self.fc = nn.Linear(3 * 4 * 4, 5)

    def forward(self, x):
        return self.forward_with_intermediates(x)[-1]

    def forward_with_intermediates(self, x):
        a0 = self.conv0(x)
        a1 = self.conv1(self.act(self.norm(a0)))
        out = self.fc(a1.flatten(start_dim=1))
        return a0, a1, out


def _reference(model, x):
    """Intermediates and their grads for loss = out.pow(2).sum(), computed by hand."""
    x = x.detach().requires_grad_()
    a0, a1, out = model.forward_with_intermediates(x)
    loss = out.pow(2).sum()
    grads = torch.autograd.grad(loss, [x, a0, a1, out])
    return dict(x=x.detach(), a0=a0.detach(), a1=a1.detach(), out=out.detach(),
                gx=grads[0], ga0=grads[1], ga1=grads[2], gout=grads[3])


def _no_hooks_left(model):
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_defaults_to_conv_and_linear():
    diag = CNNDiagnostics(_ToyConvNet())
    assert list(diag.layers) == ["conv0", "conv1", "fc"]


def test_registry_include_exclude_and_extra_types():
    model = _ToyConvNet()
    assert list(CNNDiagnostics(model, include=r"^conv").layers) == ["conv0", "conv1"]
    assert list(CNNDiagnostics(model, exclude=r"fc").layers) == ["conv0", "conv1"]
    diag = CNNDiagnostics(model, layer_types=(nn.Conv2d, nn.ReLU))
    assert list(diag.layers) == ["conv0", "act", "conv1"]
    with pytest.raises(ValueError):
        CNNDiagnostics(model, include="nothing-matches")


def test_layer_lookup_by_name_and_index():
    model = _ToyConvNet()
    diag = CNNDiagnostics(model)
    assert diag.layer(1) == ("conv1", model.conv1)
    assert diag.layer("conv1") == diag.layer(1)
    assert diag.layer(-1)[0] == "fc"
    with pytest.raises(IndexError):
        diag.layer(10)
    with pytest.raises(KeyError):
        diag.layer("missing")


# ---------------------------------------------------------------------------
# capture()
# ---------------------------------------------------------------------------

def test_capture_matches_manual_forward_and_backward():
    model = _ToyConvNet()
    x = torch.randn(6, 1, 4, 4)
    ref = _reference(model, x)
    diag = CNNDiagnostics(model, max_samples=4)

    model.zero_grad()
    with diag.capture(input_grad=True):
        model(x).pow(2).sum().backward()

    assert diag.activations["conv0"].shape == (4, 2, 4, 4)
    torch.testing.assert_close(diag.activations["conv0"], ref["a0"][:4])
    torch.testing.assert_close(diag.activations["conv1"], ref["a1"][:4])
    torch.testing.assert_close(diag.activations["fc"], ref["out"][:4])
    torch.testing.assert_close(diag.activation_grads["conv0"], ref["ga0"][:4])
    torch.testing.assert_close(diag.activation_grads["conv1"], ref["ga1"][:4])
    torch.testing.assert_close(diag.activation_grads["fc"], ref["gout"][:4])
    torch.testing.assert_close(diag.input, ref["x"][:4])
    torch.testing.assert_close(diag.input_grad, ref["gx"][:4])
    torch.testing.assert_close(diag.param_grads["conv0"], model.conv0.weight.grad)
    torch.testing.assert_close(diag.param_grads["fc.bias"], model.fc.bias.grad)
    assert diag.forward_order == ["conv0", "conv1", "fc"]
    assert _no_hooks_left(model)


def test_capture_removes_hooks_when_block_raises():
    model = _ToyConvNet()
    diag = CNNDiagnostics(model)
    with pytest.raises(RuntimeError, match="boom"):
        with diag.capture(input_grad=True):
            model(torch.randn(2, 1, 4, 4))
            raise RuntimeError("boom")
    assert _no_hooks_left(model)
    # A later capture still works.
    with diag.capture():
        model(torch.randn(2, 1, 4, 4)).sum().backward()
    assert "conv0" in diag.activation_grads


def test_capture_cannot_nest():
    model = _ToyConvNet()
    diag = CNNDiagnostics(model)
    with pytest.raises(RuntimeError):
        with diag.capture():
            with diag.capture():
                pass


def test_closed_loop_calls_pair_forward_and_backward():
    model = _ToyConvNet()
    xs = [torch.randn(2, 1, 4, 4) for _ in range(3)]
    weights = [1.0, 2.0, 3.0]
    refs = [_reference(model, x) for x in xs]
    diag = CNNDiagnostics(model)

    with diag.capture(calls="all", input_grad=True):
        loss = sum(w * model(x).pow(2).sum() for w, x in zip(weights, xs))
        loss.backward()  # one backward for all three calls, as in Trainer.train

    assert len(diag.activations["conv1"]) == 3
    assert len(diag.activation_grads["conv1"]) == 3
    for i, (w, ref) in enumerate(zip(weights, refs)):
        torch.testing.assert_close(diag.activations["conv1"][i], ref["a1"])
        torch.testing.assert_close(diag.activation_grads["conv1"][i], w * ref["ga1"])
        torch.testing.assert_close(diag.input_grad[i], w * ref["gx"])

    with diag.capture(calls="last"):
        loss = sum(w * model(x).pow(2).sum() for w, x in zip(weights, xs))
        loss.backward()
    torch.testing.assert_close(diag.activations["conv1"], refs[-1]["a1"])
    torch.testing.assert_close(diag.activation_grads["conv1"], weights[-1] * refs[-1]["ga1"])


def test_input_grad_with_upstream_graph_keeps_flowing():
    model = _ToyConvNet()
    leaf = torch.randn(2, 1, 4, 4, requires_grad=True)
    diag = CNNDiagnostics(model)

    with diag.capture(input_grad=True):
        model(2.0 * leaf).pow(2).sum().backward()

    assert leaf.grad is not None
    torch.testing.assert_close(leaf.grad, 2.0 * diag.input_grad)


def test_capture_warns_on_no_grad_forward_and_missing_backward():
    model = _ToyConvNet()
    x = torch.randn(2, 1, 4, 4)
    diag = CNNDiagnostics(model)

    with pytest.warns(UserWarning, match="no_grad"):
        with diag.capture():
            with torch.no_grad():
                model(x)
    assert "conv0" in diag.activations  # forward-only data is still stored

    with pytest.warns(UserWarning, match="backward"):
        with diag.capture():
            model(x)


def test_capture_is_purely_observational():
    model_a = _ToyConvNet()
    model_b = copy.deepcopy(model_a)
    x = torch.randn(3, 1, 4, 4)

    loss_a = model_a(x).pow(2).sum()
    loss_a.backward()

    diag = CNNDiagnostics(model_b)
    with diag.capture(input_grad=True, calls="all"):
        loss_b = model_b(x).pow(2).sum()
        loss_b.backward()

    assert torch.equal(loss_a, loss_b)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa.grad, pb.grad)


# ---------------------------------------------------------------------------
# run() and jacobian()
# ---------------------------------------------------------------------------

def test_run_matches_reference_and_leaves_grad_untouched():
    model = _ToyConvNet()
    x = torch.randn(3, 1, 4, 4)
    ref = _reference(model, x)
    preset = torch.full_like(model.conv0.weight, 7.0)
    model.conv0.weight.grad = preset.clone()

    diag = CNNDiagnostics(model)
    out = diag.run(x)

    torch.testing.assert_close(out, ref["out"])
    assert torch.equal(model.conv0.weight.grad, preset)
    assert model.conv1.weight.grad is None
    torch.testing.assert_close(diag.input_grad, ref["gx"])
    torch.testing.assert_close(diag.activation_grads["conv1"], ref["ga1"])
    loss_grad = torch.autograd.grad(model(x).pow(2).sum(), model.conv1.weight)[0]
    torch.testing.assert_close(diag.param_grads["conv1"], loss_grad)
    assert _no_hooks_left(model)


def test_jacobian_matches_autograd_functional():
    model = _ToyConvNet().eval()
    x = torch.randn(2, 1, 4, 4)
    modes = [0, 3]

    J = CNNDiagnostics(model).jacobian(x, modes)
    full = torch.autograd.functional.jacobian(model, x)  # [B, 5, B, 1, 4, 4]

    assert J.shape == (2, 2, 1, 4, 4)
    for i, k in enumerate(modes):
        for b in range(2):
            torch.testing.assert_close(J[i, b], full[b, k, b])


def test_jacobian_uses_eval_mode_and_restores_training_flag():
    model = _ToyConvNet(batchnorm=True)
    model.train()
    for _ in range(3):  # give BatchNorm non-trivial running statistics
        model(torch.randn(8, 1, 4, 4))
    x = torch.randn(3, 1, 4, 4)
    diag = CNNDiagnostics(model)

    J = diag.jacobian(x, [1])
    assert model.training
    # Eval mode -> samples are independent: batch Jacobian == per-sample Jacobians.
    for b in range(3):
        torch.testing.assert_close(J[0, b], diag.jacobian(x[b:b + 1], [1])[0, 0])

    with pytest.raises(IndexError):
        diag.jacobian(x, [99])
    assert model.training


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def test_record_requires_a_capture():
    with pytest.raises(RuntimeError):
        CNNDiagnostics(_ToyConvNet()).record(0)


def test_record_history_and_update_ratio():
    model = _ToyConvNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(4, 1, 4, 4)
    diag = CNNDiagnostics(model)

    with diag.capture():
        optimizer.zero_grad()
        model(x).pow(2).sum().backward()
    diag.record(0)
    assert diag.history["update_ratio"]["conv0"][-1] == 0.0

    with diag.capture():
        optimizer.zero_grad()
        model(x).pow(2).sum().backward()
        optimizer.step()
    diag.record(1)

    assert diag.history_steps == [0, 1]
    for metric in METRICS:
        assert len(diag.history[metric]["conv1"]) == 2
    assert diag.history["update_ratio"]["conv0"][-1] > 0
    assert diag.history["weight_grad_norm"]["fc"][-1] == pytest.approx(
        float(model.fc.weight.grad.norm()))
    assert 0.0 <= diag.history["dead_frac"]["conv0"][-1] <= 1.0

    diag.reset_history()
    assert diag.history_steps == []


def test_dead_frac_counts_channels_that_never_fire():
    model = _ToyConvNet()
    with torch.no_grad():
        model.conv0.weight.zero_()
        model.conv0.bias.copy_(torch.tensor([-1.0, 1.0]))  # channel 0 dead, 1 alive
    diag = CNNDiagnostics(model)
    diag.run(torch.randn(4, 1, 4, 4))
    diag.record(0)
    assert diag.history["dead_frac"]["conv0"][-1] == pytest.approx(0.5)


def test_dead_frac_is_nan_for_the_model_output_layer():
    model = _ToyConvNet()
    diag = CNNDiagnostics(model)
    with diag.capture():
        model(torch.randn(4, 1, 4, 4)).sum().backward()
    diag.record(0)
    assert math.isnan(diag.history["dead_frac"]["fc"][-1])
    assert not math.isnan(diag.history["dead_frac"]["conv1"][-1])


def test_mosaic_blocks_are_near_square():
    arr = np.arange(2 * 5 * 3 * 3, dtype=np.float32).reshape(2, 5, 3, 3)
    out = _mosaic(arr, gap=1, block_gap=4)
    # 5 tiles -> 3 x 2 grid per block (3*4-1=11 wide, 2*4-1=7 high); 2 blocks -> side by side.
    assert out.shape == (7, 11 + 4 + 11)
    np.testing.assert_array_equal(out[0:3, 0:3], arr[0, 0])
    np.testing.assert_array_equal(out[4:7, 4:7], arr[0, 4])
    np.testing.assert_array_equal(out[0:3, 15:18], arr[1, 0])
    assert np.isnan(out[3, 0]) and np.isnan(out[0, 11])


# ---------------------------------------------------------------------------
# Plots and text
# ---------------------------------------------------------------------------

@pytest.fixture
def captured():
    model = _ToyConvNet()
    diag = CNNDiagnostics(model, max_samples=2)
    x = torch.randn(3, 1, 4, 4)
    with diag.capture(input_grad=True):
        model(x).pow(2).sum().backward()
    diag.record(0)
    with diag.capture(input_grad=True):
        model(x).pow(2).sum().backward()
    diag.record(1)
    return diag, x


@pytest.mark.parametrize("call", [
    lambda d, x: d.plot_feature_maps("conv0"),
    lambda d, x: d.plot_feature_maps("conv1", mosaic=True),
    lambda d, x: d.plot_feature_maps("fc"),
    lambda d, x: d.plot_kernels(1),
    lambda d, x: d.plot_kernels(1, grad=True, mosaic=True),
    lambda d, x: d.plot_gradient_flow(),
    lambda d, x: d.plot_gradient_flow(source="activation"),
    lambda d, x: d.plot_histograms("activation"),
    lambda d, x: d.plot_histograms("weight_grad"),
    lambda d, x: d.plot_saliency(),
    lambda d, x: d.plot_jacobian(d.jacobian(x, [0, 2]), x, [0, 2]),
    lambda d, x: d.plot_history("weight_grad_norm"),
    lambda d, x: d.plot_history("act_mean", layers=[0, "fc"]),
])
def test_plots_create_and_update(captured, call):
    diag, x = captured

    fig, axes = call(diag, x)
    assert fig is not None and len(axes) > 0

    # Update mode: same call, handing the figure back; no new figure is created.
    n_figures = len(plt.get_fignums())
    fig2, _ = _call_with_fig(call, diag, x, fig, axes)
    assert fig2 is fig
    assert len(plt.get_fignums()) == n_figures


def _call_with_fig(call, diag, x, fig, axes):
    """Re-run a plotting lambda with fig/axes injected into the plot_* call."""
    class _Proxy:
        def __getattr__(self, attr):
            target = getattr(diag, attr)
            if attr.startswith("plot_"):
                return lambda *a, **k: target(*a, fig=fig, axes=axes, **k)
            return target
    return call(_Proxy(), x)


def test_plot_kernels_rejects_linear(captured):
    diag, _ = captured
    with pytest.raises(ValueError):
        diag.plot_kernels("fc")


def test_plot_kernels_uses_symmetric_scale(captured):
    diag, _ = captured
    _, axes = diag.plot_kernels("conv0")
    vmin, vmax = axes[0].images[0].get_clim()
    assert vmin == pytest.approx(-vmax)


def test_summary_lists_layers(captured):
    diag, _ = captured
    text = diag.summary()
    for name in ("conv0", "conv1", "fc"):
        assert name in text


def test_print_graph_handles_diamond_and_terminates(capsys):
    x = torch.tensor(2.0, requires_grad=True)
    shared = x * 2
    y = shared + shared  # diamond: shared appears twice in the autograd graph

    CNNDiagnostics.print_graph(y.grad_fn)
    out = capsys.readouterr().out
    assert "AddBackward" in out
    assert out.count("MulBackward") == 1


# ---------------------------------------------------------------------------
# A real reconstructor
# ---------------------------------------------------------------------------

def test_simplenet_end_to_end():
    model = SimpleNet({"Nmodes": 6}, None, "cpu")
    x = torch.randn(2, 1, 64, 64)
    diag = CNNDiagnostics(model)

    with diag.capture(input_grad=True):
        model(x).pow(2).sum().backward()
    diag.record(0)

    assert len(diag.layers) == 7  # six convs + output Linear
    assert set(diag.activation_grads) == set(diag.layers)
    assert diag.forward_order[-1] == "outputlayer.0"
    diag.plot_gradient_flow()
    diag.plot_saliency()
