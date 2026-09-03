"""Tests for AI4AO.Utils. Generic PyTorch/matplotlib helpers with no AO-domain
dependency -- runs entirely on CPU with the Agg backend set in conftest.py."""
import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn

from AI4AO.Utils import (
    imshow,
    imshow_multiple,
    get_activations,
    get_conv_kernel_gradients,
    get_conv_kernels,
    print_graph,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_imshow_shapes_wh_cwh_bcwh():
    fig, axes = imshow(torch.randn(8, 8))
    assert len(axes) == 1

    fig, axes = imshow(torch.randn(3, 8, 8))
    assert len(axes) == 3

    fig, axes = imshow(torch.randn(2, 3, 8, 8))
    assert len(axes) == 6


def test_imshow_rejects_bad_ndim():
    with pytest.raises(ValueError):
        imshow(torch.randn(2, 2, 2, 2, 2))


def test_imshow_update_mode_axis_count_mismatch():
    fig, axes = imshow(torch.randn(2, 3, 8, 8))
    with pytest.raises(ValueError):
        imshow(torch.randn(1, 1, 8, 8), fig=fig, axes=axes)


def test_imshow_multiple_shapes():
    tensors = [torch.randn(8, 8), torch.randn(2, 8, 8)]
    fig, axes = imshow_multiple(tensors)
    assert len(axes) == 2
    assert len(axes[0]) == 1
    assert len(axes[1]) == 2


class _ToyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv1 = nn.Conv2d(2, 2, 3, padding=1)
        self.fc = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        return self.fc(x.flatten(start_dim=1))


def test_get_activations_returns_one_entry_per_conv():
    model = _ToyConvNet()
    x = torch.randn(1, 1, 4, 4)

    activations = get_activations(model, x)

    assert set(activations.keys()) == {"conv0", "conv1"}
    assert activations["conv0"].shape == (1, 2, 4, 4)
    assert activations["conv1"].shape == (1, 2, 4, 4)


def test_get_conv_kernel_gradients_requires_backward():
    model = _ToyConvNet()

    with pytest.raises(RuntimeError):
        get_conv_kernel_gradients(model, 0)

    x = torch.randn(1, 1, 4, 4)
    model(x).sum().backward()

    grad = get_conv_kernel_gradients(model, 0)
    assert grad.shape == model.conv0.weight.shape

    with pytest.raises(ValueError):
        get_conv_kernel_gradients(model, 10)


def test_get_conv_kernels_shape_and_out_of_range():
    model = _ToyConvNet()
    kernel = get_conv_kernels(model, 1)
    assert kernel.shape == model.conv1.weight.shape

    with pytest.raises(ValueError):
        get_conv_kernels(model, 10)


def test_print_graph_handles_diamond_and_terminates(capsys):
    x = torch.tensor(2.0, requires_grad=True)
    shared = x * 2
    y = shared + shared  # diamond: shared appears twice in the autograd graph

    print_graph(y.grad_fn)
    out = capsys.readouterr().out
    assert "AddBackward" in out
