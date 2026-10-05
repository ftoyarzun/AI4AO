"""Tests for PhaseEstimators.StemEncoderCNN (BestCNN by default; switches from
Ideas/18-SearchingForABetterCNN.md) and pyramid_combinations."""
import pytest
import torch
import torch.nn as nn

from AI4AO.paths import get_data_dir
from AI4AO.PhaseEstimators import StemEncoderCNN, pyramid_combinations

# CNN-comparison instrument: 4 Pyramid pupils of 42x42, 190 modes.
N_CHANNELS, NMODES, NOUT = 4, 190, 42
PUPIL_CENTERS = [[10.0, 10.0], [10.0, 30.0], [30.0, 10.0], [30.0, 30.0]]  # [y, x], a 2x2 grid
COMPACT = dict(pupil_centers=PUPIL_CENTERS, cross_pupil=True, ceil_pool=True, spatial_head=True,
               head_channels=32, max_channels=128)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _NotebookStemEncoderCNN(nn.Module):
    """Verbatim copy of BestCNN.ipynb's StemEncoderCNN: the layout the default
    configuration must keep so existing checkpoints load."""

    def __init__(self, n_channels, Nmodes, Nout, base_channels=7, activation=nn.ELU):
        super().__init__()

        def num_downsamples(size, min_size=2):
            n = 0
            while size // 2 >= min_size:
                size //= 2
                n += 1
            return n

        def conv_stage(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), activation(), nn.Conv2d(o, o, 3, padding=1), activation(),
                    nn.MaxPool2d(2)]

        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, base_channels * n_channels, kernel_size=11, padding=5, groups=n_channels),
            activation(),
            nn.Conv2d(base_channels * n_channels, 2 * base_channels * n_channels, kernel_size=7, padding=3,
                      groups=n_channels),
            activation(),
            nn.MaxPool2d(2),
        )
        c0 = 2 * base_channels * n_channels
        n_stages = num_downsamples(Nout // 2)
        channels = [c0] + [c0 * 2 ** i for i in range(n_stages)]
        layers = []
        for i in range(n_stages):
            layers += conv_stage(channels[i], channels[i + 1])
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels[-1], Nmodes))

    def forward(self, x):
        return self.head(self.encoder(self.stem(x)))


@pytest.mark.parametrize("config", [{}, COMPACT], ids=["default", "compact"])
def test_output_shape(device, config):
    model = StemEncoderCNN(N_CHANNELS, NMODES, NOUT, **config).to(device)
    y = model(torch.randn(3, N_CHANNELS, NOUT, NOUT, device=device))
    assert y.shape == (3, NMODES)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("config", [{}, COMPACT], ids=["default", "compact"])
def test_gradient_reaches_every_parameter(device, config):
    model = StemEncoderCNN(N_CHANNELS, NMODES, NOUT, **config).to(device)
    model(torch.randn(2, N_CHANNELS, NOUT, NOUT, device=device)).pow(2).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.any(p.grad != 0)]
    assert not missing, f"no gradient for {missing}"


@pytest.mark.parametrize("config, expected", [({}, 2_791_102), (COMPACT, 1_005_170)], ids=["default", "compact"])
def test_parameter_count(config, expected):
    assert n_params(StemEncoderCNN(N_CHANNELS, NMODES, NOUT, **config)) == expected


@pytest.mark.parametrize("config", [{}, COMPACT], ids=["default", "compact"])
def test_train_and_eval_mode_agree(device, config):
    # No dropout or normalization layers: train() and eval() must give the same output.
    model = StemEncoderCNN(N_CHANNELS, NMODES, NOUT, **config).to(device)
    x = torch.randn(2, N_CHANNELS, NOUT, NOUT, device=device)
    model.train()
    y_train = model(x)
    model.eval()
    with torch.no_grad():
        y_eval = model(x)
    torch.testing.assert_close(y_train.detach(), y_eval)


def test_default_matches_notebook_bestcnn_layout():
    ref = _NotebookStemEncoderCNN(N_CHANNELS, NMODES, NOUT, base_channels=12)
    model = StemEncoderCNN(N_CHANNELS, NMODES, NOUT)
    assert {k: v.shape for k, v in model.state_dict().items()} == {k: v.shape for k, v in ref.state_dict().items()}
    model.load_state_dict(ref.state_dict())
    x = torch.randn(2, N_CHANNELS, NOUT, NOUT)
    torch.testing.assert_close(model(x), ref(x))


@pytest.mark.parametrize("path, config", [
    ("BestCNN/BestCNN.pth", {}),
    ("SearchingForABetterCNN/SEV2_cap128_lr3c_seed0.pth", COMPACT),
])
def test_checkpoints_load(path, config):
    full = get_data_dir() / path
    if not full.exists():
        pytest.skip(f"{full} not present")
    state = torch.load(full, map_location="cpu")["phase_reconstructor_state_dict"]
    StemEncoderCNN(N_CHANNELS, NMODES, NOUT, **config).load_state_dict(state)  # strict


@pytest.mark.parametrize("nout", [41, 46])
def test_ceil_pool_and_spatial_head_handle_other_sizes(nout):
    model = StemEncoderCNN(N_CHANNELS, NMODES, nout, base_channels=2, **dict(COMPACT, max_channels=16))
    assert model(torch.randn(1, N_CHANNELS, nout, nout)).shape == (1, NMODES)


def test_cross_pupil_front_initialized_to_pyramid_combinations():
    model = StemEncoderCNN(N_CHANNELS, NMODES, NOUT, pupil_centers=PUPIL_CENTERS, cross_pupil=True)
    W = model.front.weight[:, :, 0, 0]
    torch.testing.assert_close(W, pyramid_combinations(PUPIL_CENTERS))
    torch.testing.assert_close(W @ W.T, torch.eye(4))  # orthonormal: no information lost
    assert torch.all(model.front.bias == 0)


def test_cross_pupil_needs_pupil_centers_and_four_pupils():
    with pytest.raises(ValueError, match="pupil_centers"):
        StemEncoderCNN(N_CHANNELS, NMODES, NOUT, cross_pupil=True)
    with pytest.raises(ValueError, match="4 Pyramid pupils"):
        StemEncoderCNN(2, NMODES, NOUT, pupil_centers=PUPIL_CENTERS, cross_pupil=True)


def test_pyramid_combinations_rejects_non_grid_layout():
    with pytest.raises(ValueError, match="2x2 grid"):
        # Two pupils in the same (top-left) quadrant.
        pyramid_combinations([[10.0, 10.0], [10.0, 30.0], [30.0, 10.0], [12.0, 12.0]])
