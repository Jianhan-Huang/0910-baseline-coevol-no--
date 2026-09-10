import numpy as np
import torch
import torch.nn as nn

from coevol_no.wrapper import OperatorNet
from tasks.pde_benchmarks.temporal_data import (
    _tokens,
    TemporalWindowDataset,
    compute_stats,
    denormalize_fields,
    normalize_fields,
    normalize_parameter,
    normalize_static,
)
from tasks.pde_benchmarks.temporal_eval import (
    add_state_noise,
    compute_metrics,
    shift_history_tokens,
)


def test_normalization_matches_common_0821_formula():
    rng = np.random.default_rng(0)
    fields = rng.normal(size=(4, 7, 2, 3, 3)).astype(np.float32)
    para = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    static = rng.normal(size=(4, 1, 3, 3)).astype(np.float32)

    stats = compute_stats(fields, para, static)

    expected_field_min = fields.min(axis=(0, 1, 3, 4))
    expected_field_max = fields.max(axis=(0, 1, 3, 4))
    expected_static_min = static.min(axis=(0, 2, 3))
    expected_static_max = static.max(axis=(0, 2, 3))
    np.testing.assert_allclose(stats.field_min, expected_field_min)
    np.testing.assert_allclose(stats.field_max, expected_field_max)
    assert stats.parameter_min == float(para.min())
    assert stats.parameter_max == float(para.max())
    np.testing.assert_allclose(stats.static_min, expected_static_min)
    np.testing.assert_allclose(stats.static_max, expected_static_max)

    expected_norm = -1.0 + 2.0 * (
        fields - expected_field_min.reshape(1, 1, -1, 1, 1)
    ) / np.maximum(
        expected_field_max.reshape(1, 1, -1, 1, 1)
        - expected_field_min.reshape(1, 1, -1, 1, 1),
        1e-8,
    )
    np.testing.assert_allclose(normalize_fields(fields, stats), expected_norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(denormalize_fields(expected_norm, stats), fields, rtol=1e-6, atol=1e-6)

    expected_static = -1.0 + 2.0 * (
        static - expected_static_min.reshape(1, -1, 1, 1)
    ) / np.maximum(
        expected_static_max.reshape(1, -1, 1, 1)
        - expected_static_min.reshape(1, -1, 1, 1),
        1e-8,
    )
    np.testing.assert_allclose(normalize_static(static, stats), expected_static, rtol=1e-6, atol=1e-6)

    expected_para = -1.0 + 2.0 * (para - para.min()) / (para.max() - para.min())
    np.testing.assert_allclose(normalize_parameter(para, stats), expected_para, rtol=1e-6, atol=1e-6)


def test_condition_token_layout_is_frame_major_channel_concat():
    fields = np.zeros((1, 3, 2, 2, 2), dtype=np.float32)
    fields[:, :, 0] = 1.0
    fields[:, :, 1] = 2.0
    para = np.array([5.0], dtype=np.float32)
    # Use a second training trajectory value through custom stats so parameter
    # normalization does not collapse to zero.
    stats = compute_stats(np.concatenate([fields, fields + 2], axis=0), np.array([5.0, 7.0], dtype=np.float32))
    tokens = _tokens(fields, para, stats, 3, include_parameter=True)
    frames = tokens.reshape(1, 4, 3, 3)
    # Every temporal block is [state_ch0, state_ch1, scalar_parameter].
    assert frames.shape[-1] == 3
    np.testing.assert_allclose(frames[..., 2], -1.0)



def test_training_window_samples_from_full_trajectory(monkeypatch):
    fields = np.zeros((1, 6, 1, 2, 2), dtype=np.float32)
    for t in range(fields.shape[1]):
        fields[:, t] = float(t)
    parameters = np.array([1.0], dtype=np.float32)
    stats = compute_stats(fields, parameters)
    dataset = TemporalWindowDataset(
        fields,
        parameters,
        stats,
        T_in=2,
        T_out=2,
        include_parameter=False,
    )

    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return 2

    monkeypatch.setattr(np.random, "randint", fake_randint)
    pos, tokens, targets = dataset[0]
    assert calls == [(0, 3)]
    assert pos.shape == (4, 2)

    expected_history = normalize_fields(fields[:, 2:4], stats)[0]
    expected_tokens = np.transpose(expected_history, (2, 3, 0, 1)).reshape(4, 2)
    expected_targets = normalize_fields(fields[:, 4:6], stats)[0]
    expected_targets = np.transpose(expected_targets, (2, 3, 0, 1)).reshape(4, 2, 1)
    np.testing.assert_allclose(tokens.numpy(), expected_tokens, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(targets.numpy(), expected_targets, rtol=1e-6, atol=1e-6)


def test_history_shift_replaces_one_complete_frame_block():
    # [B=1,P=1,T=3, frame=(state0,state1,cond)]
    frames = torch.tensor([[[[1., 2., 9.], [3., 4., 9.], [5., 6., 9.]]]])
    tokens = frames.reshape(1, 1, -1)
    next_state = torch.tensor([[[7., 8.]]])
    shifted = shift_history_tokens(tokens, next_state, T_in=3, frame_channels=3, dynamic_channels=2)
    expected = torch.tensor([[[[3., 4., 9.], [5., 6., 9.], [7., 8., 9.]]]])
    torch.testing.assert_close(shifted.reshape_as(expected), expected)


def test_state_noise_never_changes_condition_channels():
    torch.manual_seed(0)
    frames = torch.ones(2, 4, 3, 3)
    frames[..., 2] = 17.0
    tokens = frames.reshape(2, 4, -1)
    noisy = add_state_noise(tokens, 0.1, T_in=3, frame_channels=3, dynamic_channels=2)
    noisy_frames = noisy.reshape_as(frames)
    torch.testing.assert_close(noisy_frames[..., 2], frames[..., 2])
    assert not torch.equal(noisy_frames[..., :2], frames[..., :2])


def test_metrics_match_common_formula_over_complete_future():
    target = np.arange(1, 1 + 2 * 5 * 1 * 2 * 2, dtype=np.float32).reshape(2, 5, 1, 2, 2)
    prediction = target.copy()
    prediction[:, 2:] += 1.5
    got = compute_metrics(prediction, target, input_steps=2)
    error = prediction[:, 2:] - target[:, 2:]
    expected = {
        'mse': float(np.mean(np.square(error), dtype=np.float64)),
        'mae': float(np.mean(np.abs(error), dtype=np.float64)),
        'relative_l2': float(np.sqrt(np.sum(error ** 2, dtype=np.float64) / np.sum(target[:, 2:] ** 2, dtype=np.float64))),
    }
    for key in expected:
        assert np.isclose(got[key], expected[key])


class _DummyBranch(nn.Module):
    def __init__(self, num_basis=4):
        super().__init__()
        self.num_basis = num_basis

    def forward(self, u, x, t=None):
        return torch.zeros(u.shape[0], u.shape[1], self.num_basis, dtype=u.dtype, device=u.device)


def test_operator_net_supports_multichannel_output_without_changing_scalar_shape():
    x = torch.zeros(2, 9, 3)
    scalar = OperatorNet(_DummyBranch(), num_basis=4, out_channels=1)
    vector = OperatorNet(_DummyBranch(), num_basis=4, out_channels=2)
    assert scalar(x, None).shape == (2, 9)
    assert vector(x, None).shape == (2, 9, 2)
