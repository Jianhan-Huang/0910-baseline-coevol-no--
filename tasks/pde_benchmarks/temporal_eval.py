"""Complete-trajectory evaluation for temporal CoEvol-NO benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .temporal_data import denormalize_fields


def compute_metrics(prediction: np.ndarray, target: np.ndarray, input_steps: int = 10) -> dict[str, float]:
    """Match common_0821.eval.metrics exactly on future physical frames."""
    error = prediction[:, input_steps:] - target[:, input_steps:]
    mse = float(np.mean(np.square(error), dtype=np.float64))
    mae = float(np.mean(np.abs(error), dtype=np.float64))
    numerator = np.sum(np.square(error), dtype=np.float64)
    denominator = np.sum(np.square(target[:, input_steps:]), dtype=np.float64)
    relative_l2 = float(np.sqrt(numerator / max(denominator, 1e-30)))
    return {"mse": mse, "mae": mae, "relative_l2": relative_l2}


def _as_output_frame(output: torch.Tensor, dynamic_channels: int) -> torch.Tensor:
    if dynamic_channels == 1:
        if output.ndim == 2:
            return output.unsqueeze(-1)
        if output.ndim == 3 and output.shape[-1] == 1:
            return output
    elif output.ndim == 3 and output.shape[-1] == dynamic_channels:
        return output
    raise ValueError(f"Unexpected model output shape {tuple(output.shape)} for {dynamic_channels} channels")


def shift_history_tokens(
    tokens: torch.Tensor,
    next_state: torch.Tensor,
    *,
    T_in: int,
    frame_channels: int,
    dynamic_channels: int,
) -> torch.Tensor:
    """Shift one complete [state, condition] frame block through the history."""
    b, p, _ = tokens.shape
    frames = tokens.reshape(b, p, T_in, frame_channels)
    conditions = frames[:, :, -1, dynamic_channels:]
    next_block = torch.cat([next_state, conditions], dim=-1)
    frames = torch.cat([frames[:, :, 1:], next_block.unsqueeze(2)], dim=2)
    return frames.reshape(b, p, T_in * frame_channels)


def add_state_noise(
    tokens: torch.Tensor,
    noise_std: float,
    *,
    T_in: int,
    frame_channels: int,
    dynamic_channels: int,
) -> torch.Tensor:
    """Apply the official RMS-scaled NS input noise to state channels only."""
    if noise_std <= 0:
        return tokens
    b, p, _ = tokens.shape
    frames = tokens.reshape(b, p, T_in, frame_channels)
    state = frames[..., :dynamic_channels]
    scale = float(state[0].numel()) ** 0.5
    dims = tuple(range(1, state.ndim))
    norm = torch.sqrt(torch.sum(state * state, dim=dims, keepdim=True))
    noisy_state = state + float(noise_std) * (norm / scale) * torch.randn_like(state)
    return torch.cat([noisy_state, frames[..., dynamic_channels:]], dim=-1).reshape_as(tokens)


def rollout_complete(
    model: torch.nn.Module,
    initial_tokens: torch.Tensor,
    *,
    steps: int,
    T_in: int,
    frame_channels: int,
    dynamic_channels: int,
    pos: torch.Tensor,
) -> torch.Tensor:
    predictions = []
    tokens = initial_tokens
    for _ in range(steps):
        output = _as_output_frame(model(tokens, pos), dynamic_channels)
        predictions.append(output)
        tokens = shift_history_tokens(
            tokens, output, T_in=T_in, frame_channels=frame_channels,
            dynamic_channels=dynamic_channels,
        )
    return torch.stack(predictions, dim=1)  # [B,T_future,P,C]


def evaluate_complete_trajectories(model: torch.nn.Module, data_info: dict, cfg: dict, device: torch.device) -> dict[str, dict[str, float]]:
    """Roll each configured test split through its final frame and save metrics."""
    model.eval()
    result_root = Path(cfg.get("result_dir", "./results")) / data_info["dataset"]
    result_root.mkdir(parents=True, exist_ok=True)
    batch_size = int(cfg.get("test_batch_size", 1))
    T_in = int(data_info["T_in"])
    frame_channels = int(data_info["frame_channels"])
    dynamic_channels = int(data_info["dynamic_channels"])
    stats = data_info["stats"]
    pos_base = data_info["pos"]
    all_metrics: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for split_name, split in data_info["test_splits"].items():
            x = split["x"]
            full_norm = split["full_normalized"]
            full_phys = split["full_physical"]
            parameters = split["parameters"]
            predictions = []
            for start in range(0, x.shape[0], batch_size):
                tokens = x[start:start + batch_size].to(device)
                bsz = tokens.shape[0]
                pos = pos_base.repeat(bsz, 1, 1).to(device)
                future_steps = int(full_norm.shape[1] - T_in)
                pred = rollout_complete(
                    model, tokens, steps=future_steps, T_in=T_in,
                    frame_channels=frame_channels, dynamic_channels=dynamic_channels,
                    pos=pos,
                )
                predictions.append(pred.cpu().numpy())

            future = np.concatenate(predictions, axis=0)  # [N,Tf,P,C]
            n, tf, p, c = future.shape
            h = w = int(round(p ** 0.5))
            future = future.reshape(n, tf, h, w, c).transpose(0, 1, 4, 2, 3)
            initial = full_norm[:, :T_in]
            pred_norm = np.concatenate([initial, future], axis=1)
            pred_phys = denormalize_fields(pred_norm, stats)
            metrics = compute_metrics(pred_phys, full_phys, input_steps=T_in)
            all_metrics[split_name] = metrics

            split_dir = result_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            with (split_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            np.savez_compressed(
                split_dir / "predictions.npz",
                prediction=pred_phys,
                target=full_phys,
                para=parameters,
            )

    with (result_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)
    return all_metrics
