"""Dataset adapters for the temporal PDE benchmarks used with CoEvol-NO.

The loaders in this module follow the benchmark split and min-max preprocessing
conventions from common_0821, but expose tensors in the native CoEvol-NO PDE
runner format.  Training uses the official Navier--Stokes protocol: the first
``T_in`` frames form the history and the next ``T_out`` frames are the
teacher-forced targets.  Complete trajectories are retained separately for
final autoregressive evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict

import numpy as np
import torch


TEMPORAL_TASKS = ("burger", "rd", "ns-a", "ns-b-t50", "heat", "kf-kol")


@dataclass(frozen=True)
class TemporalStats:
    field_min: np.ndarray
    field_max: np.ndarray
    parameter_min: float = 0.0
    parameter_max: float = 1.0
    static_min: np.ndarray | None = None
    static_max: np.ndarray | None = None


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError("h5py is required for NS/heat temporal datasets.") from exc
    return h5py


def _as_fields(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim != 5:
        raise ValueError(f"Expected [N,T,C,H,W], got {data.shape}.")
    return data.astype(np.float32)


def _from_last_time_axis(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim != 4:
        raise ValueError(f"Expected [N,H,W,T], got {data.shape}.")
    return np.transpose(data, (0, 3, 1, 2))[:, :, None].astype(np.float32)


def _npy(path: Path) -> np.ndarray:
    data = np.load(path)
    if data.ndim != 4:
        raise ValueError(f"Expected [N,T,H,W], got {data.shape}.")
    return data[:, :, None].astype(np.float32)


def compute_stats(
    fields: np.ndarray,
    parameters: np.ndarray | None,
    static: np.ndarray | None = None,
) -> TemporalStats:
    """Compute per-channel min/max values using training trajectories only."""

    field_min = fields.min(axis=(0, 1, 3, 4)).astype(np.float32)
    field_max = fields.max(axis=(0, 1, 3, 4)).astype(np.float32)

    if parameters is None:
        parameter_min, parameter_max = 0.0, 1.0
    else:
        parameters = np.asarray(parameters, dtype=np.float32).reshape(-1)
        parameter_min = float(parameters.min())
        parameter_max = float(parameters.max())

    static_min = static_max = None
    if static is not None:
        static = np.asarray(static, dtype=np.float32)
        static_min = static.min(axis=(0, 2, 3)).astype(np.float32)
        static_max = static.max(axis=(0, 2, 3)).astype(np.float32)

    return TemporalStats(
        field_min=field_min,
        field_max=field_max,
        parameter_min=parameter_min,
        parameter_max=parameter_max,
        static_min=static_min,
        static_max=static_max,
    )


def normalize_fields(values: np.ndarray, stats: TemporalStats) -> np.ndarray:
    minimum = stats.field_min.reshape(1, 1, -1, 1, 1)
    maximum = stats.field_max.reshape(1, 1, -1, 1, 1)
    return (-1.0 + 2.0 * (values - minimum) / np.maximum(maximum - minimum, 1e-8)).astype(np.float32)


def denormalize_fields(values: np.ndarray, stats: TemporalStats) -> np.ndarray:
    minimum = stats.field_min.reshape(1, 1, -1, 1, 1)
    maximum = stats.field_max.reshape(1, 1, -1, 1, 1)
    return ((values + 1.0) * 0.5 * (maximum - minimum) + minimum).astype(np.float32)


def normalize_parameter(values: np.ndarray, stats: TemporalStats) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    scale = stats.parameter_max - stats.parameter_min
    if abs(scale) < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (-1.0 + 2.0 * (values - stats.parameter_min) / scale).astype(np.float32)


def normalize_static(values: np.ndarray, stats: TemporalStats) -> np.ndarray:
    if stats.static_min is None or stats.static_max is None:
        raise ValueError("Static normalization requested without static statistics.")
    minimum = stats.static_min.reshape(1, -1, 1, 1)
    maximum = stats.static_max.reshape(1, -1, 1, 1)
    return (-1.0 + 2.0 * (values - minimum) / np.maximum(maximum - minimum, 1e-8)).astype(np.float32)


def _grid(resolution: int) -> torch.Tensor:
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    x, y = np.meshgrid(x, y)
    return torch.tensor(np.c_[x.ravel(), y.ravel()], dtype=torch.float32).unsqueeze(0)


def _tokens(
    fields: np.ndarray,
    parameters: np.ndarray,
    stats: TemporalStats,
    T_in: int,
    *,
    include_parameter: bool,
    static: np.ndarray | None = None,
) -> np.ndarray:
    """Build [N,H*W,T_in*(state+condition)] native CoEvol-NO tokens."""

    history = normalize_fields(fields[:, :T_in], stats)
    n, _, _, h, w = history.shape
    frame_parts = [history]

    if static is not None:
        static_norm = normalize_static(static, stats)
        static_frames = np.broadcast_to(static_norm[:, None], (n, T_in, *static_norm.shape[1:]))
        frame_parts.append(static_frames)

    if include_parameter:
        p = normalize_parameter(parameters, stats)
        p_frames = np.broadcast_to(p[:, None, None, None, None], (n, T_in, 1, h, w))
        frame_parts.append(p_frames)

    frames = np.concatenate(frame_parts, axis=2)  # [N,T,C_frame,H,W]
    frames = np.transpose(frames, (0, 3, 4, 1, 2))  # [N,H,W,T,C_frame]
    return frames.reshape(n, h * w, -1).astype(np.float32)


def _make_split(
    fields: np.ndarray,
    parameters: np.ndarray,
    stats: TemporalStats,
    T_in: int,
    T_out: int,
    *,
    include_parameter: bool,
    static: np.ndarray | None,
) -> dict[str, torch.Tensor | np.ndarray]:
    if fields.shape[1] < T_in + T_out:
        raise ValueError(f"Trajectory length {fields.shape[1]} is shorter than T_in+T_out={T_in + T_out}.")
    x = _tokens(fields, parameters, stats, T_in, include_parameter=include_parameter, static=static)
    y = normalize_fields(fields[:, T_in:T_in + T_out], stats)
    y = np.transpose(y, (0, 3, 4, 1, 2)).reshape(fields.shape[0], -1, T_out, fields.shape[2])
    return {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y.astype(np.float32)),
        "full_normalized": normalize_fields(fields, stats),
        "full_physical": fields.astype(np.float32),
        "parameters": parameters.astype(np.float32),
    }


def _finalize(
    *,
    task: str,
    train_fields: np.ndarray,
    train_parameters: np.ndarray,
    test_splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    T_in: int,
    T_out: int,
    include_parameter: bool,
    train_static: np.ndarray | None = None,
) -> dict:
    train_fields = _as_fields(train_fields)
    if train_fields.shape[-1] != train_fields.shape[-2]:
        raise ValueError("CoEvol-NO temporal adapter currently expects square spatial grids.")
    stats = compute_stats(train_fields, train_parameters, train_static)
    train = _make_split(
        train_fields, train_parameters, stats, T_in, T_out,
        include_parameter=include_parameter, static=train_static,
    )

    prepared_tests: Dict[str, dict] = {}
    for name, (fields, parameters, static) in test_splits.items():
        prepared_tests[name] = _make_split(
            _as_fields(fields), parameters, stats, T_in, T_out,
            include_parameter=include_parameter, static=static,
        )
        prepared_tests[name]["static"] = None if static is None else normalize_static(static, stats)

    dynamic_channels = int(train_fields.shape[2])
    static_channels = 0 if train_static is None else int(train_static.shape[1])
    parameter_channels = 1 if include_parameter else 0
    frame_channels = dynamic_channels + static_channels + parameter_channels
    resolution = int(train_fields.shape[-1])

    train["static"] = None if train_static is None else normalize_static(train_static, stats)
    return {
        "x_train": train["x"],
        "y_train": train["y"],
        "x_test": next(iter(prepared_tests.values()))["x"],
        "y_test": next(iter(prepared_tests.values()))["y"],
        "pos": _grid(resolution),
        "resolution": resolution,
        "in_channels": T_in * frame_channels,
        "out_channels": dynamic_channels,
        "coord_dim": 2,
        "task_type": "temporal",
        "T_in": T_in,
        "T_out": T_out,
        "dynamic_channels": dynamic_channels,
        "static_channels": static_channels,
        "include_parameter": include_parameter,
        "frame_channels": frame_channels,
        "stats": stats,
        "train_parameters": train_parameters.astype(np.float32),
        "train_static": train["static"],
        "test_splits": prepared_tests,
        "dataset": task,
    }


def _burger_or_rd(task: str, root: Path, T_in: int, T_out: int) -> dict:
    prefix = "Burger" if task == "burger" else "RD"
    train = _npz(root / f"{prefix}_train.npz")
    test = _npz(root / f"{prefix}_test.npz")
    return _finalize(
        task=task,
        train_fields=_as_fields(train["data"]),
        train_parameters=np.asarray(train["para"], dtype=np.float32).reshape(-1),
        test_splits={"external_test": (_as_fields(test["data"]), np.asarray(test["para"], dtype=np.float32).reshape(-1), None)},
        T_in=T_in,
        T_out=T_out,
        include_parameter=True,
    )


def _ns_a(root: Path, T_in: int, T_out: int) -> dict:
    h5py = _require_h5py()
    path = root / "NS" / "data" / "fno_ns_v1e-3_n1200_t50" / "data.h5"
    with h5py.File(path, "r") as handle:
        fields = _from_last_time_axis(handle["u"][()])
        parameters = np.asarray(handle["mu"][()], dtype=np.float32).reshape(-1) if "mu" in handle else np.full(len(fields), 1e-3, dtype=np.float32)
    return _finalize(
        task="ns-a",
        train_fields=fields[:1000],
        train_parameters=parameters[:1000],
        test_splits={"internal_test": (fields[1000:], parameters[1000:], None)},
        T_in=T_in,
        T_out=T_out,
        include_parameter=False,
    )


def _read_ns_group(obj) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if hasattr(obj, "keys"):
        fields = _from_last_time_axis(obj["u"][()])
        parameters = np.asarray(obj["mu"][()], dtype=np.float32).reshape(-1) if "mu" in obj else np.ones(len(fields), dtype=np.float32)
        force = np.asarray(obj["f"][()], dtype=np.float32) if "f" in obj else None
    else:
        fields = _from_last_time_axis(obj[()])
        parameters = np.ones(len(fields), dtype=np.float32)
        force = None
    return fields, parameters, force


def _ns_b_t50(root: Path, T_in: int, T_out: int) -> dict:
    h5py = _require_h5py()
    path = root / "NS" / "data" / "torus_vis_n1200_t50" / "data.h5"
    with h5py.File(path, "r") as handle:
        train_fields, train_mu, train_f = _read_ns_group(handle["train"])
        valid_fields, valid_mu, valid_f = _read_ns_group(handle["valid"])
        test_fields, test_mu, test_f = _read_ns_group(handle["test"])
    has_force = train_f is not None and valid_f is not None and test_f is not None
    train_static = train_f[:, None] if has_force else None
    eval_fields = np.concatenate([valid_fields, test_fields], axis=0)
    eval_mu = np.concatenate([valid_mu, test_mu], axis=0)
    eval_static = np.concatenate([valid_f, test_f], axis=0)[:, None] if has_force else None
    return _finalize(
        task="ns-b-t50",
        train_fields=train_fields,
        train_parameters=train_mu,
        train_static=train_static,
        test_splits={"internal_test": (eval_fields, eval_mu, eval_static)},
        T_in=T_in,
        T_out=T_out,
        include_parameter=True,
    )


def _parse_nu(path: Path) -> float:
    match = re.search(r"nu_([0-9p.eE+-]+)", path.stem)
    return float(match.group(1).replace("p", ".")) if match else 0.0


def _read_heat_split(split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    h5py = _require_h5py()
    fields, parameters = [], []
    for path in sorted(split_dir.glob("nu_*.h5")):
        with h5py.File(path, "r") as handle:
            key = "data" if "data" in handle else next(iter(handle.keys()))
            data = np.asarray(handle[key][()], dtype=np.float32)
            if data.ndim != 5:
                raise ValueError(f"{path} expected [N,T,C,H,W], got {data.shape}.")
            nu = float(handle["nu"][()]) if "nu" in handle else _parse_nu(path)
        fields.append(data)
        parameters.append(np.full(data.shape[0], nu, dtype=np.float32))
    if not fields:
        raise FileNotFoundError(f"No nu_*.h5 files found under {split_dir}")
    return np.concatenate(fields, axis=0), np.concatenate(parameters, axis=0)


def _heat(root: Path, T_in: int, T_out: int) -> dict:
    heat_root = root / "heat"
    train_fields, train_parameters = _read_heat_split(heat_root / "train")
    test_splits = {}
    for name in ("val_interp", "test_seen", "test_interp", "test_extrap_low", "test_extrap_high"):
        fields, parameters = _read_heat_split(heat_root / name)
        test_splits[name] = (fields, parameters, None)
    return _finalize(
        task="heat",
        train_fields=train_fields,
        train_parameters=train_parameters,
        test_splits=test_splits,
        T_in=T_in,
        T_out=T_out,
        include_parameter=True,
    )


def _kf_kol(root: Path, T_in: int, T_out: int) -> dict:
    kf_root = root / "KF" / "kolmogorov_apebench"
    train_fields = _npy(kf_root / "kolmogorov_train.npy")
    test_fields = _npy(kf_root / "kolmogorov_test.npy")
    train_parameters = np.full(len(train_fields), 100.0, dtype=np.float32)
    test_parameters = np.full(len(test_fields), 100.0, dtype=np.float32)
    return _finalize(
        task="kf-kol",
        train_fields=train_fields,
        train_parameters=train_parameters,
        test_splits={"external_test": (test_fields, test_parameters, None)},
        T_in=T_in,
        T_out=T_out,
        include_parameter=False,
    )


def load_temporal(data_path: str, task: str, T_in: int = 10, T_out: int = 10, **_) -> dict:
    root = Path(data_path).expanduser().resolve()
    task = task.lower()
    if task in ("burger", "rd"):
        return _burger_or_rd(task, root, T_in, T_out)
    if task == "ns-a":
        return _ns_a(root, T_in, T_out)
    if task == "ns-b-t50":
        return _ns_b_t50(root, T_in, T_out)
    if task == "heat":
        return _heat(root, T_in, T_out)
    if task == "kf-kol":
        return _kf_kol(root, T_in, T_out)
    raise ValueError(f"Unsupported temporal task {task!r}. Choices: {TEMPORAL_TASKS}")
