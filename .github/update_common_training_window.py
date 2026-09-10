from pathlib import Path
import py_compile


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one block for {label}, found {count}")
    return text.replace(old, new, 1)


data_path = Path("tasks/pde_benchmarks/temporal_data.py")
text = data_path.read_text()
text = replace_once(
    text,
    """The loaders in this module follow the benchmark split and min-max preprocessing
conventions from common_0821, but expose tensors in the native CoEvol-NO PDE
runner format.  Training uses the official Navier--Stokes protocol: the first
``T_in`` frames form the history and the next ``T_out`` frames are the
teacher-forced targets.  Complete trajectories are retained separately for
final autoregressive evaluation.""",
    """The loaders in this module follow the benchmark split and min-max preprocessing
conventions from common_0821, but expose tensors in the native CoEvol-NO PDE
runner format.  For training, each trajectory contributes one random contiguous
``T_in + T_out`` window per access, matching common_0821: the first ``T_in``
frames of that window form the history and the next ``T_out`` frames are the
teacher-forced targets.  Complete trajectories are retained separately for
final autoregressive evaluation.""",
    "module training-protocol docstring",
)

marker = "\n\ndef _make_split(\n"
if marker not in text:
    raise RuntimeError("Could not locate _make_split insertion point")
dataset_class = r'''

class TemporalWindowDataset(torch.utils.data.Dataset):
    """Random temporal windows matching common_0821 ``WindowDataset``.

    The dataset length equals the number of trajectories. Each ``__getitem__``
    samples one contiguous ``T_in + T_out`` window from the selected trajectory,
    so a shuffled epoch sees every trajectory once with a newly sampled window.
    Returned tensors already use the native CoEvol-NO point-token layout.
    """

    def __init__(
        self,
        fields: np.ndarray,
        parameters: np.ndarray,
        stats: TemporalStats,
        T_in: int,
        T_out: int,
        *,
        include_parameter: bool,
        static: np.ndarray | None = None,
        pos: torch.Tensor | None = None,
    ) -> None:
        self.fields = _as_fields(fields)
        self.parameters = np.asarray(parameters, dtype=np.float32).reshape(-1)
        self.static = None if static is None else np.asarray(static, dtype=np.float32)
        self.stats = stats
        self.T_in = int(T_in)
        self.T_out = int(T_out)
        self.include_parameter = bool(include_parameter)
        required = self.T_in + self.T_out
        if self.fields.shape[1] < required:
            raise ValueError(
                f"Trajectory length {self.fields.shape[1]} is shorter than T_in+T_out={required}."
            )
        if len(self.parameters) != len(self.fields):
            raise ValueError("Parameter and trajectory counts must match.")
        if self.static is not None and len(self.static) != len(self.fields):
            raise ValueError("Static-condition and trajectory counts must match.")
        self.max_start = self.fields.shape[1] - required
        resolution = int(self.fields.shape[-1])
        self.pos = _grid(resolution).squeeze(0) if pos is None else pos.squeeze(0)

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(item)
        start = int(np.random.randint(0, self.max_start + 1))
        stop = start + self.T_in
        target_stop = stop + self.T_out

        history = normalize_fields(self.fields[index:index + 1, start:stop], self.stats)[0]
        _, _, h, w = history.shape
        frame_parts = [history]

        if self.static is not None:
            static_norm = normalize_static(self.static[index:index + 1], self.stats)[0]
            static_frames = np.broadcast_to(
                static_norm[None], (self.T_in, *static_norm.shape)
            )
            frame_parts.append(static_frames)

        if self.include_parameter:
            parameter = float(
                normalize_parameter(self.parameters[index:index + 1], self.stats)[0]
            )
            parameter_frames = np.full(
                (self.T_in, 1, h, w), parameter, dtype=np.float32
            )
            frame_parts.append(parameter_frames)

        frames = np.concatenate(frame_parts, axis=1)
        tokens = np.transpose(frames, (2, 3, 0, 1)).reshape(h * w, -1).astype(np.float32)

        targets = normalize_fields(
            self.fields[index:index + 1, stop:target_stop], self.stats
        )[0]
        targets = np.transpose(targets, (2, 3, 0, 1)).reshape(
            h * w, self.T_out, self.fields.shape[2]
        ).astype(np.float32)

        return self.pos, torch.from_numpy(tokens), torch.from_numpy(targets)
'''
text = text.replace(marker, dataset_class + marker, 1)

text = replace_once(
    text,
    """    stats = compute_stats(train_fields, train_parameters, train_static)
    train = _make_split(
        train_fields, train_parameters, stats, T_in, T_out,
        include_parameter=include_parameter, static=train_static,
    )

    prepared_tests: Dict[str, dict] = {}""",
    """    stats = compute_stats(train_fields, train_parameters, train_static)
    pos = _grid(int(train_fields.shape[-1]))
    train_dataset = TemporalWindowDataset(
        train_fields,
        train_parameters,
        stats,
        T_in,
        T_out,
        include_parameter=include_parameter,
        static=train_static,
        pos=pos,
    )

    prepared_tests: Dict[str, dict] = {}""",
    "training dataset construction",
)

text = replace_once(
    text,
    """    train[\"static\"] = None if train_static is None else normalize_static(train_static, stats)
    return {
        \"x_train\": train[\"x\"],
        \"y_train\": train[\"y\"],
        \"x_test\": next(iter(prepared_tests.values()))[\"x\"],
        \"y_test\": next(iter(prepared_tests.values()))[\"y\"],
        \"pos\": _grid(resolution),""",
    """    train_static_normalized = None if train_static is None else normalize_static(train_static, stats)
    return {
        \"train_dataset\": train_dataset,
        \"ntrain\": len(train_dataset),
        \"x_test\": next(iter(prepared_tests.values()))[\"x\"],
        \"y_test\": next(iter(prepared_tests.values()))[\"y\"],
        \"pos\": pos,""",
    "temporal data_info return",
)
text = replace_once(
    text,
    '        "train_static": train["static"],',
    '        "train_static": train_static_normalized,',
    "normalized training static metadata",
)
data_path.write_text(text)


train_path = Path("tasks/pde_benchmarks/train.py")
text = train_path.read_text()
text = replace_once(
    text,
    "    ntrain = data_info['x_train'].shape[0]\n    ntest = data_info['x_test'].shape[0]",
    "    ntrain = data_info['ntrain'] if task_type == 'temporal' else data_info['x_train'].shape[0]\n    ntest = data_info['x_test'].shape[0]",
    "temporal ntrain source",
)
text = replace_once(
    text,
    """    # Build data loaders.  Temporal benchmark inputs are already arranged in
    # the same per-point token layout used by the official NS loader.
    if task_type in ('ns', 'temporal'):
        pos_train = data_info['pos'].repeat(ntrain, 1, 1)
        pos_test = data_info['pos'].repeat(ntest, 1, 1)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_train, data_info['x_train'], data_info['y_train']),
            batch_size=physical_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_test, data_info['x_test'], data_info['y_test']),
            batch_size=physical_batch_size, shuffle=False)
    elif task_type in ('grid',):""",
    """    # Build data loaders. Temporal training follows common_0821: every
    # trajectory samples one random T_in+T_out window on each dataset access.
    # The model-side rollout inside that window remains CoEvol-NO teacher forcing.
    if task_type == 'temporal':
        train_loader = torch.utils.data.DataLoader(
            data_info['train_dataset'], batch_size=physical_batch_size, shuffle=True)
        pos_test = data_info['pos'].repeat(ntest, 1, 1)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_test, data_info['x_test'], data_info['y_test']),
            batch_size=physical_batch_size, shuffle=False)
    elif task_type == 'ns':
        pos_train = data_info['pos'].repeat(ntrain, 1, 1)
        pos_test = data_info['pos'].repeat(ntest, 1, 1)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_train, data_info['x_train'], data_info['y_train']),
            batch_size=physical_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_test, data_info['x_test'], data_info['y_test']),
            batch_size=physical_batch_size, shuffle=False)
    elif task_type in ('grid',):""",
    "temporal train DataLoader",
)
train_path.write_text(text)


test_path = Path("tests/test_temporal_adapter.py")
text = test_path.read_text()
text = replace_once(
    text,
    "    _tokens,\n    compute_stats,",
    "    _tokens,\n    TemporalWindowDataset,\n    compute_stats,",
    "TemporalWindowDataset test import",
)
marker = "\n\ndef test_history_shift_replaces_one_complete_frame_block():\n"
if marker not in text:
    raise RuntimeError("Could not locate test insertion point")
new_test = r'''


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
'''
text = text.replace(marker, new_test + marker, 1)
test_path.write_text(text)

for path in (data_path, train_path, test_path):
    py_compile.compile(str(path), doraise=True)

updated_data = data_path.read_text()
updated_train = train_path.read_text()
assert "class TemporalWindowDataset" in updated_data
assert "np.random.randint(0, self.max_start + 1)" in updated_data
assert "data_info['train_dataset']" in updated_train
loader_block = updated_train[
    updated_train.index("if task_type == 'temporal':"):
    updated_train.index("elif task_type == 'ns':")
]
assert "data_info['x_train']" not in loader_block
