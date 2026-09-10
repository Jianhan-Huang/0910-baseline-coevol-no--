import os
import logging
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from einops import rearrange

from utils.loss import TestLoss
from utils.normalizer import UnitTransformer, UnitGaussianNormalizer


logger = logging.getLogger(__name__)


def setup_file_logger(log_path):
    """Add a file handler to the module logger if not already attached."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path):
            return
    fh = logging.FileHandler(log_path, mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def central_diff(x, h, resolution):
    x = rearrange(x, 'b (h w) c -> b h w c', h=resolution, w=resolution)
    x = F.pad(x, (0, 0, 1, 1, 1, 1), mode='constant', value=0.)
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h)
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h)
    return grad_x, grad_y


# ---------------------------------------------------------------------------
# Data loaders for each PDE task
# ---------------------------------------------------------------------------

def load_darcy(data_path, ntrain, ntest, downsample):
    r = downsample
    h = int(((421 - 1) / r) + 1)
    s = h

    train_data = scio.loadmat(os.path.join(data_path, 'piececonst_r421_N1024_smooth1.mat'))
    x_train = train_data['coeff'][:ntrain, ::r, ::r][:, :s, :s].reshape(ntrain, -1)
    x_train = torch.from_numpy(x_train).float()
    y_train = train_data['sol'][:ntrain, ::r, ::r][:, :s, :s].reshape(ntrain, -1)
    y_train = torch.from_numpy(y_train)

    test_data = scio.loadmat(os.path.join(data_path, 'piececonst_r421_N1024_smooth2.mat'))
    x_test = test_data['coeff'][:ntest, ::r, ::r][:, :s, :s].reshape(ntest, -1)
    x_test = torch.from_numpy(x_test).float()
    y_test = test_data['sol'][:ntest, ::r, ::r][:, :s, :s].reshape(ntest, -1)
    y_test = torch.from_numpy(y_test)

    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)

    x_grid = np.linspace(0, 1, s)
    y_grid = np.linspace(0, 1, s)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    pos = torch.tensor(np.c_[x_grid.ravel(), y_grid.ravel()], dtype=torch.float).unsqueeze(0)

    return {
        'x_train': x_train, 'y_train': y_train,
        'x_test': x_test, 'y_test': y_test,
        'pos': pos, 'x_normalizer': x_normalizer, 'y_normalizer': y_normalizer,
        'resolution': s, 'in_channels': 1, 'coord_dim': 2,
        'task_type': 'grid', 'use_derivative_loss': True,
    }


def load_ns(data_path, ntrain, ntest, downsample, T_in=10, T_out=10):
    r = downsample
    h = int(((64 - 1) / r) + 1)
    data = scio.loadmat(data_path)
    u_all = data['u']

    train_a = u_all[:ntrain, ::r, ::r, :T_in][:, :h, :h, :].reshape(ntrain, -1, T_in)
    train_a = torch.from_numpy(train_a)
    train_u = u_all[:ntrain, ::r, ::r, T_in:T_out + T_in][:, :h, :h, :].reshape(ntrain, -1, T_out)
    train_u = torch.from_numpy(train_u)

    test_a = u_all[-ntest:, ::r, ::r, :T_in][:, :h, :h, :].reshape(ntest, -1, T_in)
    test_a = torch.from_numpy(test_a)
    test_u = u_all[-ntest:, ::r, ::r, T_in:T_out + T_in][:, :h, :h, :].reshape(ntest, -1, T_out)
    test_u = torch.from_numpy(test_u)

    x = np.linspace(0, 1, h)
    y = np.linspace(0, 1, h)
    x, y = np.meshgrid(x, y)
    pos = torch.tensor(np.c_[x.ravel(), y.ravel()], dtype=torch.float).unsqueeze(0)

    return {
        'x_train': train_a, 'y_train': train_u,
        'x_test': test_a, 'y_test': test_u,
        'pos': pos, 'resolution': h, 'in_channels': T_in, 'coord_dim': 2,
        'task_type': 'ns', 'T_out': T_out,
    }


def load_elasticity(data_path, ntrain, ntest):
    input_s = np.load(os.path.join(data_path, 'elasticity/Meshes/Random_UnitCell_sigma_10.npy'))
    input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0)
    input_xy = np.load(os.path.join(data_path, 'elasticity/Meshes/Random_UnitCell_XY_10.npy'))
    input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)

    train_s = input_s[:ntrain]
    test_s = input_s[-ntest:]
    train_xy = input_xy[:ntrain]
    test_xy = input_xy[-ntest:]

    y_normalizer = UnitTransformer(train_s)
    train_s = y_normalizer.encode(train_s)

    return {
        'x_train': train_xy, 'y_train': train_s,
        'x_test': test_xy, 'y_test': test_s,
        'y_normalizer': y_normalizer,
        'resolution': train_xy.shape[-2], 'in_channels': 0, 'coord_dim': 2,
        'task_type': 'elasticity',
    }


def load_airfoil_pde(data_path, ntrain, ntest, downsamplex=1, downsampley=1):
    r1, r2 = downsamplex, downsampley
    s1 = int(((221 - 1) / r1) + 1)
    s2 = int(((51 - 1) / r2) + 1)

    inputX = np.load(os.path.join(data_path, 'NACA_Cylinder_X.npy'))
    inputY = np.load(os.path.join(data_path, 'NACA_Cylinder_Y.npy'))
    input = torch.stack([torch.tensor(inputX, dtype=torch.float),
                         torch.tensor(inputY, dtype=torch.float)], dim=-1)

    output = np.load(os.path.join(data_path, 'NACA_Cylinder_Q.npy'))[:, 4]
    output = torch.tensor(output, dtype=torch.float)

    x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2].reshape(ntrain, -1, 2)
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2].reshape(ntrain, -1)
    x_test = input[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2].reshape(ntest, -1, 2)
    y_test = output[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2].reshape(ntest, -1)

    return {
        'x_train': x_train, 'y_train': y_train,
        'x_test': x_test, 'y_test': y_test,
        'resolution': s1, 'in_channels': 0, 'coord_dim': 2,
        'task_type': 'airfoil_pde',
    }


def load_pipe(data_path, ntrain, ntest, downsamplex=1, downsampley=1):
    r1, r2 = downsamplex, downsampley
    s1 = int(((129 - 1) / r1) + 1)
    s2 = int(((129 - 1) / r2) + 1)
    N = ntrain + ntest

    inputX = np.load(os.path.join(data_path, 'Pipe_X.npy'))
    inputY = np.load(os.path.join(data_path, 'Pipe_Y.npy'))
    input = torch.stack([torch.tensor(inputX, dtype=torch.float),
                         torch.tensor(inputY, dtype=torch.float)], dim=-1)

    output = np.load(os.path.join(data_path, 'Pipe_Q.npy'))[:, 0]
    output = torch.tensor(output, dtype=torch.float)

    x_train = input[:N][:ntrain, ::r1, ::r2][:, :s1, :s2].reshape(ntrain, -1, 2)
    y_train = output[:N][:ntrain, ::r1, ::r2][:, :s1, :s2].reshape(ntrain, -1)
    x_test = input[:N][-ntest:, ::r1, ::r2][:, :s1, :s2].reshape(ntest, -1, 2)
    y_test = output[:N][-ntest:, ::r1, ::r2][:, :s1, :s2].reshape(ntest, -1)

    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)

    return {
        'x_train': x_train, 'y_train': y_train,
        'x_test': x_test, 'y_test': y_test,
        'x_normalizer': x_normalizer, 'y_normalizer': y_normalizer,
        'resolution': s1, 'in_channels': 0, 'coord_dim': 2,
        'task_type': 'pipe',
    }


# ---------------------------------------------------------------------------
# Unified training loop
# ---------------------------------------------------------------------------


def train_pde(model, data_info, cfg):
    from tasks.pde_benchmarks.temporal_eval import (
        _as_output_frame,
        add_state_noise,
        evaluate_complete_trajectories,
        shift_history_tokens,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    myloss = TestLoss(size_average=False)
    task_type = data_info['task_type']
    ntrain = data_info['ntrain'] if task_type == 'temporal' else data_info['x_train'].shape[0]
    ntest = data_info['x_test'].shape[0]
    configured_batch_size = int(cfg.get('batch_size', 8))
    physical_batch_size = int(cfg.get('physical_batch_size', configured_batch_size))
    if physical_batch_size < 1 or physical_batch_size > configured_batch_size:
        raise ValueError('physical_batch_size must be between 1 and batch_size')
    if configured_batch_size % physical_batch_size != 0:
        raise ValueError('batch_size must be divisible by physical_batch_size for equivalent accumulation')
    accumulation_steps = configured_batch_size // physical_batch_size
    epochs = cfg.get('epochs', 500)
    lr = cfg.get('lr', 1e-3)
    weight_decay = cfg.get('weight_decay', 1e-5)
    max_grad_norm = cfg.get('max_grad_norm', None)
    scheduler_type = cfg.get('scheduler', 'OneCycleLR')
    use_derivative_loss = data_info.get('use_derivative_loss', False)

    if use_derivative_loss:
        de_x = TestLoss(size_average=False)
        de_y = TestLoss(size_average=False)

    y_normalizer = data_info.get('y_normalizer')
    if y_normalizer is not None:
        y_normalizer.to(device)
    x_normalizer = data_info.get('x_normalizer')
    if x_normalizer is not None:
        x_normalizer.to(device)

    resolution = data_info.get('resolution', 64)
    dx = 1.0 / resolution if use_derivative_loss else None

    # Build data loaders. Temporal training follows common_0821: every
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
    elif task_type in ('grid',):
        pos_train = data_info['pos'].repeat(ntrain, 1, 1)
        pos_test = data_info['pos'].repeat(ntest, 1, 1)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_train, data_info['x_train'], data_info['y_train']),
            batch_size=physical_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_test, data_info['x_test'], data_info['y_test']),
            batch_size=physical_batch_size, shuffle=False)
    elif task_type == 'elasticity':
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_info['x_train'], data_info['x_train'], data_info['y_train']),
            batch_size=physical_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_info['x_test'], data_info['x_test'], data_info['y_test']),
            batch_size=physical_batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_info['x_train'], data_info['x_train'], data_info['y_train']),
            batch_size=physical_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_info['x_test'], data_info['x_test'], data_info['y_test']),
            batch_size=physical_batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_steps_per_epoch = (len(train_loader) + accumulation_steps - 1) // accumulation_steps
    if scheduler_type == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=optimizer_steps_per_epoch)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_dir = cfg.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    save_name = cfg.get('save_name', 'model')
    setup_file_logger(os.path.join(save_dir, 'log.txt'))

    start_epoch = 0
    resume_from = cfg.get('checkpoint') if cfg.get('eval_only', False) else cfg.get('resume_from', None)
    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f'Checkpoint not found: {resume_from}')
        logger.info(f"Loading checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            if not cfg.get('eval_only', False):
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint.get('epoch', 0) + 1
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint; start epoch {start_epoch}")

    if cfg.get('eval_only', False):
        if task_type != 'temporal':
            raise ValueError('--eval complete-trajectory mode is implemented for temporal benchmark tasks only')
        metrics = evaluate_complete_trajectories(model, data_info, cfg, device)
        logger.info(f"Complete-trajectory metrics: {metrics}")
        return model

    for ep in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0

        if task_type in ('ns', 'temporal'):
            noise_std = cfg.get('noise_std', 0.0)
            T_out = data_info['T_out']
            optimizer.zero_grad(set_to_none=True)
            micro_batches = 0
            for batch_idx, (x, fx, yy) in enumerate(train_loader):
                loss = 0
                x, fx, yy = x.to(device), fx.to(device), yy.to(device)
                bsz = x.shape[0]

                if task_type == 'ns':
                    if noise_std > 0:
                        scale_factor = fx[0].numel() ** 0.5
                        norm_x = torch.sum(fx ** 2, dim=(1, 2), keepdim=True) ** 0.5
                        fx = fx + noise_std * (norm_x / scale_factor) * torch.randn_like(fx)
                    for t in range(T_out):
                        y = yy[..., t:t + 1]
                        im = model(fx, x)
                        loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                        fx = torch.cat((fx[..., 1:], y), dim=-1)
                else:
                    T_in = data_info['T_in']
                    frame_channels = data_info['frame_channels']
                    dynamic_channels = data_info['dynamic_channels']
                    fx = add_state_noise(
                        fx, noise_std, T_in=T_in, frame_channels=frame_channels,
                        dynamic_channels=dynamic_channels)
                    for t in range(T_out):
                        y = yy[..., t, :]  # [B,P,C_state]
                        im = _as_output_frame(model(fx, x), dynamic_channels)
                        loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                        # Official teacher forcing: append ground truth, not prediction.
                        fx = shift_history_tokens(
                            fx, y, T_in=T_in, frame_channels=frame_channels,
                            dynamic_channels=dynamic_channels)

                train_loss += loss.item() / T_out
                loss.backward()
                micro_batches += 1
                do_step = micro_batches == accumulation_steps or batch_idx == len(train_loader) - 1
                if do_step:
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler_type == 'OneCycleLR':
                        scheduler.step()
                    micro_batches = 0
        else:
            for batch in train_loader:
                if len(batch) == 3:
                    pos_or_x, fx_or_pos, y = batch
                else:
                    continue
                pos_or_x = pos_or_x.to(device)
                fx_or_pos = fx_or_pos.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                if task_type in ('pipe', 'airfoil_pde', 'elasticity'):
                    out = model(None, pos_or_x).squeeze(-1)
                else:
                    out = model(fx_or_pos, pos_or_x).squeeze(-1)

                if y_normalizer is not None:
                    out = y_normalizer.decode(out)
                    y_decoded = y_normalizer.decode(y)
                else:
                    y_decoded = y

                l2loss = myloss(out, y_decoded) if y_normalizer is not None else myloss(out, y)
                if use_derivative_loss:
                    s = resolution
                    out_padded = rearrange(out.unsqueeze(-1), 'b (h w) c -> b c h w', h=s)
                    out_padded = out_padded[..., 1:-1, 1:-1].contiguous()
                    out_padded = F.pad(out_padded, (1, 1, 1, 1), "constant", 0)
                    out_padded = rearrange(out_padded, 'b c h w -> b (h w) c')
                    gt_grad_x, gt_grad_y = central_diff(y_decoded.unsqueeze(-1), dx, s)
                    pred_grad_x, pred_grad_y = central_diff(out_padded, dx, s)
                    deriv_loss = de_x(pred_grad_x, gt_grad_x) + de_y(pred_grad_y, gt_grad_y)
                    loss_val = 0.1 * deriv_loss + l2loss
                else:
                    loss_val = l2loss

                loss_val.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler_type == 'OneCycleLR':
                    scheduler.step()
                train_loss += l2loss.item()

        if scheduler_type != 'OneCycleLR':
            scheduler.step()

        train_loss /= ntrain

        # Preserve the official per-epoch test protocol.  For temporal tasks
        # this is the configured T_out-step autoregressive window; complete
        # trajectories are evaluated separately after training.
        model.eval()
        rel_err = 0.0
        test_full_loss = 0.0
        with torch.no_grad():
            if task_type in ('ns', 'temporal'):
                T_out = data_info['T_out']
                for x, fx, yy in test_loader:
                    loss = 0
                    x, fx, yy = x.to(device), fx.to(device), yy.to(device)
                    bsz = x.shape[0]
                    preds = []
                    for t in range(T_out):
                        if task_type == 'ns':
                            y = yy[..., t:t + 1]
                            im = model(fx, x)
                            loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                            im_frame = im.unsqueeze(-1)
                            fx = torch.cat((fx[..., 1:], im_frame), dim=-1)
                            preds.append(im_frame)
                        else:
                            y = yy[..., t, :]
                            im = _as_output_frame(model(fx, x), data_info['dynamic_channels'])
                            loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                            fx = shift_history_tokens(
                                fx, im, T_in=data_info['T_in'],
                                frame_channels=data_info['frame_channels'],
                                dynamic_channels=data_info['dynamic_channels'])
                            preds.append(im)
                    rel_err += loss.item()
                    if task_type == 'ns':
                        pred = torch.cat(preds, -1)
                        test_full_loss += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
                    else:
                        pred = torch.stack(preds, dim=2)
                        test_full_loss += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
            else:
                for batch in test_loader:
                    if len(batch) == 3:
                        pos_or_x, fx_or_pos, y = batch
                    else:
                        continue
                    pos_or_x = pos_or_x.to(device)
                    fx_or_pos = fx_or_pos.to(device)
                    y = y.to(device)
                    if task_type in ('pipe', 'airfoil_pde', 'elasticity'):
                        out = model(None, pos_or_x).squeeze(-1)
                    else:
                        out = model(fx_or_pos, pos_or_x).squeeze(-1)
                    if y_normalizer is not None:
                        out = y_normalizer.decode(out)
                    rel_err += myloss(out, y).item()

        rel_err /= ntest * data_info['T_out'] if task_type in ('ns', 'temporal') else ntest
        if task_type in ('ns', 'temporal'):
            test_full_loss /= ntest
            logger.info(
                f"Epoch {ep} Train loss: {train_loss:.5f} "
                f"test_step_loss: {rel_err:.5f} test_full_loss: {test_full_loss:.5f}")
        else:
            logger.info(f"Epoch {ep} Train loss: {train_loss:.5f} rel_err: {rel_err:.5f}")

        if ep % 100 == 0 or ep == epochs - 1:
            checkpoint = {
                'epoch': ep,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(save_dir, f'{save_name}.pt'))

    if task_type == 'temporal':
        metrics = evaluate_complete_trajectories(model, data_info, cfg, device)
        logger.info(f"Complete-trajectory metrics: {metrics}")
    return model
