#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier Neural Operator (FNO) based classifier/regressor for mode numbers (m or n).

This script focuses on the FNO neural network architecture and training:
- 4-channel input per sensor: [real, imag, theta, phi]
- 1D Fourier Neural Operator over the sensor axis (sorted consistently)
- Classification/regression for mode numbers (m or n)
- Training with early stopping, validation, and confusion matrix plotting

Data loading is handled by the data_caching module.
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

import matplotlib.pyplot as plt

# Import data caching utilities
from data_caching import build_or_load_cached_dataset

# -------------------------
# Data extraction and caching
# -------------------------
@dataclass
class CacheConfig:
    data_dir: str
    out_path: str
    num_timepoints: int = -1   # -1 => use all timepoints per dataset
    freq_tolerance: float = 0.1  # fractional band around mode frequency
    include_geometry: bool = True
    geometry_shot: Optional[int] = None  # If None and include_geometry, will try bp_k from this shot
    use_mode: str = 'n'  # 'n' or 'm'


def _open_datasets(data_dir: str, n_datasets = 1) -> List[xr.Dataset]:
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nc')])
    datasets = []
    for fp in files:
        if n_datasets > 0 and len(datasets) >= n_datasets:
            break
        try:
            ds = xr.open_dataset(fp)
            ds.load()  # bring into memory for speed
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: failed to open {fp}: {e}")
    return datasets


def _extract_sensor_names(ds: xr.Dataset) -> List[str]:
    # Variables without "Mode" are sensor channels, each has _real/_imag pair
    sn = [v for v in ds.data_vars if 'Mode' not in v]
    # Keep base names by stripping suffix; preserve pairing order: take unique in order
    base = []
    seen = set()
    for name in sn:
        if name.endswith('_real') or name.endswith('_imag'):
            b = name[:-5]
            if b not in seen:
                base.append(b)
                seen.add(b)
    return base


def _define_mode_regions_time(ds: xr.Dataset, F_mode_vars: List[str], time_indices: np.ndarray,
                              freq_tolerance: float) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Return regions per mode per selected time index: regions[mode][time_idx] = (freq_inds, sensor_inds)
    If frequency at time is zero, freq_inds will be empty. sensor_inds = all sensors by default.
    Expands tolerance until at least one frequency bin falls in the band.
    """
    n_sensors = len(_extract_sensor_names(ds))
    freq_vals = ds['frequency'].values
    df = float(freq_vals[1] - freq_vals[0]) if len(freq_vals) > 1 else 0.0

    regions = []
    for mode_var in F_mode_vars:
        mode_regions = []
        freq_series = ds[mode_var].values  # (time,)
        for ti in time_indices:
            f0 = float(freq_series[ti])
            if f0 == 0 or not np.isfinite(f0):
                mode_regions.append((np.array([], dtype=int), np.arange(n_sensors)))
                continue
            tol = float(freq_tolerance)
            # ensure at least one bin
            fmin = fmax = f0
            while (fmax - fmin) < max(df, 1e-12):
                fmin = f0 * (1 - tol)
                fmax = f0 * (1 + tol)
                tol *= 1.5
                if tol > 1.0:  # safety
                    break
            freq_inds = np.where((freq_vals >= fmin) & (freq_vals <= fmax))[0]
            if freq_inds.size == 0:
                mode_regions.append((np.array([], dtype=int), np.arange(n_sensors)))
            else:
                mode_regions.append((freq_inds, np.arange(n_sensors)))
        regions.append(mode_regions)
    return regions


def _avg_band_per_sensor(ds: xr.Dataset, time_index: int, freq_inds: np.ndarray,
                         sensor_bases: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Select a single representative frequency within ``freq_inds`` and average over time.

    New behavior (modified as requested):
        1. Within the provided ``freq_inds`` window, find the frequency index whose
           real component (spectrogram) has the maximum average value across ALL
           sensors AND ALL timepoints.
        2. For that chosen frequency index, compute (per sensor) the average real and
           imaginary components over time.
        3. Return (avg_real[S], avg_imag[S]) where S = number of sensors.

    Notes:
        * ``time_index`` is retained for signature compatibility but is ignored in the
          new logic (the selection uses all timepoints).
        * If ``freq_inds`` is empty, returns zeros.
    """
    S = len(sensor_bases)
    real_out = np.zeros(S, dtype=np.float32)
    imag_out = np.zeros(S, dtype=np.float32)
    if freq_inds.size == 0:
        return real_out, imag_out

    # Build a (S, T, F_sub) array of real parts for the candidate frequency window
    # This allows computing the global mean across sensors & time for each frequency.
    try:
        real_stack = np.stack([ds[f"{base}_real"].values[time_index, freq_inds] for base in sensor_bases], axis=0)
    except KeyError as e:
        # In case a sensor variable is missing; return zeros gracefully
        print(f"Warning: missing sensor variable while stacking real components: {e}")
        return real_out, imag_out

    # real_stack shape: (S, T, F_sub)
    # Mean across sensors (axis=0) and time (axis=1) -> frequency profile length F_sub
    mean_freq = real_stack.mean(axis=(0, 1))  # (F_sub,)
    # Select frequency (within freq_inds) with maximum mean real value
    local_best = int(np.argmax(mean_freq))
    chosen_freq_idx = int(freq_inds[local_best])

    # Average real & imag over time at chosen frequency index per sensor
    for si, base in enumerate(sensor_bases):
        try:
            real_vals = ds[f"{base}_real"].values[:, chosen_freq_idx]
            imag_vals = ds[f"{base}_imag"].values[:, chosen_freq_idx]
        except KeyError:
            real_vals = np.array([])
            imag_vals = np.array([])
        real_out[si] = float(real_vals.mean()) if real_vals.size else 0.0
        imag_out[si] = float(imag_vals.mean()) if imag_vals.size else 0.0

    return real_out, imag_out


def _get_mode_labels(ds: xr.Dataset, F_mode_vars: List[str], use_mode: str) -> List[int]:
    """Return per-mode integer labels from dataset attrs for 'm' or 'n'."""
    key = 'mode_n' if use_mode.lower() == 'n' else 'mode_m'
    if key not in ds.attrs:
        # Fallback: try variable or default zeros
        print(f"Warning: {key} not found in attrs; using zeros")
        return [0] * len(F_mode_vars)
    vals = ds.attrs[key]
    # Ensure list length equals number of modes
    try:
        vals_list = list(vals) if np.ndim(vals) > 0 else [int(vals)]
    except Exception:
        vals_list = [int(vals)]
    if len(vals_list) == 1 and len(F_mode_vars) > 1:
        vals_list = vals_list * len(F_mode_vars)
    if len(vals_list) != len(F_mode_vars):
        raise ValueError(f"Length of {key} in attrs ({len(vals_list)}) != number of modes ({len(F_mode_vars)})")
    return [int(v) for v in vals_list]


def _maybe_geometry_from_bp_k(geometry_shot: Optional[int], names_sorted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attempt to compute theta, phi per sensor using bp_k geometry for a given shot.
    Theta here is arctan2(Z, R - r_maj) with r_maj ~ 0.68 (C-Mod); Phi comes from bp_k.Phi.
    Names are matched case-insensitively. Returns (theta[S], phi[S]); zeros if unavailable.
    """
    S = len(names_sorted)
    th = np.zeros(S, dtype=np.float32)
    ph = np.zeros(S, dtype=np.float32)
    if __loadData is None or geometry_shot is None:
        return th, ph
    try:
        bp_k = __loadData(int(geometry_shot), pullData='bp_k', forceReload=['bp_k'] * False)['bp_k']
        name_map = {str(n).upper(): i for i, n in enumerate(bp_k.names)}
        rmaj = 0.68
        for i, nm in enumerate(names_sorted):
            u = str(nm).upper()
            if u in name_map:
                j = name_map[u]
                R = float(bp_k.R[j]); Z = float(bp_k.Z[j]); Phi = float(bp_k.Phi[j])
                th[i] = math.atan2(Z, R - rmaj)
                ph[i] = Phi
    except Exception as e:
        print(f"Warning: could not load geometry from bp_k: {e}")
    return th, ph


def cache_training_dataset(cfg: CacheConfig) -> Dict[str, np.ndarray]:
    """
    Build dataset from NetCDFs and cache to cfg.out_path (.npz).
    Saved content:
      - X_ri: (N, S, 2) real/imag averaged in the mode band for each sample
      - y:    (N,) integer labels (n or m)
      - sensor_names: (S,) strings, in sorted order
      - theta: (S,) optional if include_geometry
      - phi:   (S,) optional if include_geometry
      - meta:  dict with bookkeeping (json string)
    Returns loaded dict of arrays.
    """
    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)
    if os.path.exists(cfg.out_path):
        data = np.load(cfg.out_path, allow_pickle=True)
        print(f"Loaded cached dataset from {cfg.out_path} :: X_ri={data['X_ri'].shape}, y={data['y'].shape}")
        return {k: data[k] for k in data.files}

    datasets = _open_datasets(cfg.data_dir)
    if not datasets:
        raise FileNotFoundError(f"No NetCDF datasets found in {cfg.data_dir}")

    # Use the first dataset to determine sensor ordering, then apply globally
    base_sensor_names = _extract_sensor_names(datasets[0])
    names_sorted, names_orig, order_idx = gen_sensor_ordering(base_sensor_names)

    X_ri_list: List[np.ndarray] = []
    y_list: List[int] = []
    used_files: List[str] = []
    used_counts: List[int] = []

    for ds in datasets:
        sensor_bases = _extract_sensor_names(ds)
        # Reorder indices to match global sorted names; if any name missing, fill zeros later
        # Build mapping from sorted names to indices in this dataset
        idx_map = []
        name_to_idx = {b: i for i, b in enumerate(sensor_bases)}
        for nm in names_sorted:
            idx_map.append(name_to_idx.get(nm, -1))

        times = np.arange(len(ds['time']))
        if cfg.num_timepoints is not None and cfg.num_timepoints > 0 and cfg.num_timepoints < len(times):
            # random subset for caching speed
            rng = np.random.default_rng(42)
            times = np.sort(rng.choice(times, size=cfg.num_timepoints, replace=False))

        F_mode_vars = [v for v in ds.data_vars if v.startswith('F_Mode_')]
        mode_labels = _get_mode_labels(ds, F_mode_vars, cfg.use_mode)
        regions = _define_mode_regions_time(ds, F_mode_vars, times, cfg.freq_tolerance)

        n_added = 0
        for ti_i, ti in enumerate(times):
            for m_i, mode_var in enumerate(F_mode_vars):
                freq_inds, _sensor_inds = regions[m_i][ti_i]
                if freq_inds.size == 0:
                    continue  # skip zero-frequency times
                real_band, imag_band = _avg_band_per_sensor(ds, ti, freq_inds, sensor_bases)
                # Reorder to global sorted names; missing names -> 0
                S = len(names_sorted)
                real_sorted = np.zeros(S, dtype=np.float32)
                imag_sorted = np.zeros(S, dtype=np.float32)
                for s_idx, src in enumerate(idx_map):
                    if src >= 0:
                        real_sorted[s_idx] = real_band[src]
                        imag_sorted[s_idx] = imag_band[src]
                X_ri_list.append(np.stack([real_sorted, imag_sorted], axis=-1))  # (S, 2)
                y_list.append(mode_labels[m_i])
                n_added += 1
        used_files.append(getattr(ds, 'encoding', {}).get('source', ''))
        used_counts.append(n_added)
        print(f"Accumulated {n_added} samples from dataset")

    if not X_ri_list:
        raise RuntimeError("No samples accumulated. Check frequency bands and tolerance.")

    X_ri = np.stack(X_ri_list, axis=0)  # (N, S, 2)
    y = np.array(y_list, dtype=np.int64)

    theta = phi = None
    if cfg.include_geometry:
        theta, phi = _maybe_geometry_from_bp_k(cfg.geometry_shot, names_sorted)

    meta = {
        'data_dir': cfg.data_dir,
        'num_samples': int(X_ri.shape[0]),
        'num_sensors': int(X_ri.shape[1]),
        'used_files': used_files,
        'used_counts': used_counts,
        'use_mode': cfg.use_mode,
    }

    save_dict = {
        'X_ri': X_ri,
        'y': y,
        'sensor_names': names_sorted,
        'meta': np.array(json.dumps(meta)),
    }
    if theta is not None and phi is not None:
        save_dict['theta'] = theta
        save_dict['phi'] = phi

    np.savez(cfg.out_path, **save_dict)
    print(f"Saved cached dataset to {cfg.out_path} :: X_ri={X_ri.shape}, y={y.shape}")

    return save_dict

# -------------------------
# Dataset wrapper
# -------------------------
class SensorFourChannelDataset(Dataset):
    """Holds (N, S, 4) with per-sensor channels [real, imag, theta, phi]."""
    def __init__(self, X_ri: np.ndarray, y: np.ndarray, theta: Optional[np.ndarray]=None,
                 phi: Optional[np.ndarray]=None, mask_prob: float = 0.1, is_train: bool = True):
        assert X_ri.ndim == 3 and X_ri.shape[-1] == 2
        self.X_ri = X_ri.astype(np.float32)
        self.y = y.astype(np.int64)
        N, S, _ = self.X_ri.shape
        if theta is None:
            theta = np.zeros(S, dtype=np.float32)
        if phi is None:
            phi = np.zeros(S, dtype=np.float32)
        self.theta = theta.astype(np.float32)
        self.phi = phi.astype(np.float32)
        self.mask_prob = mask_prob
        self.is_train = is_train

    def __len__(self):
        return self.X_ri.shape[0]

    def __getitem__(self, idx):
        xri = torch.from_numpy(self.X_ri[idx])  # (S, 2)
        S = xri.shape[0]
        th = torch.from_numpy(self.theta).view(S, 1)
        ph = torch.from_numpy(self.phi).view(S, 1)
        x = torch.cat([xri, th, ph], dim=1)  # (S, 4)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)

        if self.is_train and self.mask_prob > 0:
            mask = torch.rand(S) < self.mask_prob
            x[mask] = 0.0
            # amplitude jitter on real/imag only
            noise = torch.randn_like(x[:, :2]) * 0.05
            x[:, :2] += noise
        return x, y

# -------------------------
# FNO 1D modules
# -------------------------
class SpectralConv1d(nn.Module):
    """
    1D Fourier layer. It does FFT over the sensor dimension and applies learnable
    complex weights to the lowest k modes, then inverse FFT.
    """
    def __init__(self, in_channels, out_channels, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        # Complex weights for positive frequencies
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def compl_mul1d(self, a, b):
        # (B, in, k) x (in, out, k) -> (B, out, k)
        return torch.einsum("bik,iok->bok", a, b)

    def forward(self, x):
        # x: (B, C_in, S)
        B, C, S = x.shape
        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, S_f)
        S_f = x_ft.shape[-1]
        k = min(self.modes, S_f)
        out_ft = torch.zeros(B, self.out_channels, S_f, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :k] = self.compl_mul1d(x_ft[:, :self.in_channels, :k], self.weight[:, :, :k])
        # iFFT
        x_out = torch.fft.irfft(out_ft, n=S, dim=-1)
        return x_out

class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm1d(width)

    def forward(self, x):
        # x: (B, C, S)
        y = self.spectral(x) + self.w(x)
        y = self.bn(y)
        return self.act(y)

class FNO1dClassifier(nn.Module):
    """
    1D FNO over sensor dimension with 4 input channels -> width -> blocks -> pooling -> head.
    """
    def __init__(self, in_channels=4, width=128, modes=16, depth=4, n_classes=10, dropout=0.3):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, n_classes),
        )

    def forward(self, x):
        # x: (B, S, 4) -> (B, 4, S)
        x = x.permute(0, 2, 1)
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        return self.head(x)

# -------------------------
# Training / evaluation
# -------------------------
@dataclass
class TrainConfig:
    batch_size: int = 64
    val_split: float = 0.2
    n_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    mask_prob: float = 0.1
    modes: int = 16
    width: int = 128
    depth: int = 4
    dropout: float = 0.3


def train_fno_classifier(X_ri: np.ndarray, y: np.ndarray, theta: Optional[np.ndarray], phi: Optional[np.ndarray],
                         n_classes: Optional[int] = None, cfg: TrainConfig = TrainConfig(),
                         device: Optional[torch.device] = None,
                         plot_confusion: bool = False,
                         class_names: Optional[np.ndarray] = None) -> Tuple[FNO1dClassifier, LabelEncoder, float, float]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = n_classes or int(y_enc.max()) + 1

    ds = SensorFourChannelDataset(X_ri, y_enc, theta=theta, phi=phi, mask_prob=cfg.mask_prob, is_train=True)
    N = len(ds)
    n_val = max(1, int(N * cfg.val_split))
    n_train = max(1, N - n_val)
    train_ds, val_idx = random_split(ds, [n_train, n_val])
    # Build val dataset without masking
    val_X_ri = ds.X_ri[val_idx.indices]
    val_y = y_enc[val_idx.indices]
    val_ds = SensorFourChannelDataset(val_X_ri, val_y, theta=theta, phi=phi, mask_prob=0.0, is_train=False)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = FNO1dClassifier(in_channels=4, width=cfg.width, modes=cfg.modes, depth=cfg.depth,
                            n_classes=n_classes, dropout=cfg.dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    best_val = 1e9
    best_state = None
    patience_counter = 0
    val_acc_hist = []

    for epoch in range(1, cfg.n_epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        n_seen = 0
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss /= max(1, n_seen)

        # val
        model.eval()
        val_loss = 0.0
        n_seen = 0
        correct = 0
        all_probs = []
        all_trues = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device); yb = yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                n_seen += xb.size(0)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                all_probs.append(torch.softmax(out, dim=1).cpu().numpy())
                all_trues.append(yb.cpu().numpy())
        val_loss /= max(1, n_seen)
        probs = np.concatenate(all_probs) if all_probs else np.zeros((0, n_classes))
        trues = np.concatenate(all_trues) if all_trues else np.zeros((0,), dtype=int)
        val_acc = (correct / n_seen) if n_seen else 0.0
        val_acc_hist.append(val_acc)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print("Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final eval on val set
    model.eval()
    correct = 0; total = 0
    probs_list = []; trues_list = []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device); yb = yb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            probs_list.append(torch.softmax(out, dim=1).cpu().numpy())
            trues_list.append(yb.cpu().numpy())
    val_acc = (correct / total) if total else 0.0
    probs = np.concatenate(probs_list) if probs_list else np.zeros((0, n_classes))
    trues = np.concatenate(trues_list) if trues_list else np.zeros((0,), dtype=int)
    auc = roc_auc_score(trues, probs, multi_class='ovr', average='macro') if total else 0.0

    # plot acc curve
    plt.figure(figsize=(5, 4))
    plt.plot(range(1, len(val_acc_hist) + 1), val_acc_hist, '-o')
    plt.xlabel('Epoch'); plt.ylabel('Val Acc'); plt.grid(True)
    plt.tight_layout(); plt.savefig('../output_plots/fno_val_acc.pdf', transparent=True)
    plt.close()

    if plot_confusion and total:
        from sklearn.metrics import ConfusionMatrixDisplay
        preds = np.argmax(probs, axis=1)
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5), tight_layout=True)
        ConfusionMatrixDisplay.from_predictions(trues, preds, ax=ax, display_labels=class_names)
        ax.set_title(f'FNO Confusion (Acc={val_acc:.3f}, AUC={auc:.3f})')
        fig.savefig('../output_plots/fno_confusion.pdf', transparent=True)
        plt.close(fig)

    return model, le, val_acc, auc

# -------------------------
# High-level: cache + train
# -------------------------
def build_or_load_cached_dataset(data_dir: str, out_path: str, use_mode: str = 'n',
                                 include_geometry: bool = True, geometry_shot: Optional[int] = None,
                                 num_timepoints: int = -1, freq_tolerance: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = CacheConfig(data_dir=data_dir, out_path=out_path, num_timepoints=num_timepoints,
                      freq_tolerance=freq_tolerance, include_geometry=include_geometry,
                      geometry_shot=geometry_shot, use_mode=use_mode)
    dat = cache_training_dataset(cfg)
    X_ri = dat['X_ri'] if isinstance(dat, dict) else dat.get('X_ri')
    y = dat['y'] if isinstance(dat, dict) else dat.get('y')
    sensor_names = dat['sensor_names'] if isinstance(dat, dict) else dat.get('sensor_names')
    theta = dat.get('theta', None) if isinstance(dat, dict) else None
    phi = dat.get('phi', None) if isinstance(dat, dict) else None
    return X_ri, y, sensor_names, theta, phi


def example_usage():
    """
    Example: cache dataset and train FNO classifier on 'n' labels.
    Adjust paths as needed.
    """
    data_dir = "/home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/low_m-n_testing/new_Mirnov_set/"
    out_path = "/home/rianc/Documents/Synthetic_Mirnov/data_output/cached/cached_training_slices_n.npz"
    geometry_shot = 1160714026  # for bp_k geometry; set None to skip

    # Cache or load
    X_ri, y, sensor_names, theta, phi = build_or_load_cached_dataset(
        data_dir=data_dir, out_path=out_path, use_mode='n', include_geometry=True,
        geometry_shot=geometry_shot, num_timepoints=-1, freq_tolerance=0.1)

    # Train FNO (4 channels constructed inside Dataset)
    model, le, val_acc, auc = train_fno_classifier(
        X_ri, y, theta=theta, phi=phi, n_classes=None,
        cfg=TrainConfig(batch_size=64, n_epochs=300, patience=30, modes=24, width=192, depth=4, dropout=0.3),
        plot_confusion=True, class_names=le.classes_ if hasattr(le, 'classes_') else None)

    # Save model bundle
    os.makedirs('../output_models', exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'classes': le.classes_.tolist(), 'sensor_names': sensor_names,
                'theta': theta, 'phi': phi}, '../output_models/fno_classifier_n.pth')
    print(f"Done. Val Acc={val_acc:.3f}, AUC={auc:.3f}")


if __name__ == "__main__":
    print("FNO predictor: building or loading cached dataset and training...")
    example_usage()
