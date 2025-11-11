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
# Dataset wrapper
# -------------------------
class SensorFourChannelDataset(Dataset):
    """Holds (N, S, 4) with per-sensor channels: difference from first sensor for [real, imag, theta, phi]."""
    def __init__(self, X_ri: np.ndarray, y: np.ndarray, theta: Optional[np.ndarray]=None,
                 phi: Optional[np.ndarray]=None, mask_prob: float = 0.1, is_train: bool = True,
                 scaler: Optional['StandardScaler']=None):
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
        self.scaler = scaler

        # Precompute difference features for all samples
        # For each sample: (S, 2) real/imag, (S,) theta, (S,) phi
        # Output: (S, 4) where each channel is value - value[0]
        diff_features = []
        for i in range(N):
            xri = self.X_ri[i]  # (S, 2)
            real_diff = xri[:, 0] - xri[0, 0]
            imag_diff = xri[:, 1] - xri[0, 1]
            th_diff = self.theta - self.theta[0]
            ph_diff = self.phi - self.phi[0]
            x = np.stack([real_diff, imag_diff, th_diff, ph_diff], axis=1)  # (S, 4)
            diff_features.append(x)
        self.diff_features = np.stack(diff_features, axis=0)  # (N, S, 4)

        # Apply scaler if provided
        if self.scaler is not None:
            N, S, C = self.diff_features.shape
            flat = self.diff_features.reshape(-1, C)
            flat_scaled = self.scaler.transform(flat)
            self.diff_features = flat_scaled.reshape(N, S, C)

    def __len__(self):
        return self.diff_features.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.diff_features[idx])  # (S, 4)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        S = x.shape[0]
        if self.is_train and self.mask_prob > 0:
            mask = torch.rand(S) < self.mask_prob
            x[mask] = 0.0
            # amplitude jitter on real/imag only
            noise = torch.randn_like(x[:, :2]) * 0.05
            x[:, :2] += noise
        return x, y
from sklearn.preprocessing import StandardScaler

def fit_and_apply_scaler(X_ri: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Compute difference features and fit StandardScaler, then transform and return scaled features.
    Returns scaled difference features (N, S, 4) and fitted scaler.
    """
    N, S, _ = X_ri.shape
    diff_features = []
    for i in range(N):
        xri = X_ri[i]
        real_diff = xri[:, 0] - xri[0, 0]
        imag_diff = xri[:, 1] - xri[0, 1]
        th_diff = theta - theta[0]
        ph_diff = phi - phi[0]
        x = np.stack([real_diff, imag_diff, th_diff, ph_diff], axis=1)
        diff_features.append(x)
    diff_features = np.stack(diff_features, axis=0)
    scaler = StandardScaler()
    N, S, C = diff_features.shape
    flat = diff_features.reshape(-1, C)
    scaler.fit(flat)
    flat_scaled = scaler.transform(flat)
    diff_features_scaled = flat_scaled.reshape(N, S, C)
    return diff_features_scaled, scaler

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
    y_enc = np.array(le.fit_transform(y))
    n_classes = n_classes or int(np.max(y_enc)) + 1

    ds = SensorFourChannelDataset(X_ri, y_enc, theta=theta, phi=phi, mask_prob=cfg.mask_prob, is_train=True)
    N = len(ds)
    n_val = max(1, int(N * cfg.val_split))
    n_train = max(1, N - n_val)
    train_ds, val_idx = random_split(ds, [n_train, n_val])
    # Build val dataset without masking
    val_X_ri = ds.diff_features[val_idx.indices]
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
# Example usage
# -------------------------
def example_usage():
    """
    Example: cache dataset and train FNO classifier on 'n' labels.
    Adjust paths as needed.
    """
    data_dir = "/home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/low_m-n_testing/new_Mirnov_set/"
    out_path = "/home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/low_m-n_testing/new_Mirnov_set/cached_data_-1.npz"
    geometry_shot = 1160714026  # for bp_k geometry; set None to skip

    # Cache or load
    X_ri, y, y_m, y_n, sensor_names, theta, phi = build_or_load_cached_dataset(
        data_dir=data_dir, out_path=out_path, use_mode='n', include_geometry=True,
        geometry_shot=geometry_shot, num_timepoints=10, freq_tolerance=0.1, n_datasets=-1,
        load_saved_data=True, visualize_first=False)

    # Compute difference features and fit scaler
    diff_features_scaled, scaler = fit_and_apply_scaler(X_ri, theta, phi)

    # Train FNO (4 channels constructed inside Dataset)
    model, le, val_acc, auc = train_fno_classifier(
        diff_features_scaled, y, theta=None, phi=None, n_classes=None,
        cfg=TrainConfig(batch_size=64, n_epochs=300, patience=30, modes=24, width=192, depth=4, dropout=0.3),
        plot_confusion=True, class_names=le.classes_ if hasattr(le, 'classes_') else None)

    # Save model bundle and scaler
    os.makedirs('../output_models', exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'classes': le.classes_.tolist(), 'sensor_names': sensor_names,
                'scaler': scaler, 'theta': theta, 'phi': phi}, '../output_models/fno_classifier_n.pth')
    print(f"Done. Val Acc={val_acc:.3f}, AUC={auc:.3f}")


if __name__ == "__main__":
    print("FNO predictor: building or loading cached dataset and training...")
    example_usage()
