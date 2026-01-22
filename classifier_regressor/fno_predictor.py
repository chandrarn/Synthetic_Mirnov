#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier Neural Operator (FNO) based classifier/regressor for mode numbers (m or n).

This script focuses on the FNO neural network architecture and training:
- 5-channel input per sensor: [real, sin(Δimag), cos(Δimag), theta, phi]
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
class SensorGenericDataset(Dataset):
    """Holds (N, S, C) with per-sensor channels. Assumes data is already scaled."""
    def __init__(self, diff_features_scaled: np.ndarray, y: np.ndarray, theta: Optional[np.ndarray]=None,
                 phi: Optional[np.ndarray]=None, mask_prob: float = 0.1, is_train: bool = True,
                 scaler: Optional['StandardScaler']=None):
        # assert X_ri.ndim == 3 and X_ri.shape[-1] == 2
        self.diff_features = diff_features_scaled.astype(np.float32)
        self.y = y.astype(np.int64)
        N, S, _ = self.diff_features.shape
        if theta is None:
            theta = np.zeros(S, dtype=np.float32)
        if phi is None:
            phi = np.zeros(S, dtype=np.float32)
        self.theta = theta.astype(np.float32)
        self.phi = phi.astype(np.float32)
        self.mask_prob = mask_prob
        self.is_train = is_train
        self.scaler = scaler

        # Assume that feature vector is precomputed outside of train_fno


    def __len__(self):
        return self.diff_features.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.diff_features[idx])  # (S, 5)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        S = x.shape[0]
        if self.is_train and self.mask_prob > 0:
            mask = torch.rand(S) < self.mask_prob
            x[mask] = 0.0
            # Light jitter on real channel only to avoid corrupting sin/cos encoding
            noise = torch.randn_like(x[:, 0]) * 0.08
            x[:, 0] += noise
        return x, y

class SensorFourChannelDataset(Dataset):
    """Holds (N, S, 5) with per-sensor channels: [Δreal, sin(Δimag), cos(Δimag), Δtheta, Δphi]."""
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
        # Output: (S, 5) where channels are [Δreal, sin(Δimag), cos(Δimag), Δtheta, Δphi]
        diff_features = []
        for i in range(N):
            xri = self.X_ri[i]  # (S, 2)
            real_diff = xri[:, 0] - xri[0, 0]
            delta_imag = xri[:, 1] - xri[0, 1]

            # optional: handle negative amplitudes as you do
            arg_neg  = xri[:,0] < 0
            if np.any(arg_neg):
                real_diff[arg_neg] = np.abs(real_diff[arg_neg])
                delta_imag[arg_neg] += np.pi

            # wrap-aware signed difference in (-pi, pi]
            delta_imag = np.angle(np.exp(1j*delta_imag))
            sin_imag = np.sin(delta_imag)
            cos_imag = np.cos(delta_imag)

            th_diff = self.theta - self.theta[0]
            ph_diff = self.phi - self.phi[0]
            x = np.stack([real_diff, sin_imag, cos_imag, th_diff, ph_diff], axis=1)
            diff_features.append(x)
        self.diff_features = np.stack(diff_features, axis=0)  # (N, S, 5)

        # Apply scaler if provided
        if self.scaler is not None:
            N, S, C = self.diff_features.shape
            flat = self.diff_features.reshape(-1, C)
            flat_scaled = self.scaler.transform(flat)
            self.diff_features = flat_scaled.reshape(N, S, C)

    def __len__(self):
        return self.diff_features.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.diff_features[idx])  # (S, 5)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        S = x.shape[0]
        if self.is_train and self.mask_prob > 0:
            mask = torch.rand(S) < self.mask_prob
            x[mask] = 0.0
            # Light jitter on real channel only to avoid corrupting sin/cos encoding
            noise = torch.randn_like(x[:, 0]) * 0.08
            x[:, 0] += noise
        return x, y
from sklearn.preprocessing import StandardScaler

def fit_and_apply_scaler(X_ri: np.ndarray, theta: np.ndarray, phi: np.ndarray, \
                         zero_baseline: bool = False, sincos_channels: bool = False,\
                             sincos_only: bool = True ) -> \
            Tuple[np.ndarray, StandardScaler]:
    """
    Compute difference features and fit StandardScaler, then transform and return scaled features.
    Returns scaled difference features (N, S, 5) and fitted scaler.
    """
    N, S, _ = X_ri.shape
    diff_features = []
    for i in range(N):
        xri = X_ri[i]
        real_diff = xri[:, 0] - xri[0, 0]#*(1 if not zero_baseline else 0)
        delta_imag = xri[:, 1] - xri[0, 1]*(1 if zero_baseline else 0)
        delta_imag = np.angle(np.exp(1j*delta_imag ))  # wrap-aware, with sign check
        # delta_imag *= __check_angle_slope(delta_imag)
        
        # if __check_angle_slope(delta_imag) < 0 : raise  ValueError("Angle differences are decreasing, check sensor ordering!")
        sin_imag = np.sin(delta_imag)
        cos_imag = np.cos(delta_imag)
        th_diff = theta - theta[0]
        ph_diff = phi - phi[0]

        if sincos_only: 
            x = np.stack([sin_imag, cos_imag], axis=1)
        elif sincos_channels: 
            x = np.stack([real_diff, sin_imag, cos_imag, th_diff, ph_diff], axis=1)
        else: 
            x = np.stack([real_diff, delta_imag, th_diff, ph_diff], axis=1) # Switch back to just angle

        diff_features.append(x)
    diff_features = np.stack(diff_features, axis=0)
    scaler = StandardScaler()
    N, S, C = diff_features.shape
    flat = diff_features.reshape(-1, C) # (N*S, C)
    scaler.fit(flat)
    flat_scaled = scaler.transform(flat) 
    diff_features_scaled = flat_scaled.reshape(N, S, C)
    return diff_features_scaled, scaler

def __check_angle_slope(delta_imag: np.ndarray, n_sensors: int = 5) -> int:
    # Check if angle differences are on-average increasing or decreasing
    diffs = np.diff(delta_imag)[:n_sensors]
    return np.sign(np.mean(diffs)) * -1 
def plot_scaled_features(diff_features_scaled: np.ndarray, save_path: str, saveData: bool = True):
    # For some set of samples, plot all five channels after scaling
    N, S, C = diff_features_scaled.shape

    plt.close('Scaled_Features')
    n_rows = 2
    n_cols = 2 if C <=4 else 3
    fig, ax = plt.subplots(n_rows,n_cols,figsize=(6,6), tight_layout=True,num='Scaled_Features',
                           sharex=True)
    ax = ax.flatten()

    for i in range(C):
        channel_data = diff_features_scaled[:, :, i]
        ax[i].plot(channel_data.T, alpha=0.3)
        labels = [r'$\Delta||\mathcal{F}||$', r'$\sin(\Delta\angle\mathcal{F})$',
                  r'$\cos(\Delta\angle\mathcal{F})$', r'$\Delta\theta$', r'$\Delta\phi$']
        ax[i].set_ylabel(labels[i])
        if i >= (n_rows-1)*n_cols:
            ax[i].set_xlabel('Sample Index')
        ax[i].grid()

    # Hide any unused axes
    for j in range(C, len(ax)):
        ax[j].set_visible(False)

    if save_path:
        plt.savefig(save_path, transparent=True)
        print(f"Saved scaled features plot to {save_path}")

    if saveData:
        np.savez('../data_output/scaled_features.npz', features_scaled=diff_features_scaled, \
                 channel_data=channel_data)
        print(f"Saved scaled features data to ../data_output/scaled_features.npz")

###############################################################################################
###############################################################################################

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
    def __init__(self, width: int, modes: int, dropout: float = 0.1):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, S)
        y = self.spectral(x) + self.w(x)
        y = self.bn(y)
        y = self.act(y)
        y = self.dropout(y)
        return y

class FNO1dClassifier(nn.Module):
    """
    1D FNO over sensor dimension with 5 input channels -> width -> blocks -> pooling -> head.
    """
    def __init__(self, in_channels=5, width=128, modes=16, depth=4, n_classes=10, dropout=0.3):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes, dropout=dropout*0.5) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
            nn.BatchNorm1d(width),
            nn.Dropout(dropout),
            nn.Linear(width, n_classes),
        )

    def forward(self, x):
        # x: (B, S, C=5) -> (B, 5, S)
        # Verify: S is the sensor dimension (last dim after permute becomes sensor axis)
        x = x.permute(0, 2, 1)  # Now (B, 5, S) - channels first, sensors last
        x = self.lift(x)  # (B, width, S)
        for blk in self.blocks:
            x = blk(x)  # Still (B, width, S) - FFT happens over sensor dimension (dim=-1)
        x = self.pool(x)  # (B, width, 1)
        x = x.squeeze(-1)  # (B, width)
        return self.head(x)  # (B, n_classes)


###############################################################################################
###############################################################################################

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
    patience: int = 15
    mask_prob: float = 0.0  # DISABLE masking - features too simple to regularize this way
    modes: int = 16
    width: int = 128
    depth: int = 4
    dropout: float = 0.1  # REDUCE dropout dramatically
    label_smoothing: float = 0.0  # DISABLE - hurts small datasets
    num_workers: int = 4
    grad_clip: float = 1.0
    num_channels: int = 4


def train_fno_classifier(diff_features_scaled: np.ndarray, y: np.ndarray, theta: Optional[np.ndarray], phi: Optional[np.ndarray],
                         n_classes: Optional[int] = None, cfg: TrainConfig = TrainConfig(),
                         device: Optional[torch.device] = None,
                         plot_confusion: bool = False,
                         class_names: Optional[np.ndarray] = None,
                         scaler: Optional[StandardScaler] = None) -> Tuple[FNO1dClassifier, LabelEncoder, float, float]:
   
    le = LabelEncoder()
    y_enc = np.array(le.fit_transform(y))
    n_classes = n_classes or int(np.max(y_enc)) + 1

    # Use stratified split to ensure class balance
    from sklearn.model_selection import train_test_split
    N = len(diff_features_scaled)
    n_val = max(1, int(N * cfg.val_split))
    n_train = max(1, N - n_val)
    
    
    train_indices, val_indices = train_test_split(
        np.arange(N), test_size=cfg.val_split, stratify=y_enc, random_state=42
    )
    


    # DEBUG: Print class distributions
    print(f"Train set class distribution: {np.bincount(y_enc[train_indices])}")
    print(f"Val set class distribution: {np.bincount(y_enc[val_indices])}")
    
    # Create training and validation datasets directly from splits
    train_ds = SensorGenericDataset(diff_features_scaled[train_indices], y_enc[train_indices], theta=theta, phi=phi, 
                                        mask_prob=cfg.mask_prob, is_train=True, scaler=scaler)
    val_ds = SensorGenericDataset(diff_features_scaled[val_indices], y_enc[val_indices], theta=theta, phi=phi, 
                                      mask_prob=0.0, is_train=False, scaler=scaler)

    # Use multiple workers for parallel data loading
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                         num_workers=cfg.num_workers, pin_memory=True, 
                         persistent_workers=True if cfg.num_workers > 0 else False)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True, 
                       persistent_workers=True if cfg.num_workers > 0 else False)

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = FNO1dClassifier(in_channels=cfg.num_channels, width=cfg.width, modes=cfg.modes, depth=cfg.depth,
                            n_classes=n_classes, dropout=cfg.dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Improved learning rate scheduling with warmup and cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Label smoothing for better generalization
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_enc), y=y_enc)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    # Use TWO criteria: one weighted for training, one unweighted for validation metrics
    criterion_train = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    criterion_val = nn.CrossEntropyLoss()  # Unweighted for fair validation comparison


    # DEBUG: Check for data duplication and feature variance
    print(f"\n=== DATA QUALITY CHECKS ===")
    print(f"Total samples: {N}, Train: {len(train_indices)}, Val: {len(val_indices)}")
    
    # Check if any samples are duplicates
    train_features_flat = diff_features_scaled[train_indices].reshape(len(train_indices), -1)
    val_features_flat = diff_features_scaled[val_indices].reshape(len(val_indices), -1)
    
    # Check variance per class in validation set
    for cls in np.unique(y_enc[val_indices]):
        mask = y_enc[val_indices] == cls
        class_features = val_features_flat[mask]
        print(f"Class {cls}: {mask.sum()} samples, feature mean={class_features.mean():.4f}, std={class_features.std():.4f}")
    
    # Check if features are too similar across classes
    print(f"\nFeature statistics (all):")
    print(f"  Mean: {diff_features_scaled.mean():.4f}")
    print(f"  Std:  {diff_features_scaled.std():.4f}")
    print(f"  Min:  {diff_features_scaled.min():.4f}")
    print(f"  Max:  {diff_features_scaled.max():.4f}")
    
    # Check unique encoded labels
    print(f"\nUnique y_enc values: {np.unique(y_enc)}")
    print(f"Unique in train: {np.unique(y_enc[train_indices])}")
    print(f"Unique in val: {np.unique(y_enc[val_indices])}")
    print("=" * 40 + "\n")
    
    best_val_loss = 1e9
    best_val_acc = 0.0  # Track best accuracy
    best_state = None
    patience_counter = 0
    val_acc_hist = []
    val_loss_hist = []
    train_loss_hist = []

    for epoch in range(1, cfg.n_epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        n_seen = 0
        for xb, yb in train_dl: # Run model on every sample in training set
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion_train(out, yb) # Calculate loss
            opt.zero_grad()
            loss.backward()
            # Gradient clipping for training stability
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            train_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss /= max(1, n_seen)
        train_loss_hist.append(train_loss)

        # val
        model.eval()
        val_loss_weighted = 0.0
        val_loss_unweighted = 0.0
        n_seen = 0
        correct = 0
        all_probs = []
        all_trues = []
        all_losses = []  # Track batch losses for debugging
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device); yb = yb.to(device)
                out = model(xb)
                loss_weighted = criterion_train(out, yb)
                loss_unweighted = criterion_val(out, yb)
                val_loss_weighted += loss_weighted.item() * xb.size(0)
                val_loss_unweighted += loss_unweighted.item() * xb.size(0)
                n_seen += xb.size(0)  # THIS WAS MISSING
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                all_probs.append(torch.softmax(out, dim=1).cpu().numpy())
                all_trues.append(yb.cpu().numpy())
                all_losses.append(loss_unweighted.item())
        
        val_loss_weighted /= max(1, n_seen)
        val_loss_unweighted /= max(1, n_seen)
        val_acc = (correct / n_seen) if n_seen else 0.0  # NOW val_acc is computed
        val_loss_hist.append(val_loss_unweighted)
        val_acc_hist.append(val_acc)  # Track accuracy history
        
        # DEBUG: Print per-class accuracy periodically
        if epoch == 1 or epoch % 10 == 0:
            probs = np.concatenate(all_probs)
            trues = np.concatenate(all_trues)
            print(f"  Debug: correct={correct}, n_seen={n_seen}, val_acc={val_acc:.4f}")
            print(f"  Prediction distribution: {np.bincount(np.argmax(probs, axis=1), minlength=n_classes)}")
            print(f"  Ground truth distribution: {np.bincount(trues, minlength=n_classes)}")
            
            # Per-class accuracy
            from sklearn.metrics import classification_report
            preds = np.argmax(probs, axis=1)
            print("\nPer-class validation accuracy:")
            for cls in np.unique(trues):
                mask = trues == cls
                cls_acc = (preds[mask] == cls).mean()
                print(f"  Class {cls}: {cls_acc:.4f} ({mask.sum()} samples)")
        
        print(f"Epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss_unweighted:.4f} (weighted: {val_loss_weighted:.4f}) val_acc={val_acc:.4f}")
        
        # Early stopping based on unweighted loss
        if val_loss_unweighted < best_val_loss:
            best_val_loss = val_loss_unweighted
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  → New best val_acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch}. Best val_acc: {best_val_acc:.4f} (at epoch {epoch - patience_counter})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model with val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}")

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

    # Plot training curves showing overfitting pattern
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs_range = range(1, len(train_loss_hist) + 1)
    
    # Loss curves
    ax1.plot(epochs_range, train_loss_hist, 'b-', label='Train Loss')
    ax1.plot(epochs_range, val_loss_hist, 'r-', label='Val Loss')
    ax1.axvline(len(epochs_range) - patience_counter, color='g', linestyle='--', alpha=0.7, label='Best Model')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(r'Training \& Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curve
    ax2.plot(epochs_range, val_acc_hist, 'r-', label='Val Accuracy')
    ax2.axvline(len(epochs_range) - patience_counter, color='g', linestyle='--', alpha=0.7, label='Best Model')
    ax2.axhline(best_val_acc, color='g', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Validation Accuracy (Best: {best_val_acc:.3f})')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('../output_plots/fno_training_curves.pdf', transparent=True)
    print(f"Saved training curves to ../output_plots/fno_training_curves.pdf")
    plt.close()

    # Original accuracy history plot (now redundant but keep for compatibility)
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

###############################################################################################
def train_linear_regressor(diff_features_scaled: np.ndarray, y: np.ndarray,
                           cfg: TrainConfig, sensor_names: list[str],
                           model_save_ext: str,
                           scaler: StandardScaler,
                           n_classes: Optional[int] = None):
    
    le = LabelEncoder()
    y_enc = np.array(le.fit_transform(y))
    n_classes = n_classes or int(np.max(y_enc)) + 1

    # Use stratified split to ensure class balance
    from sklearn.model_selection import train_test_split
    N = len(diff_features_scaled)
    n_val = max(1, int(N * cfg.val_split))
    n_train = max(1, N - n_val)
    
    
    train_indices, val_indices = train_test_split(
        np.arange(N), test_size=cfg.val_split, stratify=y_enc, random_state=42
    )
    

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Flatten features for sklearn
    X_flat = diff_features_scaled.reshape(len(diff_features_scaled), -1)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_flat[train_indices], y_enc[train_indices])
    lr_acc = accuracy_score(y_enc[val_indices], lr.predict(X_flat[val_indices]))
    print(f"Linear classifier accuracy: {lr_acc:.4f}")

    # Save logistic regression model
    os.makedirs('../output_models', exist_ok=True)
    import joblib
    lr_model_path = f'../output_models/logistic_regression_n{model_save_ext}.pkl'
    joblib.dump({
        'model': lr, 
        'scaler': scaler, 
        'label_encoder': le, 
        'sensor_names': sensor_names
    }, lr_model_path, compress=3)
    print(f"Saved logistic regression model to {lr_model_path}")
###############################################################################################
###############################################################################################

# -------------------------
# Example usage
# -------------------------
def example_usage():
    """
    Example: cache dataset and train FNO classifier on 'n' labels.
    Adjust paths as needed.
    """
    # data_dir = "/home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/low_m-n_testing/new_Mirnov_set/"
    data_dir = "/home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/new_helicity_low_mn/"
    out_path = data_dir + "cached_data_-1.npz"
    geometry_shot = 1160930034#1160714026#1110316031# 1051202011# # for bp_k geometry; set None to skip
    model_save_ext = f'_Sensor_Reduced_{geometry_shot}'*True
    viz_save_path = '../output_plots/'

    # Cache or load
    result = build_or_load_cached_dataset(
        data_dir=data_dir, out_path=out_path, use_mode='n', include_geometry=True,
        geometry_shot=geometry_shot, num_timepoints=10, freq_tolerance=0.1, n_datasets=-1,
        load_saved_data=True, visualize_first=True,doVisualize=True,\
            viz_save_path=viz_save_path, saveDataset=False)
    X_ri, y, y_m, y_n, sensor_names, theta, phi = result


    # Compute difference features and fit scaler
    diff_features_scaled, scaler = fit_and_apply_scaler(X_ri, theta, phi)
    plot_scaled_features(diff_features_scaled, save_path=viz_save_path + 'scaled_features.pdf')

    # Train Linear Regressor as baseline (optional)

    # train_linear_regressor(diff_features_scaled, y,
    #                         cfg=TrainConfig(
    #                              batch_size=64,
    #                              val_split=0.2,
    #                              n_epochs=100,
    #                              lr=1e-3,
    #                              weight_decay=1e-4,
    #                              patience=15,
    #                              mask_prob=0.0,
    #                              modes=16,
    #                              width=128,
    #                              depth=4,
    #                              dropout=0.1,
    #                              label_smoothing=0.0,
    #                              num_workers=4,
    #                              grad_clip=1.0,
    #                              num_channels=diff_features_scaled.shape[2] # number of independent channels
    #                         ),
    #                         sensor_names=sensor_names,
    #                         model_save_ext=model_save_ext,
    #                         scaler=scaler,
    #                         n_classes=None)
    
    # Train FNO with improved hyperparameters for better generalization
    model, le, val_acc, auc = train_fno_classifier(
        diff_features_scaled, y, theta=theta, phi=phi, n_classes=None,
        cfg=TrainConfig(
            batch_size=32,  # Smaller batch for better generalization with small dataset
            n_epochs=200, 
            patience=50,  # Stop sooner when accuracy plateaus (was 40)
            modes=32,  # Slightly more modes
            width=160,  # Reasonable width
            depth=4, 
            dropout=0.35,  # Higher dropout for small dataset
            mask_prob=0.15,  # More aggressive masking
            label_smoothing=0.1,  # Label smoothing
            lr=5e-4,  # Lower learning rate
            weight_decay=1e-3,  # Stronger L2 regularization
            num_workers=4,  # Multi-core data loading
            grad_clip=1.0,  # Gradient clipping for stability
            num_channels=diff_features_scaled.shape[2] # number of independent channels
        ),
        plot_confusion=True, 
        class_names=np.arange(np.min(y), np.max(y)+1), 
        scaler=scaler)

    # Save model bundle and scaler
    os.makedirs('../output_models', exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'classes': le.classes_.tolist(), 'sensor_names': sensor_names,
                'scaler': scaler, 'theta': theta, 'phi': phi},\
                      f'../output_models/fno_classifier_n{model_save_ext}.pth')
    plt.show()
    print(f"Done. Val Acc={val_acc:.3f}, AUC={auc:.3f}")

    print(f"Saved trained model to ../output_models/fno_classifier_n{model_save_ext}.pth")


if __name__ == "__main__":
    print("FNO predictor: building or loading cached dataset and training...")
    example_usage()
