#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-network based regressor/classifier for mode numbers (m or n).
Input per sensor: [real, imag, theta, phi] -> shape (N_samples, N_sensors, 4)
Default model: 1D conv backbone over sensors -> pooling -> MLP classifier/regressor.
Uses PyTorch. Expects prepared arrays from existing data-prep pipeline:
  X_real, X_imag: (N, S) or (N, S, T) (this script assumes single-frequency vectors -> (N, S))
  theta, phi:    (S,) giving per-sensor directional coords (radians or normalized)
  y:             integer labels (m or n)
If you already have a `prepare_training_data` function in the workspace, use it to obtain arrays,
then call build_input_tensor(...) to construct the model input.
"""
from __future__ import annotations
import os
import sys
import math
import time
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt

# sys.path.append('../signal_generation/')

sys.path.append('../C-Mod/')
from mirnov_Probe_Geometry import Mirnov_Geometry as Mirnov_Geometry_C_Mod
from regression_phase import prepare_training_data, load_training_data 
from get_Cmod_Data import __loadData

# -------------------------
# Dataset / utils
# -------------------------
class SensorDataset(Dataset):
    """X: (N, S, 4), y: (N,) integer labels"""
    def __init__(self, X: np.ndarray, y: np.ndarray, device: Optional[torch.device]=None, mask_prob: float = 0.2, is_train: bool = True):
        assert X.ndim == 3 and X.shape[-1] == 4
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.device = device
        self.mask_prob = mask_prob  # Probability to zero out each sensor
        self.is_train = is_train    # Apply masking only during training

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        
        if self.is_train:
            # Existing masking
            mask = torch.rand(x.shape[0]) < self.mask_prob
            x[mask] = 0.0
            
            # Add Gaussian noise to real/imag channels (first 2 channels)
            noise_std = 0.1  # Tune this (e.g., 0.05-0.2)
            noise = torch.randn_like(x[:, :2]) * noise_std
            x[:, :2] += noise
            
            # Random global scaling (simulate amplitude variations)
            scale_factor = torch.rand(1) * 0.4 + 0.8  # 0.8-1.2 range
            x *= scale_factor
    
        if self.device:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

def build_input_tensor(X_real: np.ndarray, X_imag: np.ndarray,
                       theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Build input tensor (N, S, 4) from components.
    X_real, X_imag: (N, S)
    theta, phi: (S,) -- will be broadcast to (N, S)
    Returns float32 numpy array.
    """
    Xr = np.asarray(X_real)
    Xi = np.asarray(X_imag)
    th = np.asarray(theta)
    ph = np.asarray(phi)
    assert Xr.shape == Xi.shape, "real/imag shapes must match"
    N, S = Xr.shape
    assert th.shape[0] == S and ph.shape[0] == S
    th_mat = np.tile(th[np.newaxis, :], (N, 1))
    ph_mat = np.tile(ph[np.newaxis, :], (N, 1))
    X = np.stack([Xr, Xi, th_mat, ph_mat], axis=-1)  # (N, S, 4)
    return X.astype(np.float32)

# -------------------------
# Model
# -------------------------
class ConvSpatialNet(nn.Module):
    """
    1D-conv backbone over sensors to learn spatial patterns.
    Input: (B, S, C=4) -> transpose to (B, C, S) for conv1d.
    Now deeper with attention for better spatial focus.
    """
    def __init__(self, in_channels=4, hidden_channels=256, n_classes=10, dropout=0.3):  # Increased hidden_channels
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            # nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            # nn.BatchNorm1d(hidden_channels),
            # nn.ReLU(),
            # nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            # nn.BatchNorm1d(hidden_channels),
            # nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),  # Added extra conv layer for depth
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # pool over sensors -> (B, hidden, 1)
        )
        # self.attention = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=4, batch_first=True)  # Added attention
        self.head = nn.Sequential(  # MLP head, now wider
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),  # Wider intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, n_classes)
        )

    def forward(self, x):
        # x: (B, S, C)
        x = x.permute(0, 2, 1)  # -> (B, C, S)
        x = self.net(x)  # (B, hidden, 1)
        # x = x.squeeze(-1).unsqueeze(1)  # (B, 1, hidden) for attention
        # x, _ = self.attention(x, x, x)  # Apply attention
        x = x.squeeze(1)  # (B, hidden)
        return self.head(x)  # (B, n_classes)

# -------------------------
# Training / evaluation
# -------------------------
def train_model(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader,
                n_epochs=50, lr=1e-3, weight_decay=1e-5, device=None,
                patience=8, verbose=True):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)  # Added scheduler
    criterion = nn.CrossEntropyLoss()
    best_val = 1e9
    best_state = None
    counter = 0
    val_acc_list = []  # Track validation accuracy over epochs
    for epoch in range(1, n_epochs+1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for xb, yb in train_dl:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss /= n_seen

        # validation
        model.eval()
        val_loss = 0.0
        n_seen = 0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                n_seen += xb.size(0)
                pred_labels = pred.argmax(dim=1)
                correct += (pred_labels == yb).sum().item()
        val_loss /= n_seen
        val_acc = correct / n_seen
        val_acc_list.append(val_acc)  # Append to list

        scheduler.step()  # Step the scheduler after each epoch

        if verbose:
            print(f"Epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print("Early stopping")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, val_acc_list

def evaluate_model(model: nn.Module, dl: DataLoader, device=None, plot_confusion=False,\
                    class_names=None, do_n=True) -> Tuple[float, np.ndarray, np.ndarray, float]:
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    preds = []
    trues = []
    probs = []  # Collect probabilities for AUC
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            out = model(xb)
            pl = out.argmax(dim=1)
            preds.append(pl.cpu().numpy())
            trues.append(yb.cpu().numpy())
            probs.append(torch.softmax(out, dim=1).cpu().numpy())  # Softmax for probabilities
            correct += (pl == yb).sum().item()
            total += yb.size(0)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    probs = np.concatenate(probs)  # (total, n_classes)
    acc = correct / total
    
    # Compute multi-class AUC (one-vs-rest)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(trues, probs, multi_class='ovr', average='macro')
    
    if plot_confusion:
        from sklearn.metrics import ConfusionMatrixDisplay
        fig, ax = plt.subplots(1, 1, figsize=(5,4.5),tight_layout=True)
        ConfusionMatrixDisplay.from_predictions(trues, preds, ax=ax, display_labels=class_names)
        ax.set_title(f'Confusion Matrix (Acc={acc:.3f}, AUC={auc:.3f})')
        fig.savefig(f"../output_plots/confusion_matrix_acc_{acc:.3f}_auc_{auc:.3f}_{('n' if do_n else 'm')}.pdf", transparent=True)
        plt.show()
    
    return acc, preds, trues, auc  # Return AUC as well

# -------------------------
# High-level helper
# -------------------------
def fit_from_arrays(X: np.ndarray, y: np.ndarray, n_classes: Optional[int]=None,
                    batch_size=64, val_split=0.2, n_epochs=50, device=None,
                    model_kwargs: dict = {}, plot_confusion=False, do_n=True, mask_prob: float = 0.2):
    """
    X: (N, S, 4) float32, y: (N,) ints (raw m/n labels)
    mask_prob: Probability to randomly zero out each sensor during training (default 0.2)
    Returns trained model and label encoder
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = n_classes or int(y_enc.max()) + 1

    ds = SensorDataset(X, y_enc, device=None, mask_prob=mask_prob, is_train=True)  # Training dataset with masking
    N = len(ds)
    n_val = int(N * val_split)
    n_train = N - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    # Create separate val dataset without masking
    val_ds = SensorDataset(X[val_ds.indices], y_enc[val_ds.indices], device=None, mask_prob=0.0, is_train=False)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ConvSpatialNet(in_channels=4, hidden_channels=model_kwargs.get('hidden', 100),
                           n_classes=n_classes, dropout=model_kwargs.get('dropout', 0.3))
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model, val_acc_list = train_model(model, train_dl, val_dl, n_epochs=n_epochs, lr=model_kwargs.get('lr', 1e-4),
                        weight_decay=model_kwargs.get('wd', 1e-4), device=device,
                        patience=model_kwargs.get('patience', 8), verbose=True)
    val_acc, preds, trues, auc = evaluate_model(model, val_dl, device=device, plot_confusion=plot_confusion,
                                                 class_names=le.classes_, do_n=do_n)
    
    # Plot validation accuracy over epochs
    plt.figure(figsize=(5, 4.5))
    plt.plot(range(1, len(val_acc_list) + 1), val_acc_list, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Over Training Epochs')
    plt.grid(True)
    plt.savefig(f'../output_plots/validation_accuracy_over_epochs{("_n" if do_n else "_m")}.pdf', transparent=True)
    plt.show()
    
    return model, le, val_acc, auc

def plot_feature_stats(X: np.ndarray, sensor_names: list, do_n: bool = True, save_ext: str = ''):
    """
    Plot average and standard deviation of each feature channel (real, imag, theta, phi)
    across the dataset for each sensor, after standard scaling.
    X: (N, S, 4) scaled feature tensor
    sensor_names: list of sensor names (length S)
    """
    import matplotlib.pyplot as plt
    
    N, S, C = X.shape
    assert C == 4, "Expected 4 channels"
    channel_names = ['Real', 'Imag', 'Theta', 'Phi']
    
    plt.close('feature_stats')
    fig, axes = plt.subplots(2, 2, figsize=(18, 5), sharex=True,num='feature_stats')
    axes = axes.flatten()
    

    for c in range(C):
        means = X[:, :, c].mean(axis=0)  # (S,)
        if len(X) > 1: stds = X[:, :, c].std(axis=0)    # (S,)
        
        
        axes[c].plot(range(S), means, '-o', color='skyblue')
        if len(X) > 1:axes[c].fill_between(range(S), means - stds, means + stds, color='skyblue', alpha=0.5)
        axes[c].set_title(f'{channel_names[c]} Channel')

        if c==0:axes[c].set_ylabel(r'$||\mathbb{R}||\,\pm\,\sigma$')
        if c==1:axes[c].set_ylabel(r'$||\mathbb{I}||\,\pm\,\sigma$')
        if c==2:axes[c].set_ylabel(r'$\theta\,\pm\,\sigma$')
        if c==3:axes[c].set_ylabel(r'$\phi\,\pm\,\sigma$')

        axes[c].set_xticks(range(S))
        axes[c].set_xticklabels(sensor_names[1:], rotation=90, ha='right')
        axes[c].grid(True)

    plt.tight_layout()
    plt.savefig(f'../output_plots/feature_stats_after_scaling{("_n" if do_n else "_m")}{save_ext}.pdf', transparent=True)
    plt.show()

# -------------------------
# Example main usage
# -------------------------
def example_usage():
    """
    Example showing how to plug this into existing preparation code.
    Replace the placeholder calls with your data-prep functions.
    """
    # --- User must replace these lines with calls to prepare_training_data or equivalent ---
    
    # If you have an existing prepare_training_data function in workspace, use it:
    # It should return: X_real (N,S), X_imag (N,S), theta (S,), phi (S,), y (N,)  (y = m or n)
    # Where N = number of samples, S = number of sensors
    sensor_set = 'C_MOD_LIM'
    mesh_file = 'C_Mod_ThinCurr_Combined-homology.h5'
    data_dir = "/home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/low_m-n_testing/new_Mirnov_set/"
    cmod_shot = 1160714026#1110316018#1160930033#,1151208900 #1051202011,#
    r_maj = 0.68
    do_n=True
    save_ext = '_test_Dropout0.2_Hidden256_Mask0.1'

    # Load in the sensors and sensor locations desired for training
    # phi, theta_pol, R, Z = Mirnov_Geometry_C_Mod(cmod_shot)
    bp_k = __loadData(cmod_shot, pullData='bp_k', forceReload=['bp_k'] * False)['bp_k']

    R = {sensor_name.upper(): bp_k.R[ind] for ind, sensor_name in enumerate(bp_k.names)}
    Z = {sensor_name.upper(): bp_k.Z[ind] for ind, sensor_name in enumerate(bp_k.names)}
    phi = {sensor_name.upper(): bp_k.Phi[ind] for ind, sensor_name in enumerate(bp_k.names)}
    theta_dict = {sensor_name.upper(): np.arctan2(bp_k.Z[ind] - 0, bp_k.R[ind] - r_maj) for ind,sensor_name in enumerate(bp_k.names)}    # 

    # phi = np.array(phi)
    # theta_pol = np.array(theta_pol)
    forceDataReload=False

    theta_dict = {sensor_name: np.arctan2(Z[sensor_name], R[sensor_name]-r_maj) for sensor_name in R}

    X, y_m, y_n, sensor_names,scaler = load_training_data(sensor_set, mesh_file, data_dir, \
            theta=theta_dict, phi=phi,n_files=1,forceDataReload=forceDataReload)
    # Xr, Xi, theta, phi = X
    y = y_n  if do_n else y_m# choose m or n labels
    
    # Plot feature stats after scaling
    plot_feature_stats(X, sensor_names, do_n=do_n, save_ext='_normalized_training')

    # X = build_input_tensor(Xr, Xi, theta, phi)  # (N, S, 4)
    model, le, val_acc, auc = fit_from_arrays(X, y, n_epochs=1000, \
        model_kwargs={'hidden':128, 'dropout':0.4, 'patience':200,'wd':5e-4,'lr':1e-4}, \
        plot_confusion=True, do_n=do_n, mask_prob=0.05)  # Add mask_prob
    print(f"Validation accuracy: {val_acc}, AUC: {auc}")
   
    # save model and label encoder
    save_fName = data_dir+f"trained_mode_classifier_C-Mod_Sensors_Shot_{cmod_shot}"+\
        f"{('_n' if do_n else '_m')}{save_ext}.pth"
    torch.save({'model_state': model.state_dict(), 'label_classes': le.classes_.tolist(),\
                'sensor_names': sensor_names, 'scaler': scaler}, save_fName
               )
    
    print('Saved trained model to: ', save_fName)
    print('Finished example usage.')
if __name__ == "__main__":
    print("This module provides fit_from_arrays(...) to train a spatial NN classifier.")
    print("Run example_usage() after wiring in your data-prep function.")
    example_usage()
    print("Done.")