#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic: Visualize feature space separability using t-SNE.

This script loads cached training data and creates a 2D t-SNE visualization
to assess whether your 9 mode classes are truly separable in feature space.

Run with: python diagnostic_feature_separability.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.ion()
from sklearn.manifold import TSNE
import sys
import os
from sklearn.preprocessing import LabelEncoder

# Add paths for imports
sys.path.insert(0, '../C-Mod/')
sys.path.insert(0, '.')

from data_caching import build_or_load_cached_dataset
from fno_predictor import fit_and_apply_scaler

def run_separability_diagnostic():
    """Load data and visualize class separability in feature space."""
    
    # Configuration (match your training config)
    use_mode = 'mn'
    n_datasets = -1
    num_timepoints = 40
    geometry_shot = 1160930034
    data_dir = "/home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/new_helicity_high_mn/"
    out_path = data_dir + f"cached_data_{geometry_shot}_{use_mode}_{num_timepoints}_{n_datasets}.npz"
    saveExt = f"_{use_mode}_{num_timepoints}_{n_datasets}_high_mn"
    print("=" * 70)
    print("DIAGNOSTIC: Feature Space Separability Analysis")
    print("=" * 70)
    out_path_features = '../output_plots/tsne_features_high_mn.npz'

    le = LabelEncoder()
    if os.path.exists(out_path_features):
        print(f"Using existing t-SNE features from: {out_path_features}")
        data = np.load(out_path_features)
        features_2d = data['features_2d']
        y = data['y']
        y_m = data['y_m']
        y_n = data['y_n']
        y_enc = np.array(le.fit_transform(y))
    else:
        # Load data
        print(f"\n1. Loading dataset from cache...")
        result = build_or_load_cached_dataset(
            data_dir=data_dir, out_path=out_path, use_mode=use_mode, include_geometry=True,
            geometry_shot=geometry_shot, num_timepoints=num_timepoints, freq_tolerance=0.1, 
            n_datasets=n_datasets, load_saved_data=True, visualize_first=False,
            doVisualize=False, saveDataset=False)
        X_ri, y, y_m, y_n, sensor_names, theta, phi = result
        
        
        y_enc = np.array(le.fit_transform(y))

        print(f"   Data shape: X_ri={X_ri.shape}, y={y.shape}")
        print(f"   Classes: {np.unique(y)}")
        print(f"   Class counts: {np.bincount(y_enc)}")
        
        # Compute difference features
        print(f"\n2. Computing scaled features...")
        diff_features_scaled, scaler = fit_and_apply_scaler(X_ri, theta, phi)
        print(f"   Features shape: {diff_features_scaled.shape}")
        print(f"   Feature stats: mean={diff_features_scaled.mean():.4f}, std={diff_features_scaled.std():.4f}")
        
        # Flatten for t-SNE
        print(f"\n3. Flattening features for t-SNE...")
        features_flat = diff_features_scaled.reshape(len(diff_features_scaled), -1)
        print(f"   Flattened shape: {features_flat.shape}")
        
        # Compute t-SNE
        print(f"\n4. Computing t-SNE (this may take 2-5 minutes)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=100, max_iter=1000, verbose=1)
        features_2d = tsne.fit_transform(features_flat)
        print(f"   t-SNE complete: shape={features_2d.shape}")
        np.savez(out_path_features, features_2d=features_2d, y=y, y_m=y_m, y_n=y_n)

    # Create visualization
    print(f"\n5. Creating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True,tight_layout=True)
    
    # Plot 1: Colored by class
    cmap = plt.get_cmap('tab10')
    bounds = np.arange(len(le.classes_)+1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, len(le.classes_))
    s_map = plt.cm.ScalarMappable(cmap='tab10', norm=norm)
    s_map.set_array([])
    
    
    for ind, mode in enumerate(le.classes_):
        mask = y == mode
        # if int(mode[0]) != 4: continue 
        scatter1 = ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], label=mode, s=20,\
                     alpha=0.6 - 0.5 * (ind/(len(le.classes_)-1)),\
                          edgecolors='none', \
                            c=ind*np.ones_like(features_2d[mask, 0]),cmap=cmap, norm=norm)
        
        # print(f"Data range: X: {np.min(features_2d[mask, 0]):.2f} to {np.max(features_2d[mask, 0]):.2f},'+\
        #       f' Y: {np.min(features_2d[mask, 1]):.2f} to {np.max(features_2d[mask, 1]):.2f}, "+\
        #         f"ind = {ind}, mode={mode}, count={mask.sum()}")

    cbar1 = fig.colorbar(s_map, ax=ax1, label='Mode Number')
    cbar1.set_ticks(np.arange(len(le.classes_)))
    cbar1.set_ticklabels(le.classes_)
    cbar1.solids.set_alpha(0.8)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_title('Feature Space (colored by mode number)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Colored by m (poloidal)
    bounds = np.arange(len(np.unique(y_m)) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, len(np.unique(y_m)))
    s_map = plt.cm.ScalarMappable(cmap='tab20', norm=norm)
    for ind, m in enumerate(np.unique(y_m)):
        mask = y_m == m
        scatter2 = ax2.scatter(features_2d[mask, 0], features_2d[mask, 1], cmap='tab20', c=ind*np.ones_like(features_2d[mask, 0]),\
                          s=20, alpha=0.6 - 0.5 * (ind/(len(y_m)-1)), edgecolors='none', norm=norm)
    # overplot the m=1 case for visibility
    mask_m1 = y_m == 1
    ax2.scatter(features_2d[mask_m1, 0], features_2d[mask_m1, 1], color=plt.get_cmap('tab20')(0),\
                 s=30, alpha=0.05, edgecolors='none')
    cbar2 = plt.colorbar(s_map, ax=ax2, label='Poloidal Mode (m)')
    cbar2.set_ticks(np.arange(len(np.unique(y_m))))
    cbar2.set_ticklabels(np.unique(y_m))
    cbar2.solids.set_alpha(0.8)
    ax2.set_xlabel('t-SNE 1')
    # ax2.set_ylabel('t-SNE 2')
    ax2.set_title('Feature Space (colored by poloidal mode m)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../output_plots/feature_separability_tsne{saveExt}.pdf', transparent=True, bbox_inches='tight')
    print(f"   Saved to: ../output_plots/feature_separability_tsne{saveExt}.pdf")
    
    # Compute class statistics in 2D space
    print(f"\n6. Computing class statistics in t-SNE space...")
    class_centers = {}
    class_spreads = {}
    for cls in np.unique(y):
        mask = y == cls
        center = features_2d[mask].mean(axis=0)
        spread = features_2d[mask].std(axis=0)
        class_centers[cls] = center
        class_spreads[cls] = spread
        print(f"   Class {cls}: center=({center[0]:.2f}, {center[1]:.2f}), spread=({spread[0]:.2f}, {spread[1]:.2f})")
    
    # Compute pairwise distances between class centers
    print(f"\n7. Computing pairwise class center distances...")
    n_classes = len(np.unique(y))
    distances = np.zeros((n_classes, n_classes))
    for i, cls_i in enumerate(np.unique(y)):
        for j, cls_j in enumerate(np.unique(y)):
            distances[i, j] = np.linalg.norm(class_centers[cls_i] - class_centers[cls_j])
    
    print(f"\n   Class center distances (average separation):")
    for i, cls_i in enumerate(np.unique(y)):
        avg_dist = distances[i, distances[i] > 0].mean()
        print(f"   Class {cls_i}: avg distance to other classes = {avg_dist:.3f}")
    
    # Separability score: ratio of between-class to within-class distances
    print(f"\n8. Separability Analysis:")
    between_class_dist = distances[distances > 0].mean()
    within_class_spread = np.array([class_spreads[c].mean() for c in np.unique(y)]).mean()
    separability_score = between_class_dist / (within_class_spread + 1e-6)
    
    print(f"   Average between-class distance: {between_class_dist:.3f}")
    print(f"   Average within-class spread: {within_class_spread:.3f}")
    print(f"   Separability score (higher=better): {separability_score:.3f}")
    
    if separability_score > 2.0:
        print(f"   ✓ Classes ARE well-separated → Model should learn this")
    elif separability_score > 1.0:
        print(f"   ⚠ Classes are moderately separated → Expect ~70-85% accuracy max")
    else:
        print(f"   ✗ Classes are poorly separated → Hard to classify, may need better features")
    
    print(f"\n" + "=" * 70)
    print(f"Diagnostic complete. Check ../output_plots/feature_separability_tsne{saveExt}.pdf")
    print("=" * 70)
    
    plt.show()

    print('Finished')

if __name__ == "__main__":
    run_separability_diagnostic()
