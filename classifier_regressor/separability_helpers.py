#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for separability diagnostics.

Includes:
  - Pure-math metrics: compute_overlap_matrix, compute_pairwise_linear_probe_auc
  - Mode-label utilities: _parse_mode_label, _normalise_mode_label, _sort_mode_labels
  - Cache-path utilities: _geometry_tag, _shared_cache_paths
  - Geometry proxy utility: find_proxy_sensors
  - Full separability workflow: load_and_compute_separability
  - Plotting helpers: plot_tsne_separability, plot_detectability_bar_chart,
      plot_auc_detectability_bar_chart, _fit_pairwise_lr_in_2d_pca,
      plot_pairwise_linear_boundaries

Imported by diagnostic_feature_separability.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

from data_caching import build_or_load_cached_dataset
from fno_predictor import fit_and_apply_scaler


# ──────────────────────────────────────────────────────────────────────────────
# Mode-label utilities
# ──────────────────────────────────────────────────────────────────────────────

def _parse_mode_label(mode_label):
    """Parse mode label formatted as 'm/n' into integer tuple; return None if unavailable."""
    try:
        m_val, n_val = str(mode_label).split('/')
        return int(m_val), int(n_val)
    except Exception:
        return None


def _normalise_mode_label(entry):
    """Convert a mode entry to the canonical 'm/n' string.

    Accepts either a string ``'m/n'`` or a tuple/list ``(m, n)``.
    """
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        return f'{int(entry[0])}/{int(entry[1])}'
    return str(entry)


def _sort_mode_labels(mode_labels):
    """Sort mode labels by toroidal n then poloidal m for consistent plotting order."""
    def _key(label):
        parsed = _parse_mode_label(label)
        if parsed is None:
            return 10**9, 10**9, str(label)
        m_val, n_val = parsed
        return n_val, m_val, str(label)

    return sorted(mode_labels, key=_key)


# ──────────────────────────────────────────────────────────────────────────────
# Cache-path utilities
# ──────────────────────────────────────────────────────────────────────────────

def _geometry_tag(geometry_shot):
    """Compact string identifier for geometry selection used in cache/output names."""
    if isinstance(geometry_shot, list):
        if len(geometry_shot) == 1:
            base = str(geometry_shot[0])
        else:
            base = f"list{len(geometry_shot)}_{'-'.join([str(v) for v in geometry_shot[:3]])}"
    else:
        base = str(geometry_shot)
    for ch in [' ', '/', ':', '[', ']', "'", '"', ',']:
        base = base.replace(ch, '_')
    return base


def _shared_cache_paths(data_dir, geometry_shot, use_mode, num_timepoints,
                        n_datasets, include_frequency_gap):
    """Return shared dataset/feature cache paths used by both single and multi workflows."""
    cfg_tag = _geometry_tag(geometry_shot)
    fgap_tag = 'fgap_on' if include_frequency_gap else 'fgap_off'

    out_path = os.path.join(
        data_dir,
        f"cached_data_{cfg_tag}_{fgap_tag}_{use_mode}_{num_timepoints}_{n_datasets}.npz",
    )
    out_path_features = os.path.join(
        '../output_plots',
        f"tsne_features_{cfg_tag}_{fgap_tag}_{use_mode}_{num_timepoints}_{n_datasets}.npz",
    )
    save_ext = f"_{use_mode}_{num_timepoints}_{n_datasets}_{cfg_tag}_{fgap_tag}"
    return out_path, out_path_features, save_ext


# ──────────────────────────────────────────────────────────────────────────────
# Geometry proxy utility
# ──────────────────────────────────────────────────────────────────────────────

def find_proxy_sensors(old_suffix, new_suffix,
                       old_json='../signal_generation/input_data/MAGX_Coordinates_CFS_Old_HV.json',
                       new_json='../signal_generation/input_data/MAGX_Coordinates_CFS.json'):
    """Return a deduplicated list of *new* sensor names that best approximate the
    geometry of *old* sensors identified by suffix letter (e.g. 'H' or 'V').

    For each old sensor matching ``_<old_suffix><digit>``, the nearest new sensor
    whose name contains ``_<new_suffix>`` is selected using Euclidean distance in
    scaled (phi_arc, Z, R) space.  Duplicate new-sensor assignments are collapsed,
    so the returned list has at most as many entries as there are distinct new sensors.

    Parameters
    ----------
    old_suffix : str
        Single letter identifying old sensors, e.g. ``'H'`` or ``'V'``.
    new_suffix : str
        Single letter identifying new sensors, e.g. ``'T'`` or ``'P'``.
    old_json, new_json : str
        Paths to the geometry JSON files (relative to this file's directory).

    Returns
    -------
    list of str
        Sorted, deduplicated new sensor names forming the proxy set.
    """
    import json as _json
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, old_json)) as _f:
        _old = _json.load(_f)['MRNV']
    with open(os.path.join(_here, new_json)) as _f:
        _new = _json.load(_f)['MRNV']

    def _c(d):
        return np.array([d['PHI'] * np.pi / 180, d['Z'], d['R']])

    old_sel = {k: v for k, v in _old.items() if f'_{old_suffix}' in k}
    new_sel_names = [n for n in _new if f'_{new_suffix}' in n]
    new_sel_coords = np.array([_c(_new[n]) for n in new_sel_names])

    # Scale dimensions: arc-length for phi, raw Z, down-weight small R variation
    mean_R = np.mean([v['R'] for v in old_sel.values()])
    scale = np.array([mean_R, 1.0, 5.0])

    matched = set()
    for ocoord_dict in old_sel.values():
        oc = _c(ocoord_dict) * scale
        dists = np.linalg.norm((new_sel_coords * scale) - oc, axis=1)
        matched.add(new_sel_names[int(np.argmin(dists))])

    return sorted(matched)


# ──────────────────────────────────────────────────────────────────────────────
# Pure-math metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_overlap_matrix(features_2d, y, classes, n_bins=60, padding_frac=0.05):
    """Compute pairwise overlap coefficient in 2D t-SNE space using normalized histograms."""
    x = features_2d[:, 0]
    y2 = features_2d[:, 1]
    x_pad = (x.max() - x.min()) * padding_frac + 1e-9
    y_pad = (y2.max() - y2.min()) * padding_frac + 1e-9
    x_edges = np.linspace(x.min() - x_pad, x.max() + x_pad, n_bins + 1)
    y_edges = np.linspace(y2.min() - y_pad, y2.max() + y_pad, n_bins + 1)

    hists = {}
    for cls in classes:
        mask = y == cls
        hist, _, _ = np.histogram2d(
            features_2d[mask, 0], features_2d[mask, 1], bins=[x_edges, y_edges]
        )
        hist = hist.astype(np.float64)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
        hists[cls] = hist

    n_classes = len(classes)
    overlap = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i, cls_i in enumerate(classes):
        overlap[i, i] = 1.0
        for j in range(i + 1, n_classes):
            cls_j = classes[j]
            prod = hists[cls_i] * hists[cls_j]
            overlap[i, j] = np.sum(hists[cls_i] * np.logical_not(prod))
            overlap[j, i] = np.sum(hists[cls_j] * np.logical_not(prod))

    return overlap


def compute_pairwise_linear_probe_auc(features_flat, y, classes, cv_splits=5, random_state=42):
    """Compute pairwise linear-probe AUC in original feature space.

    Returns
    -------
    auc_matrix : np.ndarray
        Symmetric matrix with pairwise mean cross-validated ROC-AUC.
    detectability_auc : dict
        Per-label detectability in [0, 1], defined as worst-pair transformed AUC:
        detectability = max(0, min(1, 2*(AUC - 0.5))).
    worst_auc_partners : dict
        Label with lowest pairwise AUC for each class.
    """
    n_classes = len(classes)
    auc_matrix = np.ones((n_classes, n_classes), dtype=np.float64)

    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            cls_i = classes[i]
            cls_j = classes[j]
            mask = (y == cls_i) | (y == cls_j)
            X_pair = features_flat[mask]
            y_pair = (y[mask] == cls_j).astype(np.int32)

            class_counts = np.bincount(y_pair)
            if len(class_counts) < 2 or np.min(class_counts) < 2:
                auc = 0.5
            else:
                n_splits_eff = min(cv_splits, int(np.min(class_counts)))
                if n_splits_eff < 2:
                    auc = 0.5
                else:
                    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True,
                                         random_state=random_state)
                    fold_aucs = []
                    for train_idx, test_idx in skf.split(X_pair, y_pair):
                        model = LogisticRegression(max_iter=3000, class_weight='balanced')
                        model.fit(X_pair[train_idx], y_pair[train_idx])
                        y_prob = model.predict_proba(X_pair[test_idx])[:, 1]
                        fold_auc = roc_auc_score(y_pair[test_idx], y_prob)
                        fold_auc = max(fold_auc, 1.0 - fold_auc)
                        fold_aucs.append(fold_auc)
                    auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5

            auc_matrix[i, j] = auc
            auc_matrix[j, i] = auc

    detectability_auc = {}
    worst_auc_partners = {}
    for i, cls_i in enumerate(classes):
        if n_classes <= 1:
            detectability_auc[cls_i] = 1.0
            worst_auc_partners[cls_i] = None
            continue

        aucs_i = auc_matrix[i].copy()
        aucs_i[i] = np.inf
        worst_j = int(np.argmin(aucs_i))
        worst_auc = auc_matrix[i, worst_j]
        detectability_auc[cls_i] = float(np.clip(2.0 * (worst_auc - 0.5), 0.0, 1.0))
        worst_auc_partners[cls_i] = classes[worst_j]

    return auc_matrix, detectability_auc, worst_auc_partners


# ──────────────────────────────────────────────────────────────────────────────
# Full separability workflow (t-SNE based)
# ──────────────────────────────────────────────────────────────────────────────

def load_and_compute_separability(data_dir, out_path, out_path_features, saveExt,
                                  geometry_shot, use_mode, num_timepoints, n_datasets,
                                  forceReload, include_frequency_gap=True,
                                  allow_rebuild_missing_cache=False,
                                  desired_mode_labels=None):
    """Load data, compute t-SNE features, and evaluate separability metrics."""
    print("=" * 70)
    print("DIAGNOSTIC: Feature Space Separability Analysis")
    print("=" * 70)

    has_feature_cache = os.path.exists(out_path_features)
    has_dataset_cache = os.path.exists(out_path)
    if (not forceReload) and (not has_feature_cache) and (not has_dataset_cache) and (not allow_rebuild_missing_cache):
        raise FileNotFoundError(
            "No cache available and implicit rebuild is disabled. "
            f"Missing feature cache: {out_path_features} and dataset cache: {out_path}. "
            "Set forceReload=True or allow_rebuild_missing_cache=True to rebuild explicitly."
        )

    le = LabelEncoder()
    if has_feature_cache and not forceReload:
        print(f"Using existing t-SNE features from: {out_path_features}")
        data = np.load(out_path_features)
        features_2d = data['features_2d']
        y = data['y']
        y_m = data['y_m']
        y_n = data['y_n']
        y_enc = np.array(le.fit_transform(y))
        features_flat = data['features_flat'] if 'features_flat' in data.files else None

        if features_flat is None:
            print("   Cached t-SNE file missing flattened features; recomputing feature vectors...")
            if (not has_dataset_cache) and (not allow_rebuild_missing_cache):
                raise FileNotFoundError(
                    "t-SNE cache is missing flattened features and dataset cache is unavailable. "
                    f"Missing dataset cache: {out_path}. "
                    "Set forceReload=True or allow_rebuild_missing_cache=True to rebuild explicitly."
                )
            result = build_or_load_cached_dataset(
                data_dir=data_dir, out_path=out_path, use_mode=use_mode, include_geometry=True,
                geometry_shot=geometry_shot, num_timepoints=num_timepoints, freq_tolerance=0.1,
                n_datasets=n_datasets, load_saved_data=True, visualize_first=False,
                doVisualize=False, saveDataset=False, include_frequency_gap=include_frequency_gap,
                desired_mode_labels=desired_mode_labels)
            X_ri, y, y_m, y_n, sensor_names, theta, phi = result
            y_enc = np.array(le.fit_transform(y))
            diff_features_scaled, _ = fit_and_apply_scaler(X_ri, theta, phi,
                                                           do_fgap=include_frequency_gap)
            features_flat = diff_features_scaled.reshape(len(diff_features_scaled), -1)
            if desired_mode_labels is None:
                np.savez(out_path_features, features_2d=features_2d,
                         features_flat=features_flat, y=y, y_m=y_m, y_n=y_n)
                print(f"Saved updated features with flattened data to: {out_path_features}")
    else:
        print(f"\n1. Loading dataset from cache...")
        if (not has_dataset_cache) and (not forceReload) and (not allow_rebuild_missing_cache):
            raise FileNotFoundError(
                "Dataset cache missing and implicit rebuild is disabled. "
                f"Missing dataset cache: {out_path}. "
                "Set forceReload=True or allow_rebuild_missing_cache=True to rebuild explicitly."
            )
        result = build_or_load_cached_dataset(
            data_dir=data_dir, out_path=out_path, use_mode=use_mode, include_geometry=True,
            geometry_shot=geometry_shot, num_timepoints=num_timepoints, freq_tolerance=0.1,
            n_datasets=n_datasets, load_saved_data=True, visualize_first=False,
            doVisualize=False,
            saveDataset=((forceReload or allow_rebuild_missing_cache) and (desired_mode_labels is None)),
            include_frequency_gap=include_frequency_gap,
            desired_mode_labels=desired_mode_labels)
        X_ri, y, y_m, y_n, sensor_names, theta, phi = result

        y_enc = np.array(le.fit_transform(y))
        print(f"   Data shape: X_ri={X_ri.shape}, y={y.shape}")
        print(f"   Classes: {np.unique(y)}")
        print(f"   Class counts: {np.bincount(y_enc)}")

        print(f"\n2. Computing scaled features...")
        diff_features_scaled, _ = fit_and_apply_scaler(X_ri, theta, phi,
                                                       do_fgap=include_frequency_gap)
        print(f"   Features shape: {diff_features_scaled.shape}")
        print(f"   Feature stats: mean={diff_features_scaled.mean():.4f}, std={diff_features_scaled.std():.4f}")

        print(f"\n3. Flattening features for t-SNE...")
        features_flat = diff_features_scaled.reshape(len(diff_features_scaled), -1)
        print(f"   Flattened shape: {features_flat.shape}")

        print(f"\n4. Computing t-SNE (this may take 2-5 minutes)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=100, max_iter=1000, verbose=1)
        features_2d = tsne.fit_transform(features_flat)
        print(f"   t-SNE complete: shape={features_2d.shape}")
        if desired_mode_labels is None:
            np.savez(out_path_features, features_2d=features_2d,
                     features_flat=features_flat, y=y, y_m=y_m, y_n=y_n)

    classes = le.classes_
    print(f"\n5. Computing class statistics in t-SNE space...")
    class_centers = {}
    class_spreads = {}
    for cls in classes:
        mask = y == cls
        center = features_2d[mask].mean(axis=0)
        spread = features_2d[mask].std(axis=0)
        class_centers[cls] = center
        class_spreads[cls] = spread
        print(f"   Class {cls}: center=({center[0]:.2f}, {center[1]:.2f}), "
              f"spread=({spread[0]:.2f}, {spread[1]:.2f})")

    print(f"\n6. Computing pairwise class center distances...")
    n_classes = len(classes)
    distances = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i, cls_i in enumerate(classes):
        for j, cls_j in enumerate(classes):
            distances[i, j] = np.linalg.norm(class_centers[cls_i] - class_centers[cls_j])
    for i, cls_i in enumerate(classes):
        row = np.delete(distances[i], i)
        print(f"   Class {cls_i}: avg distance to other classes = {row.mean() if len(row) else 0.0:.3f}")

    print(f"\n7. Separability Analysis:")
    between_class_dist = distances[~np.eye(n_classes, dtype=bool)].mean() if n_classes > 1 else 0.0
    within_class_spread = np.array([class_spreads[c].mean() for c in classes]).mean()
    separability_score = between_class_dist / (within_class_spread + 1e-6)
    print(f"   Average between-class distance: {between_class_dist:.3f}")
    print(f"   Average within-class spread: {within_class_spread:.3f}")
    print(f"   Separability score (higher=better): {separability_score:.3f}")
    if separability_score > 2.0:
        print("   ✓ Classes ARE well-separated")
    elif separability_score > 1.0:
        print("   ⚠ Classes are moderately separated")
    else:
        print("   ✗ Classes are poorly separated")

    print(f"\n8. Computing overlap-based detectability in t-SNE space...")
    overlap_matrix = compute_overlap_matrix(features_2d, y, classes)
    detectability_scores = {}
    worst_overlap_partners = {}
    for i, cls_i in enumerate(classes):
        if n_classes <= 1:
            detectability_scores[cls_i] = 1.0
            worst_overlap_partners[cls_i] = None
            continue
        overlaps_i = overlap_matrix[i].copy()
        worst_j = int(np.argmin(overlaps_i))
        detectability_scores[cls_i] = overlap_matrix[i, worst_j]
        worst_overlap_partners[cls_i] = classes[worst_j]
    for cls_i in classes:
        partner = worst_overlap_partners[cls_i]
        if partner is None:
            print(f"   Class {cls_i}: detectability=1.000 (single-class dataset)")
        else:
            i = np.where(classes == cls_i)[0][0]
            j = np.where(classes == partner)[0][0]
            print(f"   Class {cls_i}: detectability={detectability_scores[cls_i]:.3f}, "
                  f"min overlap={overlap_matrix[i, j]:.3f} with class {partner}")

    print(f"\n9. Computing linear-probe (pairwise ROC-AUC) detectability...")
    auc_matrix, detectability_auc, worst_auc_partners = compute_pairwise_linear_probe_auc(
        features_flat, y, classes
    )
    for cls_i in classes:
        partner = worst_auc_partners[cls_i]
        if partner is None:
            print(f"   Class {cls_i}: AUC-detectability=1.000 (single-class dataset)")
        else:
            i = np.where(classes == cls_i)[0][0]
            j = np.where(classes == partner)[0][0]
            print(f"   Class {cls_i}: AUC-detectability={detectability_auc[cls_i]:.3f}, "
                  f"worst-pair AUC={auc_matrix[i, j]:.3f} vs class {partner}")

    return {
        'features_2d': features_2d,
        'features_flat': features_flat,
        'y': y,
        'y_m': y_m,
        'y_n': y_n,
        'label_encoder': le,
        'classes': classes,
        'class_centers': class_centers,
        'class_spreads': class_spreads,
        'distances': distances,
        'separability_score': separability_score,
        'between_class_dist': between_class_dist,
        'within_class_spread': within_class_spread,
        'overlap_matrix': overlap_matrix,
        'detectability_scores': detectability_scores,
        'worst_overlap_partners': worst_overlap_partners,
        'auc_matrix': auc_matrix,
        'detectability_auc': detectability_auc,
        'worst_auc_partners': worst_auc_partners,
        'saveExt': saveExt,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_tsne_separability(results, doSave=True):
    """Plot t-SNE visualization colored by class label and poloidal mode."""
    features_2d = results['features_2d']
    y = results['y']
    y_m = results['y_m']
    classes = results['classes']
    saveExt = results['saveExt']

    print(f"\n9. Creating t-SNE visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True,
                                    tight_layout=True)

    classes_int = np.array([c.split('/') for c in classes], dtype=int)
    sorted_indices = np.argsort(classes_int[:, 1] * 100 + classes_int[:, 0])
    classes_sorted = classes_int[sorted_indices]

    cmap = 'tab20' if len(classes) <= 20 else 'turbo'
    bounds = np.arange(len(classes) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, len(classes))
    s_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    s_map.set_array([])
    class_alpha_denom = max(1, len(classes) - 1)

    for ind, mode in enumerate(classes_sorted):
        mask = y == f'{mode[0]}/{mode[1]}'
        ax1.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            label=f'{mode[0]}/{mode[1]}', s=20,
            alpha=0.6 - 0.5 * (ind / class_alpha_denom),
            edgecolors='none',
            c=ind * np.ones_like(features_2d[mask, 0]),
            cmap=cmap, norm=norm,
        )

    cbar1 = fig.colorbar(s_map, ax=ax1, label='Mode Number')
    cbar1.set_ticks(np.arange(len(classes)))
    cbar1.set_ticklabels([f'{cls[0]}/{cls[1]}' for cls in classes_sorted])
    cbar1.solids.set_alpha(0.8)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_title('Feature Space (colored by mode number)')
    ax1.grid(True, alpha=0.3)

    unique_y_m = np.unique(y_m)
    bounds_m = np.arange(len(unique_y_m) + 1) - 0.5
    norm_m = mcolors.BoundaryNorm(bounds_m, len(unique_y_m))
    cmap_m = 'tab20' if len(unique_y_m) <= 20 else 'turbo'
    s_map_m = plt.cm.ScalarMappable(cmap=cmap_m, norm=norm_m)
    m_alpha_denom = max(1, len(unique_y_m) - 1)
    for ind, m in enumerate(unique_y_m):
        mask = y_m == m
        ax2.scatter(
            features_2d[mask, 0], features_2d[mask, 1], cmap=cmap_m,
            c=ind * np.ones_like(features_2d[mask, 0]), s=20,
            alpha=0.6 - 0.5 * (ind / m_alpha_denom), edgecolors='none', norm=norm_m,
        )
    cbar2 = plt.colorbar(s_map_m, ax=ax2, label='Poloidal Mode (m)')
    cbar2.set_ticks(np.arange(len(unique_y_m)))
    cbar2.set_ticklabels(unique_y_m)
    cbar2.solids.set_alpha(0.8)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_title('Feature Space (colored by poloidal mode m)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if doSave:
        out_path = f'../output_plots/feature_separability_tsne{saveExt}.pdf'
        plt.savefig(out_path, transparent=True, bbox_inches='tight')
        print(f"   Saved to: {out_path}")


def plot_detectability_bar_chart(results, doSave):
    """Plot per-label detectability score based on worst pairwise t-SNE overlap."""
    classes = results['classes']
    detectability_scores = results['detectability_scores']
    worst_overlap_partners = results['worst_overlap_partners']
    overlap_matrix = results['overlap_matrix']
    saveExt = results['saveExt']

    print(f"\n10. Creating overlap-based detectability bar chart...")
    scores = np.array([detectability_scores[cls] for cls in classes], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True)
    ax.bar(np.arange(len(classes)), scores,
           color=plt.get_cmap('tab10')(np.arange(len(classes)) % 10), alpha=0.85)

    for i, cls_i in enumerate(classes):
        partner = worst_overlap_partners[cls_i]
        if partner is None:
            label_txt = 'single class'
        else:
            j = np.where(classes == partner)[0][0]
            overlap_val = overlap_matrix[i, j]
            label_txt = f"max ovlp {partner}: {overlap_val:.2f}"
        ax.text(i, min(0.98, scores[i] + 0.03), label_txt, rotation=90,
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel('Detectability score (1 - max overlap)')
    ax.set_xlabel('Mode label')
    ax.set_title('Per-label detectability from t-SNE overlap')
    ax.grid(True, axis='y', alpha=0.3)

    if doSave:
        out_path = f'../output_plots/feature_detectability_bar{saveExt}.pdf'
        plt.savefig(out_path, transparent=True, bbox_inches='tight')
        print(f"   Saved to: {out_path}")


def plot_auc_detectability_bar_chart(results, doSave=True):
    """Plot per-label detectability from worst-pair linear-probe ROC-AUC."""
    classes = results['classes']
    detectability_auc = results['detectability_auc']
    worst_auc_partners = results['worst_auc_partners']
    auc_matrix = results['auc_matrix']
    saveExt = results['saveExt']

    print(f"\n11. Creating linear-probe AUC detectability bar chart...")
    scores = np.array([detectability_auc[cls] for cls in classes], dtype=np.float64)

    classes_int = np.array([c.split('/') for c in classes], dtype=int)
    sorted_indices = np.argsort(classes_int[:, 1] * 100 + classes_int[:, 0])
    classes_sorted = classes_int[sorted_indices]
    scores = scores[sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True)
    ax.bar(np.arange(len(classes)), scores,
           color=plt.get_cmap('tab20')(np.arange(len(classes)) % 20), alpha=0.9)

    for i, cls_i in enumerate(classes_sorted):
        partner = worst_auc_partners[f'{cls_i[0]}/{cls_i[1]}']
        if partner is None:
            label_txt = 'single class'
        else:
            label_txt = f"worst pair: {partner}"
        ax.text(i, (scores[i] + 0.03) if scores[i] < 0.4 else 0.1,
                label_txt, rotation=90, ha='center', va='bottom', fontsize=8)

    ax.set_xticks(np.arange(len(classes_sorted)))
    ax.set_xticklabels([f'{cls[0]}/{cls[1]}' for cls in classes_sorted], rotation=45, ha='right')
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel('AUC detectability [norm]')
    ax.set_xlabel('Mode label')
    ax.set_title('Per-label detectability from linear-probe pairwise AUC')
    ax.grid(True, axis='y', alpha=0.3)

    if doSave:
        out_path = f'../output_plots/feature_detectability_auc_bar{saveExt}.pdf'
        plt.savefig(out_path, transparent=True, bbox_inches='tight')
        print(f"   Saved to: {out_path}")


def _fit_pairwise_lr_in_2d_pca(features_flat, y, cls_i, cls_j, cv_splits=5, random_state=42):
    """Fit pairwise LR on a 2D PCA projection and return plotting artifacts."""
    mask = (y == cls_i) | (y == cls_j)
    X_pair = features_flat[mask]
    y_pair = (y[mask] == cls_j).astype(np.int32)

    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_pair)

    class_counts = np.bincount(y_pair)
    if len(class_counts) < 2 or np.min(class_counts) < 2:
        auc_cv = 0.5
    else:
        n_splits_eff = min(cv_splits, int(np.min(class_counts)))
        if n_splits_eff < 2:
            auc_cv = 0.5
        else:
            skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
            fold_aucs = []
            for train_idx, test_idx in skf.split(X_2d, y_pair):
                model_cv = LogisticRegression(max_iter=3000, class_weight='balanced')
                model_cv.fit(X_2d[train_idx], y_pair[train_idx])
                y_prob = model_cv.predict_proba(X_2d[test_idx])[:, 1]
                fold_auc = roc_auc_score(y_pair[test_idx], y_prob)
                fold_aucs.append(fold_auc)
            auc_cv_raw = float(np.mean(fold_aucs)) if fold_aucs else 0.5
            auc_cv = max(auc_cv_raw, 1.0 - auc_cv_raw)

    model = LogisticRegression(max_iter=3000, class_weight='balanced')
    model.fit(X_2d, y_pair)

    x_min, x_max = X_2d[:, 0].min() - 1.0, X_2d[:, 0].max() + 1.0
    y_min, y_max = X_2d[:, 1].min() - 1.0, X_2d[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    prob = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    return X_2d, y_pair, xx, yy, prob, auc_cv


def plot_pairwise_linear_boundaries(results, doSave=True):
    """Plot and save two pairwise 2D PCA logistic boundaries (hardest and easiest class pairs)."""
    classes = results['classes']
    auc_matrix = results['auc_matrix']
    features_flat = results['features_flat']
    y = results['y']
    saveExt = results['saveExt']

    print(f"\n12. Creating pairwise linear decision-boundary plots...")

    if len(classes) < 2:
        print("   Skipping boundary plot: need at least 2 classes.")
        return

    upper_i, upper_j = np.triu_indices(len(classes), k=1)
    pair_aucs = auc_matrix[upper_i, upper_j]
    if pair_aucs.size == 0:
        print("   Skipping boundary plot: no class pairs available.")
        return

    hardest_idx = int(np.argmin(pair_aucs))
    easiest_idx = int(np.argmax(pair_aucs))

    pairs = [
        (classes[upper_i[hardest_idx]], classes[upper_j[hardest_idx]], 'Hardest pair (lowest AUC)'),
        (classes[upper_i[easiest_idx]], classes[upper_j[easiest_idx]], 'Easiest pair (highest AUC)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), tight_layout=True)
    cmap = 'RdBu_r'

    for ax, (cls_i, cls_j, subtitle) in zip(axes, pairs):
        X_2d, y_pair, xx, yy, prob, auc_cv = _fit_pairwise_lr_in_2d_pca(
            features_flat, y, cls_i, cls_j
        )
        cf = ax.contourf(xx, yy, prob, levels=np.linspace(0.0, 1.0, 21),
                         cmap=cmap, alpha=0.35)
        ax.contour(xx, yy, prob, levels=[0.5], colors='k', linewidths=2.0)
        ax.scatter(X_2d[y_pair == 0, 0], X_2d[y_pair == 0, 1], s=18, alpha=0.75, label=cls_i)
        ax.scatter(X_2d[y_pair == 1, 0], X_2d[y_pair == 1, 1], s=18, alpha=0.75, label=cls_j)
        full_auc = auc_matrix[
            np.where(classes == cls_i)[0][0],
            np.where(classes == cls_j)[0][0],
        ]
        ax.set_title(f"{subtitle}\n{cls_i} vs {cls_j} | full AUC={full_auc:.3f}, "
                     f"PCA-2D AUC={auc_cv:.3f}")
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best', framealpha=0.85)

    fig.colorbar(cf, ax=axes[-1], shrink=0.9,
                 label='Predicted probability of positive class (cls_j)')
    if doSave:
        out_path = f'../output_plots/feature_linear_boundary_examples{saveExt}.pdf'
        plt.savefig(out_path, transparent=True, bbox_inches='tight')
        print(f"   Saved to: {out_path}")
