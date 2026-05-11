#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic: AUC-based feature-space separability comparison across sensor configurations.

Helper functions (metrics, plotting, cache utilities) live in separability_helpers.py.

Run with: python diagnostic_feature_separability.py
"""
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import sys
import os
from sklearn.preprocessing import LabelEncoder

# Add paths for imports
sys.path.insert(0, '../C-Mod/')
sys.path.insert(0, '.')

from data_caching import build_or_load_cached_dataset
from fno_predictor import fit_and_apply_scaler
from separability_helpers import (
    # mode-label utilities
    _parse_mode_label,
    _normalise_mode_label,
    _sort_mode_labels,
    # cache-path utilities
    _geometry_tag,
    _shared_cache_paths,
    # geometry proxy
    find_proxy_sensors,
    # metrics
    compute_overlap_matrix,
    compute_pairwise_linear_probe_auc,
    # full t-SNE workflow
    load_and_compute_separability,
    # plotting helpers
    plot_tsne_separability,
    plot_detectability_bar_chart,
    plot_auc_detectability_bar_chart,
    plot_pairwise_linear_boundaries,
)


def compute_auc_detectability_for_configuration(
    data_dir,
    geometry_shot,
    include_frequency_gap,
    use_mode='mn',
    num_timepoints=100,
    n_datasets=-1,
    force_rebuild_cache=False,
    allow_rebuild_missing_cache=True,
    mode_subset=None,
    use_fast_path=True,
):
    """Compute per-mode AUC detectability for a single sensor/frequency-gap configuration.

    Parameters
    ----------
    mode_subset : list of str, optional
        If provided, only samples whose label is in this set are included when
        computing pairwise AUC.  This makes the distinguishability scores reflect
        only confusion *within* the chosen subset of modes. During cache rebuild,
        this list is also used to pre-filter files by dataset-level mode labels.
    """
    out_path, out_path_features, save_ext = _shared_cache_paths(
        data_dir,
        geometry_shot,
        use_mode,
        num_timepoints,
        n_datasets,
        include_frequency_gap,
    )

    print("\n" + "-" * 70)
    print(
        f"AUC detectability config: geometry={geometry_shot}, "
        f"include_frequency_gap={include_frequency_gap}, force_rebuild_cache={force_rebuild_cache}"
    )
    if mode_subset is not None:
        print(f"   mode_subset active: {mode_subset}")
        # Use subset-specific cache names so subset builds can be re-used without
        # overwriting the full-cache artifacts.
        mode_subset_norm = sorted({_normalise_mode_label(m) for m in mode_subset})

        # Build a readable, deterministic subset tag from normalized mode labels.
        # Example: subset3_1-1__2-1__3-2
        subset_parts = []
        for label in mode_subset_norm:
            safe = ''.join(ch if (ch.isalnum() or ch in ['-', '_']) else '_' for ch in label)
            safe = safe.replace('/', '-')
            subset_parts.append(safe)
        subset_tag = f"subset{len(mode_subset_norm)}_{'__'.join(subset_parts)}"

        out_path = out_path.replace('.npz', f'_{subset_tag}.npz')
        out_path_features = out_path_features.replace('.npz', f'_{subset_tag}.npz')
        save_ext = f"{save_ext}_{subset_tag}"
        print(f"   subset cache path: {out_path}")

    if use_fast_path:
        # Fast path for AUC diagnostics: avoid t-SNE and operate directly
        # on flattened/scaled features loaded from dataset cache.
        has_dataset_cache = os.path.exists(out_path)
        if (not has_dataset_cache) and (not force_rebuild_cache) and (not allow_rebuild_missing_cache):
            raise FileNotFoundError(
                "Dataset cache missing and rebuild disabled for fast AUC path. "
                f"Missing dataset cache: {out_path}."
            )

        result = build_or_load_cached_dataset(
            data_dir=data_dir,
            out_path=out_path,
            use_mode=use_mode,
            include_geometry=True,
            geometry_shot=geometry_shot,
            num_timepoints=num_timepoints,
            freq_tolerance=0.1,
            n_datasets=n_datasets,
            load_saved_data=True,
            visualize_first=False,
            doVisualize=False,
            saveDataset=(force_rebuild_cache or allow_rebuild_missing_cache),
            include_frequency_gap=include_frequency_gap,
            desired_mode_labels=mode_subset,
        )
        X_ri, y, y_m, y_n, sensor_names, theta, phi = result
        diff_features_scaled, scaler = fit_and_apply_scaler(
            X_ri, theta, phi, do_fgap=include_frequency_gap
        )
        features_flat = diff_features_scaled.reshape(len(diff_features_scaled), -1)
    else:
        results = load_and_compute_separability(
            data_dir=data_dir,
            out_path=out_path,
            out_path_features=out_path_features,
            saveExt=save_ext,
            geometry_shot=geometry_shot,
            use_mode=use_mode,
            num_timepoints=num_timepoints,
            n_datasets=n_datasets,
            forceReload=force_rebuild_cache,
            include_frequency_gap=include_frequency_gap,
            allow_rebuild_missing_cache=allow_rebuild_missing_cache,
            desired_mode_labels=mode_subset,
        )
        features_flat = results['features_flat']
        y = results['y']

    # Restrict to mode_subset if requested
    if mode_subset is not None:
        subset_set = set(_normalise_mode_label(m) for m in mode_subset)
        y_norm = np.array([_normalise_mode_label(label) for label in y], dtype=str)
        mask = np.array([label in subset_set for label in y_norm])
        features_flat = features_flat[mask]
        y = y_norm[mask]
        if len(y) == 0:
            raise ValueError(
                f"mode_subset={mode_subset} matched no samples in dataset at {out_path}."
            )

    le = LabelEncoder()
    classes = le.fit(y).classes_

    auc_matrix, detectability_auc, worst_auc_partners = compute_pairwise_linear_probe_auc(
        features_flat, y, classes
    )

    return {
        'classes': classes,
        'detectability_auc': detectability_auc,
        'worst_auc_partners': worst_auc_partners,
        'auc_matrix': auc_matrix,
        'geometry_shot': geometry_shot,
        'include_frequency_gap': include_frequency_gap,
        'cache_path': out_path,
        'feature_cache_path': out_path_features,
    }


def plot_auc_detectability_multi_configuration(
    data_dir,
    use_mode='mn',
    num_timepoints=100,
    n_datasets=-1,
    doSave=True,
    force_rebuild_all=False,
    force_rebuild_no_gap=True,
    allow_rebuild_missing_cache=True,
    split_into_two_subplots=False,
    split_index=None,
    mode_groups=None,
    mode_group_labels=None,
    force_rebuild_config_names=None,
    save_ext=''
):
    """Compare per-mode AUC detectability across multiple sensor/frequency-gap configurations.

    Parameters
    ----------
    mode_groups : list of list of str, optional
        Explicit grouping and ordering of modes for the plot, e.g.::

            [['1/1', '2/2', '3/3'], ['2/1', '4/2', '6/3']]

        The flat union of all groups defines which modes are included in the
        distinguishability computation (AUC is computed only within that subset).
        Modes are plotted in the order they appear across the groups, with each
        group visually separated by a tinted background band.
        When *None* (default), all modes found in the data are used in default
        sorted order.
    mode_group_labels : list of str, optional
        Display label for each group in *mode_groups*.  Falls back to
        ``'Group 1'``, ``'Group 2'``, … when not supplied.
    force_rebuild_config_names : list of str, optional
        Config names for which cache should be force-rebuilt regardless of
        ``force_rebuild_all`` (e.g. ['MRNV horizontal', 'MRNV vertical']).
    """
    force_rebuild_config_names = set(force_rebuild_config_names or [])

    configs = [
        {
            'name': r'MRNV all',
            'geometry_shot': ['MRNV'],
            'include_frequency_gap': force_rebuild_all,
            'color': '#1f77b4',
            'force_rebuild': force_rebuild_all,
        },
        {
            'name': r'MRNV horizontal',
            'geometry_shot': [ 'MRNV_160M_T1', 'MRNV_160M_T2', 
                       'MRNV_160M_T3', 'MRNV_160M_T4', 'MRNV_160M_T5','MRNV_160M_T6',\
                          'MRNV_340M_T1', 'MRNV_340M_T2', 'MRNV_340M_T3',\
                              'MRNV_340M_T4', 'MRNV_340M_T5', 'MRNV_340M_T6'],
            'include_frequency_gap': False,
            'color': '#2ca02c',
            'force_rebuild': force_rebuild_all,
        },
        {
            'name': r'MRNV vertical',
            'geometry_shot': ['MRNV_160M_P1', 'MRNV_160M_P2', 'MRNV_160M_P3',\
                               'MRNV_160M_P4', 
                              'MRNV_340M_P1', 'MRNV_340M_P2', 'MRNV_340M_P3',\
                                  'MRNV_340M_P4', ],
            'include_frequency_gap': False,
            'color': '#ff7f0e',
            'force_rebuild': force_rebuild_all,
        },
        # {
        #     'name': r'MRNV all (no $v_\phi$-info)',
        #     'geometry_shot': ['MRNV'],
        #     'include_frequency_gap': False,
        #     'color': '#d62728',
        #     'force_rebuild': force_rebuild_all or force_rebuild_no_gap,
        # },
    ]

    # ------------------------------------------------------------------ #
    # Build ordered mode list and group metadata                          #
    # ------------------------------------------------------------------ #
    if mode_groups is not None:
        modes_sorted = []
        _seen_modes = set()
        _group_info = []  # list of (global_start, global_end, label)
        for _g_idx, _grp in enumerate(mode_groups):
            _g_start = len(modes_sorted)
            for _m in _grp:
                _ms = _normalise_mode_label(_m)
                if _ms not in _seen_modes:
                    modes_sorted.append(_ms)
                    _seen_modes.add(_ms)
            _g_end = len(modes_sorted)
            _g_label = (
                mode_group_labels[_g_idx]
                if mode_group_labels and _g_idx < len(mode_group_labels)
                else f'Group {_g_idx + 1}'
            )
            _group_info.append((_g_start, _g_end, _g_label))
        mode_subset = modes_sorted  # restrict AUC computation to this set
    else:
        mode_subset = None
        _group_info = None

    config_results = {}
    all_modes = set()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Multi-Configuration AUC Detectability Comparison")
    print("=" * 70)
    if force_rebuild_config_names:
        print(f"Targeted force-rebuild configs: {sorted(force_rebuild_config_names)}")

    for cfg in configs:
        force_rebuild_cfg = cfg['force_rebuild'] or (cfg['name'] in force_rebuild_config_names)
        result = compute_auc_detectability_for_configuration(
            data_dir=data_dir,
            geometry_shot=cfg['geometry_shot'],
            include_frequency_gap=cfg['include_frequency_gap'],
            use_mode=use_mode,
            num_timepoints=num_timepoints,
            n_datasets=n_datasets,
            force_rebuild_cache=force_rebuild_cfg,
            allow_rebuild_missing_cache=allow_rebuild_missing_cache,
            mode_subset=mode_subset,
        )
        config_results[cfg['name']] = result
        all_modes.update([str(mode) for mode in result['classes']])

    if mode_groups is None:
        modes_sorted = _sort_mode_labels(list(all_modes))
    # else modes_sorted already built from mode_groups above

    n_cfg = len(configs)
    n_modes = len(modes_sorted)
    score_grid = np.full((n_cfg, n_modes), np.nan, dtype=np.float64)

    for i, cfg in enumerate(configs):
        detectability_map = config_results[cfg['name']]['detectability_auc']
        for j, mode in enumerate(modes_sorted):
            if mode in detectability_map:
                score_grid[i, j] = detectability_map[mode]

    # Keep per-configuration worst-pair maps so each plotted bar can show
    # its own diagnostic partner text.
    worst_pair_maps = {
        cfg['name']: config_results[cfg['name']]['worst_auc_partners']
        for cfg in configs
    }

    # Pastel background colors cycling across groups
    _GROUP_BG_COLORS = [
        '#d4e6f1', '#fde8d8', '#d5f5e3', '#f9ebea',
        '#e8daef', '#fdfcd9', '#d6eaf8', '#fdebd0',
    ]

    group_width = 0.86
    group_gap = 0.36
    bar_width = group_width / max(1, n_cfg)
    centers = np.arange(n_cfg, dtype=np.float64) - (n_cfg - 1) / 2.0
    offsets = centers * bar_width

    def _plot_mode_slice(ax, start, end):
        modes_slice = modes_sorted[start:end]
        scores_slice = score_grid[:, start:end]
        n_modes_slice = len(modes_slice)
        x = np.arange(n_modes_slice, dtype=np.float64) * (group_width + group_gap)
        half_step = (group_width + group_gap) / 2.0

        # Draw group background bands behind everything
        if _group_info is not None:
            for _g_idx, (_g_start, _g_end, _g_label) in enumerate(_group_info):
                # Translate global group indices into slice-local indices
                local_start = max(_g_start, start) - start
                local_end = min(_g_end, end) - start
                if local_end <= local_start:
                    continue
                x_left = x[local_start] - half_step
                x_right = x[local_end - 1] + half_step
                bg_color = _GROUP_BG_COLORS[_g_idx % len(_GROUP_BG_COLORS)]
                ax.axvspan(x_left, x_right, alpha=0.35, color=bg_color, zorder=0, linewidth=0)
                # Label in the upper-left corner of each shaded group with padding.
                x_pad = 0.04 * (x_right - x_left)
                y_pad = 0.03
                ax.text(
                    x_left + x_pad, 1.0 - y_pad, _g_label,
                    ha='left', va='top', fontsize=12, fontweight='bold', color='0.25',
                    transform=ax.get_xaxis_transform(),
                    style='italic',
                    bbox=dict(facecolor='white', alpha=0.55, edgecolor='0.35', linewidth=0.4, pad=0.6),
                )

        for i, cfg in enumerate(configs):
            x_offset = x + offsets[i]
            ax.bar(
                x_offset,
                scores_slice[i],
                width=bar_width * 0.9,
                label=cfg['name'],
                color=cfg['color'],
                alpha=0.7,
                edgecolor='black',
                linewidth=0.4,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(modes_slice, rotation=45, ha='right')
        if n_modes_slice > 1:
            separators = (x[:-1] + x[1:]) / 2.0
            for xpos in separators:
                ax.axvline(xpos, color='0.88', linewidth=0.8, zorder=0)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(r'Detectability: $2\times (\mathrm{AUC}-\frac{1}{2})$')
        ax.set_xlabel(r'$m/n$ Mode \#')
        ax.grid(True, axis='y', alpha=0.3)

        # Annotate each configuration's bar with its own worst-pair partner.
        for i, cfg in enumerate(configs):
            pair_map = worst_pair_maps[cfg['name']]
            for j, mode in enumerate(modes_slice):
                partner = pair_map.get(mode, None)
                if partner is None:
                    continue
                y_txt = (scores_slice[i, j] + 0.03) if scores_slice[i, j] < 0.4 else 0.08
                ax.text(
                    x[j] + offsets[i],
                    y_txt,
                    f"worst: {partner}",
                    ha='center',
                    va='bottom',
                    rotation=90,
                    fontsize=5,
                    color='0.15',
                    bbox=dict(facecolor='white', alpha=0.55, edgecolor='none', pad=0.3),
                    zorder=5,
                )

        return np.any(np.isnan(scores_slice))

    shared_title = 'Per-mode detectability from pairwise linear-probe AUC'

    if split_into_two_subplots and n_modes > 1:
        if split_index is None:
            if _group_info is not None:
                # Prefer splitting on a mode-group boundary nearest the midpoint so
                # groups do not get fragmented between the top and bottom panels.
                boundaries = [g_end for (_, g_end, _) in _group_info[:-1]]
                if len(boundaries) > 0:
                    mid = n_modes / 2.0
                    split_index = min(boundaries, key=lambda b: abs(b - mid))
                else:
                    split_index = int(np.ceil(n_modes / 2))
            else:
                split_index = int(np.ceil(n_modes / 2))
        split_index = int(np.clip(split_index, 1, n_modes - 1))

        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            figsize=(max(12, 0.9 * max(split_index, n_modes - split_index)), 10),
            tight_layout=True,
            sharey=True,
        )
        fig.suptitle(shared_title)

        _plot_mode_slice(ax_top, 0, split_index)
        _plot_mode_slice(ax_bottom, split_index, n_modes)
        ax_top.legend(loc='best', framealpha=0.9)
        ax_top.text(
            0.99,
            0.94,
            'Worst-pair shown per configuration bar',
            transform=ax_top.transAxes,
            ha='right',
            va='top',
            fontsize=8,
            alpha=0.8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='0.35', linewidth=0.6, pad=0.5),
        )
    else:
        fig, ax = plt.subplots(figsize=(max(10, 0.9 * n_modes), 6), tight_layout=True)
        ax.set_title(shared_title)
        _plot_mode_slice(ax, 0, n_modes)
        ax.legend(loc='best', framealpha=0.9)
        ax.text(
            0.99,
            0.94,
            'Worst-pair shown per configuration bar',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=8,
            alpha=0.8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='0.35', linewidth=0.6, pad=0.5),
        )

    split_tag = '_2panel' if (split_into_two_subplots and n_modes > 1) else ''
    grp_tag = f'_grp{len(mode_groups)}' if mode_groups is not None else ''
    saveExt = f"_{use_mode}_{num_timepoints}_{n_datasets}_multi_cfg_auc{split_tag}{grp_tag}{save_ext}"
    if doSave:
        out_path = f'../output_plots/feature_detectability_auc_bar{saveExt}.pdf'
        plt.savefig(out_path, transparent=True, bbox_inches='tight')
        print(f"Saved multi-configuration AUC detectability plot to: {out_path}")

    return {
        'configs': configs,
        'config_results': config_results,
        'modes_sorted': modes_sorted,
        'score_grid': score_grid,
        'saveExt': saveExt,
    }


def run_multi_configuration_auc_detectability_diagnostic():
    """Entry point for comparing AUC detectability over multiple sensor configurations."""
    data_dir = "/home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/SPARC/"
    use_mode = 'mn'
    n_datasets = -1
    num_timepoints = 100
    doSave = True
    save_ext = '_select_subset'

    # Example: group modes by resonant surface (q=1 and q=2)
    # mode_groups = [['1/1', '2/2', '3/3'], ['2/1', '4/2', '6/3']]
    # mode_groups = [
    #     ['1/1', '2/2','3/3','4/4','5/5','6/6','7/7','8/8','9/9','10/10','11/11','13/13','14/14','15/15'],
    #     [ '3/2', '5/3','6/4','8/5','9/6','11/7','12/8','14/9','15/10','17/11','18/12','20/13','21/14','23/15'],
    #     [ '2/1','4/2','6/3','8/4','10/5','12/6','14/7','18/9', '20/10','22/11','24/12','26/13','28/14','30/15'],
    #     ['3/1', '6/2','9/3', '12/4','15/5', '18/6','21/7','24/8','27/9','30/10','33/11','36/12','39/13','42/14','45/15']
    # ]
    # mode_group_labels = [r'$q=1$ surface',r'$q=1.5$ surface', r'$q=2$ surface', r'$q=3$ surface']
    # mode_groups = [[f'{n}/{n}' for n in range(1,16)]]
    # mode_group_labels = [r'$m=n$ modes',]

    mode_groups = [['5/5', '10/5', '15/5', '10/10', '15/10', '20/10', '15/15', '30/15', '45/15']]
    mode_group_labels = ['Select sensor subset']
    # mode_groups = None
    # mode_group_labels = None

    results = plot_auc_detectability_multi_configuration(
        data_dir=data_dir,
        use_mode=use_mode,
        num_timepoints=num_timepoints,
        n_datasets=n_datasets,
        doSave=doSave,
        force_rebuild_all=False,
        force_rebuild_no_gap=False,
        allow_rebuild_missing_cache=True,
        split_into_two_subplots=False,
        mode_groups=mode_groups,
        mode_group_labels=mode_group_labels,
        force_rebuild_config_names=[], #['MRNV horizontal', 'MRNV vertical'],
        save_ext=save_ext
    
    )

    print("\n" + "=" * 70)
    print("Multi-configuration AUC detectability diagnostic complete.")
    print(
        f"Modes compared: {len(results['modes_sorted'])} | "
        f"Configurations: {len(results['configs'])}"
    )
    print("=" * 70)


def run_separability_diagnostic():
    use_mode = 'mn'
    n_datasets = -1
    num_timepoints = 100
    forceReload = False
    allow_rebuild_missing_cache = False
    include_frequency_gap = True
    # geometry_shot = ['BP'] #1160930034
    # geometry_shot =  ['BP-IOL1-110U', 'BP-IOL2-190L', 'BP-DVT1-010L', 'BP-DT23-290U', 'MRNV_160M_H1',\
    #                    'MRNV_160M_H2', 'MRNV_160M_H3', 'MRNV_160M_H5', 'MRNV_160M_V1', 'MRNV_160M_V2',\
    #                       'MRNV_160M_V4', 'MRNV_160M_V5', 'MRNV_340M_H1', 'MRNV_340M_H3', 'MRNV_340M_H4',\
    #                           'MRNV_340M_H5', 'MRNV_340M_V1', 'MRNV_340M_V2', 'MRNV_340M_V3', 'MRNV_340M_V4']
    geometry_shot =  [  
                    #    'MRNV_160M_H1', 'MRNV_160M_H2', 
                       'MRNV_160M_H3', 'MRNV_160M_H4', 'MRNV_160M_H5',\
                    #    'MRNV_160M_V1', 'MRNV_160M_V2',
                         'MRNV_160M_V3', 'MRNV_160M_V4', 'MRNV_160M_V5',\
                    #    'MRNV_340M_H1', 'MRNV_340M_H2', 
                       'MRNV_340M_H3', 'MRNV_340M_H4',  'MRNV_340M_H5',\
                    #    'MRNV_340M_V1', 'MRNV_340M_V2', 
                       'MRNV_340M_V3', 'MRNV_340M_V4', 'MRNV_340M_V5',
                       ]
    # geometry_shot = ['MRNV']

    """Run full feature separability workflow (compute + plotting)."""
    data_dir = "/home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/SPARC/"
    out_path, out_path_features, saveExt = _shared_cache_paths(
        data_dir,
        geometry_shot,
        use_mode,
        num_timepoints,
        n_datasets,
        include_frequency_gap,
    )
    doSave=True

    results = load_and_compute_separability(data_dir, out_path, out_path_features,\
            saveExt, geometry_shot, use_mode, num_timepoints, n_datasets, forceReload,
            include_frequency_gap=include_frequency_gap,
            allow_rebuild_missing_cache=allow_rebuild_missing_cache)
    plot_tsne_separability(results, doSave)
    plot_detectability_bar_chart(results, doSave)
    plot_auc_detectability_bar_chart(results, doSave)
    plot_pairwise_linear_boundaries(results, doSave)

    print(f"\n" + "=" * 70)
    print(f"Diagnostic complete. Check ../output_plots/feature_separability_tsne{results['saveExt']}.pdf")
    print(f"Also check ../output_plots/feature_detectability_bar{results['saveExt']}.pdf")
    print(f"Also check ../output_plots/feature_detectability_auc_bar{results['saveExt']}.pdf")
    print(f"Also check ../output_plots/feature_linear_boundary_examples{results['saveExt']}.pdf")
    print("=" * 70)

    plt.show()
    print('Finished')

if __name__ == "__main__":
    # run_separability_diagnostic()
    run_multi_configuration_auc_detectability_diagnostic()
    '''
    No cache available and implicit rebuild is disabled. 
    Missing feature cache: ../output_plots/tsne_features_MRNV_fgap_on_mn_100_-1.npz 
    and dataset cache: /home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/SPARC/cached_data_MRNV_fgap_on_mn_100_-1.npz. Set forceReload=True or allow_rebuild_missing_cache=True to rebuild explicitly.
'''
    print("Done")
