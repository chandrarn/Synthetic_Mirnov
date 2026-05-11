# MRNV Horizontal vs Vertical Detectability: Quick Reasonableness Readout

Date: 2026-05-08

## Scope
This readout checks whether the observed result (vertical-only occasionally outperforming horizontal-only) looks physically and statistically reasonable, without triggering expensive dataset/t-SNE rebuilds.

Data sources used:
- Cached feature spaces in `output_plots/tsne_features_*.npz` (fast path only).
- Sensor geometry from `signal_generation/input_data/MAGX_Coordinates_CFS.json`.
- Mode groups from `classifier_regressor/diagnostic_feature_separability.py`.
- Mode `6/3` excluded.

## 1) Geometry coverage sanity check (current MRNV T/P arrays)
Horizontal subset (`T` arrays):
- N = 12 sensors
- unique toroidal angles = 12
- toroidal span = 188.96 deg
- median toroidal gap = 1.8 deg
- vertical span (`Z`) ~ 1e-4 m (effectively none)

Vertical subset (`P` arrays):
- N = 8 sensors
- unique toroidal angles = 2
- toroidal span = 188.96 deg
- median toroidal gap = 180 deg
- vertical span (`Z`) = 0.348 m

Interpretation:
- Horizontal gives dense toroidal sampling, almost no poloidal (vertical) leverage.
- Vertical gives strong poloidal/vertical leverage, sparse toroidal sampling.
- So either subset can win depending on whether separability is dominated by toroidal vs poloidal structure for the selected mode family.

## 2) Geometry-only mode aliasing check (no learned features)
Using phase-signature distances from sensor geometry only (global phase-invariant):
- Horizontal: mean nearest-neighbor mode distance = 0.1783
- Vertical: mean nearest-neighbor mode distance = 0.1551

Interpretation:
- Geometry-only check does **not** show a universal vertical advantage.
- If vertical wins in classifier metrics, that is not explained by simple sensor spacing alone.

## 3) Cache-only feature-space checks (same mode subset)
Using precomputed `features_flat` with no rebuild:

From prior quick classifier proxies:
- PCA+multinomial logistic CV accuracy:
  - horizontal-only (5-sensor cache): 0.8752
  - vertical-only (5-sensor cache): 0.8953
  - all sensors: 0.9680

Additional unsupervised separability checks:
- Silhouette (higher better): horizontal 0.0198, vertical 0.0152
- Davies-Bouldin (lower better): horizontal 28.611, vertical 18.127
- Fisher ratio (higher better): horizontal 0.8872, vertical 0.8708

Interpretation:
- No single unsupervised metric gives a decisive winner between H-only and V-only.
- Supervised proxy (CV logistic) still shows vertical > horizontal for this cached comparison.
- This is consistent with the observed bar-plot behavior being a feature-space/classifier effect, not a trivial geometry bug.

## 4) Unexpected issues check
No hard anomalies found:
- Label-shuffle silhouette baselines are strongly below observed values, so structure is real.
- Geometry coverage values are consistent with sensor definitions.
- Hard mode confusions are plausible (neighboring mode families like 2/2 vs 4/2, 6/4 vs 8/4, etc.).

Caveat (historical section):
- The comparison above used pre-existing 5-sensor H-only and 5-sensor V-only feature caches, so it was directional.

## 5) Why runs looked "hung" and what fixed it
Root cause:
- Exact cache files for current sets did not exist initially:
  - `cached_data_list12_MRNV_160M_T1-MRNV_160M_T2-MRNV_160M_T3_fgap_off_mn_100_-1.npz`
  - `cached_data_list8_MRNV_160M_P1-MRNV_160M_P2-MRNV_160M_P3_fgap_off_mn_100_-1.npz`
- So the helper launched a first-time dataset build from raw NetCDFs (56 files) plus pairwise AUC CV work.
- This is long and can appear stalled unless run unbuffered with progress output.

Fix used:
- Added a fast AUC path in `compute_auc_detectability_for_configuration()` that avoids t-SNE.
- Ran explicit unbuffered build+compute with progress prints, then reused the generated exact caches.

## 6) Exact current-sensor results (12T vs 8P)
Exact subsets:
- Horizontal: 12 T sensors (`MRNV_160M_T1..T6`, `MRNV_340M_T1..T6`)
- Vertical: 8 P sensors (`MRNV_160M_P1..P4`, `MRNV_340M_P1..P4`)

Exact AUC-detectability summary (same mode groups, excluding 6/3):
- Horizontal 12T: `n_classes=56`, mean `0.3082`, median `0.1464`, min `0.0589`, max `0.9661`
- Vertical 8P: `n_classes=56`, mean `0.4890`, median `0.4383`, min `0.0799`, max `1.0000`
- Delta (vertical - horizontal): mean `+0.1808`, median `+0.1223`

Interpretation:
- For the exact current sensor sets, vertical is clearly better on this metric across most modes.

## 7) Unsupervised separability checks: what they mean and what they say here
Definitions:
- Silhouette: compares each sample's distance to its own class vs nearest other class. Higher is better (can be negative for overlapping classes).
- Calinski-Harabasz (CH): ratio of between-class to within-class dispersion. Higher is better.
- Davies-Bouldin (DB): average similarity of each class to its most similar neighbor. Lower is better.
- Fisher ratio: total between-class variance divided by within-class variance. Higher is better.
- Permutation baseline (for silhouette): recompute silhouette after shuffling labels; if real silhouette is much better than shuffled, labels align with real structure.

Exact-set unsupervised results (from the exact 12T/8P caches):
- Horizontal 12T:
  - silhouette `-0.0487` (perm baseline `-0.0802 ± 0.0034`)
  - CH `31.15`
  - DB `21.50`
  - Fisher `0.4431`
- Vertical 8P:
  - silhouette `-0.0043` (perm baseline `-0.0662 ± 0.0031`)
  - CH `29.54`
  - DB `14.95`
  - Fisher `0.4202`

How to read this mix:
- Both sets are far better than label-shuffled baselines (so structure is real, not noise).
- Vertical has better silhouette and much better DB (cleaner nearest-neighbor class boundaries).
- Horizontal has slightly better CH and Fisher (some stronger global spread dimensions).
- This mixed unsupervised picture is common in high-class-count problems and is consistent with the supervised AUC result showing vertical advantage for this mode set.

## Bottom line
The result is reasonable: with the exact current sensor sets, vertical-only outperforms horizontal-only on the pairwise AUC detectability metric, and nothing in geometry-only or unsupervised checks indicates a bug. The behavior is best explained as a feature-space effect where poloidal/vertical information is more discriminative for much of the selected mode family.