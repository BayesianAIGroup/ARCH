# Data and checkpoint provenance

All retained data remain byte-for-byte unchanged. Original relative paths and SHA-256 values, plus current release sizes and hashes, are in `provenance/source_manifest.json`. No data were regenerated, re-split, or normalized on disk.

## Real data

| Dataset | Train | Validation | Test | Paper time unit |
|---|---:|---:|---:|---|
| Earthquake | 950 | 50 | 50 | day |
| COVID19 | 1450 | 100 | 100 | day |
| Citibike | 2440 | 300 | 320 | hour |

Counts match between prediction and Seahorse; contents do not generally match.

- `data/prediction/{earthquakes,COVID19,Citibike}` comes from `transformer_events`; prediction/recovery tasks read it.
- `data/seahorse/{earthquakes,COVID19,Citibike}` comes from `transformer_events/Seahorse`; likelihood reads it.

The duplicate `seahorse_original` directory has been removed: all nine split pairs matched `seahorse` exactly, event-by-event, despite different pickle serialization. No experiment loaded the duplicate directory. Comparison counts, hashes, and original source paths are recorded in [the deduplication report](validation/data_cleanup.json). Prediction and Seahorse remain separate because their contents differ.

Format: list of sequences, each event `[absolute_time, longitude, latitude]`. `load_earthquake_st_ori` converts times to gaps, applies epsilon for first/nonpositive gaps, and normalizes locations to [0,1]. The original loader derives spatial bounds and maximum length from **all three splits**. This is preserved, not described as train-only preprocessing. Changing it requires a separately reported experiment.

The paper attributes these processed benchmarks to Chen et al. (2021) and Yuan et al. (2023). The actual processed inputs are provided; a verified raw-download-to-split pipeline is not. Unused raw earthquake CSV/NPZ and unrelated downloads are omitted. Dataset licensing is not inferred from local file availability.

## Synthetic data

- `stpp_out`: original Hawkes train/val/test and midpoint assets.
- `stscp_out`: original self-correcting train/val/test and midpoint assets.
- `Spatial_Temporal_Plotting`: event-time test inputs and AutoSTPP/DeepSTPP/NSTPP predictions.
- `stscp_midpoint_baselines`: STSCP midpoint query data and saved AutoSTPP/DeepSTPP/NSTPP predictions. All four files are loaded by `self_correcting_midpoints.py` for midpoint intensity comparison plots (`python run.py synthetic --process self_correcting --action midpoints`).
- `HP_stpp_tmid_S900`: Hawkes midpoint data/predictions; its unused duplicate self-correcting branch is omitted.
- `HP_stscp_t_int9`: both processes' dense temporal-query data and predictions.

These all live under `data/synthetic`. Baseline layouts and scale conversions are preserved. Only the plotting-data directory actually referenced by the curated source was included; similarly named old alternatives were not substituted. Large comparison arrays are required inputs, not output caches. Baseline training repositories are not bundled.

Generation and query-construction helpers are in `synthetic/`; the supplied splits are authoritative for these runs. Utility modules remain separate per family to preserve their actual preprocessing behavior.

## Checkpoints and historical results

Four `best_model_{time,loc}[_SC].pth` files are copied from `EventsFM-zhe` into `checkpoints/synthetic`. Their exact historical training provenance is not independently verified. New training writes separate weights under `outputs`; use `--checkpoint-dir` to evaluate them.

`reference_results/prediction` holds selected old task logs; `reference_results/seahorse` holds the three existing LL JSONs. They can describe different seeds or exploratory settings and are not regenerated release results. No real-data checkpoint was selected by guessing from ambiguous filenames.
