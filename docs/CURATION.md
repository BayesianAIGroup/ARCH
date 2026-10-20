# Curation scope and paper audit

This release was curated from the two supplied source trees and the local 32-page paper. The original trees are unchanged. Small checks validate inputs and execution; they do not establish numerical reproduction of all paper tables.

## Source mapping

Real-data source filenames begin with `Traceformer_earthquakes_flow_matching_zhe_two_model_v2_fixed_mask`.

| Original file or suffix | Release file | Purpose |
|---|---|---|
| `EventsFM-zhe/run_simu_new_rc_two_models.py` | `synthetic/hawkes.py` | Hawkes training and intensity plots |
| `EventsFM-zhe/run_simu_new_rc_two_models_SC.py` | `synthetic/self_correcting.py` | Self-correcting training and intensity plots |
| `.py` | `real_world/next_event.py` | Next-event prediction |
| `_multi_pred.py` | `real_world/forecast.py` | Joint prediction of the next K events |
| `_imputation_completion_v2.py` | `real_world/random_imputation.py` | Random missing-event recovery |
| `_imputation_completion_CM.py` | `real_world/trajectory_recovery.py` | Contiguous segment recovery, lengths 5 and 10 |
| `_imputation_completion_CM_partial_missing.py` | `real_world/partial_attributes.py` | Time/location/full-attribute recovery at 5% and 10% |
| `_imputation_completion_CM_inverse_infer.py` | `real_world/inverse_inference.py` | Inference of the first, second, or first two events |
| `Seahorse/..._compute_LL.py` | `likelihood/seahorse_likelihood.py` | Validation-selected Seahorse likelihood evaluation |

Intensity L2 evaluation, midpoint and temporal plots, generators, and query builders are retained. The synthetic dependency `EventsFM6.py` is distributed as `flow_backbone.py`. Full source mappings and hashes are in `provenance/source_manifest.json`; release changes are described in `provenance/CHANGES.md`.

## Release behavior

- `run.py` selects the task, dataset, device, epochs, seeds, missing ratio, and forecast length. Real-data tasks run ARCH only.
- Paths resolve within the release; each run writes logs, metrics, and weights to a separate output directory.
- Model and loss implementations are preserved. Launcher defaults are 300 epochs for prediction/recovery, 200 for likelihood, 70 for Hawkes, and 160 for self-correcting. These explicit overrides do not establish historical training budgets.
- Several original recovery entry points specify one epoch. The launcher overrides this; `--epochs 1` remains available for smoke checks.
- Summaries report mean, population standard deviation, and population standard error separately.
- Historical logs and four synthetic checkpoint files are retained; their exact training provenance is not independently verified.
- The duplicate Seahorse serialization was removed after exact numerical comparison of all nine splits. `data/seahorse` supplies likelihood inputs and remains separate from prediction data.

## Source/paper differences

| Item | Preserved behavior |
|---|---|
| CM task naming | The active loops evaluate contiguous segments; random recovery uses v2 separately. |
| Mixed masking | Next-event, forecast, and random-v2 use AR plus random conditioning; CM tasks add a segment loss. |
| Contiguous masks | CM training uses a fixed segment ratio; the paper describes two random endpoints. |
| Training epochs | Several recovery files specify one epoch; the release launcher defaults to 300. Historical budgets need separate confirmation. |
| Width/dropout | Prediction: 32/0.15; recovery: 32/0.1; likelihood: 64/0.05. |
| Synthetic training | Adam, four layers, 70/160 epochs; the paper overview describes AdamW and up to 300 epochs. |
| Normalization | Utilities derive spatial bounds and maximum length from train, validation, and test together. Changing this protocol requires a separate experiment. |
| Prediction model selection | Best test flow loss is printed, but the final epoch weights are evaluated and saved. |
| Synthetic model selection | Best weights are selected using test loss; supplied validation splits are not used for selection. |
| Likelihood model selection | Validation likelihood selects the restored checkpoint. |
| Standard deviation/error | Historical aggregation often computes standard error; release summaries distinguish both quantities. |
| Cross-task weights | Each task creates its own temporal and spatial models. No shared checkpoint across tasks is implied. |
| Likelihood seeds | Original lists were overwritten and the active entry point selected earthquakes. Explicit CLI seeds avoid guessing the historical Citibike list. |
| Synthetic comparisons | Saved baseline predictions support plots and evaluation, without supplying full baseline training repositories. |

These are reproduction-protocol differences, not revised experimental conclusions. Short execution checks are not paper-result reproduction.

## Exclusions

- Real-data mean/KNN/iterative-imputer, M-Transformer, DualTPP, and NHPS implementations and execution paths; forecast autoregressive comparison.
- Mask, decoder-only, and joint-flow ablations/architecture comparisons, including Tables 7–12.
- Old experimental versions, unrelated datasets, duplicate archives, caches, empty outputs, and duplicate images.
- External HPP/STHP/NJSDE/NSTPP/DeepSTPP/DSTPP training repositories.

`reference_results` contains historical logs, which may still include comparison rows and exploratory settings. For example, 15% random-missing logs do not match the paper's 10%, 20%, and 30% table settings. No new repository or dataset license is assigned; the historical NHPS BSD notice remains in `licenses/`.
