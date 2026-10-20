# Validation of the cleaned release

The installation files now match the author's live `keyan_1` environment: Python 3.10.20, PyTorch 2.7.1+cu126, NumPy 2.2.6, SciPy 1.15.3, Matplotlib 3.10.8, torchdiffeq 0.2.5, python-dateutil 2.9.0.post0, and joblib 1.5.3. The interpreter used for these checks was `/home/mercury/anaconda3/envs/keyan_1/bin/python`.

The initial cleanup used the default Python 3.9.13/PyTorch 2.4.0+cu121 environment, and the initial dependency extraction incorrectly reflected that environment. Requirements and installation instructions have been corrected. All seven real-data smoke runs, four synthetic input checks, and loading of all four synthetic checkpoints passed again in `keyan_1`. Training checks used CPU with one PyTorch thread. No packages in the existing environment were changed.

All seven ARCH dependency pins and Python match the installed versions; `environment.yml` and `requirements.txt` agree. `python -m pip check` reported no broken requirements. PyTorch reports a CUDA 12.6 build, but CUDA devices were unavailable to this execution session, so GPU execution was not tested. A fresh environment installation was not tested.

## Checks performed

- Syntax parsing of all 29 Python files, including the launcher and verification tool.
- Executable AST comparisons against the pre-cleanup copy: ARCH model classes, training loops, evaluation masks, and sampling functions are unchanged. The language cleanup changed one synthetic error message; the subsequent data cleanup updated STSCP midpoint input paths. Real-data comparison orchestration and result structures were intentionally simplified.
- One-epoch CPU runs for all seven real-data tasks on earthquakes, with seed 42 and one sequence per split: next event, forecast, random imputation, trajectory recovery, partial attributes, inverse inference, and likelihood.
- Each real-data run saved `metrics.json`, `summary.json`, and `models.pt`. Prediction/recovery results contain only ARCH metrics. Likelihood exercised validation-based checkpoint selection and held-out LL/NLL output.
- Hawkes and self-correcting midpoint and dense-temporal `--check-data` commands: all four passed.
- SHA-256 verification of the 105 remaining source/data/reference artifacts against the updated manifest, plus syntax and real-data structure checks using `tools/verify_release.py`.
- All 79 retained data, checkpoint, and historical-result artifacts preserve their original release hashes. Nine duplicate Seahorse files were removed after exact numerical comparison; four STSCP plotting inputs were renamed without changing their contents.
- Chinese text and filenames were removed from the distributed text files. Real-data baseline modules, imports, functions, and launcher settings were removed. Documentation links and release checksums were refreshed.

Machine-readable records: [keyan_1 environment and run results](validation/keyan_1_environment.json), [integrity and split statistics](validation/integrity.json), and [earlier default-environment metrics](validation/smoke_results.json). These small runs are **not paper results**. Full temporary checkpoints and smoke outputs are outside the release directory.

## Data cleanup checks

All nine Seahorse split pairs were compared again, sequence-by-sequence and event-by-event, using exact numerical equality before deleting the duplicate `seahorse_original` directory. The original source files outside this release were verified against their recorded hashes. The retained `seahorse` and `prediction` families contain 18 split files; the integrity report validates their counts, event structure, and timestamp ordering.

The STSCP midpoint comparison directory is now `data/synthetic/stscp_midpoint_baselines`. Its query file and three saved baseline prediction files are all used by `self_correcting_midpoints.py`. The midpoint `--check-data` command was rerun from outside the release directory to verify the renamed default path. The [data cleanup report](validation/data_cleanup.json) records removed duplicates and renamed inputs.

All four synthetic checkpoints were loaded into their temporal/spatial models under `keyan_1`, with finite parameters verified. The earlier curation also checked finite synthetic temporal densities on selected queries; that inference check was not rerun during this environment audit.

## Not performed

- Full 70/160/200/300-epoch training or five-seed paper sweeps.
- Complete event-time spatial L2 evaluation or regeneration of every 3D figure.
- Fresh-environment dependency installation or CUDA execution.
- Reproduction of excluded ablations or external baseline training.
- Verification that historical logs/checkpoints correspond exactly to the final paper tables.

The source/paper differences in [curation notes](CURATION.md) remain explicit. Model and loss implementations were preserved.

## Repeat integrity checks

From the release directory:

```bash
python tools/verify_release.py --report /tmp/arch_integrity.json
sha256sum -c SHA256SUMS
```

Optional original-tree comparison on the author's machine:

```bash
python tools/verify_release.py --original-root ../keyan_odin_bak
```

`SHA256SUMS` covers all distributed files other than itself, excluding generated caches and outputs. Intentional source edits require updating the checksums; training outputs do not change the distributed inputs.
