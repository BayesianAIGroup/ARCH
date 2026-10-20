# ARCH

**Arbitrarily Conditioned Hierarchical Flows for Spatiotemporal Events**

ARCH models event times and locations using conditional temporal and spatial flow models. This repository provides training and evaluation code for next-event prediction, multi-event forecasting, missing-event recovery, inverse inference, held-out likelihood evaluation, and synthetic intensity estimation.

Real-world training and evaluation cover ARCH. Synthetic comparisons additionally use precomputed baseline predictions, as described below.

## Installation

The software versions for this release are listed below.

| Component | Version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu126 (CUDA 12.6 build) |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| Matplotlib | 3.10.8 |
| torchdiffeq | 0.2.5 |
| python-dateutil | 2.9.0.post0 |
| joblib | 1.5.3 |

GPU execution requires an NVIDIA driver compatible with the CUDA 12.6 build of PyTorch. The launcher automatically selects CUDA when available. Use `--device cpu` or `--device cuda:0` to select a device explicitly.

## Data preparation

Download the [data archive from Google Drive](https://drive.google.com/file/d/1zRYz6cA068DdPm19sRd5VEi-O4XaTFhI/view?usp=drive_link) and extract its contents into the repository root (the directory containing `run.py`) before running the experiments.

After extraction, the data should be accessible under `data/`:

```text
ARCH/
  run.py
  data/
    prediction/
    seahorse/
    synthetic/
  ...
```

Ensure that the extracted files are placed directly in the repository root without an additional enclosing directory. See [data and checkpoints](docs/DATA.md) for details about the expected inputs and checkpoint files.

## Quick start

Run the following command from the repository root to train and evaluate ARCH for next-event prediction on the earthquake dataset:

```bash
python run.py real --task next_event --dataset earthquakes
```

The launcher resolves input paths relative to the repository root and saves each run in a new directory under `outputs/`.

## Prediction and recovery

Select a dataset using `--dataset earthquakes`, `--dataset COVID19`, or `--dataset Citibike`.

| Task | Command |
|---|---|
| Next-event prediction | `python run.py real --task next_event --dataset earthquakes` |
| Joint forecasting of the next K events | `python run.py real --task forecast --dataset earthquakes --suffix-length 5` |
| Random missing-event recovery | `python run.py real --task random_imputation --dataset earthquakes --missing-ratio 0.2` |
| Contiguous segment recovery | `python run.py real --task trajectory_recovery --dataset earthquakes` |
| Partial attribute recovery | `python run.py real --task partial_attributes --dataset earthquakes` |
| Inverse inference | `python run.py real --task inverse_inference --dataset earthquakes` |

By default, prediction and recovery experiments train for **300 epochs** using seeds **42, 39, 41, 16, and 46**. Use `--epochs N` to change the training budget or `--seeds 42` to run a single seed. Each task trains its own temporal and spatial models.

- **Forecasting:** Use `--suffix-length K`. The settings used in the paper are 5, 10, 15, 20, and 30.
- **Random missing-event recovery:** Use `--missing-ratio R`. The settings used in the paper are 0.1, 0.2, and 0.3.
- **Contiguous segment recovery:** Segment lengths of 5 and 10 are evaluated in a single run.
- **Partial attribute recovery:** Missing ratios of 5% and 10% are evaluated in a single run.
- **Inverse inference:** The first event, the second event, and the first two events are evaluated as separate recovery settings.

## Held-out likelihood

Held-out likelihood experiments use the data splits in `data/seahorse/`, which are separate from those in `data/prediction/`. Model checkpoints are selected based on validation likelihood. The default training budget is **200 epochs**.

```bash
python run.py real --task likelihood --dataset earthquakes --seeds 46 41 39
python run.py real --task likelihood --dataset COVID19 --seeds 16 42 39
python run.py real --task likelihood --dataset Citibike --seeds 42 39 41
```

The Citibike seed configuration above is provided as a reproducible example; it has not been verified against the seeds used in the original experiments.

Reported results include **NLL = −average joint log-likelihood**, evaluated in raw longitude/latitude coordinates. See [metric definitions](docs/PAPER_MAP.md) for output fields and coordinate conventions.

## Synthetic experiments

Select a synthetic process using `--process hawkes` or `--process self_correcting`. The default training budgets are **70 epochs** for Hawkes and **160 epochs** for the self-correcting process.

```bash
# Train ARCH on each synthetic process.
python run.py synthetic --process hawkes --action train
python run.py synthetic --process self_correcting --action train

# Evaluate and visualize the Hawkes process.
python run.py synthetic --process hawkes --action plot
python run.py synthetic --process hawkes --action intensity-error --max-sequences 200
python run.py synthetic --process hawkes --action midpoints
python run.py synthetic --process hawkes --action temporal
```

The same evaluation actions are available for `self_correcting`. Add `--check-data` to `midpoints` or `temporal` to validate the required inputs without running inference.

By default, evaluation loads the supplied checkpoints from `checkpoints/synthetic/`. To evaluate newly trained models, pass `--checkpoint-dir outputs/<training-run>`.

Synthetic plots and intensity comparisons use precomputed predictions from AutoSTPP, DeepSTPP, and NSTPP. These prediction arrays are required inputs; the evaluation commands do not train the baseline models.

For the self-correcting process, midpoint queries and the three precomputed baseline predictions are stored in `data/synthetic/stscp_midpoint_baselines/` and used by `--process self_correcting --action midpoints`.

## Results and outputs

Each run creates a directory such as `outputs/next_event_earthquakes_<timestamp>/`, containing `run_config.json` and `console.log`. Real-world runs additionally save model weights and evaluation metrics:

```text
outputs/next_event_earthquakes_<timestamp>/
  run_config.json
  console.log
  seed_42/
    models.pt       # Temporal and spatial model weights
    metrics.json    # Evaluation results for this seed
  summary.json      # Per-seed values, mean, population standard deviation,
                    # and standard error
```

Additional seeds are saved in separate `seed_<seed>/` directories. Use `--output-dir <new-directory>` to specify an output location; the directory must not already exist. Synthetic experiments save checkpoints, figures, or numerical results in their respective run directories.

## Repository structure

```text
run.py                  Experiment launcher
real_world/             ARCH prediction and recovery
likelihood/             Seahorse likelihood evaluation
synthetic/              Synthetic models, evaluation, and data helpers
data/                   Real-world and synthetic inputs
checkpoints/synthetic/  Supplied synthetic model weights
reference_results/      Historical logs and metrics
docs/                   Data documentation, experiment mapping, and validation notes
provenance/             Source mappings, hashes, and change notes
tools/                  Release verification utilities
```

## Verification and documentation

Run the following commands to verify the release files and inspect the available command-line options:

```bash
python tools/verify_release.py
sha256sum -c SHA256SUMS
python run.py real --help
python run.py synthetic --help
```

Additional documentation is available in [data and checkpoints](docs/DATA.md), [paper-to-code mapping](docs/PAPER_MAP.md), [validation](docs/VALIDATION.md), and [implementation notes](docs/CURATION.md).

## Attribution and licensing

Source attribution is recorded in `provenance/`, and retained third-party notices are provided in `licenses/`. This release does not assign a repository-wide license or a dataset license.
