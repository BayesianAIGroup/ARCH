# ARCH

**Arbitrarily Conditioned Hierarchical Flows for Spatiotemporal Events**

ARCH models event times and locations with two conditional flow models. This repository includes training and evaluation for next-event prediction, forecasting, missing-event recovery, inverse inference, held-out likelihood, and synthetic intensity estimation.

Real-world experiments train and evaluate ARCH only.

## Installation

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

GPU execution requires a driver compatible with the CUDA 12.6 PyTorch build. The launcher selects CUDA when available; use `--device cpu` or `--device cuda:0` to select a device explicitly. 

## Quick start

For a full next-event training run:

```bash
python run.py real --task next_event --dataset earthquakes
```

The launcher resolves input paths relative to this repository and saves each run in a new directory under `outputs/`.

## Prediction and recovery

Choose a dataset with `--dataset earthquakes`, `--dataset COVID19`, or `--dataset Citibike`.

| Task | Command |
|---|---|
| Next-event prediction | `python run.py real --task next_event --dataset earthquakes` |
| Joint forecasting of the next K events | `python run.py real --task forecast --dataset earthquakes --suffix-length 5` |
| Random missing-event recovery | `python run.py real --task random_imputation --dataset earthquakes --missing-ratio 0.2` |
| Contiguous segment recovery | `python run.py real --task trajectory_recovery --dataset earthquakes` |
| Partial attribute recovery | `python run.py real --task partial_attributes --dataset earthquakes` |
| Inverse inference | `python run.py real --task inverse_inference --dataset earthquakes` |

Prediction and recovery default to **300 epochs** and seeds **42, 39, 41, 16, 46**. Use `--epochs N` and `--seeds 42` to change the budget or run a single seed. Each task trains its own temporal and spatial models.

- Forecasting uses `--suffix-length K`; paper settings are 5, 10, 15, 20, and 30.
- Random recovery uses `--missing-ratio R`; paper settings are 0.1, 0.2, and 0.3.
- Contiguous recovery evaluates segment lengths 5 and 10 in one run.
- Partial attribute recovery evaluates missing ratios 5% and 10% in one run.
- Inverse inference evaluates the first event, second event, and first two events.

## Held-out likelihood

Likelihood reads `data/seahorse/` and selects model weights using validation likelihood. These splits are separate from `data/prediction/`; the numerically identical `seahorse_original` copy has been removed. The default training budget is **200 epochs**.

```bash
python run.py real --task likelihood --dataset earthquakes --seeds 46 41 39
python run.py real --task likelihood --dataset COVID19 --seeds 16 42 39
python run.py real --task likelihood --dataset Citibike --seeds 42 39 41
```

The Citibike seeds above are a reproducible example; their historical use has not been verified. Results include **NLL = -average joint LL** in raw longitude/latitude coordinates. See [metric definitions](docs/PAPER_MAP.md) for output fields and coordinate conventions.

## Synthetic experiments

Use `--process hawkes` or `--process self_correcting`. Default training budgets are **70** and **160** epochs, respectively.

```bash
python run.py synthetic --process hawkes --action train
python run.py synthetic --process self_correcting --action train
python run.py synthetic --process hawkes --action plot
python run.py synthetic --process hawkes --action intensity-error --max-sequences 200
python run.py synthetic --process hawkes --action midpoints
python run.py synthetic --process hawkes --action temporal
```

The same evaluation actions are available for `self_correcting`. Add `--check-data` to `midpoints` or `temporal` to validate their inputs without running inference.

Evaluation loads the supplied checkpoints from `checkpoints/synthetic/`. To use newly trained weights, pass `--checkpoint-dir outputs/<training-run>`.

Synthetic plots and intensity comparisons use bundled AutoSTPP, DeepSTPP, and NSTPP predictions. These saved arrays remain required inputs; the commands do not train those baselines.

STSCP midpoint comparison inputs are in `data/synthetic/stscp_midpoint_baselines/`. This directory contains the midpoint queries and the three saved baseline predictions used by `--process self_correcting --action midpoints`.

## Results

Each run creates a directory such as `outputs/next_event_earthquakes_<timestamp>/` containing `run_config.json` and `console.log`. Real-world runs also save:

```text
seed_42/
  models.pt       temporal and spatial model weights
  metrics.json    ARCH evaluation results for this seed
summary.json     per-seed values, mean, population standard deviation and standard error
```

Additional seeds create their own `seed_<seed>/` directories. Use `--output-dir <new-directory>` to choose an output location; the directory must not already exist. Synthetic actions save their checkpoints, figures, or numerical results in their run directory.

## Repository layout

```text
run.py                  experiment launcher
real_world/             ARCH prediction and recovery
likelihood/             Seahorse likelihood evaluation
synthetic/              synthetic models, evaluation, and data helpers
data/                   real-world and synthetic inputs
checkpoints/synthetic/  supplied synthetic model weights
reference_results/      historical logs and metrics
docs/                   data, experiment mapping, and validation notes
provenance/             source mappings, hashes, and change notes
tools/                  release verification
```

## Verification and documentation

```bash
python tools/verify_release.py
sha256sum -c SHA256SUMS
python run.py real --help
python run.py synthetic --help
```

See [data and checkpoints](docs/DATA.md), [paper-to-code mapping](docs/PAPER_MAP.md), [validation](docs/VALIDATION.md), and [implementation notes](docs/CURATION.md). Source attribution is recorded in `provenance/`; retained third-party notices are in `licenses/`. No repository-wide or dataset license is assigned by this release.
