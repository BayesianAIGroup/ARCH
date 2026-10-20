#!/usr/bin/env python3
"""Portable experiment entry point. Run --help before starting a full experiment."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
TASKS = ("next_event", "forecast", "random_imputation", "trajectory_recovery",
         "partial_attributes", "inverse_inference", "likelihood")


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="experiment", required=True)
    real = sub.add_parser("real", help="Real-world prediction, recovery, and Seahorse LL")
    real.add_argument("--task", choices=TASKS, required=True)
    real.add_argument("--dataset", choices=("earthquakes", "COVID19", "Citibike"), required=True)
    real.add_argument("--epochs", type=positive, help="Default: 300; likelihood: 200")
    real.add_argument("--seeds", type=int, nargs="+", default=[42, 39, 41, 16, 46])
    real.add_argument("--suffix-length", type=positive, default=5)
    real.add_argument("--missing-ratio", type=float, default=0.2)
    real.add_argument("--limit-sequences", type=positive, help="Smoke checks only: truncate each split")
    syn = sub.add_parser("synthetic", help="Synthetic intensity experiments")
    syn.add_argument("--process", choices=("hawkes", "self_correcting"), required=True)
    syn.add_argument("--action", choices=("train", "plot", "intensity-error", "midpoints", "temporal"), required=True)
    syn.add_argument("--epochs", type=positive, help="Default: Hawkes 70; self-correcting 160")
    syn.add_argument("--seed", type=int, default=42)
    syn.add_argument("--checkpoint-dir", type=Path, help="Default: supplied synthetic checkpoints")
    syn.add_argument("--max-sequences", type=positive, default=200)
    syn.add_argument("--check-data", action="store_true", help="Validate midpoint/temporal inputs without inference")
    for parser in (real, syn):
        parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
        parser.add_argument("--cpu-threads", type=positive, default=1)
        parser.add_argument("--output-dir", type=Path, help="New directory; refuses to overwrite existing runs")
    args = p.parse_args()
    if args.experiment == "real" and not 0 < args.missing_ratio < 1:
        p.error("--missing-ratio must be between 0 and 1")
    if args.experiment == "real" and len(set(args.seeds)) != len(args.seeds):
        p.error("--seeds must be unique")
    if args.experiment == "synthetic" and args.check_data and args.action not in ("midpoints", "temporal"):
        p.error("--check-data is supported for midpoints and temporal")
    return args


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            result.update(flatten(v, f"{prefix}/{k}".strip("/")))
        return result
    if isinstance(value, (tuple, list)):
        return flatten(dict(enumerate(value)), prefix)
    if isinstance(value, (float, int)) and math.isfinite(value):
        return {prefix: float(value)}
    return {}


def run_real(args, torch):
    from release_config import settings
    for key in ("suffix_length", "missing_ratio"):
        setattr(settings, key, getattr(args, key))
    folder = "likelihood" if args.task == "likelihood" else "real_world"
    sys.path.insert(0, str(ROOT / folder))
    module = importlib.import_module("seahorse_likelihood" if args.task == "likelihood" else args.task)
    if args.limit_sequences:
        original_loader = module.load_earthquake_st_ori
        def limited_loader(*a, **kw):
            data = original_loader(*a, **kw)
            for split in ("train", "dev", "test"):
                data[split] = data[split][:args.limit_sequences]
            data["max_len"] = max(int(x["length"]) for split in ("train", "dev", "test") for x in data[split])
            return data
        module.load_earthquake_st_ori = limited_loader
    epochs = args.epochs or (200 if args.task == "likelihood" else 300)
    original = module.run_two_stage
    signature = inspect.signature(original)

    def configured_run(*positional, **kwargs):
        bound = signature.bind(*positional, **kwargs)
        bound.arguments["max_epochs"] = epochs
        if args.limit_sequences:
            for split in ("train_data", "val_data", "test_data"):
                bound.arguments[split] = bound.arguments[split][:args.limit_sequences]
            bound.arguments["max_len"] = 1 + max(
                int(x["length"]) for split in ("train_data", "val_data", "test_data")
                for x in bound.arguments[split])
        result = original(*bound.args, **bound.kwargs)
        # Prediction evaluates the final epoch. LL restores its validation-selected state.
        torch.save({"model_time": bound.arguments["model_time"].state_dict(),
                    "model_loc": bound.arguments["model_loc"].state_dict(),
                    "config": {**vars(args), "epochs": epochs},
                    "selection": "validation LL" if args.task == "likelihood" else "final epoch"},
                   "models.pt")
        return result

    module.run_two_stage = configured_run
    records = []
    base = Path.cwd()
    for index, seed in enumerate(args.seeds):
        work = base / f"seed_{seed}"
        work.mkdir()
        os.chdir(work)
        module.set_seed(seed)
        call_args = {}
        if "eval_seed" in inspect.signature(module.test_event_st_two_stage).parameters:
            call_args["eval_seed"] = 1000 + index
        if "num_run" in inspect.signature(module.test_event_st_two_stage).parameters:
            call_args["num_run"] = index
        value = module.test_event_st_two_stage(args.device, args.dataset, **call_args)
        if args.task == "likelihood":
            value[2]["avg_nll_joint_raw_lon_lat"] = -value[2]["avg_loglik_joint_raw_lon_lat"]
        json_write("metrics.json", {"seed": seed, "result": value})
        records.append({"seed": seed, "metrics": flatten(value)})
        os.chdir(base)
    import numpy as np
    summary = {}
    for key in sorted(set.intersection(*(set(r["metrics"]) for r in records))):
        values = [r["metrics"][key] for r in records]
        summary[key] = {"mean": float(np.mean(values)), "std_population": float(np.std(values)),
                        "standard_error_population": float(np.std(values) / np.sqrt(len(values))),
                        "values": values}
    json_write("summary.json", {"runs": records, "metrics": summary,
                               "epochs": epochs, "smoke_only": bool(args.limit_sequences)})


def run_synthetic(args, torch):
    sys.path.insert(0, str(ROOT / "synthetic"))
    sc = args.process == "self_correcting"
    ckpts = args.checkpoint_dir or ROOT / "checkpoints" / "synthetic"
    paths = {"model_time_path": str(ckpts / ("best_model_time_SC.pth" if sc else "best_model_time.pth")),
             "model_loc_path": str(ckpts / ("best_model_loc_SC.pth" if sc else "best_model_loc.pth"))}
    module = importlib.import_module(args.process)
    module.set_seed(args.seed)
    if args.action == "train":
        module.train(device=args.device, save_best=True, epochs=args.epochs)
    elif args.action == "plot":
        module.visualize(device=args.device, **paths)
    elif args.action == "intensity-error":
        module = importlib.import_module(args.process + "_intensity_error")
        fn = getattr(module, "compute_saved_models_event_lambda_l2_sc" if sc else "compute_saved_models_event_lambda_l2")
        fn(device=args.device, max_sequences=args.max_sequences, **paths)
    elif args.action == "midpoints":
        module = importlib.import_module(args.process + "_midpoints")
        module.visualize(device=args.device, check_data=args.check_data, output_dir=str(Path.cwd() / "figures"),
                         seed=args.seed, **paths)
    else:
        module = importlib.import_module("temporal_intensity")
        # This module uses the supplied temporal checkpoints; custom paths are explicit here.
        if args.checkpoint_dir:
            module.CHECKPOINT_ROOT = ckpts
        module.visualize(dataset="STSCP" if sc else "HP", device=args.device,
                         check_data=args.check_data, output_dir=str(Path.cwd() / "figures"))


def main():
    args = arguments()
    os.environ.setdefault("MPLBACKEND", "Agg")
    import torch
    torch.set_num_threads(args.cpu_threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --device cpu or a CUDA environment.")
    # ODE divergence requires differentiable attention on CPU as well as CUDA.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if args.experiment == "synthetic" and args.checkpoint_dir:
        args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    tag = (args.task + "_" + args.dataset) if args.experiment == "real" else (args.process + "_" + args.action)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output_dir or ROOT / "outputs" / f"{tag}_{timestamp}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    json_write("run_config.json", {**vars(args), "output_dir": str(output), "torch": torch.__version__,
                                   "python": sys.version, "command": sys.argv})
    print(f"Results: {output}")
    with open("console.log", "w", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            if args.experiment == "real":
                run_real(args, torch)
            else:
                run_synthetic(args, torch)


if __name__ == "__main__":
    main()
