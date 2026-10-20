"""Plot HP and STSCP lambda(t) using the actual HP_stscp_t_int9 predictions.

    python temporal_intensity.py
    python temporal_intensity.py --dataset HP --seq-idx 21
    python temporal_intensity.py --check-data

Each interval supplies nine interior queries followed by its right event.
ARCH is evaluated at these same queries, with only real events in history.
All five curves directly connect their predictions; no PCHIP is used.
Default: sequence 20, a plot ending at t=50 AND a full-sequence plot, each as
PNG/PDF, plus the full numerical arrays and a manifest. No 3D inference runs.
"""

from __future__ import annotations

from release_paths import DATA_ROOT, CHECKPOINT_ROOT, data_path

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from self_correcting_midpoints import (
    BASELINE_NAMES, FIRST_RIGHT_EVENT, SCRIPT_DIR,
    GroundTruth as SCGroundTruth, GT_PARAMETERS as SC_GT_PARAMETERS,
    SpatioTemporalTraceformer, _pickle_load_cpu_compat, as_numpy,
    cpu_safe_attention, get_intensity_function_dopri5, make_query_instance,
    resolve_path, set_seed,
)
from hawkes_midpoints import (
    GroundTruth as HPGroundTruth, GT_PARAMETERS as HP_GT_PARAMETERS,
)
from build_hawkes_dense_queries import sha256_file


def load_dense_results(dataset, seq_idx=20, baseline_root="HP_stscp_t_int9"):
    """Validate every sequence and align all methods by original event index."""
    root = resolve_path(baseline_root)
    key = "hawkess_process" if dataset == "HP" else "stscp"
    original_path = SCRIPT_DIR / ("stpp_out" if dataset == "HP" else "stscp_out") / "data_test.pkl"
    dataset_path = root / "datasets" / key / (
        "data_test_HP_with_midpoints_bundle.pkl" if dataset == "HP"
        else "data_test_with_inter_event_txy_padded_SC.pkl"
    )
    original = _pickle_load_cpu_compat(original_path)
    exported = _pickle_load_cpu_compat(dataset_path)
    raw_sequences = exported["test_sequences" if dataset == "HP" else "flat6_sequences"]
    if len(raw_sequences) != len(original) or not 0 <= seq_idx < len(original):
        raise ValueError("Dataset sequence count or requested sequence index is invalid")
    all_events, all_times = [], []
    fractions = np.linspace(0.0, 1.0, 11)[1:]
    for i, (raw, real) in enumerate(zip(raw_sequences, original)):
        raw = as_numpy(raw)
        if dataset == "HP":
            if raw.shape != (2 * len(real) - 1, 3):
                raise ValueError(f"HP sequence {i}: invalid midpoint bundle shape")
            events = raw[::2]
        else:
            if raw.shape != (len(real), 6):
                raise ValueError(f"STSCP sequence {i}: invalid flat6 shape")
            events = raw[:, :3]
        if not np.array_equal(events, real) or not np.isfinite(events).all():
            raise ValueError(f"{dataset} sequence {i}: real events differ from original data")
        # Match the supplied data_extract.py, including source-dtype subtraction.
        event_t = events[:, :1]
        times = (event_t[:-1] + (event_t[1:] - event_t[:-1]) * fractions)[FIRST_RIGHT_EVENT - 1:]
        if times.shape != (len(events) - FIRST_RIGHT_EVENT, 10):
            raise ValueError(f"{dataset} sequence {i}: invalid interval grid")
        if not np.isfinite(times).all() or np.any(np.diff(times.ravel()) <= 0):
            raise ValueError(f"{dataset} sequence {i}: query times must strictly increase")
        np.testing.assert_array_equal(times[:, -1], events[FIRST_RIGHT_EVENT:, 0])
        expected = np.repeat(np.arange(FIRST_RIGHT_EVENT, len(events)), 10)
        np.testing.assert_array_equal(np.searchsorted(events[:, 0], times.ravel(), side="left"), expected)
        all_events.append(events.astype(np.float64))
        all_times.append(times)

    result_dir = root / "result" / f"{key}_tint"
    auto = _pickle_load_cpu_compat(result_dir / "AUTOSTPP.pkl")
    scales = as_numpy(auto["scales"])
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Invalid baseline scaling factors")
    predictions = {"AutoSTPP": [as_numpy(x).copy() for x in auto["test_t_ints"]]}
    del auto

    deep = _pickle_load_cpu_compat(result_dir / "DeepSTPP.pkl")
    indices = as_numpy(deep["indices"])
    mid = np.concatenate([as_numpy(x) for x in deep["t_int_between"]])
    event = np.concatenate([as_numpy(x) for x in deep["t_int"]])
    if indices.shape != (len(event), 2) or mid.shape != (len(event), 9):
        raise ValueError("Unexpected DeepSTPP query/index dimensions")
    if (len(indices) != sum(len(x) - FIRST_RIGHT_EVENT for x in all_events)
            or np.any(indices[:, 0] < 0) or np.any(indices[:, 0] >= len(all_events))):
        raise ValueError("DeepSTPP index table does not cover the original test set")
    values = np.column_stack((mid, event)) / scales[-1]
    predictions["DeepSTPP"] = []
    for i, events in enumerate(all_events):
        selected = np.flatnonzero(indices[:, 0] == i)
        selected = selected[np.argsort(indices[selected, 1])]
        np.testing.assert_array_equal(indices[selected, 1], np.arange(FIRST_RIGHT_EVENT, len(events)))
        predictions["DeepSTPP"].append(values[selected].copy())
    del deep, values, mid, event

    nst = _pickle_load_cpu_compat(result_dir / "NSTPP.pkl")
    if len(nst["t_ints"]) != len(nst["mask"]):
        raise ValueError("NSTPP batch/mask counts differ")
    predictions["NSTPP"] = []
    for batch, mask_batch in zip(nst["t_ints"], nst["mask"]):
        if len(batch) != len(mask_batch):
            raise ValueError("NSTPP batch and mask dimensions differ")
        for values, mask in zip(batch, mask_batch):
            i = len(predictions["NSTPP"])
            valid = as_numpy(values)[as_numpy(mask).astype(bool)]
            if i >= len(all_events) or valid.shape != (len(all_events[i]), 11):
                raise ValueError("NSTPP valid sequence length or query count is incorrect")
            predictions["NSTPP"].append(valid[FIRST_RIGHT_EVENT:, 1:].copy())
    del nst
    input_issues = []
    for name, sequences in predictions.items():
        if len(sequences) != len(all_events):
            raise ValueError(f"{name}: sequence count mismatch")
        for i, values in enumerate(sequences):
            if values.shape != all_times[i].shape:
                raise ValueError(f"{name} sequence {i}: temporal shape does not match query times")
            bad = ~np.isfinite(values) | (values < 0)
            if bad.any():
                issue = dict(method=name, sequence_index=i,
                             nonfinite_count=int((~np.isfinite(values)).sum()),
                             negative_count=int((values < 0).sum()),
                             invalid_interval_and_slot=np.argwhere(bad).tolist())
                input_issues.append(issue)
                if i == seq_idx:
                    raise ValueError(f"Selected {dataset} {name} sequence {i}: invalid intensities: {issue}")
                print(f"Input note ({dataset}, unselected sequence): {issue}", flush=True)
    sources = [original_path, dataset_path] + [result_dir / f"{name}.pkl" for name in ("AUTOSTPP", "DeepSTPP", "NSTPP")]
    print(f"{dataset}: aligned all {len(all_events)} sequences; validated selected seq={seq_idx}, "
          f"{len(all_events[seq_idx])} events, {all_times[seq_idx].size} actual query values per method.", flush=True)
    return dict(dataset=dataset, seq_idx=seq_idx, events=all_events[seq_idx],
                query_times=all_times[seq_idx],
                temporal={name: sequences[seq_idx] for name, sequences in predictions.items()},
                scales=scales, sources=sources, input_issues=input_issues)


def load_time_model(dataset, device):
    path = CHECKPOINT_ROOT / ("best_model_time.pth" if dataset == "HP" else "best_model_time_SC.pth")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = SpatioTemporalTraceformer(loc_dim=2, d_model=32, n_heads=1, n_layers=4,
                                     d_ff=32, max_len=int(state["pos_emb"].shape[0]), dropout=0)
    model.set_log_transform(True)
    model.set_RN_train(False)
    model.set_AR_train(True)
    model.load_state_dict(state)
    return model.to(device).eval(), path


def evaluate_arch(model, result, device):
    events, queries = result["events"], result["query_times"]
    if len(events) + 1 > model.max_len:
        raise ValueError("Sequence exceeds checkpoint positional encoding length")
    values = []
    with cpu_safe_attention(device):
        for j, times in enumerate(queries):
            n = j + FIRST_RIGHT_EVENT
            instance = make_query_instance(events, n, float(events[n, 0]))
            delta = times - events[n - 1, 0]
            _, cif, returned_delta, _ = get_intensity_function_dopri5(
                model, instance, n=n, device=device,
                query_deltas=np.concatenate(([0.0], delta)),
            )
            np.testing.assert_array_equal(returned_delta, delta.astype(np.float32))
            if cif.shape != (10,) or not np.isfinite(cif).all() or np.any(cif < 0):
                raise ValueError(f"Invalid ARCH intensity in interval {n}")
            values.append(cif)
            if j == 0 or (j + 1) % 10 == 0 or j + 1 == len(queries):
                print(f"{result['dataset']} ARCH: {j + 1}/{len(queries)} intervals complete", flush=True)
    return np.stack(values)


def draw(result, arch, gt, end, output_dir):
    times = result["query_times"].ravel()
    if not times[0] < end <= times[-1]:
        raise ValueError("Invalid temporal display window")
    fig, ax = plt.subplots(figsize=(20, 5), dpi=220)
    ax.plot(times, arch.ravel(), color="#1f77b4", linewidth=1.35, alpha=.9, label=r"ARCH $\lambda(t)$")
    ax.plot(times, gt.ravel(), color="#ff7f0e", linewidth=1.35, alpha=.9,
            linestyle="--", label=r"GT $\lambda(t|H)$")
    for name, color in zip(BASELINE_NAMES, ("#2ca02c", "#d62728", "#9467bd")):
        ax.plot(times, result["temporal"][name].ravel(), color=color, linewidth=1.2, alpha=.8, label=name)
    event_times = result["events"][FIRST_RIGHT_EVENT:, 0]
    event_times = event_times[(event_times >= times[0]) & (event_times <= end)]
    # Scale y to the visible window, including the line segment crossing its end.
    series = [arch.ravel(), gt.ravel()] + [result["temporal"][name].ravel() for name in BASELINE_NAMES]
    y = np.concatenate([np.append(v[times <= end], np.interp(end, times, v)) for v in series])
    low, high = float(y.min()), float(y.max())
    margin = .05 * max(high - low, high * .1, 1e-6)
    ax.set_ylim(max(0.0, low - margin), high + margin)
    ymin, ymax = ax.get_ylim()
    ax.vlines(event_times[::max(1, int(np.ceil(len(event_times) / 60)))], ymin,
              ymin + .06 * (ymax - ymin), alpha=.35, linewidth=.8, color="black")
    ax.set_title(f"{result['dataset']}: temporal intensity (seq={result['seq_idx']}; 10 samples per event interval)")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\lambda(t)$")
    ax.set_xlim(float(times[0]), end)
    ax.grid(True, alpha=.25)
    ax.legend()
    fig.tight_layout()
    tag = "full" if end == times[-1] else f"tmax{end:g}"
    stem = f"lambdaT_{result['dataset']}_seq{result['seq_idx']}_int9_n10_{tag}"
    paths = []
    for extension in ("png", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path)
        paths.append(str(path))
        print(f"Saved: {path}", flush=True)
    plt.close(fig)
    return paths


def visualize(dataset="both", seq_idx=20, baseline_root="HP_stscp_t_int9",
              output_dir="lambdaT_int9_figures", lambda_t_max=50., device=None,
              cpu_threads=1, check_data=False):
    if cpu_threads < 1 or not np.isfinite(lambda_t_max):
        raise ValueError("Use a positive CPU thread count and a finite time limit")
    torch.set_num_threads(cpu_threads)
    output_dir = resolve_path(output_dir)
    for name in (("HP", "STSCP") if dataset == "both" else (dataset,)):
        set_seed(42)
        result = load_dense_results(name, seq_idx, baseline_root)
        if check_data:
            continue
        end = min(lambda_t_max, float(result["query_times"][-1, -1]))
        if end <= result["query_times"][0, 0]:
            raise ValueError("No available queries inside the requested plotting window")
        inference_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{name}: inference device={inference_device}", flush=True)
        model, checkpoint = load_time_model(name, inference_device)
        arch = evaluate_arch(model, result, inference_device)
        ground_truth = (HPGroundTruth if name == "HP" else SCGroundTruth)(result["events"])
        gt = ground_truth.temporal(result["query_times"].ravel()).reshape(result["query_times"].shape)
        if not np.isfinite(gt).all() or np.any(gt < 0):
            raise ValueError("Invalid GT temporal intensity")
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"lambdaT_{name}_seq{seq_idx}_int9_n10"
        data_file = output_dir / f"{stem}_data.npz"
        np.savez_compressed(
            data_file, events=result["events"], query_times=result["query_times"],
            right_event_indices=np.arange(FIRST_RIGHT_EVENT, len(result["events"])),
            lambda_arch=arch, lambda_gt=gt,
            **{f"lambda_{key}": value for key, value in result["temporal"].items()},
        )
        figures = []
        for limit in sorted({end, float(result["query_times"][-1, -1])}):
            figures.extend(draw(result, arch, gt, limit, output_dir))
        manifest = dict(
            dataset=name, sequence_index=seq_idx, seed=42, device=inference_device,
            source_files={str(p): sha256_file(p) for p in result["sources"]},
            checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
            query_layout="[interval, 10]: nine interior queries followed by the real right event",
            first_right_event_index=FIRST_RIGHT_EVENT,
            history="Original real events[:right_event_index]; evaluate before inserting the right event",
            baseline_interpolation=False,
            baseline_conversions="AutoSTPP and NSTPP as supplied; DeepSTPP divided by AutoSTPP scales[-1]",
            baseline_scales=result["scales"].tolist(),
            unselected_sequence_input_issues=result["input_issues"],
            sampling="Use collaborator query timestamps; ARCH evaluates supplied deltas cast to model float32",
            gt_parameters=HP_GT_PARAMETERS if name == "HP" else SC_GT_PARAMETERS,
            window=[float(result["query_times"][0, 0]), end],
            full_end=float(result["query_times"][-1, -1]), plot_data=str(data_file), figures=figures,
        )
        (output_dir / f"{stem}_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=lambda x: x.tolist()) + "\n", encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("both", "HP", "STSCP"), default="both")
    parser.add_argument("--seq-idx", type=int, default=20)
    parser.add_argument("--baseline-root", default="HP_stscp_t_int9")
    parser.add_argument("--output-dir", default="lambdaT_int9_figures")
    parser.add_argument("--lambda-t-max", type=float, default=50.)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--check-data", action="store_true")
    visualize(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
