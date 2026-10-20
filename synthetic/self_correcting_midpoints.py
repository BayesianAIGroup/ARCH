"""Plot STSCP midpoint comparisons using the stscp_midpoint_baselines results.

Run directly to plot sequence 20, with five randomly selected midpoints in
[30, 100]. The 1D plot starts after the ten-event history warmup and stops at
t=50 by default. ARCH and GT use 10 query points per event interval by default,
connected in time order across intervals, as in the original plot. Baselines use PCHIP display
interpolation through both midpoint and event samples, with raw sample markers.
Interpolated baseline values are for visualization, not new model predictions.

    python self_correcting_midpoints.py
    python self_correcting_midpoints.py --seq-idx 0 --device cpu
    python self_correcting_midpoints.py --temporal-only --lambda-t-max 50
    python self_correcting_midpoints.py --baseline-time-points midpoints
    python self_correcting_midpoints.py --check-data

Inference and model definitions are reused from the original script. No training
is performed. Midpoints are queries, never events in the conditioning history.
All 3D rows show joint intensity lambda(t,x,y), not spatial density p(x,y|t).
Output filenames include ``mid_point`` and are saved under ``mid_point_figures``.
The 1D plot is saved as PNG and the 3D comparison as PDF.
"""

from __future__ import annotations

from release_paths import DATA_ROOT, CHECKPOINT_ROOT, data_path

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

from self_correcting import (
    SpatioTemporalTraceformer,
    _pickle_load_cpu_compat,
    _precompute_stscp_kernels,
    _stoi_from_grid,
    get_intensity_function_dopri5,
    get_pxy_heatmap,
    set_seed,
)


SCRIPT_DIR = DATA_ROOT
BASELINE_NAMES = ("AutoSTPP", "DeepSTPP", "NSTPP")
ROW_NAMES = ("GT", "ARCH") + BASELINE_NAMES
FIRST_RIGHT_EVENT = 10
GT_PARAMETERS = dict(
    g0_cov=np.diag([0.25, 0.25]),
    g2_cov=np.diag([0.2, 0.2]),
    alpha=0.4, beta=0.2, mu=1.0, gamma=2.0, lamb_max=10.0,
    x_num=51, y_num=51, x_lim=(0.0, 1.0), y_lim=(0.0, 1.0),
)


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if (DATA_ROOT / path).exists():
        return DATA_ROOT / path
    if (CHECKPOINT_ROOT / path).exists():
        return CHECKPOINT_ROOT / path
    return Path.cwd() / path


def as_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class MidpointResults:
    seq_idx: int
    events: np.ndarray
    right_event_indices: np.ndarray
    # Axis 1: [midpoint, right event], exactly as in data_extract.py.
    query_times: np.ndarray
    temporal: dict
    joint_spatial: dict
    grid_size: int


def load_midpoint_results(baseline_root="stscp_midpoint_baselines", seq_idx=20):
    """Align exports by original right-event index, with explicit shape checks.

    AutoSTPP/DeepSTPP row j corresponds to right-event index j+10.
    NSTPP valid row n corresponds to right-event index n; its three slots are
    [interval start, midpoint, right event], so retain slots 1: and rows 10:.
    Dataset pair j is [midpoint_(j,j+1), event_(j+1)], so retain pairs 9:.
    """
    root = resolve_path(baseline_root)
    dataset = _pickle_load_cpu_compat(
        root / "datasets/stscp/data_test_with_inter_event_txy_padded_SC.pkl"
    )
    sequences = dataset["flat6_sequences"]
    if not 0 <= seq_idx < len(sequences):
        raise ValueError(f"seq_idx must be in [0, {len(sequences) - 1}]")

    # flat6 row = [event_t,x,y, midpoint_t,x,y]; the last midpoint is padding.
    packed = np.asarray(sequences[seq_idx], dtype=np.float64)
    if packed.ndim != 2 or packed.shape[1] != 6:
        raise ValueError(f"Expected [L, 6] dataset rows, got {packed.shape}")
    events = packed[:, :3].copy()
    if not np.isfinite(packed).all() or np.any(np.diff(events[:, 0]) <= 0):
        raise ValueError("Dataset must contain finite values and increasing event times")
    if not np.allclose(packed[-1, 3:], 0.0):
        raise ValueError("Expected a padding midpoint after the final event")
    midpoints = packed[:-1, 3:]
    expected_mid_times = 0.5 * (events[:-1, 0] + events[1:, 0])
    if not np.allclose(midpoints[:, 0], expected_mid_times, rtol=1e-6, atol=1e-6):
        raise ValueError("Exported query times are not event-interval midpoints")
    if np.any(midpoints[:, 0] <= events[:-1, 0]) or np.any(midpoints[:, 0] >= events[1:, 0]):
        raise ValueError("Each midpoint must be strictly inside its event interval")

    # Check correspondence to the test set used with the existing checkpoints.
    original = _pickle_load_cpu_compat(SCRIPT_DIR / data_path("stscp_out/data_test.pkl"))
    if len(original) != len(sequences) or not np.array_equal(events, original[seq_idx]):
        raise ValueError("Collaborator dataset does not match the original STSCP test sequence")
    length = len(events)
    right_indices = np.arange(FIRST_RIGHT_EVENT, length)
    if right_indices.size == 0:
        raise ValueError("Sequence has no baseline queries after the history warmup")
    query_times = np.column_stack((midpoints[right_indices - 1, 0], events[right_indices, 0]))

    result_dir = root / "result/stscp_tmid_s900"
    auto = _pickle_load_cpu_compat(result_dir / "AUTOSTPP.pkl")
    scales = as_numpy(auto["scales"])
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError(f"Invalid baseline scales: {scales}")
    if len(auto["test_t_ints"]) != len(sequences) or len(auto["test_st_sampling"]) != len(sequences):
        raise ValueError("AutoSTPP sequence count does not match the dataset")
    # AutoSTPP's exported intensities are already in original coordinates.
    temporal = {"AutoSTPP": as_numpy(auto["test_t_ints"][seq_idx]).copy()}
    joint = {"AutoSTPP": as_numpy(auto["test_st_sampling"][seq_idx]).copy()}
    del auto

    deep = _pickle_load_cpu_compat(result_dir / "DeepSTPP.pkl")
    indices = as_numpy(deep["indices"])
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f"Invalid DeepSTPP indices shape: {indices.shape}")
    selected = np.flatnonzero(indices[:, 0] == seq_idx)
    selected = selected[np.argsort(indices[selected, 1])]
    if not np.array_equal(indices[selected, 1], right_indices):
        raise ValueError("DeepSTPP right-event indices do not match the dataset")
    deep_event_t = np.concatenate([as_numpy(x) for x in deep["t_int"]])
    deep_mid_t = np.concatenate([as_numpy(x) for x in deep["t_int_between"]])
    deep_cond_s = np.concatenate([as_numpy(x) for x in deep["s_int_between"]])
    if not (len(deep_event_t) == len(deep_mid_t) == len(deep_cond_s) == len(indices)):
        raise ValueError("DeepSTPP batch lengths do not match its index table")
    if deep_mid_t.shape != (len(indices), 1):
        raise ValueError("Expected exactly one DeepSTPP midpoint per event interval")
    temporal["DeepSTPP"] = np.column_stack((deep_mid_t[selected, 0], deep_event_t[selected])) / scales[-1]
    # s_int_between is p(x,y|t); multiply by lambda(t) for joint intensity.
    joint["DeepSTPP"] = (deep_cond_s[selected] / np.prod(scales[:2])) * temporal["DeepSTPP"][..., None]
    del deep, deep_event_t, deep_mid_t, deep_cond_s

    nst = _pickle_load_cpu_compat(result_dir / "NSTPP.pkl")
    if not (len(nst["t_ints"]) == len(nst["mask"]) == len(nst["test_s_ints"])):
        raise ValueError("NSTPP batch counts are inconsistent")
    offset = 0
    found = False
    for t_batch, mask_batch, s_batch in zip(nst["t_ints"], nst["mask"], nst["test_s_ints"]):
        batch_size = len(t_batch)
        if not (batch_size == len(mask_batch) == len(s_batch)):
            raise ValueError("NSTPP sequence counts are inconsistent")
        if offset <= seq_idx < offset + batch_size:
            j = seq_idx - offset
            mask = as_numpy(mask_batch[j]).astype(bool)
            t_valid = as_numpy(t_batch[j])[mask]
            s_valid = as_numpy(s_batch[j])[mask]
            if t_valid.shape != (length, 3) or s_valid.shape != (length, 900, 3, 2):
                raise ValueError(f"Unexpected NSTPP shapes: t={t_valid.shape}, s={s_valid.shape}")
            temporal["NSTPP"] = t_valid[right_indices, 1:].copy()
            # Match data_extract.NSTPP_Processing: multiply the two exported
            # spatial factors. These already represent joint intensity.
            joint["NSTPP"] = s_valid[right_indices, :, 1:, :].prod(axis=-1).transpose(0, 2, 1)
            found = True
        offset += batch_size
    if not found or offset != len(sequences):
        raise ValueError("NSTPP sequence count does not match the dataset")
    del nst

    for name in BASELINE_NAMES:
        if temporal[name].shape != (len(right_indices), 2):
            raise ValueError(f"{name}: invalid temporal shape {temporal[name].shape}")
        if joint[name].shape != (len(right_indices), 2, 900):
            raise ValueError(f"{name}: invalid spatial shape {joint[name].shape}")
        for values in (temporal[name], joint[name]):
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{name}: intensities must be finite and nonnegative")
    print(f"Aligned seq={seq_idx}: {length} real events, {len(right_indices)} midpoint/event pairs; 30x30 spatial grid.", flush=True)
    return MidpointResults(seq_idx, events, right_indices, query_times, temporal, joint, 30)


def make_query_instance(events, right_event_index, t_query):
    """Original events 0..n-1 plus ONE query token; no midpoint history."""
    n = int(right_event_index)
    if not 0 <= n < len(events):
        raise ValueError("Query right-event index is outside the sequence")
    left_t = float(events[n - 1, 0]) if n else 0.0
    if not left_t < t_query <= events[n, 0]:
        raise ValueError("Query must follow its history and not pass the right event")
    times = np.concatenate(([0.0], events[:n, 0], [t_query]))
    locs = np.zeros((n + 1, 2), dtype=np.float32)
    locs[:n] = events[:n, 1:3]
    return {
        "length": n + 1,
        "deltas": torch.tensor(np.diff(times), dtype=torch.float32),
        "locs": torch.tensor(locs),
    }


class GroundTruth:
    """Same discrete-grid STSCP formula as the original script; cache kernels."""

    def __init__(self, events):
        self.events = events
        self.params = GT_PARAMETERS
        kernel_args = {key: self.params[key] for key in (
            "g0_cov", "g2_cov", "mu", "gamma", "x_num", "y_num", "x_lim", "y_lim"
        )}
        self.cache = _precompute_stscp_kernels(**kernel_args)
        event_cells = _stoi_from_grid(events[:, 1:3], self.params["x_num"], self.params["y_num"])
        self.suppression = np.concatenate((
            np.zeros((1, len(self.cache["g0_mat"]))),
            np.cumsum(self.cache["g2_mats"][event_cells], axis=0),
        ))

    def spatial_vectors(self, times):
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        # At an event time, evaluate BEFORE that event updates the history.
        history_lengths = np.searchsorted(self.events[:, 0], times, side="left")
        exponent = (times[:, None] * self.cache["g0_mat"] * self.params["beta"]
                    - self.params["alpha"] * self.suppression[history_lengths])
        exponent = np.minimum(exponent, np.log(self.params["lamb_max"] / self.params["mu"]))
        return self.params["mu"] * np.exp(exponent)

    def temporal(self, times):
        times = np.asarray(times)
        return np.concatenate([
            self.spatial_vectors(times[start:start + 256]).mean(axis=1)
            for start in range(0, len(times), 256)
        ])

    def heatmap(self, t_query, grid_size):
        samples = np.linspace(0.0, 1.0, grid_size)
        xx, yy = np.meshgrid(samples, samples, indexing="xy")
        cells = _stoi_from_grid(np.column_stack((xx.ravel(), yy.ravel())), self.params["x_num"], self.params["y_num"])
        return xx, yy, self.spatial_vectors([t_query])[0, cells].reshape(grid_size, grid_size)


def load_models(device, model_time_path, model_loc_path):
    models = []
    for path in (model_time_path, model_loc_path):
        state = torch.load(resolve_path(path), map_location="cpu", weights_only=True)
        model = SpatioTemporalTraceformer(
            loc_dim=2, d_model=32, n_heads=1, n_layers=4, d_ff=32,
            max_len=int(state["pos_emb"].shape[0]), dropout=0,
        )
        model.set_log_transform(True)
        model.set_RN_train(False)
        model.set_AR_train(True)
        model.load_state_dict(state)
        models.append(model.to(device).eval())
    return tuple(models)


@contextmanager
def cpu_safe_attention(device):
    """Give fully masked attention rows zero output on older CPU backends.

    PyTorch 2.4 CPU SDPA produces NaN for the history encoder's masked dummy
    row. Subsequent layers propagate it to real events and the ODE fails.
    Use zero attention for that empty row, preserving every nonempty row and
    its gradients. This is scoped to CPU plotting; restore SDPA on exit.
    """
    if torch.device(device).type != "cpu":
        yield
        return
    functional = torch.nn.functional
    original = functional.scaled_dot_product_attention

    def safe_attention(query, key, value, attn_mask=None, *args, **kwargs):
        if attn_mask is None:
            return original(query, key, value, attn_mask, *args, **kwargs)
        if attn_mask.dtype == torch.bool:
            empty_rows = ~attn_mask.any(dim=-1, keepdim=True)
            fill_value = True
        else:
            empty_rows = torch.isneginf(attn_mask).all(dim=-1, keepdim=True)
            fill_value = 0.0
        if not empty_rows.any():
            return original(query, key, value, attn_mask, *args, **kwargs)
        # Avoid NaN in the forward AND backward pass: do not compute an
        # all-masked softmax and then attempt to sanitize its NaN result.
        safe_mask = attn_mask.masked_fill(empty_rows, fill_value)
        output = original(query, key, value, safe_mask, *args, **kwargs)
        return output.masked_fill(empty_rows, 0.0)

    functional.scaled_dot_product_attention = safe_attention
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = original


def temporal_query_grid(results, baseline_time_points="both"):
    """Use exactly the baseline samples after the first ten history events."""
    if baseline_time_points == "both":
        return results.query_times.ravel(), np.repeat(results.right_event_indices, 2)
    if baseline_time_points == "midpoints":
        return results.query_times[:, 0], results.right_event_indices
    raise ValueError("baseline_time_points must be 'midpoints' or 'both'")


def sampled_learned_curve(model_time, results, device, baseline_time_points="both"):
    times, right_indices = temporal_query_grid(results, baseline_time_points)
    intensities = []
    for i, (n, t_query) in enumerate(zip(right_indices, times)):
        instance = make_query_instance(results.events, int(n), float(t_query))
        _, cif, _, _ = get_intensity_function_dopri5(model_time, instance, n=int(n), device=device)
        # Evaluate at the exact query time using the ODE endpoint. Do not
        # interpolate the previous dense curve or retain its interior samples.
        intensities.append(float(cif[-1]))
        if i == 0 or (i + 1) % 5 == 0 or i + 1 == len(times):
            print(f"ARCH temporal samples: {i + 1}/{len(times)} (right-event index >= 10)", flush=True)
    return times.copy(), np.asarray(intensities)


def temporal_time_window(results, lambda_t_max=50.0, baseline_time_points="both"):
    times, _ = temporal_query_grid(results, baseline_time_points)
    if not np.isfinite(lambda_t_max):
        raise ValueError("lambda_t_max must be finite")
    start, end = float(times[0]), min(float(lambda_t_max), float(times[-1]))
    if end <= start:
        raise ValueError(f"No temporal plotting interval between first query {start} and {end}")
    return start, end


def dense_learned_curve(model_time, results, device, lambda_t_max=50.0,
                        temporal_points=10, baseline_time_points="both"):
    """Evaluate the model itself densely, keeping real-event history intact."""
    if temporal_points < 2:
        raise ValueError("temporal_points must be at least 2 per event interval")
    start, end = temporal_time_window(results, lambda_t_max, baseline_time_points)
    all_times, all_intensities = [], []
    indices = [int(n) for n in results.right_event_indices
               if results.events[n, 0] >= start and results.events[n - 1, 0] < end]
    for i, n in enumerate(indices):
        left_t = float(results.events[n - 1, 0])
        right_t = min(float(results.events[n, 0]), end)
        instance = make_query_instance(results.events, n, right_t)
        _, cif, delta_grid, _ = get_intensity_function_dopri5(
            model_time, instance, n=n, device=device, num_points=temporal_points + 1,
        )
        times = left_t + np.asarray(delta_grid, dtype=np.float64)
        times[-1] = right_t  # retain the exact stored event/cutoff time
        keep = (times >= start) & (times <= end) & (times > left_t)
        times, values = times[keep], np.asarray(cif)[keep]
        if i == 0 and (len(times) == 0 or times[0] > start):
            first_query = make_query_instance(results.events, n, start)
            _, first_cif, _, _ = get_intensity_function_dopri5(
                model_time, first_query, n=n, device=device,
            )
            times = np.concatenate(([start], times))
            values = np.concatenate(([float(first_cif[-1])], values))
        all_times.append(times)
        all_intensities.append(values)
        if i == 0 or (i + 1) % 5 == 0 or i + 1 == len(indices):
            print(f"ARCH dense temporal evaluation: {i + 1}/{len(indices)} intervals (t <= {end:g})", flush=True)
    return np.concatenate(all_times), np.concatenate(all_intensities)


def interpolate_baselines(results, lambda_t_max=50.0, baseline_time_points="both"):
    """Shape-preserving visualization through the original baseline samples.

    Use the full knot sequence so t=50 can be interpolated between its actual
    neighbors even when the right neighbor is later than 50. Never extrapolate.
    These exports do not contain post-event limits, so an interpolated join
    across an event cannot recover that event's true instantaneous jump.
    """
    start, end = temporal_time_window(results, lambda_t_max, baseline_time_points)
    knots, _ = temporal_query_grid(results, baseline_time_points)
    if len(knots) < 2 or np.any(np.diff(knots) <= 0):
        raise ValueError("Baseline interpolation needs distinct increasing query times")
    visible_knots = knots[(knots >= start) & (knots <= end)]
    times = np.unique(np.concatenate((np.linspace(start, end, 2000), visible_knots)))
    curves = {}
    for name in BASELINE_NAMES:
        values = (results.temporal[name].ravel() if baseline_time_points == "both"
                  else results.temporal[name][:, 0])
        curves[name] = PchipInterpolator(knots, values, extrapolate=False)(times)
        if not np.isfinite(curves[name]).all() or np.any(curves[name] < 0):
            raise ValueError(f"Invalid interpolated {name} intensity")
    return times, curves


def plot_lambda_t_with_baselines(results, t_learn, lam_learn, ground_truth, out_file,
                                 baseline_time_points="both", lambda_t_max=50.0,
                                 dataset_label="STSCP"):
    start, end = temporal_time_window(results, lambda_t_max, baseline_time_points)
    t_learn, lam_learn = np.asarray(t_learn), np.asarray(lam_learn)
    if (t_learn.ndim != 1 or t_learn.size < 2 or lam_learn.shape != t_learn.shape
            or not np.isfinite(t_learn).all() or not np.isfinite(lam_learn).all()
            or np.any(np.diff(t_learn) <= 0) or t_learn[0] < start or t_learn[-1] > end):
        raise ValueError("Dense ARCH samples must be finite, increasing, and inside the plotting window")
    fig, ax = plt.subplots(figsize=(20, 5), dpi=220)
    lam_gt = ground_truth.temporal(t_learn)
    # Match the original plot: one time-ordered line for each method, including
    # the connections between neighboring event intervals. History is still
    # evaluated correctly at each query; only the line rendering is continuous.
    ax.plot(t_learn, lam_learn, color="#1f77b4", linewidth=1.35,
            alpha=0.9, label=r"ARCH $\lambda(t)$")
    ax.plot(t_learn, lam_gt, color="#ff7f0e", linewidth=1.35,
            alpha=0.9, linestyle="--", label=r"GT $\lambda(t|H)$")
    interp_times, interp_values = interpolate_baselines(results, lambda_t_max, baseline_time_points)
    knots, _ = temporal_query_grid(results, baseline_time_points)
    visible = (knots >= start) & (knots <= end)
    styles = (("#2ca02c", "o"), ("#d62728", "s"), ("#9467bd", "^"))
    for name, (color, marker) in zip(BASELINE_NAMES, styles):
        values = (results.temporal[name].ravel() if baseline_time_points == "both"
                  else results.temporal[name][:, 0])
        ax.plot(interp_times, interp_values[name], color=color, linewidth=1.2,
                alpha=0.8, label=f"{name} (PCHIP)")
        ax.scatter(knots[visible], values[visible], color=color, marker=marker,
                   s=9, alpha=0.5, zorder=3, linewidths=0)
    ev = results.events[FIRST_RIGHT_EVENT:, 0]
    ev = ev[(ev >= start) & (ev <= end)]
    ymin, ymax = ax.get_ylim()
    ax.vlines(ev[::max(1, int(np.ceil(len(ev) / 60)))], ymin, ymin + 0.06 * (ymax - ymin),
              alpha=0.35, linewidth=0.8, color="black")
    sample_label = "midpoints + events" if baseline_time_points == "both" else "midpoints"
    ax.set_title(f"{dataset_label}: temporal intensity (seq={results.seq_idx}; "
                 f"connected ARCH/GT; baseline interpolation through {sample_label})")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\lambda(t)$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.margins(x=0)
    # Event ticks must not extend the plot into the ten-event warmup period.
    ax.set_xlim(start, end)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved: {out_file}", flush=True)
    return lam_gt


def choose_midpoints(results, pick_k_times=5, time_range=(30.0, 100.0), seed=42):
    if pick_k_times <= 0:
        raise ValueError("pick_k_times must be positive")
    t_lo, t_hi = sorted(time_range)
    pool = np.flatnonzero((results.query_times[:, 0] >= t_lo) & (results.query_times[:, 0] <= t_hi))
    if len(pool) == 0:
        raise ValueError(f"No baseline midpoint queries in time range [{t_lo}, {t_hi}]")
    k = min(pick_k_times, len(pool))
    if k < pick_k_times:
        print(f"Only {k} midpoints are available in the requested time range.", flush=True)
    return np.sort(np.random.default_rng(seed).choice(pool, size=k, replace=False))


def plot_sc_3d_with_baselines(model_time, model_loc, results, chosen, ground_truth,
                             device, out_file):
    grid_size = results.grid_size
    columns = []
    for column_index, local_idx in enumerate(chosen):
        n = int(results.right_event_indices[local_idx])
        t_query = float(results.query_times[local_idx, 0])
        instance = make_query_instance(results.events, n, t_query)
        # Run the original ODE on [0, delta_mid]. Its final sample is exactly
        # the query; selecting the nearest point on the event grid is biased.
        _, cif, _, _ = get_intensity_function_dopri5(model_time, instance, n=n, device=device)
        delta_query = float(instance["deltas"][n])
        xx, yy, pxy = get_pxy_heatmap(model_loc, instance, n=n, delta_query=delta_query,
                                      grid_size=grid_size, device=device)
        _, _, gt_map = ground_truth.heatmap(t_query, grid_size)
        maps = {"GT": gt_map, "ARCH": float(cif[-1]) * pxy}
        for name in BASELINE_NAMES:
            # Export uses torch.meshgrid's ij ordering; our mesh is xy.
            maps[name] = results.joint_spatial[name][local_idx, 0].reshape(grid_size, grid_size).T
        for name, values in maps.items():
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Invalid {name} spatial intensity at t={t_query}")
        columns.append((t_query, maps))
        print(f"3D midpoint {column_index + 1}/{len(chosen)}: events {n - 1}/{n}, t={t_query:.6f}", flush=True)

    draw_sc_3d_comparison(columns, grid_size, out_file)
    return {name: np.stack([maps[name] for _, maps in columns]) for name in ROW_NAMES}


def draw_sc_3d_comparison(columns, grid_size, out_file):
    """Render computed spatial maps; PDF export does not require new inference."""
    samples = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(samples, samples, indexing="xy")
    n_rows, n_cols = len(ROW_NAMES), len(columns)
    fig = plt.figure(figsize=(3.6 * n_cols, 2.95 * n_rows), dpi=200)
    for c, (t_query, maps) in enumerate(columns):
        for r, name in enumerate(ROW_NAMES):
            values = maps[name]
            vmax = max(float(np.max(values)), 1e-8)
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection="3d")
            ax.set_anchor("NW")
            try:
                ax.set_box_aspect((1, 1, 0.58), zoom=1.04)
            except TypeError:
                ax.set_box_aspect((1, 1, 0.58))
            ax.plot_surface(xx, yy, values, cmap="YlGnBu", vmin=0.0, vmax=vmax,
                            linewidth=0, antialiased=True, shade=False, alpha=0.9)
            if r == 0:
                ax.set_title(f"$t_{{mid}}$={t_query:.2f}", fontsize=11, fontweight="bold", pad=2.0)
            ax.set_xlabel("x", labelpad=-2, fontsize=10, fontweight="bold")
            ax.set_ylabel("y", labelpad=-2, fontsize=10, fontweight="bold")
            ax.set_zlabel("λ", labelpad=-1, fontsize=10, fontweight="bold")
            ax.tick_params(axis="both", which="major", labelsize=8, pad=0)
            ax.tick_params(axis="z", which="major", labelsize=8, pad=0)
            for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
                tick.set_fontweight("bold")
            # Keep the original script's per-panel adaptive z limits.
            ax.set_zlim(0.0, 1.05 * vmax)
            ax.view_init(elev=28, azim=-60)
    for r, name in enumerate(ROW_NAMES):
        fig.text(0.078, 1.0 - (r + 0.5) / n_rows, name, va="center", ha="right",
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.998, top=0.94, bottom=0.04, wspace=-0.05, hspace=0.0)
    fig.savefig(out_file, bbox_inches="tight",
                bbox_extra_artists=list(fig.texts) + [ax.zaxis.label for ax in fig.axes])
    plt.close(fig)
    print(f"Saved: {out_file}", flush=True)


def visualize(device=None, model_time_path="best_model_time_SC.pth",
              model_loc_path="best_model_loc_SC.pth", seq_idx=20, pick_k_times=5,
              time_range=(30.0, 100.0), seed=42, baseline_root="stscp_midpoint_baselines",
              output_dir="mid_point_figures", baseline_time_points="both",
              check_data=False, cpu_threads=1, lambda_t_max=50.0,
              temporal_points=10, temporal_only=False):
    if baseline_time_points not in ("midpoints", "both"):
        raise ValueError("baseline_time_points must be 'midpoints' or 'both'")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(cpu_threads)
    set_seed(seed)
    results = load_midpoint_results(baseline_root, seq_idx)
    start, end = temporal_time_window(results, lambda_t_max, baseline_time_points)
    chosen = (np.array([], dtype=int) if temporal_only
              else choose_midpoints(results, pick_k_times, time_range, seed))
    if not temporal_only:
        print("Selected midpoint times:", results.query_times[chosen, 0].tolist(), flush=True)
    if check_data:
        return results
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}", flush=True)
    model_time, model_loc = load_models(device, model_time_path, model_loc_path)
    if len(results.events) + 1 > min(model_time.max_len, model_loc.max_len):
        raise ValueError("Sequence plus padding exceeds checkpoint positional encoding length")
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"SC_mid_point_seq{seq_idx}_smooth_{baseline_time_points}_tmax{end:g}"
           f"_n{temporal_points}_connected")
    temporal_file = output_dir / f"learned_vs_gt_baselines_lambdaT_{tag}.png"
    spatial_file = output_dir / f"stscp_3d_gt_arch_autostpp_deepstpp_nstpp_mid_point_seq{seq_idx}.pdf"
    ground_truth = GroundTruth(results.events)
    with cpu_safe_attention(device):
        times, intensities = dense_learned_curve(
            model_time, results, device, lambda_t_max, temporal_points, baseline_time_points,
        )
    if not np.isfinite(intensities).all():
        raise ValueError("ARCH temporal curve contains non-finite intensities")
    lam_gt = plot_lambda_t_with_baselines(results, times, intensities, ground_truth,
                                         temporal_file, baseline_time_points, lambda_t_max)
    interp_times, interp_values = interpolate_baselines(results, lambda_t_max, baseline_time_points)
    spatial_maps = {}
    if not temporal_only:
        with cpu_safe_attention(device):
            spatial_maps = plot_sc_3d_with_baselines(model_time, model_loc, results, chosen,
                                                    ground_truth, device, spatial_file)
    # Save the plotted values and chosen indices for reproducibility/replotting.
    arrays = dict(events=results.events, t_learn=times, lambda_arch=intensities,
                  lambda_gt=lam_gt, baseline_query_times=results.query_times,
                  baseline_right_event_indices=results.right_event_indices,
                  selected_local_indices=chosen,
                  selected_midpoint_times=results.query_times[chosen, 0])
    arrays.update({f"lambda_{name}": results.temporal[name] for name in BASELINE_NAMES})
    arrays["baseline_interpolation_times"] = interp_times
    arrays.update({f"lambda_{name}_interpolated": interp_values[name] for name in BASELINE_NAMES})
    arrays.update({f"spatial_{name}": values for name, values in spatial_maps.items()})
    np.savez_compressed(output_dir / f"plot_data_{tag}.npz", **arrays)
    manifest = dict(
        sequence_index=seq_idx, seed=seed, time_range=list(time_range),
        baseline_time_points=baseline_time_points,
        temporal_sampling="direct ARCH/GT evaluations, connected across event intervals",
        temporal_points_per_interval=temporal_points,
        temporal_time_range=[start, end],
        baseline_interpolation="PCHIP for visualization only; raw midpoint/event samples retained",
        baseline_event_discontinuities="post-event limits are unavailable and cannot be recovered by interpolation",
        temporal_first_right_event_index=FIRST_RIGHT_EVENT,
        baseline_pair_order=["midpoint", "right_event"],
        spatial_quantity="joint intensity lambda(t,x,y)",
        spatial_z_limits="adaptive per panel, as in the original script",
        selected_right_event_indices=results.right_event_indices[chosen].tolist(),
        selected_midpoint_times=results.query_times[chosen, 0].tolist(),
        baseline_root=str(resolve_path(baseline_root)),
        model_time_path=str(resolve_path(model_time_path)),
        model_loc_path=str(resolve_path(model_loc_path)),
        temporal_figure=str(temporal_file), spatial_figure=None if temporal_only else str(spatial_file),
    )
    (output_dir / f"plot_manifest_{tag}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-idx", type=int, default=20)
    parser.add_argument("--pick-k-times", type=int, default=5)
    parser.add_argument("--time-range", type=float, nargs=2, default=(30.0, 100.0), metavar=("START", "END"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Default: CUDA if available, otherwise CPU")
    parser.add_argument("--model-time-path", default="best_model_time_SC.pth")
    parser.add_argument("--model-loc-path", default="best_model_loc_SC.pth")
    parser.add_argument("--baseline-root", default="stscp_midpoint_baselines")
    parser.add_argument("--output-dir", default="mid_point_figures")
    parser.add_argument("--baseline-time-points", choices=("both", "midpoints"), default="both",
                        help="Baseline interpolation knots from right-event index 10: midpoints plus events (default), or only midpoints")
    parser.add_argument("--lambda-t-max", type=float, default=50.0,
                        help="Stop the 1D time axis at this value (default: 50); independent of the 3D time range")
    parser.add_argument("--temporal-points", type=int, default=10,
                        help="Number of directly evaluated positive ARCH query times per event interval (default: 10)")
    parser.add_argument("--temporal-only", action="store_true", help="Generate only the temporal comparison")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--check-data", action="store_true", help="Validate data alignment without model inference or plotting")
    visualize(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
