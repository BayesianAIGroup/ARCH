"""Plot HP midpoint baselines using HP_stpp_tmid_S900.

Defaults follow the latest STSCP midpoint plots: sequence 20, five random
midpoints, joint 3D intensity as PDF, and a temporal plot ending at t=50.
ARCH/GT use 10 points per event interval and connected lines. Baselines use
PCHIP through both midpoint and event samples, starting at right-event index 10.

    python hawkes_midpoints.py
    python hawkes_midpoints.py --temporal-only
    python hawkes_midpoints.py --seq-idx 0 --seed 7
    python hawkes_midpoints.py --check-data
    python hawkes_midpoints.py --spatial-only --midpoint-right-indices 24 32 49 55 63

The last command redraws five alternative midpoints with distinct GT peak
patterns. Indices refer to the original, zero-based right-event indices.
Explicit selections use separate output names so previous figures are retained.

HP ground truth and checkpoints follow hawkes.py. Its
time/space inference functions are identical to the SC versions; shared
utilities keep the plotting protocol consistent. Midpoints never update the
history. Spatial ODE queries are batched, then normalized over the whole grid.
All outputs are saved under hp_mid_point_figures with HP-specific filenames.
"""

from __future__ import annotations

from release_paths import DATA_ROOT, CHECKPOINT_ROOT, data_path

import argparse
import json

import numpy as np
import torch

from self_correcting_midpoints import (
    BASELINE_NAMES, FIRST_RIGHT_EVENT, ROW_NAMES, SCRIPT_DIR, MidpointResults,
    _pickle_load_cpu_compat, as_numpy, choose_midpoints, cpu_safe_attention,
    dense_learned_curve, draw_sc_3d_comparison, get_intensity_function_dopri5,
    interpolate_baselines, load_models, make_query_instance,
    plot_lambda_t_with_baselines, resolve_path, set_seed, temporal_time_window,
)
from hawkes import (
    get_logprob_loc_dopri5, gt_lambda_txy_heatmap, lambda_t_given_history,
)


GT_PARAMETERS = dict(mu=1.0, alpha=0.72, beta=1.2, sigma=0.05,
                     x_lim=(0.0, 1.0), y_lim=(0.0, 1.0))


class GroundTruth:
    """Original HP GT, including Gaussian mass inside the spatial window."""

    def __init__(self, events):
        self.events = np.asarray(events, dtype=np.float64)

    def temporal(self, times):
        return lambda_t_given_history(np.asarray(times, dtype=np.float64),
                                      self.events, **GT_PARAMETERS)

    def heatmap(self, t_query, grid_size):
        return gt_lambda_txy_heatmap(t_query, self.events, grid_size=grid_size,
                                    **GT_PARAMETERS)


def _load_baseline_arrays(result_dir, seq_idx, length, sequence_count):
    """Read the collaborator's [midpoint, right event] intensity pairs."""
    right_indices = np.arange(FIRST_RIGHT_EVENT, length)
    auto = _pickle_load_cpu_compat(result_dir / "AUTOSTPP.pkl")
    scales = as_numpy(auto["scales"])
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError(f"Invalid baseline scales: {scales}")
    if len(auto["test_t_ints"]) != sequence_count or len(auto["test_st_sampling"]) != sequence_count:
        raise ValueError("AutoSTPP sequence count does not match HP data")
    # AutoSTPP outputs are already in the original time/space coordinates.
    temporal = {"AutoSTPP": as_numpy(auto["test_t_ints"][seq_idx]).copy()}
    joint = {"AutoSTPP": as_numpy(auto["test_st_sampling"][seq_idx]).copy()}
    del auto

    deep = _pickle_load_cpu_compat(result_dir / "DeepSTPP.pkl")
    indices = as_numpy(deep["indices"])
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f"Invalid DeepSTPP indices: {indices.shape}")
    selected = np.flatnonzero(indices[:, 0] == seq_idx)
    selected = selected[np.argsort(indices[selected, 1])]
    if not np.array_equal(indices[selected, 1], right_indices):
        raise ValueError("DeepSTPP right-event indices do not match HP data")
    event_t = np.concatenate([as_numpy(x) for x in deep["t_int"]])
    midpoint_t = np.concatenate([as_numpy(x) for x in deep["t_int_between"]])
    conditional_s = np.concatenate([as_numpy(x) for x in deep["s_int_between"]])
    if not (len(event_t) == len(midpoint_t) == len(conditional_s) == len(indices)):
        raise ValueError("DeepSTPP batch lengths do not match the index table")
    if midpoint_t.shape != (len(indices), 1):
        raise ValueError("Expected one DeepSTPP midpoint per interval")
    temporal["DeepSTPP"] = np.column_stack((midpoint_t[selected, 0], event_t[selected])) / scales[-1]
    # DeepSTPP stores p(x,y|t). Convert to joint lambda(t,x,y) for all 3D rows.
    joint["DeepSTPP"] = conditional_s[selected] / np.prod(scales[:2]) * temporal["DeepSTPP"][..., None]
    del deep, event_t, midpoint_t, conditional_s

    nst = _pickle_load_cpu_compat(result_dir / "NSTPP.pkl")
    if not (len(nst["t_ints"]) == len(nst["mask"]) == len(nst["test_s_ints"])):
        raise ValueError("NSTPP batch counts are inconsistent")
    offset, found = 0, False
    for t_batch, mask_batch, s_batch in zip(nst["t_ints"], nst["mask"], nst["test_s_ints"]):
        batch_size = len(t_batch)
        if not (batch_size == len(mask_batch) == len(s_batch)):
            raise ValueError("NSTPP sequence counts are inconsistent")
        if offset <= seq_idx < offset + batch_size:
            j = seq_idx - offset
            mask = as_numpy(mask_batch[j]).astype(bool)
            t_valid, s_valid = as_numpy(t_batch[j])[mask], as_numpy(s_batch[j])[mask]
            if t_valid.shape != (length, 3) or s_valid.shape != (length, 900, 3, 2):
                raise ValueError(f"Unexpected NSTPP shapes: {t_valid.shape}, {s_valid.shape}")
            # NSTPP slots: [interval start, midpoint, right event].
            temporal["NSTPP"] = t_valid[right_indices, 1:].copy()
            joint["NSTPP"] = s_valid[right_indices, :, 1:, :].prod(axis=-1).transpose(0, 2, 1)
            found = True
        offset += batch_size
    if not found or offset != sequence_count:
        raise ValueError("NSTPP sequence count does not match HP data")
    del nst

    for name in BASELINE_NAMES:
        if temporal[name].shape != (len(right_indices), 2):
            raise ValueError(f"Invalid {name} temporal shape: {temporal[name].shape}")
        if joint[name].shape != (len(right_indices), 2, 900):
            raise ValueError(f"Invalid {name} spatial shape: {joint[name].shape}")
        for values in (temporal[name], joint[name]):
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{name} intensities must be finite and nonnegative")
    return temporal, joint


def load_midpoint_results(baseline_root="HP_stpp_tmid_S900", seq_idx=20):
    root = resolve_path(baseline_root)
    bundle = _pickle_load_cpu_compat(
        root / "datasets/hawkess_process/data_test_HP_with_midpoints_bundle.pkl"
    )
    if bundle.get("dataset") != "HP":
        raise ValueError("Expected the HP midpoint bundle")
    sequences, masks = bundle["test_sequences"], bundle["is_event"]
    original = _pickle_load_cpu_compat(SCRIPT_DIR / data_path("stpp_out/data_test.pkl"))
    if not (len(sequences) == len(masks) == len(original)):
        raise ValueError("HP bundle sequence count differs from the original test set")
    if not 0 <= seq_idx < len(sequences):
        raise ValueError(f"seq_idx must be in [0, {len(sequences) - 1}]")
    # Validate every sequence, including the float64 midpoint in the HP data
    # that would round onto an event if absolute times were cast to float32.
    for i, (raw, event_mask, real_events) in enumerate(zip(sequences, masks, original)):
        augmented = np.asarray(raw, dtype=np.float64)
        mask = as_numpy(event_mask)
        if augmented.shape != (2 * len(real_events) - 1, 3):
            raise ValueError(f"HP sequence {i}: invalid augmented shape")
        if mask.dtype != np.bool_ or not np.array_equal(mask, np.arange(len(augmented)) % 2 == 0):
            raise ValueError(f"HP sequence {i}: invalid event/midpoint mask")
        if not np.isfinite(augmented).all() or np.any(np.diff(augmented[:, 0]) <= 0):
            raise ValueError(f"HP sequence {i}: non-finite data or non-increasing times")
        if not np.array_equal(augmented[mask], real_events):
            raise ValueError(f"HP sequence {i}: real events differ from the original test set")
        expected_midpoints = 0.5 * (augmented[:-2:2, 0] + augmented[2::2, 0])
        if not np.array_equal(augmented[1::2, 0], expected_midpoints):
            raise ValueError(f"HP sequence {i}: query times are not exact midpoints")
        if not np.array_equal(bundle["history_length_before_row"][i], np.cumsum(mask) - mask):
            raise ValueError(f"HP sequence {i}: history length counts artificial events")

    augmented = np.asarray(sequences[seq_idx], dtype=np.float64)
    events = augmented[::2].copy()
    right_indices = np.arange(FIRST_RIGHT_EVENT, len(events))
    if len(right_indices) == 0:
        raise ValueError("No HP queries after the ten-event warmup")
    query_times = np.column_stack((augmented[2 * right_indices - 1, 0], events[right_indices, 0]))
    # This directory also includes STSCP results. Select only the HP results.
    temporal, joint = _load_baseline_arrays(
        root / "result/hawkess_process_tmid_s900", seq_idx, len(events), len(sequences),
    )
    print(f"HP seq={seq_idx}: {len(events)} real events; {len(right_indices)} aligned midpoint/event pairs; 30x30 grid.", flush=True)
    return MidpointResults(seq_idx, events, right_indices, query_times, temporal, joint, 30)


def get_pxy_heatmap_batched(model_loc, instance, n, delta_query, grid_size=30,
                           device="cpu", spatial_batch_size=128):
    """Same spatial density as the original HP script, with bounded ODE batches."""
    if spatial_batch_size < 1:
        raise ValueError("spatial_batch_size must be positive")
    samples = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(samples, samples, indexing="xy")
    points = torch.tensor(np.column_stack((xx.ravel(), yy.ravel())), dtype=torch.float32)
    log_probs = []
    for start in range(0, len(points), spatial_batch_size):
        logp = get_logprob_loc_dopri5(
            model_loc, instance, n=n, delta_query=delta_query,
            xy_query=points[start:start + spatial_batch_size], device=device,
        )
        log_probs.append(logp.detach().cpu())
    density = torch.cat(log_probs).exp().reshape(grid_size, grid_size).numpy()
    cell_area = 1.0 / (grid_size - 1) ** 2
    density = density / (density.sum() * cell_area + 1e-12)
    return xx, yy, density


def plot_hp_3d_with_baselines(model_time, model_loc, results, chosen, ground_truth,
                             device, out_file, spatial_batch_size=128):
    columns = []
    for i, local_idx in enumerate(chosen):
        n = int(results.right_event_indices[local_idx])
        t_query = float(results.query_times[local_idx, 0])
        print(f"HP 3D midpoint {i + 1}/{len(chosen)}: events {n - 1}/{n}, t={t_query:.6f}", flush=True)
        instance = make_query_instance(results.events, n, t_query)
        _, cif, _, _ = get_intensity_function_dopri5(model_time, instance, n=n, device=device)
        _, _, density = get_pxy_heatmap_batched(
            model_loc, instance, n, float(instance["deltas"][n]), results.grid_size,
            device, spatial_batch_size,
        )
        _, _, gt_map = ground_truth.heatmap(t_query, results.grid_size)
        maps = {"GT": gt_map, "ARCH": float(cif[-1]) * density}
        for name in BASELINE_NAMES:
            # Export: ij mesh ordering. Plot: xy mesh ordering.
            maps[name] = results.joint_spatial[name][local_idx, 0].reshape(results.grid_size, results.grid_size).T
        for name, values in maps.items():
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Invalid {name} spatial intensity at HP midpoint {t_query}")
        columns.append((t_query, maps))
        print(f"HP 3D midpoint {i + 1}/{len(chosen)} complete.", flush=True)
    draw_sc_3d_comparison(columns, results.grid_size, out_file)
    return {name: np.stack([maps[name] for _, maps in columns]) for name in ROW_NAMES}


def visualize(device=None, model_time_path="best_model_time.pth",
              model_loc_path="best_model_loc.pth", seq_idx=20, pick_k_times=5,
              time_range=(10.0, 50.0), seed=42, baseline_root="HP_stpp_tmid_S900",
              output_dir="hp_mid_point_figures", baseline_time_points="both",
              lambda_t_max=50.0, temporal_points=10, temporal_only=False,
              check_data=False, cpu_threads=1, spatial_batch_size=128,
              spatial_only=False, midpoint_right_indices=None):
    if cpu_threads < 1 or spatial_batch_size < 1 or temporal_points < 2:
        raise ValueError("Use positive thread/batch counts and at least two temporal points")
    if spatial_only and temporal_only:
        raise ValueError("Choose either spatial_only or temporal_only")
    if temporal_only and midpoint_right_indices is not None:
        raise ValueError("Explicit midpoint selection requires a spatial plot")
    torch.set_num_threads(cpu_threads)
    set_seed(seed)
    results = load_midpoint_results(baseline_root, seq_idx)
    start, end = temporal_time_window(results, lambda_t_max, baseline_time_points)
    if midpoint_right_indices is not None:
        requested = np.asarray(midpoint_right_indices)
        if (requested.ndim != 1 or requested.size == 0
                or not np.issubdtype(requested.dtype, np.integer)
                or len(np.unique(requested)) != len(requested)
                or not np.isin(requested, results.right_event_indices).all()):
            raise ValueError("Use distinct original right-event indices with available baseline results")
        chosen = np.flatnonzero(np.isin(results.right_event_indices, requested))
    else:
        chosen = (np.array([], dtype=int) if temporal_only else
                  choose_midpoints(results, pick_k_times, time_range, seed))
    if len(chosen):
        print("Selected HP midpoint times:", results.query_times[chosen, 0].tolist(), flush=True)
    if check_data:
        return results
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}", flush=True)
    model_time, model_loc = load_models(device, model_time_path, model_loc_path)
    if len(results.events) + 1 > min(model_time.max_len, model_loc.max_len):
        raise ValueError("HP sequence plus padding exceeds the checkpoint positional encoding")
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"HP_mid_point_seq{seq_idx}_smooth_{baseline_time_points}_tmax{end:g}_n{temporal_points}_connected"
    selection_tag = "n" + "_".join(str(int(n)) for n in results.right_event_indices[chosen])
    selection_suffix = f"_{selection_tag}" if midpoint_right_indices is not None or spatial_only else ""
    temporal_file = output_dir / f"learned_vs_gt_baselines_lambdaT_{tag}{selection_suffix}.png"
    spatial_file = output_dir / f"hp_3d_gt_arch_autostpp_deepstpp_nstpp_mid_point_seq{seq_idx}{selection_suffix}.pdf"
    ground_truth = GroundTruth(results.events)
    if spatial_only:
        with cpu_safe_attention(device):
            maps = plot_hp_3d_with_baselines(model_time, model_loc, results, chosen,
                                           ground_truth, device, spatial_file, spatial_batch_size)
        spatial_tag = f"HP_mid_point_seq{seq_idx}_3d_{selection_tag}"
        data_file = output_dir / f"plot_data_{spatial_tag}.npz"
        np.savez_compressed(
            data_file, events=results.events,
            selected_local_indices=chosen,
            selected_right_event_indices=results.right_event_indices[chosen],
            selected_midpoint_times=results.query_times[chosen, 0],
            **{f"spatial_{name}": values for name, values in maps.items()},
        )
        manifest = dict(
            dataset="HP", sequence_index=seq_idx, seed=seed,
            selection="explicit right-event indices" if midpoint_right_indices is not None else "random",
            selected_right_event_indices=results.right_event_indices[chosen].tolist(),
            selected_midpoint_times=results.query_times[chosen, 0].tolist(),
            gt_parameters=GT_PARAMETERS, spatial_quantity="joint intensity lambda(t,x,y)",
            spatial_z_limits="adaptive per panel", spatial_batch_size=spatial_batch_size,
            baseline_root=str(resolve_path(baseline_root)),
            model_time_path=str(resolve_path(model_time_path)), model_loc_path=str(resolve_path(model_loc_path)),
            temporal_figure=None, spatial_figure=str(spatial_file), plot_data=str(data_file),
        )
        (output_dir / f"plot_manifest_{spatial_tag}.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    with cpu_safe_attention(device):
        times, intensities = dense_learned_curve(
            model_time, results, device, lambda_t_max, temporal_points, baseline_time_points,
        )
    lambda_gt = plot_lambda_t_with_baselines(
        results, times, intensities, ground_truth, temporal_file,
        baseline_time_points, lambda_t_max, dataset_label="HP",
    )
    interp_times, interp_values = interpolate_baselines(results, lambda_t_max, baseline_time_points)
    arrays = dict(events=results.events, t_learn=times, lambda_arch=intensities,
                  lambda_gt=lambda_gt, baseline_query_times=results.query_times,
                  baseline_right_event_indices=results.right_event_indices,
                  selected_local_indices=chosen, selected_midpoint_times=results.query_times[chosen, 0],
                  baseline_interpolation_times=interp_times)
    arrays.update({f"lambda_{name}": results.temporal[name] for name in BASELINE_NAMES})
    arrays.update({f"lambda_{name}_interpolated": interp_values[name] for name in BASELINE_NAMES})
    # Preserve the temporal result before the more expensive spatial inference.
    data_file = output_dir / f"plot_data_{tag}{selection_suffix}.npz"
    np.savez_compressed(data_file, **arrays)
    if not temporal_only:
        with cpu_safe_attention(device):
            maps = plot_hp_3d_with_baselines(model_time, model_loc, results, chosen,
                                           ground_truth, device, spatial_file, spatial_batch_size)
        arrays.update({f"spatial_{name}": values for name, values in maps.items()})
        np.savez_compressed(data_file, **arrays)
    manifest = dict(
        dataset="HP", sequence_index=seq_idx, seed=seed, time_range=list(time_range),
        gt_parameters=GT_PARAMETERS, baseline_time_points=baseline_time_points,
        temporal_sampling="direct ARCH/GT evaluations, connected across event intervals",
        temporal_points_per_interval=temporal_points, temporal_time_range=[start, end],
        temporal_first_right_event_index=FIRST_RIGHT_EVENT,
        baseline_interpolation="PCHIP for visualization only; raw midpoint/event samples retained",
        baseline_pair_order=["midpoint", "right_event"],
        spatial_quantity="joint intensity lambda(t,x,y)", spatial_z_limits="adaptive per panel",
        spatial_batch_size=spatial_batch_size,
        selected_right_event_indices=results.right_event_indices[chosen].tolist(),
        selected_midpoint_times=results.query_times[chosen, 0].tolist(),
        baseline_root=str(resolve_path(baseline_root)),
        model_time_path=str(resolve_path(model_time_path)), model_loc_path=str(resolve_path(model_loc_path)),
        temporal_figure=str(temporal_file), spatial_figure=None if temporal_only else str(spatial_file),
    )
    (output_dir / f"plot_manifest_{tag}{selection_suffix}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-idx", type=int, default=20)
    parser.add_argument("--pick-k-times", type=int, default=5)
    parser.add_argument("--time-range", type=float, nargs=2, default=(10.0, 50.0), metavar=("START", "END"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--midpoint-right-indices", type=int, nargs="+", default=None,
                        help="Explicit original right-event indices (zero-based); overrides random selection and time range")
    parser.add_argument("--device", default=None, help="Default: CUDA if available, otherwise CPU")
    parser.add_argument("--model-time-path", default="best_model_time.pth")
    parser.add_argument("--model-loc-path", default="best_model_loc.pth")
    parser.add_argument("--baseline-root", default="HP_stpp_tmid_S900")
    parser.add_argument("--output-dir", default="hp_mid_point_figures")
    parser.add_argument("--baseline-time-points", choices=("both", "midpoints"), default="both")
    parser.add_argument("--lambda-t-max", type=float, default=50.0)
    parser.add_argument("--temporal-points", type=int, default=10)
    plot_mode = parser.add_mutually_exclusive_group()
    plot_mode.add_argument("--temporal-only", action="store_true", help="Generate only the temporal plot")
    plot_mode.add_argument("--spatial-only", action="store_true", help="Generate only the 3D PDF, without recomputing the temporal plot")
    parser.add_argument("--check-data", action="store_true", help="Validate data and indices without inference")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--spatial-batch-size", type=int, default=128,
                        help="Maximum spatial query points per ODE batch")
    visualize(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
