"""Export HP lambda(t) queries on the same grid as the midpoint plot scripts.

    python build_hawkes_dense_queries.py
    python build_hawkes_dense_queries.py --lambda-t-max 100
    python build_hawkes_dense_queries.py --full-horizon

Ten subdivisions mean eleven endpoints, then the left endpoint is removed:
nine interior queries plus the right event. The first displayed interval is
cropped at its midpoint; an interval crossing t=50 is subdivided up to t=50.
Only queries in the plotting window are inserted, but ALL original events
are retained. Artificial queries never update history. Exported absolute
times are float64, preserving even the very short HP event intervals.

This file also supplies the common exporter used by the STSCP entry point.
No model checkpoints or baseline result files are needed.
"""

from __future__ import annotations

from release_paths import DATA_ROOT, CHECKPOINT_ROOT, data_path

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = DATA_ROOT
DATA_DIRS = {"HP": "stpp_out", "STSCP": "stscp_out"}


def resolve_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else Path.cwd() / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sequence(sequence, sequence_index):
    if sequence.ndim != 2 or sequence.shape[1] < 3 or not len(sequence):
        raise ValueError(f"Sequence {sequence_index}: expected nonempty [L, >=3] data")
    if not np.isfinite(sequence).all() or np.any(np.diff(sequence[:, 0]) <= 0):
        raise ValueError(f"Sequence {sequence_index}: need finite values and strictly increasing times")


def plotting_query_grid(sequence, dataset, temporal_points=10,
                        first_right_event_index=10, lambda_t_max=50.0):
    """Reproduce dense_learned_curve's CPU query times without running its ODE.

    Keep the plotting code's float32 delta / torch.linspace arithmetic, then
    add the float64 left event time and anchor the right endpoint exactly.
    STSCP's exported starting midpoint was stored in the source event dtype;
    HP's was stored in float64. This difference matters for exact alignment.
    """
    if dataset not in DATA_DIRS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if temporal_points < 2 or first_right_event_index < 1:
        raise ValueError("Use at least two subdivisions and a right-event index >= 1")
    if lambda_t_max is not None and not np.isfinite(lambda_t_max):
        raise ValueError("lambda_t_max must be finite, or use full_horizon")
    raw = np.asarray(sequence)
    events = np.asarray(raw, dtype=np.float64)
    validate_sequence(events, "input")
    n0 = first_right_event_index
    empty = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64), None)
    if len(events) <= n0:
        return empty
    start = .5 * (events[n0 - 1, 0] + events[n0, 0])
    if dataset == "STSCP":
        start = float(np.asarray(start, dtype=raw.dtype))
    end = float(events[-1, 0])
    if lambda_t_max is not None:
        end = min(end, float(lambda_t_max))
    if end <= start:
        return empty

    all_times, all_right_indices = [], []
    for n in range(n0, len(events)):
        left_t = float(events[n - 1, 0])
        if left_t >= end:
            break
        right_t = min(float(events[n, 0]), end)
        # make_query_instance stores this delta as float32 before linspace.
        target_delta = float(np.float32(right_t - left_t))
        deltas = torch.linspace(0.0, target_delta, steps=temporal_points + 1,
                                dtype=torch.float32, device="cpu").numpy()[1:]
        times = left_t + deltas.astype(np.float64)
        times[-1] = right_t
        times = times[(times >= start) & (times <= end) & (times > left_t)]
        if n == n0 and (not len(times) or times[0] > start):
            times = np.concatenate(([start], times))
        all_times.append(times)
        all_right_indices.append(np.full(len(times), n, dtype=np.int64))
    times, right_indices = np.concatenate(all_times), np.concatenate(all_right_indices)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("The requested density produces invalid or duplicate query times")
    if not np.array_equal(np.searchsorted(events[:, 0], times, side="left"), right_indices):
        raise ValueError("A query escaped its original history interval")
    return times, right_indices, [float(start), end]


def interleave_sequence(sequence, query_times, right_indices, location_mode="zero"):
    """Merge exact plot queries with all original events and record row masks."""
    events = np.asarray(sequence, dtype=np.float64)
    length, dimensions = events.shape
    times, inverse = np.unique(np.concatenate((events[:, 0], query_times)), return_inverse=True)
    event_rows, query_rows = inverse[:length], inverse[length:]
    augmented = np.zeros((len(times), dimensions), dtype=np.float64)
    augmented[:, 0] = times
    augmented[event_rows] = events
    is_event = np.zeros(len(times), dtype=np.bool_)
    is_event[event_rows] = True
    evaluate = np.zeros(len(times), dtype=np.bool_)
    evaluate[query_rows] = True
    artificial = ~is_event[query_rows]
    rows, right = query_rows[artificial], right_indices[artificial]
    if location_mode == "average":
        augmented[rows, 1:] = .5 * (events[right - 1, 1:] + events[right, 1:])
    elif location_mode == "left":
        augmented[rows, 1:] = events[right - 1, 1:]
    elif location_mode == "right":
        augmented[rows, 1:] = events[right, 1:]
    elif location_mode != "zero":
        raise ValueError(f"Unknown query location mode: {location_mode}")
    source_event_index = np.full(len(times), -1, dtype=np.int64)
    source_event_index[event_rows] = np.arange(length)
    history = np.cumsum(is_event, dtype=np.int64) - is_event
    maps = dict(
        is_event=is_event, is_query=~is_event, evaluate_lambda_t=evaluate,
        source_event_index=source_event_index, history_length_before_row=history,
        lambda_t_query_times=query_times,
        lambda_t_query_row_indices=query_rows,
        lambda_t_query_right_event_indices=right_indices,
        lambda_t_query_is_event=is_event[query_rows],
    )
    validate_sequence(augmented, "augmented")
    if not np.array_equal(augmented[is_event], events):
        raise ValueError("Original event values were modified")
    if not np.array_equal(history[query_rows], right_indices):
        raise ValueError("An artificial query incorrectly changed the real-event history")
    if not np.array_equal(augmented[evaluate, 0], query_times):
        raise ValueError("Evaluation mask does not select the exact plotting grid")
    return augmented, maps


def build_test_set(args, dataset="HP"):
    input_path, output_dir = resolve_path(args.input), resolve_path(args.output_dir)
    limit = None if args.full_horizon else args.lambda_t_max
    with input_path.open("rb") as stream:
        original = pickle.load(stream)
    if not isinstance(original, (list, tuple)) or not len(original):
        raise TypeError("Expected a nonempty list/tuple of event sequences")
    sequences, collected, windows = [], {}, []
    for i, raw in enumerate(original):
        raw = np.asarray(raw)
        validate_sequence(raw.astype(np.float64), i)
        times, indices, window = plotting_query_grid(
            raw, dataset, args.temporal_points, args.first_right_event_index, limit,
        )
        augmented, maps = interleave_sequence(raw, times, indices, args.query_location)
        sequences.append(augmented)
        windows.append(window)
        for name, values in maps.items():
            collected.setdefault(name, []).append(values)
    horizon_tag = "full" if limit is None else f"tmax{limit:g}"
    tag = f"data_test_with_dense_queries_n{args.temporal_points}_{horizon_tag}"
    paths = {name: output_dir / f"{tag}{suffix}" for name, suffix in (
        ("sequences", ".pkl"), ("is_event", "_is_event.pkl"),
        ("evaluate_lambda_t", "_evaluate_lambda_t.pkl"),
        ("bundle", "_bundle.pkl"), ("metadata", "_metadata.json"),
    )}
    protocol = dict(
        interval_rule="Split into K equal subintervals; omit left endpoint: K-1 interior times plus right endpoint.",
        time_arithmetic="float32 delta and CPU torch.linspace(K+1)[1:], added to float64 left time; right endpoint anchored exactly",
        window_rule=(
            "Start at midpoint before original right-event index N. First interval is cropped. "
            + ("Continue to the final real event, including its pre-event intensity; no time cutoff."
               if limit is None else
               "Stop at min(last event, cutoff); final interval is subdivided up to cutoff.")
        ),
        original_events="All original events are retained, including history warmup and events beyond the plotting cutoff.",
        is_event_true="Real event: evaluate lambda(t|H) BEFORE updating history, then append this real event.",
        is_event_false="Artificial query: NEVER update event history or event counters; x/y are placeholders.",
        evaluate_lambda_t="Evaluate only rows with this mask True. It includes artificial queries AND plotted real-event rows.",
        conditioning="Query with right-event index n uses original events[:n]; n is zero-based and also equals the number of historical real events.",
        time_since_last_event="Use t_query - last REAL event time; never use the previous artificial query as the last event.",
        output="One scalar temporal intensity lambda(t) per lambda_t_query_times entry, in the original time units. Preserve sequence and query order; no spatial grid is required.",
        boundary_precision="Use the supplied float64 absolute times as-is; casting absolute times to float32 can collapse close HP queries.",
        short_sequences="A sequence with no eligible window has empty query arrays; its original events are still retained.",
    )
    settings = dict(
        temporal_points_per_interval=args.temporal_points,
        first_right_event_index=args.first_right_event_index,
        lambda_t_max=limit, query_location_mode=args.query_location,
        output_dtype="float64", grid_device="cpu", torch_version=str(torch.__version__),
    )
    bundle = dict(
        format_version=1, dataset=dataset, source_file=str(input_path),
        source_sha256=sha256_file(input_path), settings=settings,
        test_sequences=sequences, temporal_time_ranges=windows,
        protocol=protocol, **collected,
    )
    event_counts = [int(mask.sum()) for mask in collected["is_event"]]
    artificial_counts = [int(mask.sum()) for mask in collected["is_query"]]
    evaluation_counts = [len(q) for q in collected["lambda_t_query_times"]]
    event_evaluation_counts = [int(mask.sum()) for mask in collected["lambda_t_query_is_event"]]
    metadata = dict(
        format_version=1, dataset=dataset, settings=settings, protocol=protocol,
        source_file=str(input_path), source_sha256=bundle["source_sha256"],
        files={name: str(path) for name, path in paths.items()},
        sequence_count=len(sequences), original_event_count=sum(event_counts),
        inserted_query_count=sum(artificial_counts), lambda_t_evaluation_count=sum(evaluation_counts),
        event_time_evaluation_count=sum(event_evaluation_counts),
        augmented_row_count=sum(map(len, sequences)),
        original_lengths=event_counts, augmented_lengths=list(map(len, sequences)),
        inserted_query_counts=artificial_counts, lambda_t_evaluation_counts=evaluation_counts,
        event_time_evaluation_counts=event_evaluation_counts, temporal_time_ranges=windows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("sequences", sequences), ("is_event", collected["is_event"]),
                        ("evaluate_lambda_t", collected["evaluate_lambda_t"]), ("bundle", bundle)):
        with paths[name].open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{dataset}: {len(sequences)} sequences, {sum(event_counts)} original events, "
          f"{sum(artificial_counts)} inserted queries, {sum(evaluation_counts)} lambda(t) evaluations.")
    print(f"Bundle: {paths['bundle']}")
    print(f"Metadata: {paths['metadata']}")
    return metadata


def parse_args(dataset="HP"):
    parser = argparse.ArgumentParser(description=f"Export {dataset} baseline query times matching the current dense lambda(t) plots.")
    parser.add_argument("--input", default=str(SCRIPT_DIR / DATA_DIRS[dataset] / "data_test.pkl"))
    parser.add_argument("--output-dir", default=str(Path.cwd() / "generated_queries" / "lambdaT_dense_baseline_data" / dataset))
    parser.add_argument("--temporal-points", type=int, default=10,
                        help="Number of subdivisions: 10 means 9 interior samples plus the right endpoint")
    parser.add_argument("--first-right-event-index", type=int, default=10)
    horizon = parser.add_mutually_exclusive_group()
    horizon.add_argument("--lambda-t-max", type=float, default=50.0)
    horizon.add_argument("--full-horizon", action="store_true", help="Extend queries to the last event instead of t=50")
    parser.add_argument("--query-location", choices=("zero", "average", "left", "right"), default="zero",
                        help="Placeholder x/y for artificial queries; never update history with it")
    return parser.parse_args()


def main():
    build_test_set(parse_args("HP"), "HP")


if __name__ == "__main__":
    main()
