"""Export between-event midpoint queries for STSCP baseline inference.

The existing baseline results in ``self_correcting.py`` are
evaluated at observed event times and are aligned from event index 10 onward.
This script creates arbitrary-time queries at

    t_mid(i) = (t_event[i - 1] + t_event[i]) / 2

with history restricted to events ``0 .. i-1``.  It does not create pseudo
events and does not alter the test sequences.

Outputs:
  - ``<prefix>.pkl``: complete collaborator-facing payload
  - ``<prefix>.csv``: flat, human-readable query manifest
  - ``<prefix>_metadata.json``: protocol and shape summary
"""

from __future__ import annotations

from release_paths import DATA_ROOT, CHECKPOINT_ROOT, data_path

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = DATA_ROOT
DEFAULT_INPUT = (
    SCRIPT_DIR
    / "Spatial_Temporal_Plotting"
    / "test_dataset"
    / "stscp_out"
    / "data_test.pkl"
)
DEFAULT_OUTPUT_PREFIX = (
    Path.cwd() / "generated_queries" / "stscp_interval_midpoint_queries"
)

CSV_COLUMNS = (
    "sequence_index",
    "query_index_in_sequence",
    "left_event_index",
    "right_event_index",
    "history_length",
    "t_left",
    "t_mid",
    "t_right",
    "interval_duration",
    "delta_since_last_event",
    "delta_to_next_event",
    "left_x",
    "left_y",
    "right_x",
    "right_y",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sequence_indices(
    text: Optional[str],
    number_of_sequences: int,
) -> List[int]:
    if text is None or text.strip().lower() in ("", "all"):
        return list(range(number_of_sequences))

    indices = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        index = int(token)
        if index < 0 or index >= number_of_sequences:
            raise ValueError(
                f"sequence index {index} is outside "
                f"[0, {number_of_sequences - 1}]"
            )
        indices.append(index)
    if not indices:
        raise ValueError("No sequence indices were selected.")
    return list(dict.fromkeys(indices))


def validate_sequence(sequence: np.ndarray, sequence_index: int) -> None:
    if sequence.ndim != 2 or sequence.shape[1] < 3:
        raise ValueError(
            f"sequence {sequence_index} must have shape [L, >=3], "
            f"got {sequence.shape}"
        )
    if sequence.shape[0] < 2:
        raise ValueError(
            f"sequence {sequence_index} has fewer than two events"
        )
    if not np.isfinite(sequence[:, :3]).all():
        raise ValueError(
            f"sequence {sequence_index} contains non-finite t/x/y values"
        )
    differences = np.diff(sequence[:, 0].astype(np.float64))
    if np.any(differences <= 0):
        raise ValueError(
            f"sequence {sequence_index} event times are not strictly "
            "increasing"
        )


def build_spatial_grid(
    grid_size: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(x_range[0], x_range[1], grid_size, dtype=np.float64)
    ys = np.linspace(y_range[0], y_range[1], grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid_xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    return xs, ys, grid_xy


def make_queries_for_sequence(
    sequence: np.ndarray,
    start_right_event_index: int,
    time_range: Optional[Tuple[float, float]],
):
    times = sequence[:, 0].astype(np.float64)
    first_right_index = max(1, start_right_event_index)
    right_indices = np.arange(
        first_right_index,
        sequence.shape[0],
        dtype=np.int64,
    )
    left_indices = right_indices - 1

    t_left = times[left_indices]
    t_right = times[right_indices]
    t_mid = 0.5 * (t_left + t_right)

    if time_range is not None:
        low, high = sorted((float(time_range[0]), float(time_range[1])))
        keep = (t_mid >= low) & (t_mid <= high)
        left_indices = left_indices[keep]
        right_indices = right_indices[keep]
        t_left = t_left[keep]
        t_mid = t_mid[keep]
        t_right = t_right[keep]

    interval_duration = t_right - t_left
    # right_event_index i means events 0..i-1 are strictly before t_mid.
    history_lengths = right_indices.copy()

    return {
        "query_times": t_mid,
        "history_lengths": history_lengths,
        "left_event_indices": left_indices,
        "right_event_indices": right_indices,
        "t_left": t_left,
        "t_right": t_right,
        "interval_durations": interval_duration,
        "delta_since_last_event": t_mid - t_left,
        "delta_to_next_event": t_right - t_mid,
    }


def iter_csv_rows(
    sequence_index: int,
    sequence: np.ndarray,
    queries,
) -> Iterable[dict]:
    for query_index in range(len(queries["query_times"])):
        left_index = int(queries["left_event_indices"][query_index])
        right_index = int(queries["right_event_indices"][query_index])
        left_location = sequence[left_index, 1:3]
        right_location = sequence[right_index, 1:3]
        yield {
            "sequence_index": sequence_index,
            "query_index_in_sequence": query_index,
            "left_event_index": left_index,
            "right_event_index": right_index,
            "history_length": int(
                queries["history_lengths"][query_index]
            ),
            "t_left": float(queries["t_left"][query_index]),
            "t_mid": float(queries["query_times"][query_index]),
            "t_right": float(queries["t_right"][query_index]),
            "interval_duration": float(
                queries["interval_durations"][query_index]
            ),
            "delta_since_last_event": float(
                queries["delta_since_last_event"][query_index]
            ),
            "delta_to_next_event": float(
                queries["delta_to_next_event"][query_index]
            ),
            "left_x": float(left_location[0]),
            "left_y": float(left_location[1]),
            "right_x": float(right_location[0]),
            "right_y": float(right_location[1]),
        }


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def export_queries(args: argparse.Namespace) -> dict:
    input_path = resolve_path(args.input)
    output_prefix = resolve_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("rb") as handle:
        raw_sequences = pickle.load(handle)
    if not isinstance(raw_sequences, (list, tuple)):
        raise TypeError(
            "Expected the test pickle to contain a list/tuple of sequences."
        )

    selected_indices = parse_sequence_indices(
        args.sequence_indices,
        len(raw_sequences),
    )
    time_range = (
        tuple(args.time_range)
        if args.time_range is not None
        else None
    )
    xs, ys, spatial_grid_xy = build_spatial_grid(
        grid_size=args.grid_size,
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
    )

    selected_sequences: List[np.ndarray] = []
    query_times: List[np.ndarray] = []
    history_lengths: List[np.ndarray] = []
    left_event_indices: List[np.ndarray] = []
    right_event_indices: List[np.ndarray] = []
    interval_durations: List[np.ndarray] = []
    csv_rows = []

    for sequence_index in selected_indices:
        sequence = np.asarray(raw_sequences[sequence_index])
        validate_sequence(sequence, sequence_index)
        sequence = sequence.copy()
        queries = make_queries_for_sequence(
            sequence=sequence,
            start_right_event_index=args.start_right_event_index,
            time_range=time_range,
        )

        selected_sequences.append(sequence)
        query_times.append(queries["query_times"])
        history_lengths.append(queries["history_lengths"])
        left_event_indices.append(queries["left_event_indices"])
        right_event_indices.append(queries["right_event_indices"])
        interval_durations.append(queries["interval_durations"])
        csv_rows.extend(
            iter_csv_rows(sequence_index, sequence, queries)
        )

    query_counts = [len(values) for values in query_times]
    total_queries = int(sum(query_counts))
    if total_queries == 0:
        raise ValueError(
            "The selected sequences and filters produced zero midpoint "
            "queries."
        )

    protocol = {
        "query_time_definition": (
            "t_mid = (t_event[left_event_index] + "
            "t_event[right_event_index]) / 2"
        ),
        "history_definition": (
            "For right_event_index=i, condition only on original events "
            "0..i-1. history_length=i. The right event i is not in history."
        ),
        "temporal_intensity_request": (
            "Evaluate lambda_t(t_mid | history events 0..i-1)."
        ),
        "spatiotemporal_intensity_request": (
            "Evaluate lambda(t_mid, x, y | history events 0..i-1) at every "
            "explicit point in spatial_grid_xy, preserving its row order."
        ),
        "pseudo_events_added": False,
        "time_unit": "same absolute-time unit as data_test.pkl",
        "location_unit": "normalized STSCP x/y coordinates",
        "baseline_alignment": (
            "Default start_right_event_index=10 matches the existing "
            "AutoSTPP/DeepSTPP/NSTPP event-time output alignment seq[10:]."
        ),
        "requested_baseline_output": {
            "sequence_indices": (
                "Copy sequence_indices from this payload without reordering."
            ),
            "query_times": (
                "Copy query_times from this payload without reordering."
            ),
            "history_lengths": (
                "Copy history_lengths; query j must use exactly "
                "events[:history_lengths[j]]."
            ),
            "temporal_intensity": (
                "List with one float array [Q_i] per selected sequence."
            ),
            "spatiotemporal_intensity": (
                "List with one float array [Q_i, 900] per selected sequence, "
                "aligned exactly with spatial_grid_xy. Returning "
                "[Q_i, 30, 30] in the declared xy/C layout is also accepted."
            ),
            "model_name": "Include the baseline method name.",
        },
    }

    payload = {
        "format_version": 1,
        "dataset": "STSCP",
        "source_file": str(input_path),
        "source_sha256": sha256_file(input_path),
        "sequence_indices": np.asarray(selected_indices, dtype=np.int64),
        "events": selected_sequences,
        "query_times": query_times,
        "history_lengths": history_lengths,
        "left_event_indices": left_event_indices,
        "right_event_indices": right_event_indices,
        "interval_durations": interval_durations,
        "spatial_grid_xy": spatial_grid_xy,
        "spatial_grid_x": xs,
        "spatial_grid_y": ys,
        "spatial_grid_shape": (args.grid_size, args.grid_size),
        "spatial_grid_indexing": "xy",
        "spatial_grid_flatten_order": "C",
        "start_right_event_index": args.start_right_event_index,
        "time_range": time_range,
        "protocol": protocol,
    }

    pickle_path = output_prefix.with_suffix(".pkl")
    csv_path = output_prefix.with_suffix(".csv")
    metadata_path = output_prefix.parent / (
        output_prefix.name + "_metadata.json"
    )

    with pickle_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)

    metadata = {
        "format_version": payload["format_version"],
        "dataset": payload["dataset"],
        "source_file": payload["source_file"],
        "source_sha256": payload["source_sha256"],
        "pickle_file": str(pickle_path),
        "csv_file": str(csv_path),
        "selected_sequence_count": len(selected_indices),
        "selected_sequence_indices": selected_indices,
        "query_count_per_sequence": query_counts,
        "total_query_count": total_queries,
        "start_right_event_index": args.start_right_event_index,
        "time_range": time_range,
        "spatial_grid_shape": [args.grid_size, args.grid_size],
        "spatial_grid_point_count": int(spatial_grid_xy.shape[0]),
        "spatial_grid_indexing": "xy",
        "spatial_grid_flatten_order": "C",
        "protocol": protocol,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Saved pickle : {pickle_path}")
    print(f"Saved CSV    : {csv_path}")
    print(f"Saved metadata: {metadata_path}")
    print(
        f"Sequences={len(selected_indices)}, "
        f"midpoint queries={total_queries}, "
        f"spatial grid={args.grid_size}x{args.grid_size}"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export between-event midpoint query times and explicit spatial "
            "grid points for STSCP baselines."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="STSCP data_test.pkl containing [absolute_t, x, y] sequences.",
    )
    parser.add_argument(
        "--output-prefix",
        default=str(DEFAULT_OUTPUT_PREFIX),
        help="Output path without extension.",
    )
    parser.add_argument(
        "--start-right-event-index",
        type=int,
        default=10,
        help=(
            "First right event i used for midpoint (i-1,i). Default 10 "
            "matches existing baseline alignment."
        ),
    )
    parser.add_argument(
        "--sequence-indices",
        default="all",
        help='Comma-separated original sequence indices, or "all".',
    )
    parser.add_argument(
        "--time-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Optionally retain midpoint times inside this inclusive range.",
    )
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("X_MIN", "X_MAX"),
    )
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("Y_MIN", "Y_MAX"),
    )
    args = parser.parse_args()

    if args.start_right_event_index < 1:
        parser.error("--start-right-event-index must be at least 1")
    if args.grid_size <= 1:
        parser.error("--grid-size must be greater than 1")
    if args.x_range[0] >= args.x_range[1]:
        parser.error("--x-range must be increasing")
    if args.y_range[0] >= args.y_range[1]:
        parser.error("--y-range must be increasing")
    return args


def main() -> None:
    export_queries(parse_args())


if __name__ == "__main__":
    main()
