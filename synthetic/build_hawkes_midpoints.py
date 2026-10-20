"""Build an HP test set with one midpoint inserted between every event.

For an original sequence

    event_0, event_1, ..., event_(L-1)

the new sequence is

    event_0, midpoint_0_1, event_1, ..., midpoint_(L-2)_(L-1), event_(L-1)

and has a same-length boolean mask

    True, False, True, ..., False, True

``True`` rows are real events and may update a baseline's history/state.
``False`` rows are artificial query times: evaluate intensity there but never
update the history/state.  Midpoint x/y values are numerical placeholders
computed as the average of the adjacent event locations.

The HP test data used by hawkes.py is stpp_out/data_test.pkl.
Outputs use float64: an extremely short interval in the HP float32 data would
otherwise round its midpoint onto one of its adjacent event times. Original
event values are preserved exactly when promoted from float32 to float64.
"""

from __future__ import annotations

from release_paths import DATA_ROOT, CHECKPOINT_ROOT, data_path

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np


SCRIPT_DIR = DATA_ROOT
DEFAULT_INPUT = SCRIPT_DIR / "stpp_out" / "data_test.pkl"
DEFAULT_OUTPUT_DIR = Path.cwd() / "generated_queries" / "stpp_out"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def validate_sequence(sequence: np.ndarray, sequence_index: int) -> None:
    if sequence.ndim != 2 or sequence.shape[1] < 3:
        raise ValueError(
            f"sequence {sequence_index} must have shape [L, >=3], "
            f"got {sequence.shape}"
        )
    if sequence.shape[0] < 1:
        raise ValueError(f"sequence {sequence_index} is empty")
    if not np.isfinite(sequence).all():
        raise ValueError(
            f"sequence {sequence_index} contains non-finite values"
        )
    if np.any(np.diff(sequence[:, 0].astype(np.float64)) <= 0):
        raise ValueError(
            f"sequence {sequence_index} times are not strictly increasing"
        )


def midpoint_locations(
    left_locations: np.ndarray,
    right_locations: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "average":
        return 0.5 * (left_locations + right_locations)
    if mode == "left":
        return left_locations.copy()
    if mode == "right":
        return right_locations.copy()
    if mode == "zero":
        return np.zeros_like(left_locations)
    raise ValueError(f"Unsupported midpoint location mode: {mode}")


def interleave_sequence(
    sequence: np.ndarray,
    location_mode: str = "average",
) -> Tuple[np.ndarray, dict]:
    """Interleave events and strictly interior midpoint queries in float64."""
    sequence = np.asarray(sequence, dtype=np.float64)
    length, dimensions = sequence.shape
    augmented_length = max(1, 2 * length - 1)
    augmented = np.empty(
        (augmented_length, dimensions),
        dtype=sequence.dtype,
    )

    is_event = np.zeros(augmented_length, dtype=np.bool_)
    source_event_index = np.full(
        augmented_length,
        -1,
        dtype=np.int64,
    )
    midpoint_left_event_index = np.full(
        augmented_length,
        -1,
        dtype=np.int64,
    )
    midpoint_right_event_index = np.full(
        augmented_length,
        -1,
        dtype=np.int64,
    )
    history_length = np.zeros(augmented_length, dtype=np.int64)

    event_positions = np.arange(0, augmented_length, 2)
    augmented[event_positions] = sequence
    is_event[event_positions] = True
    source_event_index[event_positions] = np.arange(length)
    # If the loop evaluates an event before updating, event i sees i previous
    # events.  After processing it, the next midpoint sees i+1 events.
    history_length[event_positions] = np.arange(length)

    if length > 1:
        midpoint_positions = np.arange(1, augmented_length, 2)
        left_indices = np.arange(length - 1, dtype=np.int64)
        right_indices = left_indices + 1

        left_times = sequence[left_indices, 0].astype(np.float64)
        right_times = sequence[right_indices, 0].astype(np.float64)
        midpoint_times = 0.5 * (left_times + right_times)
        augmented[midpoint_positions, 0] = midpoint_times.astype(
            sequence.dtype
        )
        augmented[midpoint_positions, 1:] = midpoint_locations(
            sequence[left_indices, 1:],
            sequence[right_indices, 1:],
            location_mode,
        )

        midpoint_left_event_index[midpoint_positions] = left_indices
        midpoint_right_event_index[midpoint_positions] = right_indices
        history_length[midpoint_positions] = right_indices

    midpoint_mask = ~is_event
    # Existing baseline plots align from original right-event index 10.
    baseline_aligned_midpoint_mask = (
        midpoint_mask & (midpoint_right_event_index >= 10)
    )
    maps = {
        "is_event": is_event,
        "is_midpoint": midpoint_mask,
        "baseline_aligned_midpoint": baseline_aligned_midpoint_mask,
        "source_event_index": source_event_index,
        "midpoint_left_event_index": midpoint_left_event_index,
        "midpoint_right_event_index": midpoint_right_event_index,
        "history_length_before_row": history_length,
    }
    return augmented, maps


def build_test_set(args: argparse.Namespace) -> dict:
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("rb") as handle:
        original_sequences = pickle.load(handle)
    if not isinstance(original_sequences, (list, tuple)):
        raise TypeError(
            "Expected data_test.pkl to contain a list/tuple of sequences."
        )

    augmented_sequences: List[np.ndarray] = []
    is_event_masks: List[np.ndarray] = []
    is_midpoint_masks: List[np.ndarray] = []
    baseline_aligned_midpoint_masks: List[np.ndarray] = []
    source_event_indices: List[np.ndarray] = []
    midpoint_left_indices: List[np.ndarray] = []
    midpoint_right_indices: List[np.ndarray] = []
    history_lengths: List[np.ndarray] = []

    for sequence_index, raw_sequence in enumerate(original_sequences):
        sequence = np.asarray(raw_sequence)
        validate_sequence(sequence, sequence_index)
        augmented, maps = interleave_sequence(
            sequence,
            location_mode=args.midpoint_location,
        )
        # Catch a query time rounded onto an event before writing any output.
        validate_sequence(augmented, sequence_index)
        augmented_sequences.append(augmented)
        is_event_masks.append(maps["is_event"])
        is_midpoint_masks.append(maps["is_midpoint"])
        baseline_aligned_midpoint_masks.append(
            maps["baseline_aligned_midpoint"]
        )
        source_event_indices.append(maps["source_event_index"])
        midpoint_left_indices.append(
            maps["midpoint_left_event_index"]
        )
        midpoint_right_indices.append(
            maps["midpoint_right_event_index"]
        )
        history_lengths.append(maps["history_length_before_row"])

    sequence_path = output_dir / "data_test_with_midpoints.pkl"
    mask_path = output_dir / "data_test_with_midpoints_is_event.pkl"
    bundle_path = output_dir / "data_test_with_midpoints_bundle.pkl"
    metadata_path = (
        output_dir / "data_test_with_midpoints_metadata.json"
    )

    # Drop-in sequence list: same outer structure and [time,x,y] columns as
    # the original test pickle.
    with sequence_path.open("wb") as handle:
        pickle.dump(
            augmented_sequences,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    # Separate list of masks, requested as [True, False, True, False, ...].
    with mask_path.open("wb") as handle:
        pickle.dump(
            is_event_masks,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    protocol = {
        "row_order": (
            "event_0, midpoint_0_1, event_1, ..., "
            "midpoint_(L-2)_(L-1), event_(L-1)"
        ),
        "is_event_true": (
            "Real event row. Evaluate event-time intensity before updating "
            "if needed, then update the model history/state with this row."
        ),
        "is_event_false": (
            "Artificial midpoint query. Evaluate intensity, but DO NOT "
            "update the model history/state with this row."
        ),
        "midpoint_time": "0.5 * (left_event_time + right_event_time)",
        "midpoint_location": (
            f"{args.midpoint_location} placeholder; it is not an observed "
            "event location and must not update history"
        ),
        "baseline_alignment": (
            "Use baseline_aligned_midpoint_mask to select midpoint queries "
            "whose original right-event index is >=10, matching seq[10:]."
        ),
    }
    bundle = {
        "format_version": 1,
        "dataset": "HP",
        "output_dtype": "float64",
        "source_file": str(input_path),
        "source_sha256": sha256_file(input_path),
        "test_sequences": augmented_sequences,
        "is_event": is_event_masks,
        "is_midpoint": is_midpoint_masks,
        "baseline_aligned_midpoint": baseline_aligned_midpoint_masks,
        "source_event_index": source_event_indices,
        "midpoint_left_event_index": midpoint_left_indices,
        "midpoint_right_event_index": midpoint_right_indices,
        "history_length_before_row": history_lengths,
        "protocol": protocol,
    }
    with bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    original_lengths = [len(sequence) for sequence in original_sequences]
    augmented_lengths = [len(sequence) for sequence in augmented_sequences]
    event_counts = [int(mask.sum()) for mask in is_event_masks]
    midpoint_counts = [
        int(mask.sum()) for mask in is_midpoint_masks
    ]
    aligned_midpoint_counts = [
        int(mask.sum()) for mask in baseline_aligned_midpoint_masks
    ]
    metadata = {
        "format_version": 1,
        "dataset": "HP",
        "output_dtype": "float64",
        "source_file": str(input_path),
        "source_sha256": bundle["source_sha256"],
        "sequence_file": str(sequence_path),
        "is_event_mask_file": str(mask_path),
        "bundle_file": str(bundle_path),
        "sequence_count": len(augmented_sequences),
        "original_event_count": int(sum(original_lengths)),
        "augmented_row_count": int(sum(augmented_lengths)),
        "midpoint_count": int(sum(midpoint_counts)),
        "baseline_aligned_midpoint_count": int(
            sum(aligned_midpoint_counts)
        ),
        "original_lengths": original_lengths,
        "augmented_lengths": augmented_lengths,
        "event_counts": event_counts,
        "midpoint_counts": midpoint_counts,
        "baseline_aligned_midpoint_counts": aligned_midpoint_counts,
        "midpoint_location_mode": args.midpoint_location,
        "protocol": protocol,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Saved sequences: {sequence_path}")
    print(f"Saved masks    : {mask_path}")
    print(f"Saved bundle   : {bundle_path}")
    print(f"Saved metadata : {metadata_path}")
    print(
        f"Sequences={len(augmented_sequences)}, "
        f"real events={sum(event_counts)}, "
        f"inserted midpoints={sum(midpoint_counts)}, "
        f"total rows={sum(augmented_lengths)}"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Insert a query-time midpoint between every pair of HP test "
            "events and export matching event/midpoint masks."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Original HP data_test.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--midpoint-location",
        choices=("average", "left", "right", "zero"),
        default="average",
        help=(
            "Numerical x/y placeholder for midpoint rows. It must not be "
            "used to update model history."
        ),
    )
    return parser.parse_args()


def main() -> None:
    build_test_set(parse_args())


if __name__ == "__main__":
    main()
