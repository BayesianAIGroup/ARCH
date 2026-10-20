"""Export STSCP lambda(t) queries matching the current midpoint plotting script.

    python build_self_correcting_dense_queries.py
    python build_self_correcting_dense_queries.py --lambda-t-max 100
    python build_self_correcting_dense_queries.py --full-horizon

Uses the common exporter in build_hawkes_dense_queries.py so both
datasets follow the same ten-subdivision rule (nine interior times plus the
right event), warmup mask, t=50 cutoff, and real-event-only history protocol.
The STSCP starting midpoint preserves the original float32 midpoint export.
"""

from release_paths import DATA_ROOT, CHECKPOINT_ROOT, data_path

from build_hawkes_dense_queries import (
    build_test_set as _build_test_set,
    parse_args as _parse_args,
)


def build_test_set(args):
    return _build_test_set(args, dataset="STSCP")


def parse_args():
    return _parse_args(dataset="STSCP")


def main():
    build_test_set(parse_args())


if __name__ == "__main__":
    main()
