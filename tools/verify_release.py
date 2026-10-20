#!/usr/bin/env python3
"""Check curated source/data integrity without running training."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(original_root=None):
    problems = []
    manifest = json.loads((ROOT / 'provenance/source_manifest.json').read_text())
    for entry in manifest:
        p = ROOT / entry['destination']
        if not p.is_file() or sha256(p) != entry['release_sha256']:
            problems.append(f"Missing or modified release file: {p.relative_to(ROOT)}")
        if original_root:
            source = original_root / entry['source']
            if not source.is_file() or sha256(source) != entry['source_sha256']:
                problems.append(f"Original changed: {entry['source']}")
    python_files = [p for p in ROOT.rglob('*.py') if 'outputs' not in p.relative_to(ROOT).parts]
    for p in python_files:
        try:
            ast.parse(p.read_text())
        except SyntaxError as exc:
            problems.append(f'{p.relative_to(ROOT)}: {exc}')
    datasets = []
    for family in ['prediction', 'seahorse']:
        for dataset in sorted((ROOT / 'data' / family).iterdir()):
            for split in ['train', 'val', 'test']:
                p = dataset / f'data_{split}.pkl'
                seqs = pickle.loads(p.read_bytes())
                lengths = [len(seq) for seq in seqs]
                bad_times = 0
                for seq in seqs:
                    if any(len(event) != 3 for event in seq):
                        problems.append(f'Non-[time, lon, lat] event: {p.relative_to(ROOT)}')
                    times = [float(event[0]) for event in seq]
                    bad_times += sum(t1 > t2 for t1, t2 in zip(times, times[1:]))
                if bad_times:
                    problems.append(f'Non-monotone timestamps: {p.relative_to(ROOT)}: {bad_times}')
                datasets.append(dict(family=family, dataset=dataset.name, split=split,
                                     sequences=len(seqs), events=sum(lengths),
                                     min_length=min(lengths), max_length=max(lengths),
                                     mean_length=sum(lengths) / len(lengths), sha256=sha256(p)))
    return dict(passed=not problems, source_files_checked=len(manifest),
                python_files_checked=len(python_files), problems=problems, datasets=datasets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, help='Optional local original directory to verify unchanged inputs')
    parser.add_argument('--report', type=Path, help='Optional JSON report destination')
    args = parser.parse_args()
    result = verify(args.original_root)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'datasets'}, indent=2))
    sys.exit(0 if result['passed'] else 1)
