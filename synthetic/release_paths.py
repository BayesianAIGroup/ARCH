"""Resolve immutable inputs relative to this release, independently of cwd."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "synthetic"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "synthetic"


def data_path(path):
    path = Path(path)
    return str(path if path.is_absolute() else DATA_ROOT / path)
