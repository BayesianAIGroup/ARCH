"""Explicit runtime settings used by the curated experiment scripts."""
from dataclasses import dataclass


@dataclass
class Settings:
    suffix_length: int = 5
    missing_ratio: float = 0.2


settings = Settings()
