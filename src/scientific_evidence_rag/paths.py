"""Project-local paths; runtime caches never default to the user's home."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_DIR = ROOT / ".cache"
REPORTS_DIR = ROOT / "reports"
