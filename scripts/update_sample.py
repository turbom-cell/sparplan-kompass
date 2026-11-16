"""Aktualisiert den Kurs-Snapshot (Notfall-Daten, falls Yahoo Finance nicht antwortet).

Aufruf im Projektordner:  python scripts/update_sample.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data_service import CATALOG, SAMPLE_FILE, _download  # noqa: E402

prices = _download(list(CATALOG), years=20).round(4)
prices.index.name = "Date"
SAMPLE_FILE.parent.mkdir(exist_ok=True)
prices.to_csv(SAMPLE_FILE)
print(f"{len(prices)} Tage, {prices.shape[1]} Ticker -> {SAMPLE_FILE}")
