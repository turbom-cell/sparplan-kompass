"""Aktualisiert die gespeicherten Kursdaten.

  data/sample_prices.csv  – ETF-Kurse als Notfall-Stand, falls Yahoo Finance nicht antwortet
  data/proxy_history.csv  – Ersatz-Reihen in Euro für die Zeit vor ETF-Auflage (ab 1999)

Aufruf im Projektordner:  python scripts/update_sample.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data_service import (CATALOG, PRICE_INDEX_DIVIDEND, PROXIES, PROXY_FILE,  # noqa: E402
                               SAMPLE_FILE, _download, repair_etf)

FRED_EURUSD = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU&cosd=1999-01-01"
START = pd.Timestamp("1999-01-01")

# 1) ETF-Kurse
etfs = _download(list(CATALOG), pd.Timestamp("2005-01-01")).round(4)
etfs.index.name = "Date"
SAMPLE_FILE.parent.mkdir(exist_ok=True)
etfs.to_csv(SAMPLE_FILE)
print(f"ETF-Kurse: {len(etfs)} Tage, {etfs.shape[1]} Ticker -> {SAMPLE_FILE.name}")

# 2) Ersatz-Reihen in Euro
symbols = sorted({sym for sym, _, _ in PROXIES.values()})
raw = _download(symbols, START)
fx = pd.read_csv(FRED_EURUSD, index_col=0, parse_dates=True).iloc[:, 0]
fx = pd.to_numeric(fx, errors="coerce").dropna()          # US-Dollar je Euro

out = {}
for ticker, (symbol, kind, _) in PROXIES.items():
    series = raw[symbol].dropna()
    if kind in ("usd", "usd_price"):
        series = (series / fx.reindex(series.index).ffill().bfill()).dropna()
    if kind == "usd_price":
        # Der Kursindex bei Yahoo hat einzelne falsche Niveausprünge (z. B. 04.04.2001: -14 %,
        # während der S&P 500 kaum fiel). Ein breiter Weltindex bewegt sich an keinem Tag um
        # mehr als 10 %; solche Tage werden als Datenfehler behandelt und auf 0 % gesetzt.
        r = series.pct_change().fillna(0.0)
        r[r.abs() > 0.10] = 0.0
        series = series.iloc[0] * (1.0 + r).cumprod()
        # Dividenden pauschal ergänzen
        days = (series.index - series.index[0]).days
        series = series * (1.0 + PRICE_INDEX_DIVIDEND) ** (days / 365.25)
    first_etf = repair_etf(etfs[ticker]).index[0]             # erster verlässliche ETF-Kurs
    out[ticker] = series.loc[:first_etf + pd.Timedelta(days=7)]   # nur bis zum ETF-Start nötig
    print(f"  {ticker:8s} <- {symbol:18s} {out[ticker].index[0].date()} bis {out[ticker].index[-1].date()}")

proxies = pd.DataFrame(out).sort_index()
proxies = (proxies / proxies.bfill().iloc[0] * 100).round(4)  # auf 100 normiert, nur Verlauf zählt
proxies.index.name = "Date"
proxies.to_csv(PROXY_FILE)
print(f"Ersatz-Reihen: {len(proxies)} Tage -> {PROXY_FILE.name}")
