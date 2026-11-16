"""Datenschicht: Kurse laden, bereinigen und in Tagesrenditen umrechnen.

Die Schicht kennt weder Streamlit noch die Optimierung. Sie liefert nur
saubere Kurs- und Renditetabellen in Euro (alle Ticker sind Xetra-Notierungen).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SAMPLE_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_prices.csv"

# Ticker -> (Anzeigename, Teilfreistellung in %). Aktienfonds: 30 %, sonst 0 %.
CATALOG: dict[str, tuple[str, int]] = {
    "EUNL.DE": ("MSCI World", 30),
    "VWCE.DE": ("FTSE All-World", 30),
    "SXR8.DE": ("S&P 500", 30),
    "SXRV.DE": ("Nasdaq 100", 30),
    "XDWT.DE": ("MSCI World IT", 30),
    "IS3S.DE": ("MSCI World Value", 30),
    "IS3Q.DE": ("MSCI World Quality", 30),
    "IS3N.DE": ("Emerging Markets", 30),
    "EXSA.DE": ("STOXX Europe 600", 30),
    "EXS1.DE": ("DAX", 30),
    "EUNH.DE": ("Euro-Staatsanleihen", 0),
    "4GLD.DE": ("Xetra-Gold", 0),
}


def display_name(ticker: str) -> str:
    return CATALOG.get(ticker, (ticker, 0))[0]


def default_teilfreistellung(ticker: str) -> int:
    return CATALOG.get(ticker, (ticker, 30))[1]


@dataclass
class PriceData:
    prices: pd.DataFrame      # Tagesschlusskurse, gemeinsame Historie aller Ticker
    source: str               # "live" oder "snapshot"
    missing: list[str]        # Ticker, für die es keine Daten gab


def _download(tickers: list[str], years: int) -> pd.DataFrame:
    import yfinance as yf

    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=years)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close


def _snapshot(tickers: list[str], years: int) -> pd.DataFrame:
    if not SAMPLE_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(SAMPLE_FILE, index_col=0, parse_dates=True)
    start = df.index[-1] - pd.DateOffset(years=years)
    return df.loc[start:, [t for t in tickers if t in df.columns]]


def clean_prices(raw: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Behält nur Ticker mit Daten und kürzt auf die gemeinsame Historie."""
    have = [t for t in tickers if t in raw.columns and raw[t].notna().sum() > 60]
    missing = [t for t in tickers if t not in have]
    if not have:
        return pd.DataFrame(), missing
    prices = raw[have].sort_index().ffill(limit=3).dropna()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices, missing


def load_prices(tickers: list[str], years: int = 15, allow_live: bool = True) -> PriceData:
    """Lädt Kurse live von Yahoo Finance; fällt bei Problemen auf den Snapshot zurück."""
    tickers = list(dict.fromkeys(tickers))
    if allow_live:
        try:
            prices, missing = clean_prices(_download(tickers, years), tickers)
            if not prices.empty and not missing:
                return PriceData(prices, "live", missing)
        except Exception:
            pass
    prices, missing = clean_prices(_snapshot(tickers, years), tickers)
    return PriceData(prices, "snapshot", missing)


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()
