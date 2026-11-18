"""Datenschicht: Kurse laden, bereinigen und in Tagesrenditen umrechnen.

Die Schicht kennt weder Streamlit noch die Optimierung. Sie liefert nur
saubere Kurs- und Renditetabellen in Euro (alle Ticker sind Xetra-Notierungen).

Lange Historie: Die ETFs gibt es erst seit etwa 2008 bis 2019. Für die Zeit davor
wird eine Ersatz-Reihe (Index oder alter Fonds, in Euro umgerechnet) vor den
ETF-Kurs gesetzt. Diese Reihen liegen fertig in `data/proxy_history.csv` und werden
von `scripts/update_sample.py` erzeugt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_FILE = DATA_DIR / "sample_prices.csv"
PROXY_FILE = DATA_DIR / "proxy_history.csv"
EARLIEST = pd.Timestamp("2000-01-01")

# Ticker -> (Anzeigename, Teilfreistellung in %, Gruppe). Aktienfonds: 30 %, sonst 0 %.
CATALOG: dict[str, tuple[str, int, str]] = {
    "EUNL.DE": ("MSCI World", 30, "Aktien Welt"),
    "VWCE.DE": ("FTSE All-World", 30, "Aktien Welt"),
    "IUSN.DE": ("MSCI World Small Cap", 30, "Aktien Welt"),
    "SXR8.DE": ("S&P 500", 30, "Aktien Regionen"),
    "SXRV.DE": ("Nasdaq 100", 30, "Aktien Regionen"),
    "EXSA.DE": ("STOXX Europe 600", 30, "Aktien Regionen"),
    "EXW1.DE": ("EURO STOXX 50", 30, "Aktien Regionen"),
    "EXS1.DE": ("DAX", 30, "Aktien Regionen"),
    "IS3N.DE": ("Emerging Markets", 30, "Aktien Regionen"),
    "EUNN.DE": ("Japan", 30, "Aktien Regionen"),
    "SXR1.DE": ("Pazifik ohne Japan", 30, "Aktien Regionen"),
    "XDWT.DE": ("MSCI World IT", 30, "Branchen & Faktoren"),
    "XDWH.DE": ("MSCI World Gesundheit", 30, "Branchen & Faktoren"),
    "XDWS.DE": ("MSCI World Basiskonsum", 30, "Branchen & Faktoren"),
    "XDWU.DE": ("MSCI World Versorger", 30, "Branchen & Faktoren"),
    "XDWI.DE": ("MSCI World Industrie", 30, "Branchen & Faktoren"),
    "XDWF.DE": ("MSCI World Finanzen", 30, "Branchen & Faktoren"),
    "XDW0.DE": ("MSCI World Energie", 30, "Branchen & Faktoren"),
    "XDWM.DE": ("MSCI World Grundstoffe", 30, "Branchen & Faktoren"),
    "XDWC.DE": ("MSCI World Zyklischer Konsum", 30, "Branchen & Faktoren"),
    "IQQ0.DE": ("MSCI World Minimum Volatility", 30, "Branchen & Faktoren"),
    "IS3S.DE": ("MSCI World Value", 30, "Branchen & Faktoren"),
    "IS3Q.DE": ("MSCI World Quality", 30, "Branchen & Faktoren"),
    "IQQ6.DE": ("Immobilien (REITs) Welt", 30, "Immobilien & Rohstoffe"),
    "4GLD.DE": ("Xetra-Gold", 0, "Immobilien & Rohstoffe"),
    "EXXY.DE": ("Rohstoffe breit", 0, "Immobilien & Rohstoffe"),
    "EUNH.DE": ("Euro-Staatsanleihen", 0, "Anleihen & Geldmarkt"),
    "EUN5.DE": ("Euro-Unternehmensanleihen", 0, "Anleihen & Geldmarkt"),
    "IBCI.DE": ("Euro-Inflationsanleihen", 0, "Anleihen & Geldmarkt"),
    "XUTD.DE": ("US-Staatsanleihen", 0, "Anleihen & Geldmarkt"),
    "IUS7.DE": ("Schwellenländer-Anleihen", 0, "Anleihen & Geldmarkt"),
    "XEON.DE": ("Geldmarkt Euro", 0, "Anleihen & Geldmarkt"),
}
GROUPS = list(dict.fromkeys(g for _, _, g in CATALOG.values()))

# Ersatz-Reihen für die Zeit vor ETF-Auflage: Ticker -> (Yahoo-Symbol, Art, Beschreibung)
#   "usd"        = in US-Dollar, wird mit dem EUR/USD-Kurs in Euro umgerechnet
#   "usd_hedged" = in US-Dollar, ohne Umrechnung (Näherung für währungsgesicherte Anleihen)
#   "eur"        = schon in Euro
#   "usd_price"  = wie "usd", aber Kursindex ohne Dividenden: pauschal +1,8 % p.a. ergänzt
PROXIES: dict[str, tuple[str, str, str]] = {
    "EUNL.DE": ("^990100-USD-STRD", "usd_price", "MSCI-World-Kursindex + 1,8 % Dividende p.a."),
    "VWCE.DE": ("^990100-USD-STRD", "usd_price", "MSCI-World-Kursindex + 1,8 % Dividende p.a. (Näherung)"),
    "SXR8.DE": ("^SP500TR", "usd", "S&P 500 Total Return Index"),
    "SXRV.DE": ("QQQ", "usd", "Nasdaq-100-ETF QQQ (USA)"),
    "XDWT.DE": ("XLK", "usd", "US-Technologie-ETF XLK (Näherung)"),
    "IS3S.DE": ("VIVAX", "usd", "Vanguard Value Index, nur USA (Näherung)"),
    "IS3N.DE": ("VEIEX", "usd", "Vanguard Emerging Markets Index"),
    "EXSA.DE": ("VEURX", "usd", "Vanguard European Stock Index"),
    "EXS1.DE": ("^GDAXI", "eur", "DAX Performance-Index"),
    "EUNH.DE": ("VFITX", "usd_hedged", "US-Staatsanleihen mittlerer Laufzeit, ohne Währungseffekt (Näherung)"),
    "4GLD.DE": ("GC=F", "usd", "Gold-Future in US-Dollar (ab 08/2000)"),
    "IUSN.DE": ("NAESX", "usd", "US-Nebenwerte-Indexfonds (Näherung)"),
    "EXW1.DE": ("VEURX", "usd", "Vanguard European Stock Index (Näherung)"),
    "XDWH.DE": ("VGHCX", "usd", "US-Gesundheitsfonds (Näherung)"),
    "XDWS.DE": ("FDFAX", "usd", "US-Basiskonsumfonds (Näherung)"),
    "IQQ6.DE": ("VGSIX", "usd", "US-Immobilienindexfonds (Näherung)"),
    "EXXY.DE": ("^SPGSCI", "usd", "S&P-GSCI-Rohstoffindex (Näherung)"),
    "EUN5.DE": ("DODIX", "usd_hedged", "US-Anleihefonds mit Unternehmensanleihen, ohne Währungseffekt (Näherung)"),
    "IBCI.DE": ("VIPSX", "usd_hedged", "US-Inflationsanleihen, ohne Währungseffekt (Näherung, ab 06/2000)"),
    "XEON.DE": ("VFISX", "usd_hedged", "Kurzlaufende US-Staatsanleihen, ohne Währungseffekt (Näherung)"),
    "EUNN.DE": ("EWJ", "usd", "iShares MSCI Japan (USA)"),
    "SXR1.DE": ("EPP", "usd", "iShares MSCI Pacific ex Japan (USA, ab 10/2001)"),
    "XDWU.DE": ("FSUTX", "usd", "US-Versorgerfonds (Näherung)"),
    "XDWI.DE": ("FCYIX", "usd", "US-Industriefonds (Näherung)"),
    "XDWF.DE": ("FIDSX", "usd", "US-Finanzfonds (Näherung)"),
    "XDW0.DE": ("FSENX", "usd", "US-Energiefonds (Näherung)"),
    "XDWM.DE": ("FSDPX", "usd", "US-Grundstofffonds (Näherung)"),
    "XDWC.DE": ("FSCPX", "usd", "US-Konsumfonds (Näherung)"),
    "XUTD.DE": ("VFITX", "usd", "US-Staatsanleihen mittlerer Laufzeit, in Euro"),
    "IUS7.DE": ("PREMX", "usd", "Schwellenländer-Anleihefonds (USA)"),
}
PRICE_INDEX_DIVIDEND = 0.018


CUSTOM_NAMES: dict[str, str] = {}      # Namen von Titeln aus der Suche (für Grafiken und Tabellen)


def display_name(ticker: str) -> str:
    return CATALOG.get(ticker, (CUSTOM_NAMES.get(ticker, ticker), 0, ""))[0]


def default_teilfreistellung(ticker: str) -> int:
    return CATALOG.get(ticker, (ticker, 30, ""))[1]


def group_of(ticker: str) -> str:
    return CATALOG.get(ticker, (ticker, 30, "Eigene Titel"))[2]


@dataclass
class PriceData:
    prices: pd.DataFrame      # Tagesschlusskurse, gemeinsame Historie aller Ticker
    source: str               # "live" oder "snapshot"
    missing: list[str]        # Ticker, für die es keine Daten gab
    etf_start: dict[str, pd.Timestamp] = field(default_factory=dict)  # ab wann echte ETF-Kurse


def _download(tickers: list[str], start: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf

    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def _snapshot(tickers: list[str]) -> pd.DataFrame:
    if not SAMPLE_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(SAMPLE_FILE, index_col=0, parse_dates=True)
    return df[[t for t in tickers if t in df.columns]]


def _proxy_history() -> pd.DataFrame:
    if not PROXY_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(PROXY_FILE, index_col=0, parse_dates=True)


def repair_etf(series: pd.Series, spike: float = 0.08, shift: float = 0.18) -> pd.Series:
    """Entfernt offensichtliche Fehler aus Yahoo-Kursen eines ETFs.

    1. Ausreißer: ein Tagessprung über 8 %, der am nächsten Tag fast ganz zurückläuft,
       ist ein falscher Einzelkurs und wird durch den Vortageskurs ersetzt.
    2. Niveausprung: bleibt ein Sprung über 18 % bestehen (z. B. Wechsel der Handelswährung
       in den Daten), sind die Kurse davor nicht vergleichbar und werden verworfen.
    Echte Crash-Tage breiter ETFs liegen unter diesen Grenzen (März 2020: rund -12 %).
    """
    s = series.dropna().copy()
    for _ in range(3):
        r = s.pct_change()
        round_trip = s.shift(-1) / s.shift(1) - 1.0          # Kurs danach gegen Kurs davor
        bad = (r.abs() > spike) & (round_trip.abs() < 0.03)
        if not bad.any():
            break
        s[bad] = float("nan")
        s = s.ffill()
    r = s.pct_change().abs()
    jumps = r[r > shift]
    if len(jumps):
        s = s.loc[jumps.index[-1]:]
    return s


def splice(etf: pd.Series, proxy: pd.Series) -> pd.Series:
    """Setzt die Ersatz-Reihe vor den ETF-Kurs und skaliert sie auf den ersten ETF-Kurs."""
    etf, proxy = etf.dropna(), proxy.dropna()
    if etf.empty or proxy.empty:
        return etf
    first = etf.index[0]
    before = proxy.loc[:first]
    if len(before) < 2:
        return etf
    scaled = before / before.iloc[-1] * etf.iloc[0]
    return pd.concat([scaled.loc[scaled.index < first], etf])


def extend_with_proxies(raw: pd.DataFrame, proxies: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Verlängert jede ETF-Spalte nach hinten. Gibt auch das Datum des ersten echten Kurses zurück."""
    columns, etf_start = {}, {}
    for t in raw.columns:
        etf = repair_etf(raw[t]) if t in CATALOG else raw[t].dropna()
        if etf.empty:
            continue
        etf_start[t] = etf.index[0]
        columns[t] = splice(etf, proxies[t]) if t in proxies.columns else etf
    return pd.DataFrame(columns), etf_start


def clean_prices(raw: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Behält nur Ticker mit Daten und kürzt auf die gemeinsame Historie."""
    have = [t for t in tickers if t in raw.columns and raw[t].notna().sum() > 60]
    missing = [t for t in tickers if t not in have]
    if not have:
        return pd.DataFrame(), missing
    prices = raw[have].sort_index().ffill(limit=3).dropna()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices, missing


def _finish(raw: pd.DataFrame, tickers: list[str], start: pd.Timestamp, source: str,
            use_proxies: bool = True) -> PriceData:
    proxies = _proxy_history() if use_proxies else pd.DataFrame()
    extended, etf_start = extend_with_proxies(raw, proxies)
    prices, missing = clean_prices(extended, tickers)
    if not prices.empty:
        prices = prices.loc[max(start, EARLIEST):]
    return PriceData(prices, source, missing, {t: etf_start[t] for t in prices.columns})


def load_prices(tickers: list[str], start_year: int = 2000, allow_live: bool = True,
                use_proxies: bool = True) -> PriceData:
    """Lädt Kurse ab `start_year`: live von Yahoo Finance, sonst aus dem Snapshot.

    Mit `use_proxies` wird vor einen ETF, der nicht so weit zurückreicht, seine Ersatz-Reihe
    gesetzt. Ohne zählt jeder Titel erst ab seinem echten Start.
    """
    tickers = list(dict.fromkeys(tickers))
    start = pd.Timestamp(year=int(start_year), month=1, day=1)
    known = [t for t in tickers if t in CATALOG]
    custom = [t for t in tickers if t not in CATALOG]
    extra = _custom_prices(custom) if (custom and allow_live) else pd.DataFrame()
    if allow_live and known:
        try:
            raw = _download(known, pd.Timestamp("2005-01-01"))
            data = _finish(raw.join(extra, how="outer") if not extra.empty else raw, tickers, start, "live",
                           use_proxies)
            if not data.prices.empty and not set(data.missing) & set(known):
                return data
        except Exception:
            pass
    snap = _snapshot(known)
    if not extra.empty:
        snap = snap.join(extra, how="outer") if not snap.empty else extra
    return _finish(snap, tickers, start, "snapshot" if known else "live", use_proxies)


EUR_SUFFIXES = (".DE", ".F", ".AS", ".PA", ".MI", ".MC", ".VI", ".BR", ".HE", ".IR", ".LS")


def currency_of(ticker: str) -> str:
    """Grobe Währungs-Erkennung über das Börsenkürzel: Xetra & Euro-Börsen = EUR, ohne Endung = USD."""
    if ticker.upper().endswith(EUR_SUFFIXES):
        return "EUR"
    return "USD" if "." not in ticker and not ticker.startswith("^") else "?"


def _custom_prices(tickers: list[str]) -> pd.DataFrame:
    """Frei eingegebene Yahoo-Kürzel live laden; US-Werte in Euro umrechnen."""
    try:
        raw = _download(tickers + ["EURUSD=X"], pd.Timestamp("1999-01-01"))
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    fx = raw["EURUSD=X"].dropna() if "EURUSD=X" in raw.columns else pd.Series(dtype=float)
    out = {}
    for t in tickers:
        if t not in raw.columns or raw[t].dropna().empty:
            continue
        s = raw[t].dropna()
        if currency_of(t) == "USD":
            if fx.empty:
                continue
            s = (s / fx.reindex(s.index).ffill()).dropna()   # Euro-Kurs erst ab 12/2003 verfügbar
        out[t] = s
    return pd.DataFrame(out)


SEARCH_TYPES = {"ETF": "ETF", "EQUITY": "Aktie", "MUTUALFUND": "Fonds", "CRYPTOCURRENCY": "Krypto"}


def search_titles(query: str, limit: int = 12, _retry: bool = True) -> list[dict]:
    """Yahoo-Finance-Suche nach Name, Kürzel oder ISIN.

    Liefert nur Titel, deren Kurs sich in Euro rechnen lässt (Euro-Börsen, US-Börsen, Krypto in USD)
    und sortiert Euro-Börsen (Xetra zuerst) nach vorne.
    """
    query = query.strip()
    if len(query) < 2:
        return []
    try:
        import yfinance as yf
        quotes = yf.Search(query, max_results=25, news_count=0).quotes
    except Exception:
        return []
    hits = []
    for q in quotes:
        sym, kind = str(q.get("symbol", "")).upper(), q.get("quoteType", "")
        if not sym or kind not in SEARCH_TYPES:
            continue
        cur = "USD" if kind == "CRYPTOCURRENCY" and sym.endswith("-USD") else               "EUR" if kind == "CRYPTOCURRENCY" and sym.endswith("-EUR") else currency_of(sym)
        if kind == "CRYPTOCURRENCY" and cur == "USD":
            pass
        elif cur not in ("EUR", "USD") or "-" in sym or "=" in sym:
            continue
        name = (q.get("longname") or q.get("shortname") or sym).strip()
        hits.append({"ticker": sym, "name": name, "type": SEARCH_TYPES[kind],
                     "exchange": q.get("exchDisp", ""), "currency": cur})
    if not hits and quotes and _retry:
        # ISIN liefert oft nur die Londoner Notierung: mit deren Namen nach Euro-Börsen weitersuchen
        name = quotes[0].get("longname") or quotes[0].get("shortname")
        if name:
            return search_titles(name, limit, _retry=False)
    rank = {".DE": 0}
    hits.sort(key=lambda h: (rank.get(h["ticker"][-3:], 1 if h["currency"] == "EUR" else 2)))
    return hits[:limit]


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()
