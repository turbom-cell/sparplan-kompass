"""Datenschicht: Kurse laden, bereinigen und in Tagesrenditen umrechnen.

Die Schicht kennt weder Streamlit noch die Optimierung. Sie liefert nur
saubere Kurs- und Renditetabellen in Euro (alle Ticker sind Xetra-Notierungen).

Lange Historie: Die ETFs gibt es erst seit etwa 2008 bis 2019. Für die Zeit davor
wird eine Ersatz-Reihe (Index oder alter Fonds, in Euro umgerechnet) vor den
ETF-Kurs gesetzt. Diese Reihen liegen fertig in `data/proxy_history.csv` und werden
von `scripts/update_sample.py` erzeugt.
"""
from __future__ import annotations

import re
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


NON_EQUITY_WORDS = ("gold", "silber", "silver", "metal", "bond", "anleihe", "treasury", "aggregate",
                    "govt", "government", "corporate", "geldmarkt", "money market", "euwax", "rohstoff",
                    "commodit", "bitcoin", "krypto", "crypto")


def default_teilfreistellung(ticker: str) -> int:
    """Aktien-ETFs 30 %, sonst 0 %. Gesuchte Titel: an Namen wie Gold/Anleihe erkennen."""
    if ticker in CATALOG:
        return CATALOG[ticker][1]
    name = CUSTOM_NAMES.get(ticker, "").lower()
    if ticker.endswith(("-USD", "-EUR")) or any(w in name for w in NON_EQUITY_WORDS):
        return 0
    return 30


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


EUR_SUFFIXES = (".DE", ".F", ".SG", ".MU", ".DU", ".HM", ".BE", ".HA", ".AS", ".PA", ".MI", ".MC",
                ".VI", ".BR", ".HE", ".IR", ".LS")


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
# OpenFIGI-Börsencode -> Yahoo-Endung und Anzeigename (deutsche Börsen zuerst)
FIGI_EXCHANGES = {"GY": (".DE", "XETRA"), "GF": (".F", "Frankfurt"), "GS": (".SG", "Stuttgart"),
                  "GM": (".MU", "München"), "GD": (".DU", "Düsseldorf"), "GH": (".HM", "Hamburg"),
                  "GB": (".BE", "Berlin"), "GI": (".HA", "Hannover"), "NA": (".AS", "Amsterdam"),
                  "FP": (".PA", "Paris"), "IM": (".MI", "Mailand"), "SM": (".MC", "Madrid"),
                  "BB": (".BR", "Brüssel"), "AV": (".VI", "Wien"), "US": ("", "USA")}
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
WKN_RE = re.compile(r"^[A-Z0-9]{6}$")


def isin_valid(isin: str) -> bool:
    """Prüfziffer nach ISO 6166 (Luhn über die in Zahlen umgewandelten Zeichen)."""
    if not ISIN_RE.match(isin):
        return False
    digits = "".join(str(int(c, 36)) for c in isin[:-1])
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d) * (2 if i % 2 == 0 else 1)
        total += n // 10 + n % 10
    return (10 - total % 10) % 10 == int(isin[-1])


def search_kind(query: str) -> str:
    """Was wurde eingegeben: ISIN, (fast-)ISIN mit Tippfehler, WKN oder Text?"""
    q = query.strip().upper().replace(" ", "")
    if len(q) == 12 and q[:2].isalpha() and q.isalnum():
        return "isin" if isin_valid(q) else "bad_isin"
    if 10 <= len(q) <= 13 and q[:2] in ("DE", "IE", "LU", "FR", "NL", "AT", "US", "GB", "CH", "JE") \
            and q[2:4].isdigit() and q.isalnum():
        return "bad_isin"
    if WKN_RE.match(q) and any(c.isdigit() for c in q):
        return "wkn"
    return "text"


def _yahoo_search(query: str) -> list[dict]:
    import yfinance as yf
    return yf.Search(query, max_results=25, news_count=0).quotes


def _figi(id_type: str, value: str) -> list[dict]:
    """OpenFIGI (kostenlos, ohne Schlüssel): alle Börsennotierungen zu ISIN oder WKN."""
    import json
    import urllib.request
    req = urllib.request.Request("https://api.openfigi.com/v3/mapping",
                                 data=json.dumps([{"idType": id_type, "idValue": value}]).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)[0].get("data", [])


def _has_prices(symbols: list[str]) -> list[str]:
    """Welche Yahoo-Kürzel haben wirklich aktuelle Kurse?"""
    if not symbols:
        return []
    import yfinance as yf
    raw = yf.download(symbols, period="1mo", progress=False, auto_adjust=True, threads=True)
    if raw is None or raw.empty:
        return []
    close = raw["Close"] if "Close" in raw else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    return [s for s in symbols if s in close.columns and close[s].notna().sum() >= 3]


def _suffix(sym: str) -> str:
    return sym[sym.rfind("."):] if "." in sym else ""


def _from_figi(id_type: str, value: str) -> list[dict]:
    rank = {v[0]: i for i, v in enumerate(FIGI_EXCHANGES.values())}
    cands: dict[str, dict] = {}
    _figi_rows = _figi(id_type, value)
    for d in _figi_rows:
        code, ticker = d.get("exchCode"), str(d.get("ticker", "")).upper()
        if code not in FIGI_EXCHANGES or not ticker or " " in ticker or "/" in ticker:
            continue
        suffix, exch = FIGI_EXCHANGES[code]
        sym = ticker + suffix
        fund = "Fund" in str(d.get("securityType2", "")) or "ETP" in str(d.get("securityType", ""))
        cands.setdefault(sym, {"ticker": sym, "name": str(d.get("name") or sym).title(),
                               "type": "ETF/Fonds" if fund else "Aktie", "exchange": exch,
                               "currency": "EUR" if suffix else "USD"})
    # Bloomberg-Kürzel weichen oft vom Börsenkürzel ab (iShares DAX: DAXEX, Xetra: EXS1).
    # Daher aus allen Varianten Xetra-/Frankfurt-Kandidaten bilden und auf echte Kurse prüfen.
    bases: dict[str, None] = {}
    for d in _figi_rows:
        t = str(d.get("ticker", "")).upper()
        if not t or " " in t or "/" in t:
            continue
        for cut in ("EUR", "USD", "CHF", "GBP"):
            if t.endswith(cut) and len(t) > len(cut) + 2:
                t = t[: -len(cut)]
        bases[t] = None
        if len(t) == 5 and t[-1] in "DNMZ":
            bases[t[:-1]] = None
    template = next(iter(cands.values()), None) or (_figi_rows and {
        "name": str(_figi_rows[0].get("name") or value).title(), "type": "Titel"})
    for b in list(bases)[:12]:
        for suffix, exch in ((".DE", "XETRA"), (".F", "Frankfurt")):
            if b + suffix not in cands and template:
                cands[b + suffix] = {**template, "ticker": b + suffix, "exchange": exch, "currency": "EUR"}
    syms = sorted(cands, key=lambda s: rank.get(_suffix(s), 99))[:24]
    ok = set(_has_prices(syms))
    hits = [cands[s] for s in syms if s in ok][:8]
    if hits:   # schöneren, vollständigen Namen von Yahoo holen
        try:
            q = next((q for q in _yahoo_search(hits[0]["ticker"]) if q.get("longname")), None)
            if q:
                for h in hits:
                    h["name"] = q["longname"]
        except Exception:
            pass
    return hits


def _from_yahoo(query: str, retry: bool = True) -> list[dict]:
    quotes = _yahoo_search(query)
    hits = []
    for q in quotes:
        sym, kind = str(q.get("symbol", "")).upper(), q.get("quoteType", "")
        if not sym or kind not in SEARCH_TYPES:
            continue
        if kind == "CRYPTOCURRENCY":
            if not sym.endswith(("-USD", "-EUR")):
                continue
            cur = sym[-3:]
        else:
            cur = currency_of(sym)
            if cur not in ("EUR", "USD") or "-" in sym or "=" in sym:
                continue
        name = (q.get("longname") or q.get("shortname") or sym).strip()
        hits.append({"ticker": sym, "name": name, "type": SEARCH_TYPES[kind],
                     "exchange": q.get("exchDisp", ""), "currency": cur})
    if not hits and quotes and retry and search_kind(query) == "isin":
        # ISIN liefert bei Yahoo oft nur die Londoner Notierung: mit deren Namen weitersuchen
        name = quotes[0].get("longname") or quotes[0].get("shortname")
        if name:
            return _from_yahoo(name, retry=False)
    return hits


def search_titles(query: str, limit: int = 12) -> tuple[list[dict], str]:
    """Suche nach Name, Kürzel, ISIN oder WKN. Gibt (Treffer, Hinweis) zurück.

    Name/Kürzel: Yahoo Finance. ISIN/WKN: zusätzlich OpenFIGI (alle Börsennotierungen weltweit),
    geprüft auf echte Yahoo-Kurse. Nur Titel, die sich in Euro rechnen lassen; Euro-Börsen vorn.
    """
    query = query.strip()
    if len(query) < 2:
        return [], ""
    kind = search_kind(query)
    if kind == "bad_isin":
        return [], ("Das sieht nach einer ISIN aus, ist aber keine gültige: Eine ISIN hat genau "
                    "12 Zeichen und eine passende Prüfziffer. Bitte auf Tippfehler prüfen.")
    key = query.upper().replace(" ", "")
    hits, errors = [], 0
    if kind in ("isin", "wkn"):
        try:
            hits = _from_figi("ID_ISIN" if kind == "isin" else "ID_WERTPAPIER", key)
        except Exception:
            errors += 1
    try:
        seen = {h["ticker"] for h in hits}
        hits += [h for h in _from_yahoo(key if kind != "text" else query) if h["ticker"] not in seen]
    except Exception:
        errors += 1
    if not hits and kind == "text" and len(query) <= 15 and " " not in query:
        sym = query.upper()   # Yahoo-Kürzel direkt probieren, z. B. EWG2.SG
        try:
            if currency_of(sym) in ("EUR", "USD") and _has_prices([sym]):
                hits = [{"ticker": sym, "name": sym, "type": "Titel", "exchange": "",
                         "currency": currency_of(sym)}]
        except Exception:
            pass
    hits.sort(key=lambda h: 0 if h["ticker"].endswith(".DE") else 1 if h["currency"] == "EUR" else 2)
    if kind in ("isin", "wkn") and hits:
        # ISIN/WKN = ein Wertpapier: überall denselben, vollständigsten Namen zeigen
        best = max((h["name"] for h in hits), key=lambda n: (not n.isupper() and not n.istitle(), len(n)))
        for h in hits:
            h["name"] = best
    note = ""
    if not hits:
        note = ("Die Suche ist gerade nicht erreichbar, bitte gleich nochmal versuchen." if errors >= 2 else
                "Nichts gefunden, das an einer Euro- oder US-Börse mit Kursen gehandelt wird. "
                "Probiere Name, ISIN, WKN oder das Yahoo-Kürzel (z. B. SAP.DE).")
    return hits[:limit], note


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()
