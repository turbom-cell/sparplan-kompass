"""Sparplan-Kompass: deutlich weniger Depot-Risiko, ohne zu verkaufen.

Start:  streamlit run app.py
Diese Datei enthält nur die Oberfläche. Gerechnet wird in `core/`, gezeichnet in `ui/`.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from core import analytics, tax
from core.data_service import (CATALOG, CUSTOM_NAMES, GROUPS, default_teilfreistellung, display_name,
                               group_of, load_prices, search_titles, to_returns)
from core.optimizer import cvar, diluted_weights, earliest_month, mix_weights, portfolio_stats
from core.rebalancer import build_plan, sell_rebalance
from ui import charts, theme

st.set_page_config(page_title="Sparplan-Kompass", page_icon="🧭", layout="wide",
                   initial_sidebar_state="collapsed")
theme.inject()

KOMPASS, HOLD = "Sparplan-Kompass", "Einfach halten"
BENCH = "EUNL.DE"                 # MSCI-World-ETF: bestimmt, ab wann echte Vergleichskurse vorliegen
STRATEGIES = {
    "Ausgewogen (empfohlen)": {"Sharpe": 1.0},
    "Extra vorsichtig (50/35/15)": {"Sharpe": 0.50, "CVaR": 0.35, "Varianz": 0.15},
}
STRATEGY_HELP = {
    "Ausgewogen (empfohlen)": "Bestes Verhältnis aus Rendite und Schwankung, solange die "
                              "Schmerzgrenze hält.",
    "Extra vorsichtig (50/35/15)": "50 % bestes Rendite-Risiko-Verhältnis, 35 % kleinster "
                                   "Crash-Verlust, 15 % kleinste Schwankung. Noch weniger Risiko, "
                                   "etwas weniger Rendite.",
}
DIVERSIFIERS = ["4GLD.DE", "EUNH.DE"]   # im Backtest die wirksamsten Risikopuffer
# Vorlagen: (Ticker, Bestand €, Einstand €, Kernposition)
PRESETS: dict[str, tuple[list[tuple[str, int, int, bool]], list[str]]] = {
    "Tech-lastig (Klumpenrisiko)": (
        [("SXRV.DE", 6000, 3600, True), ("SXR8.DE", 3000, 2200, False),
         ("EUNL.DE", 2500, 2050, False), ("4GLD.DE", 500, 400, False)], DIVERSIFIERS),
    "Welt-Depot mit viel DAX (Heimatmarkt-Klumpen)": (
        [("EUNL.DE", 5000, 3800, True), ("EXS1.DE", 4500, 3300, False),
         ("IS3N.DE", 1500, 1400, False)], DIVERSIFIERS),
    "Langjähriger Sparer (60.000 €)": (
        [("SXRV.DE", 26000, 9000, True), ("SXR8.DE", 18000, 9500, False),
         ("EUNL.DE", 12000, 8000, False), ("XDWT.DE", 4000, 2000, False)], DIVERSIFIERS),
    "Leeres Depot zum Selberbauen": ([("EUNL.DE", 1000, 1000, True)], []),
}
LABEL = {t: f"{name} · {t}" for t, (name, _, _) in CATALOG.items()}


def label_of(ticker: str) -> str:
    return LABEL.get(ticker, f"{display_name(ticker)} · {ticker}" if ticker in CUSTOM_NAMES
                     else f"Eigener Titel · {ticker}")


def ticker_of(label: str) -> str:
    return label.split(" · ")[-1].strip().upper()


def is_equity(ticker: str) -> bool:
    """Aktien-ETFs (Teilfreistellung 30 %) erholen sich nach Korrekturen meist; Anleihen und Gold nicht sicher."""
    return default_teilfreistellung(ticker) == 30


# --------------------------------------------------------------------------- Formatierung
def eur(x: float) -> str:
    return f"{x:,.0f} €".replace(",", ".")


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f} %".replace(".", ",").replace("-", "−")


def num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def vs(mine: str, other: str) -> str:
    return f"{mine} statt {other}"


# --------------------------------------------------------------------------- Rechnen (gecacht)
@st.cache_data(ttl=3600, show_spinner="Lade Kurse …")
def cached_prices(tickers: tuple[str, ...]):
    data = load_prices(list(tickers), 2000, True, False)   # nur echte Kurse ab Börsenstart
    return data.prices, data.source, data.missing, data.etf_start


@st.cache_data(show_spinner="Berechne Ziel-Depot …")
def cached_target(returns: pd.DataFrame, rf: float, limit: float, w_max: float,
                  w_min: tuple[float, ...] | None, mix: tuple):
    t = mix_weights(returns.values, rf, limit, w_max, None if w_min is None else np.array(w_min),
                    dict(mix))
    return t.weights, t.status


@st.cache_data(show_spinner=False)
def cached_earliest(returns: pd.DataFrame, holdings: tuple[float, ...], budget: float,
                    limit: float, w_max: float):
    return earliest_month(returns.values, np.array(holdings), budget, limit, w_max)


@st.cache_data(show_spinner="Spiele die Jahre mit echten Kursen nach …")
def cached_walk_forward(prices: pd.DataFrame, bench: pd.Series, holdings: tuple[float, ...],
                        budget: float, years: int, rf: float, limit: float, w_max: float,
                        horizon: int, mix: tuple, core: tuple[bool, ...], not_before: pd.Timestamp,
                        dip: tuple | None, dip_mask: tuple[bool, ...]):
    return analytics.walk_forward(prices, bench, np.array(holdings), budget, years, rf, limit,
                                  w_max, horizon, mix=dict(mix), core=np.array(core),
                                  not_before=not_before, dip=dip, dip_mask=np.array(dip_mask))


@st.cache_data(ttl=86400, show_spinner="Suche bei Yahoo Finance und OpenFIGI …")
def cached_search(query: str):
    return search_titles(query)


def remember(hit: dict) -> str:
    """Titel aus der Suche merken (Name für Tabellen und Grafiken)."""
    t = hit["ticker"]
    if t not in CATALOG:
        CUSTOM_NAMES[t] = st.session_state.setdefault("own_names", {})[t] = hit["name"]
        if t not in st.session_state["own"]:
            st.session_state["own"].append(t)
    return t


DEPOT_COLS = ["Position", "Anteil %", "Gewinn %", "Kern"]


def depot_frame(rows: list[tuple[str, float, float, bool]]) -> pd.DataFrame:
    """Tabelle für den Editor aus Euro-Werten (Ticker, Bestand €, Einstand €, Kern).

    Eingegeben wird in Prozent: Anteil am Depot und Gewinn seit Kauf. Ticker werden zu Labels.
    """
    total = sum(float(b) for _, b, _, _ in rows) or 1.0
    return pd.DataFrame({"Position": [label_of(t) for t, _, _, _ in rows],
                         "Anteil %": [round(float(b) / total * 100, 1) for _, b, _, _ in rows],
                         "Gewinn %": [round((float(b) / float(e) - 1) * 100, 1) if float(e) > 0 else 0.0
                                      for _, b, e, _ in rows],
                         "Kern": [bool(c) for _, _, _, c in rows],
                         "Entfernen": [False] * len(rows)})


def to_euro(df: pd.DataFrame, value: float) -> pd.DataFrame:
    """Prozent-Eingaben in Euro umrechnen: Anteile werden auf 100 % hochgerechnet."""
    df = df.copy()
    share = pd.to_numeric(df["Anteil %"], errors="coerce").fillna(0.0).clip(lower=0)
    gain = pd.to_numeric(df["Gewinn %"], errors="coerce").fillna(0.0).clip(lower=-99)
    df["Bestand €"] = share / (share.sum() or 1.0) * value
    df["Einstand €"] = df["Bestand €"] / (1 + gain / 100)
    return df


def set_base(df: pd.DataFrame, extras: list[str] | None = None):
    """Neue Ausgangstabelle für den Editor (neuer Schlüssel = Editor übernimmt sie sauber)."""
    ss = st.session_state
    old = ss.get(f"extras_{ss['base_ver']}", [])
    ss["base_df"] = df.reset_index(drop=True)
    ss["base_ver"] += 1
    ss[f"extras_{ss['base_ver']}"] = list(old if extras is None else extras)


def current_depot() -> pd.DataFrame:
    """Ausgangstabelle plus alle Änderungen, die gerade im Editor gemacht wurden."""
    ss = st.session_state
    df = ss["base_df"].copy().reset_index(drop=True)
    edits = ss.get(f"depot_{ss['base_ver']}") or {}
    for i, change in edits.get("edited_rows", {}).items():
        for col, val in change.items():
            df.loc[int(i), col] = val
    added = [r for r in edits.get("added_rows", []) if r]
    if added:
        df = pd.concat([df, pd.DataFrame(added)], ignore_index=True)
    deleted = [int(i) for i in edits.get("deleted_rows", [])]
    df = df.drop(index=[i for i in deleted if i in df.index]).reset_index(drop=True)
    if "Entfernen" not in df:
        df["Entfernen"] = False
    df["Entfernen"] = df["Entfernen"].fillna(False).astype(bool)
    return df


def on_depot_edit():
    """Haken bei „Entfernen“ gesetzt: Zeile sofort aus dem Depot nehmen."""
    df = current_depot()
    if df["Entfernen"].any():
        gone = [str(p).split(" · ")[0] for p in df.loc[df["Entfernen"], "Position"]]
        set_base(df[~df["Entfernen"]])
        st.session_state["toast"] = "Entfernt: " + ", ".join(gone)


def depot_file() -> str:
    """Depot als JSON zum Herunterladen (bleibt beim Nutzer, auf dem Server wird nichts gespeichert)."""
    ss = st.session_state
    df = to_euro(current_depot().dropna(subset=["Position"]), float(ss.get("depot_value", 10000)))
    data = {"app": "Sparplan-Kompass", "version": 2, "gespeichert": date.today().isoformat(),
            "sparrate": ss.get("budget", 500), "depotwert": float(ss.get("depot_value", 10000)),
            "positionen": [{"ticker": ticker_of(r["Position"]),
                            "name": str(r["Position"]).split(" · ")[0],
                            "bestand": float(r["Bestand €"] or 0),
                            "einstand": float(r["Einstand €"] or 0),
                            "kern": bool(r["Kern"])} for _, r in df.iterrows()],
            "bausteine": [{"ticker": ticker_of(x), "name": x.split(" · ")[0]}
                          for x in ss.get(f"extras_{ss['base_ver']}", [])]}
    return json.dumps(data, ensure_ascii=False, indent=2)


def load_depot_file():
    ss = st.session_state
    f = ss.get("depot_upload")
    if f is None:
        return
    try:
        data = json.loads(f.getvalue().decode("utf-8"))
        for item in data.get("positionen", []) + data.get("bausteine", []):
            t = str(item["ticker"]).upper()
            if t not in CATALOG:
                remember({"ticker": t, "name": item.get("name", t)})
        rows = [(str(p["ticker"]).upper(), float(p.get("bestand", 0)), float(p.get("einstand", 0)),
                 bool(p.get("kern", False))) for p in data["positionen"]]
        held = {r[0] for r in rows}
        set_base(depot_frame(rows), [label_of(str(b["ticker"]).upper()) for b in data.get("bausteine", [])
                                     if str(b["ticker"]).upper() not in held])
        ss["budget"] = int(min(3000, max(50, round(float(data.get("sparrate", 500)) / 50) * 50)))
        ss["depot_value"] = int(round(sum(r[1] for r in rows))) or 10000
        ss["toast"] = f"Depot geladen: {len(rows)} Positionen."
    except Exception:
        ss["toast_err"] = ("Die Datei konnte nicht gelesen werden. Bitte eine Datei wählen, "
                           "die mit „Depot speichern“ erstellt wurde.")


def add_to_depot(hit: dict):
    ss = st.session_state
    t = remember(hit)
    df = current_depot().dropna(subset=["Position"])
    if label_of(t) not in set(df["Position"]):
        new = pd.DataFrame({"Position": [label_of(t)], "Anteil %": [10.0], "Gewinn %": [0.0],
                            "Kern": [False], "Entfernen": [False]})
        df = pd.concat([df, new], ignore_index=True)
    set_base(df)
    ss["toast"] = f"{hit['name']} steht jetzt in deinem Depot. Trage Anteil und Gewinn in % ein."


def add_extra(hit: dict, key: str):
    ss = st.session_state
    t = remember(hit)
    if label_of(t) not in ss[key]:
        ss[key] = ss[key] + [label_of(t)]
    ss["toast"] = f"{hit['name']} darf jetzt vom Kompass zugekauft werden."


def search_box() -> dict | None:
    """Suchfeld für jeden Titel von Yahoo Finance: Name, Kürzel oder ISIN."""
    query = st.text_input("🔎 Titel suchen", placeholder="Name, ISIN, WKN oder Kürzel – z. B. Vanguard All-World, "
                          "IE00B4L5Y983, A0S9GB, Rheinmetall",
                          help="Sucht ETFs, ETCs, Aktien, Fonds und Krypto bei Yahoo Finance und im "
                               "ISIN-Verzeichnis OpenFIGI. Euro-Börsen (Xetra) stehen oben, US-Werte "
                               "rechnet der Kompass in Euro um.")
    if not query.strip():
        return None
    hits, note = cached_search(query.strip())
    if not hits:
        st.caption(note)
        if "nicht erreichbar" in note:
            cached_search.clear()
        return None
    pick = st.selectbox(f"{len(hits)} Treffer", range(len(hits)), label_visibility="collapsed",
                        format_func=lambda i: f"{hits[i]['name']} · {hits[i]['ticker']} · "
                                              f"{hits[i]['type']} · {hits[i]['exchange']}")
    return hits[pick]


hero_slot = st.container()
tab_home, tab_depot, tab_proof = st.tabs(["Überblick", "Dein Depot", "Beweis"])
if st.session_state.get("toast"):
    st.toast(st.session_state.pop("toast"), icon="✅")
if st.session_state.get("toast_err"):
    st.toast(st.session_state.pop("toast_err"), icon="⚠️")

# =========================================================================== Eingaben
with tab_depot:
    theme.section("Schritt 1", "Dein Depot",
                  "Trage ein, was du heute besitzt. Rechts wählst du, was der Kompass zukaufen darf.")
    ss = st.session_state
    ss.setdefault("own", [])                      # Titel aus der Suche: [Ticker]
    CUSTOM_NAMES.update(ss.get("own_names", {}))  # Namen gehören zur Sitzung
    left, right = st.columns([3, 2], gap="large")
    with left:
        preset = st.selectbox("Vorlage", list(PRESETS))
    rows, preset_extras = PRESETS[preset]
    if ss.get("base_preset") != preset:           # neue Vorlage: Tabelle und Bausteine zurücksetzen
        ss["base_preset"], ss["base_ver"] = preset, ss.get("base_ver", 0) + 1
        ss["base_df"] = depot_frame(rows)
        ss["depot_value"] = int(sum(b for _, b, _, _ in rows))
        ss["extras_sel"] = [LABEL[t] for t in preset_extras if t not in [r[0] for r in rows]]
    extras_key = f"extras_{ss['base_ver']}"
    ss.setdefault(extras_key, list(ss["extras_sel"]))

    with left:
        found = search_box()
        if found:
            b1, b2 = st.columns(2)
            b1.button("➕ Ins Depot übernehmen", on_click=add_to_depot, args=(found,),
                      width="stretch", type="primary")
            b2.button("🧩 Als Baustein zukaufen lassen", on_click=add_extra, args=(found, extras_key),
                      width="stretch")
    options = list(LABEL.values()) + [label_of(t) for t in ss["own"] if t not in LABEL]
    with left:
        V_input = float(st.number_input("Depotwert gesamt in €", min_value=0, max_value=10_000_000,
                                        step=500, key="depot_value",
                                        help="Was dein Depot heute insgesamt wert ist. Die Anteile in % "
                                             "rechnet der Kompass damit in Euro um."))
        depot = st.data_editor(
            ss["base_df"],
            key=f"depot_{ss['base_ver']}", num_rows="dynamic", hide_index=True, width="stretch",
            on_change=on_depot_edit,
            column_order=["Entfernen", "Position", "Anteil %", "Gewinn %", "Kern"],
            column_config={
                "Entfernen": st.column_config.CheckboxColumn(
                    "🗑️", width="small", help="Haken setzen: Position wird sofort aus dem Depot genommen."),
                "Position": st.column_config.SelectboxColumn(options=options, required=True,
                                                             width="medium"),
                "Kern": st.column_config.CheckboxColumn(
                    help="Kernposition: behält mindestens ihren heutigen Anteil und bekommt weiter "
                         "ihren Teil der Sparrate."),
                "Anteil %": st.column_config.NumberColumn(
                    min_value=0.0, max_value=100.0, step=0.1, format="%.1f %%",
                    help="Anteil der Position an deinem Depot. Zusammen 100 %."),
                "Gewinn %": st.column_config.NumberColumn(
                    min_value=-99.0, step=0.1, format="%.1f %%",
                    help="Wie viel die Position seit dem Kauf im Plus (oder Minus) ist – steht in "
                         "deiner Broker-App. Braucht der Kompass für die Steuerrechnung.")})
        share_sum = float(pd.to_numeric(depot["Anteil %"], errors="coerce").fillna(0).sum())
        if abs(share_sum - 100) > 0.5 and share_sum > 0:
            st.caption(f"⚠️ Die Anteile ergeben zusammen {pct(share_sum)}. Der Kompass rechnet sie "
                       f"anteilig auf 100 % um.")
        st.caption("Neue Zeile: unten auf „+“ klicken oder oben suchen. Entfernen: Haken bei 🗑️ setzen.")
        ss.setdefault("budget", 500)
        budget = float(st.slider("Monatliche Sparrate in €", min_value=50, max_value=3000, step=50,
                                 key="budget"))
        sv, ld = st.columns(2)
        sv.download_button("💾 Depot speichern", depot_file(), file_name="mein-depot.json",
                           mime="application/json", width="stretch",
                           help="Lädt dein Depot als kleine Datei auf dein Gerät. Gespeichert wird nur "
                                "bei dir, nicht auf dem Server.")
        with ld.popover("📂 Depot laden", width="stretch"):
            st.file_uploader("Gespeicherte Depot-Datei wählen", type=["json"], key="depot_upload",
                             on_change=load_depot_file)
    depot = depot.dropna(subset=["Position"]).drop_duplicates("Position")[DEPOT_COLS].copy()
    depot = to_euro(depot, V_input)
    depot["Kern"] = depot["Kern"].fillna(False).astype(bool)
    depot[["Bestand €", "Einstand €"]] = depot[["Bestand €", "Einstand €"]].fillna(0.0)
    held = [ticker_of(p) for p in depot["Position"]]
    with right:
        own_extra = [label_of(t) for t in ss["own"] if t not in held and t not in LABEL]
        ss[extras_key] = [x for x in ss[extras_key] if ticker_of(x) not in held]
        extras_labels = st.multiselect(
            "Bausteine, die der Kompass zukaufen darf",
            own_extra + [LABEL[t] for g in GROUPS for t in CATALOG if group_of(t) == g and t not in held],
            key=extras_key,
            help="Gold und Euro-Staatsanleihen senken das Risiko am wirksamsten. Weitere Titel "
                 "findest du links über die Suche.")
        with st.expander("Steuer-Annahmen"):
            allowance = float(st.number_input("Freier Sparerpauschbetrag in €", 0, 2000, 1000, 100))
            fee = float(st.number_input("Ordergebühr je Verkauf in €", 0.0, 20.0, 1.0, 0.5))
    rf = 0.025

if depot.empty or depot["Bestand €"].sum() <= 0:
    with hero_slot:
        theme.hero("Sparplan-Kompass", "Trage unter „Dein Depot“ mindestens eine Position mit "
                   "Bestand ein.", [], eyebrow="ETF-Sparplan · Risiko · Steuern")
    st.stop()

universe = list(dict.fromkeys(held + [ticker_of(x) for x in extras_labels]))
prices, source, missing, etf_start = cached_prices(tuple(universe))
if missing:
    st.warning("Keine Kursdaten für: " + ", ".join(display_name(t) for t in missing) + ".")
if prices.empty:
    st.error("Für diese Auswahl liegen keine Kursdaten vor.")
    st.stop()

tickers = list(prices.columns)
names = [display_name(t) for t in tickers]
book = depot.assign(Ticker=held).set_index("Ticker")
h = np.array([float(book.loc[t, "Bestand €"]) if t in book.index else 0.0 for t in tickers])
basis = np.array([float(book.loc[t, "Einstand €"]) if t in book.index else 0.0 for t in tickers])
core = np.array([bool(book.loc[t, "Kern"]) if t in book.index else False for t in tickers])
equity = np.array([is_equity(t) for t in tickers])
if h.sum() <= 0:
    st.error("Für die Positionen mit Bestand liegen keine Kursdaten vor.")
    st.stop()
tf = np.array([default_teilfreistellung(t) for t in tickers])
returns = to_returns(prices)
R = returns.values
V = float(h.sum())
w_now = h / V
now = portfolio_stats(w_now, R, rf)
loss_now_pct = now.cvar * 100

with tab_depot:
    theme.section("Schritt 2", "Deine Regeln",
                  "Drei Regeln gelten immer: Schmerzgrenze, maximale Positionsgröße und "
                  "„nichts verkaufen“.")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        default_limit = float(np.clip(np.ceil(loss_now_pct * 10) / 10, 0.5, 6.0))
        limit_pct = st.slider("Schmerzgrenze: Verlust an einem Schrecktag höchstens",
                              0.5, 6.0, default_limit, 0.1, format="%.1f %%",
                              key=f"limit_{preset}_{len(tickers)}",
                              help="Schrecktag = Durchschnitt der 5 % schlechtesten Börsentage. "
                                   "Angegeben in Prozent des Depots.")
        st.caption(f"Heute: {pct(loss_now_pct, 1)} · bei deinem Depot sind {pct(limit_pct, 1)} "
                   f"rund {eur(limit_pct / 100 * V)}")
    with c2:
        w_max = st.slider("Max. Positionsgröße", 10, 100, 60, 5, format="%d %%") / 100
        st.caption("Keine Position darf im Ziel größer werden. Kernpositionen behalten ihren Anteil.")
    with c3:
        months = st.slider("Umbau-Zeitraum in Monaten", 6, 120, 36, 6,
                           help="Ohne Verkauf geht der Umbau nur so schnell, wie Sparraten nachfließen.")
        limit = limit_pct / 100
        first = cached_earliest(returns, tuple(h), budget, limit, w_max)
        st.caption("Schmerzgrenze ist schon heute erreichbar." if first == 0 else
                   f"Ohne Verkauf frühestens nach {first} Monaten erreichbar." if first else
                   "Ohne Verkauf in 10 Jahren nicht erreichbar.")
    s1, s2 = st.columns([1, 1], gap="large")
    with s1:
        strategy = st.segmented_control("Strategie", list(STRATEGIES), default="Ausgewogen (empfohlen)")
        strategy = strategy or "Ausgewogen (empfohlen)"
        st.caption(STRATEGY_HELP[strategy])
    with s2:
        use_dip = st.toggle("Nachkaufen in Korrekturen", value=True,
                            help="Liegt ein Aktien-ETF unter seiner 50-, 100- oder 200-Tage-Linie, "
                                 "bekommt er mehr von der Sparrate, die anderen entsprechend weniger. "
                                 "Gilt nicht für Anleihen und Gold.")
        st.caption("Aktien-ETFs unter ihrer 50-, 100- oder 200-Tage-Linie bekommen mehr von der "
                   "Sparrate (bis doppelt so viel). Anleihen und Gold nie. Die Aufteilung dafür "
                   "einmal im Monat in der App nachsehen und beim Broker anpassen.")
    mix = STRATEGIES[strategy]
    mix_key = tuple(sorted((k, float(v)) for k, v in mix.items()))
    dip = analytics.DIP_BOOST if use_dip else None

# =========================================================================== Ergebnis berechnen
lo = diluted_weights(h, budget, months, core)
w_goal, status = cached_target(returns, rf, limit, w_max, tuple(np.round(lo, 6)), mix_key)
w_free, _ = cached_target(returns, rf, limit, w_max, None, mix_key)
goal = portfolio_stats(w_goal, R, rf)
plan = build_plan(h, w_goal, budget, months, names)
cost = tax.sell_cost(sell_rebalance(h, w_free), h, np.where(h > 0, basis, 0.0), tf, allowance, fee)
top_now, top_goal = int(np.argmax(w_now)), int(np.argmax(w_goal))
dip_today = analytics.dip_factors(prices, analytics.DIP_BOOST).iloc[-1].to_numpy()
this_month = (analytics.apply_dip(plan.monthly.to_numpy(), dip_today, equity, budget)
              if use_dip else plan.monthly.to_numpy())

bench_prices, _, _, bench_start = cached_prices((BENCH,))
bench_since = bench_start.get(BENCH, prices.index[0])
first_sim = max(bench_since, prices.index[0] + pd.Timedelta(days=730))
avail = (prices.index[-1] - first_sim).days / 365.25

with tab_proof:
    theme.section("Beweis", "Mit echten Kursen nachgerechnet",
                  "Zwei Anleger starten mit deinem Depot und zahlen jeden Monat dieselbe Sparrate ein. "
                  "Der eine nutzt den Kompass, der andere hält einfach weiter.")
    options = [y for y in (5, 10) if y < avail - 0.5] + [int(np.ceil(avail))]
    sim_years = st.radio("Zeitraum", options, index=len(options) - 1, horizontal=True,
                         format_func=lambda y: (f"letzte {y} Jahre" if y < avail - 0.5
                                                else f"so lange wie möglich: seit {first_sim:%m/%Y}"))
period = f"letzte {sim_years} Jahre" if sim_years < avail - 0.5 else f"seit {first_sim:%m/%Y}"
sim = None if bench_prices.empty else cached_walk_forward(
    prices, bench_prices.iloc[:, 0], tuple(h), budget, sim_years, rf, limit, w_max, months, mix_key,
    tuple(core), first_sim, dip, tuple(equity))

if sim is not None:
    values, rets, _ = sim
    stats = analytics.compare_stats(values, rets, rf).drop(index="Markt")
    crises = analytics.crisis_from_rets(rets).drop(columns="Markt", errors="ignore")
    K, H = stats.loc[KOMPASS], stats.loc[HOLD]
    crisis_wins = int((crises[KOMPASS] > crises[HOLD]).sum()) if len(crises) else 0
    risk_cols = ["Größter Einbruch %", "Schlechtestes Jahr %", "Schwankung p.a. %", "Schrecktag %",
                 "Längste Durststrecke Monate"]
    risk_wins = sum(abs(K[c]) < abs(H[c]) - 1e-9 for c in risk_cols)
    money_diff = K["Endwert €"] / H["Endwert €"] - 1
    dd_gain = K["Größter Einbruch %"] - H["Größter Einbruch %"]     # in Prozentpunkten, positiv = besser
    risk_word = ("Deutlich weniger Risiko" if dd_gain >= 3 else
                 "Weniger Risiko" if dd_gain > 0.5 else "Ähnliches Risiko")
    if abs(money_diff) <= 0.03:
        money_word = "Gleich viel Geld"
    elif money_diff > 0:
        money_word = "Mehr Geld"
    else:
        money_word = "Etwas weniger Geld" if money_diff > -0.10 else "Weniger Geld"
    headline = f"{risk_word}. {money_word}."
    # zweites Risikomaß für die Kurzfassung: das mit der größten relativen Verbesserung
    second = {"Schwankung p.a. %": ("Schwankung pro Jahr", "Weniger Schwankung",
                                    "Wie stark das Depot im Jahr schwankt"),
              "Schrecktag %": ("Verlust am Schrecktag", "Kleinere Schrecktage",
                               "Durchschnitt der 5 % schlechtesten Börsentage"),
              "Schlechtestes Jahr %": ("Schlechtestes Jahr", "Ruhigere Jahre",
                                       "Schlechteste Entwicklung über 12 Monate")}
    best_col = max(second, key=lambda c: 1 - abs(K[c]) / abs(H[c]) if H[c] else 0)
    best_label, best_title, best_text = second[best_col]

# =========================================================================== Kopfbereich
with hero_slot:
    if sim is not None:
        proofs = [
            (vs(pct(K["Größter Einbruch %"], 1), pct(H["Größter Einbruch %"], 1)),
             f"größter Absturz mit Kompass statt einfach halten, {period}"),
            (vs(pct(K[best_col], 1), pct(H[best_col], 1)),
             f"{best_label} mit Kompass statt einfach halten"),
            (eur(K["Endwert €"]), f"Endwert mit Kompass – einfach halten: {eur(H['Endwert €'])}"),
        ]
        title = f"{risk_word}.<br>{money_word}."
    else:
        proofs, title = [], "Deutlich weniger Risiko.<br>Ohne zu verkaufen."
    theme.hero(
        title,
        "Der <b>Sparplan-Kompass</b> baut dein ETF-Depot nur über die Aufteilung deiner Sparrate "
        "um. Du verkaufst nichts, zahlst keine Steuer auf Kursgewinne, und dein Depot stürzt in "
        "Krisen weniger tief ab als beim einfachen Halten – nachgerechnet mit echten Kursen.",
        [("#34d399" if source == "live" else "#fbbf24",
          "Live-Kurse" if source == "live" else "Gespeicherter Kurs-Stand"),
         ("", f"Vergleich {period}"), ("", f"Depot {eur(V)}"),
         ("", f"Sparrate {eur(budget)} / Monat")],
        proofs, eyebrow="Sparplan-Kompass · für ETF-Sparer")

# =========================================================================== Überblick
with tab_home:
    theme.section("Das Problem", "Wer einfach hält, sitzt auf einem Klumpen")
    theme.cards([
        {"icon": "📈", "title": "Klumpen wachsen von selbst",
         "text": f"Was gut läuft, wird im Depot immer größer. In deinem Depot sind es heute "
                 f"<b>{pct(w_now[top_now] * 100, 0)} {theme.esc(names[top_now])}</b>."},
        {"icon": "🎢", "title": "Klumpen stürzen tief",
         "text": f"An einem schlechten Börsentag verlierst du heute im Schnitt "
                 f"<b>{pct(now.cvar * 100, 1)}</b> deines Depots. Wer in so einer Phase aussteigt, "
                 "verpasst die Erholung."},
        {"icon": "💸", "title": "Umschichten kostet Steuern",
         "text": ("Wer verkauft, zahlt 26,375 % Steuer auf den Gewinn. Bei deinem Depot wären das "
                  f"<b>{eur(cost.total)}</b>." if cost.total >= 100 else
                  "Wer verkauft, zahlt 26,375 % Steuer auf den Gewinn. Bei großen Depots mit hohen "
                  "Gewinnen sind das schnell Tausende Euro (Vorlage „Langjähriger Sparer“).")},
    ])
    if sim is not None:
        theme.section("Das Ergebnis", headline,
                      f"Dein Depot mit Kompass und beim einfachen Halten, echte Tageskurse {period}.")
        theme.cards([
            {"num": vs(pct(K["Größter Einbruch %"], 1), pct(H["Größter Einbruch %"], 1)),
             "title": "Kleinerer Absturz",
             "text": "Größter Fall vom Höchststand bis zum Tief, mit Kompass statt einfach halten.",
             "src": f"in {crisis_wins} von {len(crises)} Krisen weniger Verlust" if len(crises) else ""},
            {"num": vs(pct(K[best_col], 1), pct(H[best_col], 1)),
             "title": best_title,
             "text": f"{best_text}, mit Kompass statt einfach halten.",
             "src": f"{risk_wins} von 5 Risikomaßen kleiner"},
            {"num": eur(K["Endwert €"]), "title": money_word,
             "text": f"Endwert mit Kompass, einfach halten: {eur(H['Endwert €'])}. Rendite pro Jahr "
                     f"{vs(pct(K['Rendite p.a. %']), pct(H['Rendite p.a. %']))}.",
             "src": "gleiche Einzahlungen, gleiche Kurse"},
            {"num": "0 €", "title": "Keine Steuer beim Umbau",
             "text": f"Sofort umschichten würde {eur(cost.sold)} Verkäufe auslösen. Der Kompass "
                     "kauft nur zu.",
             "src": "Abgeltungsteuer, Teilfreistellung, Pauschbetrag"},
        ])
    theme.section("Der Unterschied", "Drei Wege mit deinem Depot")
    na = "nicht nachgerechnet"
    hs = sim is not None
    theme.compare_table(
        ["Einfach halten", "Umschichten durch Verkauf", "Sparplan-Kompass"],
        [[f"Größter Absturz, {period}",
          f'<span class="no">{pct(H["Größter Einbruch %"])}</span>' if hs else "–", na,
          f'<span class="yes">{pct(K["Größter Einbruch %"])}</span>' if hs else "–"],
         [f"Schwankung pro Jahr, {period}",
          f'<span class="no">{pct(H["Schwankung p.a. %"])}</span>' if hs else "–", na,
          f'<span class="yes">{pct(K["Schwankung p.a. %"])}</span>' if hs else "–"],
         [f"Endwert, {period}", eur(H["Endwert €"]) if hs else "–", na,
          f'<b>{eur(K["Endwert €"])}</b>' if hs else "–"],
         ["Steuer beim Umbau", '<span class="yes">0 €</span>',
          f'<span class="no">{eur(cost.total)}</span>', '<span class="yes">0 €</span>'],
         ["Klumpenrisiko", '<span class="no">bleibt</span>', '<span class="yes">sofort weg</span>',
          f'<span class="yes">schrittweise weg</span>'],
         ["Was du tun musst", "nichts", "verkaufen und neu kaufen",
          "Sparplan einstellen, einmal im Monat prüfen"]],
        highlight=2)
    st.caption("Studienprojekt, keine Anlageberatung. Vergangene Kurse sind keine Garantie für die "
               "Zukunft. Alle Zahlen dieser Seite werden live aus deinen Eingaben berechnet.")

# =========================================================================== Dein Depot: Ergebnis
with tab_depot:
    theme.section("Ergebnis", "Dein Sparplan",
                  "Verkaufe nichts. Stelle deinen Sparplan so ein und schau einmal im Monat hier vorbei.")
    if status == "unreachable":
        st.warning(f"Die Schmerzgrenze von {pct(limit_pct)} ist in {months} Monaten ohne Verkauf nicht "
                   "erreichbar. Gezeigt wird das sicherste Depot, das bis dahin möglich ist. Tipp: "
                   "Umbau-Zeitraum verlängern, Sparrate erhöhen oder mehr Bausteine zulassen.")
    theme.kpi_row([
        {"label": "Schrecktag", "value": vs(pct(goal.cvar * 100, 1), pct(now.cvar * 100, 1)),
         "sub": f"nach {months} Monaten statt heute", "tone": "good", "accent": theme.GREEN},
        {"label": "Schwankung pro Jahr", "value": vs(pct(goal.vol * 100, 1), pct(now.vol * 100, 1)),
         "sub": f"nach {months} Monaten statt heute", "tone": "good", "accent": theme.GREEN},
        {"label": "Krisenpuffer (Gold, Anleihen)",
         "value": vs(pct(w_goal[~equity].sum() * 100, 0), pct(w_now[~equity].sum() * 100, 0)),
         "sub": f"Anteil am Depot nach {months} Monaten statt heute", "tone": "good", "accent": theme.BLUE},
        {"label": "Steuer beim Umbau", "value": "0 €", "sub": f"statt {eur(cost.total)} beim Verkaufen",
         "tone": "good", "accent": theme.GREEN},
    ])
    below = {w: prices.iloc[-1] < prices.rolling(w).mean().iloc[-1] for w in analytics.SMA_WINDOWS}
    status_txt = []
    for t in tickers:
        lines = [str(w) for w in analytics.SMA_WINDOWS if bool(below[w][t])]
        status_txt.append("–" if not is_equity(t) else
                          ("unter " + "/".join(lines) + "-Tage-Linie") if lines else "keine Korrektur")
    table = pd.DataFrame({
        "Position": names,
        "Sparrate €/Monat": plan.monthly.round(0).to_numpy(),
        "Diesen Monat €": np.round(this_month, 0),
        "Korrektur": status_txt,
        "Anteil heute %": (w_now * 100).round(1),
        "Anteil im Ziel %": (w_goal * 100).round(1)})
    if not use_dip:
        table = table.drop(columns=["Diesen Monat €", "Korrektur"])
    keep = (table["Sparrate €/Monat"] > 0) | (table["Anteil heute %"] > 0)
    st.dataframe(table[keep].sort_values("Sparrate €/Monat", ascending=False), hide_index=True,
                 width="stretch")
    if use_dip:
        st.caption("„Sparrate €/Monat“ ist die Grundaufteilung. „Diesen Monat €“ berücksichtigt das "
                   "Nachkaufen: Aktien-ETFs unter ihren Tageslinien bekommen mehr, die anderen weniger.")
    paused = [n for n, m, w in zip(names, plan.monthly, w_now) if m < 0.5 and w > 0]
    if paused:
        st.caption("⏸️ Sparplan pausieren für: " + ", ".join(paused) +
                   ". Diese Positionen werden nicht verkauft, nur durch die Zukäufe der anderen verwässert.")
    a, b = st.columns([3, 2], gap="large")
    with a:
        st.plotly_chart(charts.plan_animation(plan.weights.loc[:, (plan.weights > 0.001).any()],
                                              w_goal[(plan.weights > 0.001).any().to_numpy()]),
                        width="stretch")
        st.caption("▶ drücken: Monat für Monat zur Zielmarke (schwarzer Strich).")
    with b:
        path = [cvar(w, R) * 100 for w in plan.weights.to_numpy()]
        st.plotly_chart(charts.risk_path(list(plan.weights.index), path, limit_pct), width="stretch")
        st.caption("Verlust an einem Schrecktag in % des Depots, Monat für Monat. Rot: deine Schmerzgrenze.")
    if use_dip and equity.any():
        theme.section("Nachkaufen in Korrekturen", "Wo gerade mehr gekauft wird",
                      "Kurs eines Aktien-ETFs mit seinen Tageslinien. Farbig hinterlegt: Phasen, in denen "
                      "der Kurs darunter lag und dieser ETF mehr von der Sparrate bekommen hat.")
        eq_tickers = [t for t in tickers if is_equity(t)]
        d1, d2 = st.columns([2, 3])
        with d1:
            dip_pick = st.selectbox("Aktien-ETF", eq_tickers, format_func=display_name, key="dip_pick")
        with d2:
            dip_years = st.radio("Zeitraum", [1, 3, 5], index=1, horizontal=True, key="dip_years",
                                 format_func=lambda y: f"letzte {y} Jahre" if y > 1 else "letztes Jahr")
        dip_all = analytics.dip_factors(prices, analytics.DIP_BOOST)
        dip_start = prices.index[-1] - pd.DateOffset(years=dip_years)
        st.plotly_chart(charts.dip_price_chart(prices[dip_pick], dip_all[dip_pick], display_name(dip_pick),
                                               dip_start), width="stretch")
        f_now = float(dip_all[dip_pick].iloc[-1])
        st.caption(f"Heute: Faktor {num(f_now, 2)} – {display_name(dip_pick)} bekommt "
                   + ("die normale Sparrate." if f_now <= 1.0 else
                      f"{num(f_now, 2)}-mal so viel Sparrate wie im Plan (die anderen entsprechend weniger).")
                   + " Hellgelb = unter der 50-Tage-Linie, orange = zusätzlich unter der 100-Tage-Linie, "
                     "rot = unter allen drei Linien (doppelt).")

# =========================================================================== Beweis
with tab_proof:
    st.caption("**Einfach halten** = die Sparrate fließt weiter im heutigen Verhältnis, Klumpen und "
               "Risiko bleiben. **Sparplan-Kompass** = deine Einstellungen; er rechnet alle 6 Monate "
               "neu, nutzt nur Kurse, die es damals schon gab, und verkauft nie. Jeder Titel zählt ab "
               "seinem echten Börsenstart.")
    if sim is None:
        st.info("Für diesen Zeitraum reicht die gemeinsame Kurshistorie nicht aus.")
    else:
        theme.section("Die Antwort", headline)

        def card(label, k, hval, fmt, lower_better, accent):
            better = (abs(k) < abs(hval)) if lower_better else (k >= hval * 0.97)
            return {"label": label, "value": fmt(k), "sub": f"einfach halten: {fmt(hval)}",
                    "tone": "good" if better else "", "accent": accent}

        g1, g2 = st.columns([3, 2], gap="large")
        with g1:
            theme.panel_title("Risiko")
            theme.kpi_row([
                card("Größter Absturz", K["Größter Einbruch %"], H["Größter Einbruch %"], pct, True, theme.GREEN),
                card("Verlust am Schrecktag", K["Schrecktag %"], H["Schrecktag %"], pct, True, theme.GREEN),
                card("Schwankung pro Jahr", K["Schwankung p.a. %"], H["Schwankung p.a. %"], pct, True,
                     theme.GREEN),
            ])
        with g2:
            theme.panel_title("Geld")
            theme.kpi_row([
                card("Endwert", K["Endwert €"], H["Endwert €"], eur, False, theme.BLUE),
                card("Rendite pro Jahr", K["Rendite p.a. %"], H["Rendite p.a. %"], pct, False, theme.BLUE),
            ])
        f1, f2 = st.columns([1, 1], gap="large")
        with f1:
            fc_years = st.segmented_control(
                "Prognose in die Zukunft", [0, 10, 20, 30], default=10, key="forecast_years",
                format_func=lambda y: "aus" if y == 0 else f"{y} Jahre",
                help="Beide Depots wachsen ab heute mit derselben Sparrate weiter. Der Kompass hält "
                     "dabei sein neues Ziel-Depot, einfach halten bleibt beim alten Verhältnis.") or 0
        with f2:
            fc_mode = st.segmented_control(
                "Renditeannahme", ["Vorsichtig (7 % Basis)", "Wie im Rückblick"],
                default="Vorsichtig (7 % Basis)", key="forecast_mode",
                help="Vorsichtig: einfach halten wächst mit 7 % pro Jahr (langfristiger Schnitt des "
                     "Weltaktienmarkts), der Kompass mit 7 % plus seinem Vorsprung aus dem Rückblick. "
                     "Wie im Rückblick: beide mit ihrer tatsächlichen Durchschnittsrendite im "
                     "gewählten Zeitraum.") or "Vorsichtig (7 % Basis)"
        forecast = None
        if fc_years:
            r_k, r_h = K["Rendite p.a. %"] / 100, H["Rendite p.a. %"] / 100
            if fc_mode.startswith("Vorsichtig"):
                r_k, r_h = 0.07 + (r_k - r_h), 0.07
            rates = {KOMPASS: r_k, HOLD: r_h}
            forecast = analytics.project(values, rates, budget, int(fc_years))
        st.plotly_chart(charts.money_and_calm(
            values, rets, "",
            f"{money_word}: Depotwert in € mit Kompass und beim einfachen Halten",
            f"{risk_word}: so tief lag das Depot jeweils unter seinem Höchststand (in %)",
            analytics.CRISES, forecast), width="stretch")
        st.caption("Oben: gleiche Einzahlungen, Depotwert über die Zeit. Unten: Die blaue Fläche "
                   "(Kompass) ist kleiner als die graue (einfach halten) – das Depot fiel in Krisen "
                   "weniger tief. Rot hinterlegt: Krisenphasen.")
        if forecast is not None:
            fk, fh = forecast[KOMPASS].iloc[-1], forecast[HOLD].iloc[-1]
            gap = fk - fh
            st.caption(
                f"**Prognose (gestrichelt):** Beide Depots wachsen ab heute mit gleichbleibender "
                f"Durchschnittsrendite weiter – Kompass {pct(rates[KOMPASS] * 100, 2)} pro Jahr, einfach "
                f"halten {pct(rates[HOLD] * 100, 2)} pro Jahr – und bekommen weiter {eur(budget)} im Monat. "
                f"In {fc_years} Jahren: **{eur(fk)} statt {eur(fh)}**, "
                + (f"also **{eur(gap)} mehr** durch den Zinseszins." if gap > 0 else
                   f"also {eur(-gap)} weniger – der Preis für das ruhigere Depot." if gap < 0 else
                   "also gleich viel.")
                + " Die Prognose ist eine Rechnung mit Durchschnittswerten, keine Garantie: echte "
                  "Kurse schwanken, und beim Kompass fallen die Rückgänge auf dem Weg kleiner aus.")
        if use_dip and equity.any():
            theme.section("Nachkaufen im Rückblick", "Wann der Kompass mehr gekauft hat",
                          "Jeder Kasten ist ein Monat. Je röter, desto mehr Sparrate ging an diesem "
                          "Monatsanfang in den Aktien-ETF, weil er unter seinen Tageslinien lag.")
            month_days = values.index.to_series().groupby(values.index.to_period("M")).first()
            dip_hist = analytics.dip_factors(prices, analytics.DIP_BOOST).loc[month_days.values,
                                                                              [t for t in tickers if is_equity(t)]]
            dip_hist.columns = [display_name(t) for t in dip_hist.columns]
            st.plotly_chart(charts.dip_timeline(dip_hist), width="stretch")
            n_months = int((dip_hist > 1.0).any(axis=1).sum())
            st.caption(f"In {n_months} von {len(dip_hist)} Monaten lag mindestens ein Aktien-ETF unter "
                       "einer Tageslinie. Der Aufschlag gilt nur für ETFs, die laut Plan gerade Sparrate "
                       "bekommen; die Summe bleibt immer die normale Sparrate.")

        theme.section("Risiko-Bilanz", f"{risk_wins} von 5 Risikomaßen kleiner als beim einfachen Halten",
                      "Grau = einfach halten, Blau = Kompass. Je kürzer der blaue Balken, desto weniger Risiko.")
        st.plotly_chart(charts.calm_bars(stats), width="stretch")
        if len(crises):
            theme.section("Krisen", f"In {crisis_wins} von {len(crises)} Krisen weniger Verlust")
            st.plotly_chart(charts.crisis_compare_bar(crises, ""), width="stretch")
        with st.expander("Alle Kennzahlen zum Nachprüfen"):
            st.dataframe(stats, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="%.0f" if "€" in c or "Monate" in c else "%.2f")
                for c in stats.columns})
        with st.expander("Strategien vergleichen: ausgewogen gegen extra vorsichtig"):
            st.caption("Gleiche Regeln, gleiche Kurse, gleicher Zeitraum. Nur die Strategie ändert sich.")
            if st.button("Vergleich starten (dauert etwas)", type="primary"):
                st.session_state["compare_strategies"] = True
            if st.session_state.get("compare_strategies"):
                rows_c = []
                for sname, smix in STRATEGIES.items():
                    skey = tuple(sorted((k, float(v)) for k, v in smix.items()))
                    out = cached_walk_forward(prices, bench_prices.iloc[:, 0], tuple(h), budget, sim_years,
                                              rf, limit, w_max, months, skey, tuple(core), first_sim, dip,
                                              tuple(equity))
                    rows_c.append(analytics.compare_stats(out[0], out[1], rf).loc[KOMPASS].rename(sname))
                comp = pd.DataFrame(rows_c)
                comp.loc[HOLD] = H
                st.plotly_chart(charts.strategy_scatter(comp), width="stretch")
                st.caption("Weiter rechts = kleinerer Absturz, weiter oben = mehr Rendite.")
        limiting = max(tickers, key=lambda t: etf_start[t])
        with st.expander(f"Datenquellen: echte Kurse ab {prices.index[0]:%m/%Y}"):
            st.dataframe(pd.DataFrame({
                "Titel": [display_name(t) for t in tickers],
                "Börsenstart": [f"{etf_start[t]:%m/%Y}" for t in tickers],
            }), hide_index=True, width="stretch")
            st.caption(f"Jeder Titel zählt erst ab seinem Börsenstart; die gemeinsame Historie beginnt "
                       f"mit dem jüngsten ({display_name(limiting)}). Der Vergleich beginnt zwei Jahre "
                       "später, damit der Kompass vorher lernen kann, und frühestens mit dem "
                       "MSCI-World-ETF (09/2009). Offensichtliche Datenfehler von Yahoo Finance werden "
                       "herausgefiltert.")
