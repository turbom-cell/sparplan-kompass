"""Sparplan-Kompass: Depot-Risiko senken, ohne zu verkaufen.

Start:  streamlit run app.py
Diese Datei enthält nur die Oberfläche. Gerechnet wird in `core/`, gezeichnet in `ui/`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import analytics, tax
from core.data_service import (CATALOG, GROUPS, PROXIES, default_teilfreistellung, display_name,
                               group_of, load_prices, to_returns)
from core.optimizer import cvar, diluted_weights, earliest_month, mix_weights, portfolio_stats
from core.rebalancer import build_plan, sell_rebalance
from ui import charts, theme

st.set_page_config(page_title="Sparplan-Kompass", page_icon="🧭", layout="wide",
                   initial_sidebar_state="collapsed")
theme.inject()

BENCHMARKS = {"MSCI World": "EUNL.DE", "S&P 500": "SXR8.DE", "FTSE All-World": "VWCE.DE",
              "DAX": "EXS1.DE", "STOXX Europe 600": "EXSA.DE"}
STRATEGIES: dict[str, dict[str, float] | None] = {
    "Sharpe-Fokus": {"Sharpe": 1.0},
    "Rendite-Fokus": {"Rendite": 1.0},
    "Ausgewogen 50/35/15": {"Sharpe": 0.50, "CVaR": 0.35, "Varianz": 0.15},
    "Eigene Mischung": None,
}
STRATEGY_HELP = {
    "Sharpe-Fokus": "Empfohlen: das beste Verhältnis aus Rendite und Schwankung, solange die "
                    "Schmerzgrenze hält. Im Backtest die höchste Sharpe Ratio.",
    "Rendite-Fokus": "Höchste erwartete Rendite, solange die Schmerzgrenze hält.",
    "Ausgewogen 50/35/15": "50 % bestes Rendite-Risiko-Verhältnis (Sharpe), 35 % kleinster "
                           "Crash-Verlust (CVaR), 15 % kleinste Schwankung (Varianz).",
    "Eigene Mischung": "Die vier Bausteine frei gewichten.",
}
DIVERSIFIERS = ["4GLD.DE", "EUNH.DE"]   # im Backtest die wirksamsten Risikopuffer
# Vorlagen: (Ticker, Bestand €, Einstand €, Kernposition)
PRESETS: dict[str, tuple[list[tuple[str, int, int, bool]], list[str]]] = {
    "Tech-lastig (Klumpenrisiko)": (
        [("SXRV.DE", 6000, 3600, True), ("SXR8.DE", 3000, 2200, False),
         ("EUNL.DE", 2000, 1600, False), ("IS3S.DE", 500, 450, False),
         ("4GLD.DE", 500, 400, False)], DIVERSIFIERS),
    "Welt-Depot mit viel DAX": (
        [("EUNL.DE", 5000, 3800, True), ("EXS1.DE", 4500, 3300, False),
         ("IS3N.DE", 1500, 1400, False)], DIVERSIFIERS),
    "Langjähriger Sparer (60.000 €)": (
        [("SXRV.DE", 26000, 9000, True), ("SXR8.DE", 18000, 9500, False),
         ("EUNL.DE", 12000, 8000, False), ("XDWT.DE", 4000, 2000, False)], DIVERSIFIERS),
    "Berufseinsteiger (ein ETF)": ([("SXR8.DE", 2500, 2100, True)], DIVERSIFIERS),
    "Leeres Depot zum Selberbauen": ([("EUNL.DE", 1000, 1000, True)], []),
}
LABEL = {t: f"{name} · {t}" for t, (name, _, _) in CATALOG.items()}


def label_of(ticker: str) -> str:
    return LABEL.get(ticker, f"Eigener Titel · {ticker}")


def ticker_of(label: str) -> str:
    return label.split(" · ")[-1].strip().upper()


# --------------------------------------------------------------------------- Formatierung
def eur(x: float) -> str:
    return f"{x:,.0f} €".replace(",", ".")


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f} %".replace(".", ",")


def num(x: float, digits: int = 1, sign: bool = False) -> str:
    return f"{x:{'+' if sign else ''}.{digits}f}".replace(".", ",")


# --------------------------------------------------------------------------- Rechnen (gecacht)
@st.cache_data(ttl=3600, show_spinner="Lade Kurse …")
def cached_prices(tickers: tuple[str, ...], start_year: int, allow_live: bool):
    data = load_prices(list(tickers), start_year, allow_live)
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
                        horizon: int, mix: tuple, core: tuple[bool, ...]):
    return analytics.walk_forward(prices, bench, np.array(holdings), budget, years, rf, limit,
                                  w_max, horizon, mix=dict(mix), core=np.array(core))


hero_slot = st.container()
tab_home, tab_depot, tab_proof = st.tabs(["Überblick", "Dein Depot", "Beweis"])

# =========================================================================== Eingaben
with tab_depot:
    theme.section("Schritt 1", "Dein Depot",
                  "Trage ein, was du heute besitzt. Rechts wählst du, in welche Bausteine der "
                  "Kompass zusätzlich investieren darf.")
    left, right = st.columns([3, 2], gap="large")
    with right:
        own_raw = st.text_input("Eigene Titel hinzufügen (Yahoo-Kürzel, mit Komma getrennt)",
                                placeholder="z. B. AAPL, SAP.DE, IWDA.AS",
                                help="Jeder Titel von Yahoo Finance: ETFs, Einzelaktien, Fonds. "
                                     "US-Werte ohne Endung werden in Euro umgerechnet. Ohne "
                                     "Ersatz-Reihe beginnt die Historie erst mit dem Titel selbst.")
    own = [x.strip().upper() for x in own_raw.split(",") if x.strip()]
    options = list(LABEL.values()) + [label_of(t) for t in own if t not in LABEL]
    with left:
        preset = st.selectbox("Vorlage", list(PRESETS))
        rows, preset_extras = PRESETS[preset]
        depot = st.data_editor(
            pd.DataFrame({"Position": [label_of(t) for t, _, _, _ in rows],
                          "Bestand €": [float(b) for _, b, _, _ in rows],
                          "Einstand €": [float(e) for _, _, e, _ in rows],
                          "Kern": [bool(c) for _, _, _, c in rows]}),
            key=f"depot_{preset}_{len(options)}", num_rows="dynamic", hide_index=True, width="stretch",
            column_config={
                "Position": st.column_config.SelectboxColumn(options=options, required=True,
                                                             width="large"),
                "Kern": st.column_config.CheckboxColumn(
                    help="Kernposition: wird nicht verwässert, behält mindestens ihren heutigen "
                         "Anteil und bekommt weiter ihren Teil der Sparrate."),
                "Bestand €": st.column_config.NumberColumn(min_value=0, step=100, format="%.0f"),
                "Einstand €": st.column_config.NumberColumn(
                    min_value=0, step=100, format="%.0f",
                    help="Was du insgesamt für die Position bezahlt hast (für die Steuerrechnung)")})
        budget = float(st.slider("Monatliche Sparrate in €", 50, 3000, 500, 50))
    depot = depot.dropna(subset=["Position"]).drop_duplicates("Position")
    depot["Kern"] = depot["Kern"].fillna(False).astype(bool)
    depot[["Bestand €", "Einstand €"]] = depot[["Bestand €", "Einstand €"]].fillna(0.0)
    held = [ticker_of(p) for p in depot["Position"]]
    with right:
        extras_labels = st.multiselect(
            "Bausteine, die der Kompass zukaufen darf",
            [label_of(t) for t in own if t not in held] +
            [LABEL[t] for g in GROUPS for t in CATALOG if group_of(t) == g and t not in held],
            default=[LABEL[t] for t in preset_extras if t not in held],
            key=f"extras_{preset}_{len(own)}",
            help="Gold und Euro-Staatsanleihen waren im Backtest die wirksamsten Risikopuffer. "
                 "Mehr Bausteine heißt nicht automatisch besser.")
        with st.expander("Erweiterte Annahmen"):
            start_year = st.slider("Kurse ab Jahr", 2000, 2021, 2000)
            rf = st.slider("Risikofreier Zins in %", 0.0, 5.0, 2.5, 0.25) / 100
            allowance = float(st.number_input("Freier Sparerpauschbetrag in €", 0, 2000, 1000, 100))
            fee = float(st.number_input("Ordergebühr je Verkauf in €", 0.0, 20.0, 1.0, 0.5))
            allow_live = st.toggle("Live-Kurse von Yahoo Finance", value=True)

if depot.empty or depot["Bestand €"].sum() <= 0:
    with hero_slot:
        theme.hero("Sparplan-Kompass", "Trage unter „Dein Depot“ mindestens eine Position mit "
                   "Bestand ein.", [], eyebrow="ETF-Sparplan · Risiko · Steuern")
    st.stop()

universe = list(dict.fromkeys(held + [ticker_of(x) for x in extras_labels]))
prices, source, missing, etf_start = cached_prices(tuple(universe), start_year, allow_live)
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
                  "„nichts verkaufen“. Innerhalb dieser Regeln wählst du die Strategie.")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        default_limit = float(np.clip(np.ceil(loss_now_pct * 10) / 10, 0.5, 6.0))
        limit_pct = st.slider("Schmerzgrenze: Verlust an einem Schrecktag höchstens (% des Depots)",
                              0.5, 6.0, default_limit, 0.1, format="%.1f %%",
                              key=f"limit_{preset}_{len(tickers)}",
                              help="Schrecktag = Durchschnitt der 5 % schlechtesten Börsentage "
                                   "(95 %-CVaR).")
        st.caption(f"Heute: {pct(loss_now_pct, 2)} · bei deinem Depot entspricht die Grenze "
                   f"{eur(limit_pct / 100 * V)}")
    with c2:
        w_max = st.slider("Max. Positionsgröße (% des Depots)", 10, 100, 60, 5, format="%d %%") / 100
        st.caption("Keine Position darf im Ziel größer werden (Kernpositionen behalten mindestens "
                   "ihren heutigen Anteil).")
    with c3:
        months = st.slider("Umbau-Zeitraum in Monaten", 6, 120, 36, 6,
                           help="Ohne Verkauf geht der Umbau nur so schnell, wie Sparraten "
                                "nachfließen. Länger = mehr Spielraum, kürzer = schneller am Ziel.")
        limit = limit_pct / 100
        first = cached_earliest(returns, tuple(h), budget, limit, w_max)
        st.caption("Schmerzgrenze ist schon heute erreichbar." if first == 0 else
                   f"Ohne Verkauf frühestens nach {first} Monaten erreichbar." if first else
                   "Ohne Verkauf in 10 Jahren nicht erreichbar.")
    s1, s2 = st.columns([1, 1], gap="large")
    with s1:
        strategy = st.segmented_control("Strategie", list(STRATEGIES), default="Sharpe-Fokus")
        strategy = strategy or "Sharpe-Fokus"
        st.caption(STRATEGY_HELP[strategy])
    with s2:
        if STRATEGIES[strategy] is None:
            m1, m2, m3, m4 = st.columns(4)
            mix = {"Rendite": m1.slider("Rendite", 0, 100, 25, 5),
                   "Sharpe": m2.slider("Sharpe", 0, 100, 25, 5),
                   "CVaR": m3.slider("CVaR", 0, 100, 25, 5),
                   "Varianz": m4.slider("Varianz", 0, 100, 25, 5)}
            if sum(mix.values()) == 0:
                mix = {"Rendite": 100}
        else:
            mix = STRATEGIES[strategy]
    mix_key = tuple(sorted((k, float(v)) for k, v in mix.items()))

# =========================================================================== Ergebnis berechnen
lo = diluted_weights(h, budget, months, core)
w_goal, status = cached_target(returns, rf, limit, w_max, tuple(np.round(lo, 6)), mix_key)
w_free, _ = cached_target(returns, rf, limit, w_max, None, mix_key)
goal = portfolio_stats(w_goal, R, rf)
free = portfolio_stats(w_free, R, rf)
plan = build_plan(h, w_goal, budget, months, names)
cost = tax.sell_cost(sell_rebalance(h, w_free), h, np.where(h > 0, basis, 0.0), tf, allowance, fee)
top_now, top_goal = int(np.argmax(w_now)), int(np.argmax(w_goal))

with tab_proof:
    theme.section("Beweis", "Mit echten Kursen nachgerechnet",
                  "Drei Anleger starten mit deinem Depot und zahlen dieselbe Sparrate ein. Der "
                  "Kompass rechnet alle 6 Monate neu, nutzt nur Kurse, die es damals schon gab, "
                  "und verkauft nie.")
    p1, p2 = st.columns([3, 2])
    with p1:
        avail = (prices.index[-1] - prices.index[0]).days / 365.25
        options = [y for y in (3, 5, 10, 20) if y <= avail - 2] or [3]
        sim_years = st.radio("Zeitraum", options, index=len(options) - 1, horizontal=True,
                             format_func=lambda y: f"letzte {y} Jahre")
    with p2:
        bench_name = st.selectbox("Markt-Vergleich (alles in einen ETF)", list(BENCHMARKS))
bench_prices, _, bench_missing, _ = cached_prices((BENCHMARKS[bench_name],), start_year, allow_live)
sim = None if bench_missing or bench_prices.empty else cached_walk_forward(
    prices, bench_prices.iloc[:, 0], tuple(h), budget, sim_years, rf, limit, w_max, months, mix_key,
    tuple(core))
if sim is not None:
    values, rets, _ = sim
    stats = analytics.compare_stats(values, rets, rf)
    crises = analytics.crisis_from_rets(rets)
    K, O, M = stats.loc["Sparplan-Kompass"], stats.loc["Ohne Kompass"], stats.loc["Markt"]
    crisis_wins = int((crises["Sparplan-Kompass"] > crises["Ohne Kompass"]).sum()) if len(crises) else 0

# =========================================================================== Kopfbereich
with hero_slot:
    proofs = []
    if sim is not None:
        proofs.append((f"{num(K['Sharpe'], 2)} statt {num(O['Sharpe'], 2)}",
                       f"Sharpe Ratio mit statt ohne Kompass, letzte {sim_years} Jahre "
                       f"({bench_name}: {num(M['Sharpe'], 2)})"))
        proofs.append((f"{num(K['Größter Einbruch %'], 0)} % statt {num(O['Größter Einbruch %'], 0)} %",
                       "größter Einbruch mit statt ohne Kompass"))
    proofs.append((eur(cost.total), "Steuern und Gebühren, die du dir gegenüber Umschichten "
                                    "durch Verkauf sparst"))
    theme.hero(
        "Mehr Rendite pro Risiko.<br>Keine Steuer beim Umbau.",
        "Der <b>Sparplan-Kompass</b> baut dein ETF-Depot nur über die Aufteilung deiner Sparrate "
        "um. Du verkaufst nichts, zahlst keine Steuer auf Kursgewinne und legst selbst fest, wie "
        "viel ein schlechter Börsentag kosten darf.",
        [("#34d399" if source == "live" else "#fbbf24",
          "Live-Kurse" if source == "live" else "Gespeicherter Kurs-Stand"),
         ("", f"Kurse {prices.index[0]:%m/%Y} – {prices.index[-1]:%m/%Y}"),
         ("", f"Depot {eur(V)}"), ("", f"Sparrate {eur(budget)} / Monat"), ("", strategy)],
        proofs, eyebrow="Sparplan-Kompass · für ETF-Sparer")

# =========================================================================== Überblick
with tab_home:
    theme.section("Das Problem", "Drei Dinge, die fast jeder ETF-Sparer falsch macht, ohne es zu merken")
    theme.cards([
        {"icon": "📈", "title": "Klumpen wachsen von selbst",
         "text": f"Was gut läuft, wird im Depot immer größer. In deinem Depot sind es heute "
                 f"<b>{pct(w_now[top_now] * 100, 0)} {theme.esc(names[top_now])}</b>."},
        {"icon": "❓", "title": "Risiko in Prozent sagt niemandem etwas",
         "text": f"„Volatilität {pct(now.vol * 100, 0)}“ ist abstrakt. Konkret heißt es: An einem "
                 f"schlechten Börsentag verlierst du im Schnitt <b>{eur(now.cvar * V)}</b>."},
        {"icon": "💸", "title": "Umschichten kostet Geld",
         "text": f"Wer verkauft, zahlt 26,375 % Steuer auf den Gewinn plus Gebühren. Bei deinem "
                 f"Depot wären das <b>{eur(cost.total)}</b>."},
    ])
    theme.section("Der Unterschied", "Drei Wege mit deinem Depot – mit deinen Zahlen")
    dd_k = f"{num(K['Größter Einbruch %'])} %" if sim is not None else "–"
    dd_o = f"{num(O['Größter Einbruch %'])} %" if sim is not None else "–"
    theme.compare_table(
        ["Weiter sparen wie bisher", "Umschichten durch Verkauf", "Sparplan-Kompass"],
        [["Klumpenrisiko", '<span class="no">bleibt</span>', '<span class="yes">sofort weg</span>',
          f'<span class="yes">schrittweise weg</span> ({months} Monate)'],
         ["Steuer und Gebühren beim Umbau", '<span class="yes">0 €</span>',
          f'<span class="no">{eur(cost.total)}</span>', '<span class="yes">0 €</span>'],
         ["Verlust an einem Schrecktag", eur(now.cvar * V), f"{eur(free.cvar * V)} (sofort)",
          f"{eur(goal.cvar * V)} (nach {months} Monaten)"],
         [f"Sharpe Ratio, letzte {sim_years} Jahre",
          num(O["Sharpe"], 2) if sim is not None else "–", "nicht nachgerechnet",
          f'<b>{num(K["Sharpe"], 2)}</b>' if sim is not None else "–"],
         [f"Größter Einbruch, letzte {sim_years} Jahre", dd_o, "nicht nachgerechnet", dd_k],
         ["Was du tun musst", "nichts", "Verkaufs- und Kauforders", "Sparplan einmal neu einstellen"]],
        highlight=2)
    theme.section("Die Belege", "Was wir behaupten – und wo man es nachprüfen kann")
    proof_cards = []
    if sim is not None:
        proof_cards += [
            {"num": f"{num(K['Sharpe'], 2)} statt {num(O['Sharpe'], 2)}",
             "title": "Mehr Rendite pro Risiko",
             "text": f"Sharpe Ratio mit und ohne Kompass, {bench_name}: {num(M['Sharpe'], 2)}. "
                     f"Rendite p.a. {num(K['Rendite p.a. %'])} % gegen {num(O['Rendite p.a. %'])} %.",
             "src": f"Tageskurse, letzte {sim_years} Jahre, risikofreier Zins {pct(rf * 100, 2)}"},
            {"num": f"{num(K['Größter Einbruch %'] - O['Größter Einbruch %'], 1, True)} Punkte",
             "title": "Kleinerer Absturz",
             "text": f"Größter Einbruch {num(K['Größter Einbruch %'])} % statt "
                     f"{num(O['Größter Einbruch %'])} % ohne Kompass und "
                     f"{num(M['Größter Einbruch %'])} % im {bench_name}.",
             "src": f"Tageskurse, letzte {sim_years} Jahre, Tab „Beweis“"},
            {"num": f"{num(K['Längste Durststrecke Monate'], 0)} statt {num(O['Längste Durststrecke Monate'], 0)} Monate",
             "title": "Kürzere Durststrecke",
             "text": "So lange lag das Depot höchstens unter seinem alten Höchststand. Kürzer "
                     "heißt: weniger Grund, in der Krise aufzugeben.",
             "src": "berechnet ohne Effekt der Einzahlungen"},
        ]
    proof_cards.append({"num": eur(cost.total), "title": "Keine Steuer beim Umbau",
                        "text": f"Sofort umschichten würde {eur(cost.sold)} Verkäufe auslösen. "
                                "Der Kompass erreicht das Ziel nur über Zukäufe.",
                        "src": "Abgeltungsteuer, Teilfreistellung, Pauschbetrag"})
    theme.cards(proof_cards)
    theme.section("So funktioniert es", "In drei Schritten zum ruhigeren Depot")
    theme.cards([
        {"icon": "1", "title": "Depot eintragen", "text": "Bestand je ETF, Sparrate und die "
                                                            "Bausteine, die dazukommen dürfen."},
        {"icon": "2", "title": "Regeln festlegen", "text": "Schmerzgrenze in Prozent, maximale "
                                                            "Positionsgröße, Umbau-Zeitraum, Strategie."},
        {"icon": "3", "title": "Sparplan einstellen", "text": "Die App sagt, wie viel Euro jeden "
                                                               "Monat in welchen ETF gehen. Verkauft wird nichts."},
    ])
    st.caption("Studienprojekt, keine Anlageberatung. Vergangene Kurse sind keine Garantie für die "
               "Zukunft. Alle Zahlen dieser Seite werden live aus deinen Eingaben berechnet.")

# =========================================================================== Dein Depot: Ergebnis
with tab_depot:
    theme.section("Ergebnis", f"Dein Sparplan für die nächsten {months} Monate",
                  "Verkaufe nichts. Stelle deinen Sparplan einmal so ein:")
    if status == "unreachable":
        st.warning(f"Die Schmerzgrenze von {pct(limit_pct)} ist in {months} Monaten ohne Verkauf nicht "
                   "erreichbar. Gezeigt wird das sicherste Depot, das bis dahin möglich ist. Tipp: "
                   "Umbau-Zeitraum verlängern, Sparrate erhöhen oder mehr Bausteine zulassen.")
    theme.kpi_row([
        {"label": "Schrecktag heute", "value": eur(-now.cvar * V), "sub": pct(-now.cvar * 100, 2),
         "accent": theme.SLATE},
        {"label": f"Schrecktag nach {months} Monaten", "value": eur(-goal.cvar * V),
         "sub": f"{pct(-goal.cvar * 100, 2)} · Grenze {pct(limit_pct)}",
         "tone": "good" if goal.cvar <= limit * 1.003 else "bad", "accent": theme.BLUE},
        {"label": "Größte Position", "value": f"{pct(w_now[top_now] * 100, 0)} → {pct(w_goal[top_goal] * 100, 0)}",
         "sub": f"{theme.esc(names[top_now])} → {theme.esc(names[top_goal])}", "accent": theme.AMBER},
        {"label": "Steuer gespart", "value": eur(cost.total), "sub": "gegenüber Umschichten",
         "tone": "good", "accent": theme.GREEN},
    ])
    table = pd.DataFrame({
        "Position": names, "Ticker": tickers,
        "Sparrate €/Monat": plan.monthly.round(0).to_numpy(),
        "Anteil an der Sparrate %": (plan.monthly / budget * 100).round(1).to_numpy(),
        "Gewicht heute %": (w_now * 100).round(1),
        "Gewicht im Ziel %": (w_goal * 100).round(1)})
    table = table[(table["Sparrate €/Monat"] > 0) | (table["Gewicht heute %"] > 0)]
    st.dataframe(table.sort_values("Sparrate €/Monat", ascending=False), hide_index=True,
                 width="stretch", column_config={
                     "Anteil an der Sparrate %": st.column_config.ProgressColumn(
                         format="%.1f %%", min_value=0, max_value=100)})
    paused = [n for n, m, w in zip(names, plan.monthly, w_now) if m < 0.5 and w > 0]
    if paused:
        st.caption("⏸️ Sparplan pausieren für: " + ", ".join(paused) +
                   ". Diese Positionen werden durch die Zukäufe der anderen verwässert.")
    a, b = st.columns([3, 2], gap="large")
    with a:
        st.plotly_chart(charts.plan_animation(plan.weights.loc[:, (plan.weights > 0.001).any()],
                                              w_goal[(plan.weights > 0.001).any().to_numpy()]),
                        width="stretch")
        st.caption("▶ drücken: Monat für Monat zur Zielmarke (schwarzer Strich).")
    with b:
        path = [cvar(w, R) * V for w in plan.weights.to_numpy()]
        st.plotly_chart(charts.risk_path(list(plan.weights.index), path, limit * V), width="stretch")
        st.caption("Verlust an einem Schrecktag in €, Monat für Monat.")

# =========================================================================== Beweis
with tab_proof:
    st.caption("**Kompass** = deine Einstellungen aus „Dein Depot“. **Ohne Kompass** = du sparst weiter wie bisher: Die Sparrate fließt im selben "
               "Verhältnis wie dein heutiges Depot. Klumpen und Risiko bleiben. "
               f"**Markt** = alles in einen {bench_name}-ETF.")
    if sim is None:
        st.info("Für diesen Zeitraum reicht die gemeinsame Kurshistorie nicht aus.")
    else:
        def vs(mine: float, other: float, unit: str, lower_better: bool, digits: int = 1):
            diff = mine - other
            good = diff < 0 if lower_better else diff > 0
            word = "weniger" if diff < 0 else "mehr"
            text = (f"{eur(abs(diff))} {word}" if unit == "€" else f"{num(abs(diff), digits)} {unit} {word}")
            return {"sub": text + " als ohne Kompass", "tone": "good" if good else "bad"}

        theme.kpi_row([
            {"label": "Größter Einbruch", "value": f"{num(K['Größter Einbruch %'])} %",
             **vs(-K["Größter Einbruch %"], -O["Größter Einbruch %"], "Punkte", True)},
            {"label": "Schlechtestes Jahr", "value": f"{num(K['Schlechtestes Jahr %'])} %",
             **vs(-K["Schlechtestes Jahr %"], -O["Schlechtestes Jahr %"], "Punkte", True)},
            {"label": "Längste Durststrecke", "value": f"{num(K['Längste Durststrecke Monate'], 0)} Mon.",
             **vs(K["Längste Durststrecke Monate"], O["Längste Durststrecke Monate"], "Monate", True, 0)},
            {"label": "Sharpe Ratio", "value": num(K["Sharpe"], 2),
             **vs(K["Sharpe"], O["Sharpe"], "", False, 2)},
            {"label": "Endwert", "value": eur(K["Endwert €"]),
             **vs(K["Endwert €"], O["Endwert €"], "€", False)},
        ])
        richer = K["Endwert €"] >= O["Endwert €"]
        theme.callout(
            f"Mit Kompass fiel das Depot höchstens um <b>{num(K['Größter Einbruch %'])} %</b>, ohne Kompass "
            f"um {num(O['Größter Einbruch %'])} %, im {bench_name} um {num(M['Größter Einbruch %'])} %. "
            + (f"Und am Ende stehen sogar <b>{eur(K['Endwert €'] - O['Endwert €'])} mehr</b> auf dem Depot."
               if richer else
               f"Der Preis: <b>{eur(O['Endwert €'] - K['Endwert €'])} weniger Endwert</b> als ohne Kompass. "
               "Wer den größeren Absturz durchgehalten hätte, hätte mehr verdient."))
        st.plotly_chart(charts.compare_lines(values.rename(columns={}), f"Markt ({bench_name})",
                                             analytics.CRISES), width="stretch")
        st.plotly_chart(charts.underwater_lines(analytics.underwater(rets), f"Markt ({bench_name})",
                                                analytics.CRISES), width="stretch")
        if len(crises):
            theme.section("Krisen", f"In {crisis_wins} von {len(crises)} Krisen weniger Verlust als ohne Kompass")
            st.plotly_chart(charts.crisis_compare_bar(crises, f"Markt ({bench_name})"), width="stretch")
        theme.section("Alle Kennzahlen", "Zum Nachprüfen")
        st.dataframe(stats.rename(index={"Markt": f"Markt ({bench_name})"}), width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.0f" if "€" in c or "Monate" in c
                                                                     else "%.2f")
                                    for c in stats.columns})

        theme.section("Welche Strategie?", "Rendite-Fokus, 50/35/15 oder eigene Mischung im Vergleich",
                      "Gleiche Regeln, gleiche Kurse, gleicher Zeitraum. Nur die Strategie ändert sich.")
        if st.button("Strategien vergleichen (dauert etwas)", type="primary"):
            st.session_state["compare_strategies"] = True
        if st.session_state.get("compare_strategies"):
            rows = []
            for sname, smix in STRATEGIES.items():
                smix = mix if smix is None else smix
                skey = tuple(sorted((k, float(v)) for k, v in smix.items()))
                out = cached_walk_forward(prices, bench_prices.iloc[:, 0], tuple(h), budget, sim_years,
                                          rf, limit, w_max, months, skey, tuple(core))
                s = analytics.compare_stats(out[0], out[1], rf).loc["Sparplan-Kompass"]
                rows.append(s.rename(sname))
            comp = pd.DataFrame(rows)
            comp.loc["Ohne Kompass"] = O
            comp.loc[f"Markt ({bench_name})"] = M
            st.plotly_chart(charts.strategy_scatter(comp), width="stretch")
            st.dataframe(comp[["Endwert €", "Rendite p.a. %", "Größter Einbruch %", "Schlechtestes Jahr %",
                               "Schrecktag %", "Sharpe"]], width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.0f" if "€" in c else "%.2f")
                                        for c in comp.columns})
            st.caption("Oben rechts ist besser: mehr Rendite bei kleinerem Einbruch.")

    proxied = [t for t in tickers if etf_start[t] > prices.index[0] + pd.Timedelta(days=10)]
    if proxied:
        with st.expander(f"Datenquellen: Kurse ab {prices.index[0]:%m/%Y}, bei {len(proxied)} "
                         "Bausteinen vor ETF-Auflage mit Ersatz-Reihe"):
            st.dataframe(pd.DataFrame({
                "Baustein": [display_name(t) for t in proxied],
                "Echter ETF-Kurs ab": [f"{etf_start[t]:%m/%Y}" for t in proxied],
                "Davor verwendet": [PROXIES[t][2] if t in PROXIES else "–" for t in proxied],
            }), hide_index=True, width="stretch")
            st.caption("Die ETFs gibt es erst seit 2008 bis 2019. Davor steht eine Ersatz-Reihe "
                       "derselben Anlageklasse in Euro. Offensichtliche Datenfehler von Yahoo "
                       "Finance werden herausgefiltert.")
