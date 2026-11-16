"""Sparplan-Kompass: Depot-Risiko senken, ohne zu verkaufen.

Start:  streamlit run app.py
Diese Datei enthält nur die Oberfläche. Gerechnet wird in `core/`, gezeichnet in `ui/`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import analytics, tax
from core.data_service import (CATALOG, default_teilfreistellung, display_name, load_prices,
                               to_returns)
from core.optimizer import (cvar, diluted_weights, earliest_month, portfolio_stats,
                            target_weights)
from core.rebalancer import build_plan, sell_rebalance
from ui import charts, theme

REPO_URL = "https://github.com/turbom-cell/sparplan-kompass"
BENCHMARKS = {"MSCI World": "EUNL.DE", "S&P 500": "SXR8.DE", "Nasdaq 100": "SXRV.DE",
              "DAX": "EXS1.DE", "STOXX Europe 600": "EXSA.DE"}

st.set_page_config(page_title="Sparplan-Kompass", page_icon="🧭", layout="wide")
theme.inject()

PRESETS: dict[str, list[tuple[str, int, int]]] = {   # (Ticker, Bestand €, Einstand €)
    "Tech-lastig (Klumpenrisiko)": [
        ("SXRV.DE", 6000, 3600), ("SXR8.DE", 3000, 2200), ("EUNL.DE", 2000, 1600),
        ("IS3S.DE", 500, 450), ("EUNH.DE", 0, 0), ("4GLD.DE", 500, 400)],
    "Welt-Depot mit viel DAX": [
        ("EUNL.DE", 5000, 3800), ("EXS1.DE", 4500, 3300), ("IS3N.DE", 1500, 1400),
        ("EUNH.DE", 0, 0), ("4GLD.DE", 0, 0)],
    "Langjähriger Sparer (60.000 €, hohe Gewinne)": [
        ("SXRV.DE", 26000, 9000), ("SXR8.DE", 18000, 9500), ("EUNL.DE", 12000, 8000),
        ("XDWT.DE", 4000, 2000), ("IS3S.DE", 0, 0), ("EUNH.DE", 0, 0), ("4GLD.DE", 0, 0)],
    "Berufseinsteiger (klein, ein ETF)": [
        ("SXR8.DE", 2500, 2100), ("EUNL.DE", 0, 0), ("IS3S.DE", 0, 0),
        ("EUNH.DE", 0, 0), ("4GLD.DE", 0, 0)],
}


def eur(x: float, digits: int = 0) -> str:
    return f"{x:,.{digits}f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f} %".replace(".", ",")


def num(x: float, digits: int = 1, sign: bool = False) -> str:
    return f"{x:{'+' if sign else ''}.{digits}f}".replace(".", ",")


def preset_frame(name: str) -> pd.DataFrame:
    rows = PRESETS[name]
    return pd.DataFrame({
        "Ticker": [r[0] for r in rows],
        "Bestand €": [float(r[1]) for r in rows],
        "Einstand €": [float(r[2]) for r in rows],
    })


@st.cache_data(ttl=3600, show_spinner="Lade Kurse …")
def cached_prices(tickers: tuple[str, ...], years: int, allow_live: bool):
    data = load_prices(list(tickers), years, allow_live)
    return data.prices, data.source, data.missing


@st.cache_data(show_spinner="Berechne Ziel-Depot …")
def cached_target(returns: pd.DataFrame, rf: float, limit: float, w_max: float,
                  w_min: tuple[float, ...] | None):
    t = target_weights(returns.values, rf, limit, w_max, None if w_min is None else np.array(w_min))
    return t.weights, t.status, t.cvar_min


@st.cache_data(show_spinner="Suche den frühesten Monat …")
def cached_earliest(returns: pd.DataFrame, holdings: tuple[float, ...], budget: float,
                    limit: float, w_max: float):
    return earliest_month(returns.values, np.array(holdings), budget, limit, w_max)


@st.cache_data(show_spinner="Rechne Blindtest …")
def cached_blind_test(returns: pd.DataFrame, w_now: tuple[float, ...], rf: float, limit: float,
                      w_max: float, years: int, value: float):
    return analytics.blind_test(returns, np.array(w_now), rf, limit, w_max, years, value)


@st.cache_data(show_spinner="Spiele die letzten Jahre mit echten Kursen nach …")
def cached_walk_forward(prices: pd.DataFrame, bench: pd.Series, holdings: tuple[float, ...],
                        budget: float, years: int, rf: float, limit: float, w_max: float,
                        horizon: int):
    return analytics.walk_forward(prices, bench, np.array(holdings), budget, years, rf, limit,
                                  w_max, horizon)


# --------------------------------------------------------------------------- Seitenleiste 1
with st.sidebar:
    st.header("1 · Dein Depot")
    preset = st.selectbox("Beispiel-Depot laden", list(PRESETS))
    st.caption("Zeilen mit 0 € Bestand sind Kauf-Kandidaten für den Sparplan. "
               "Tabelle frei bearbeitbar, Zeilen unten hinzufügen.")
    depot = st.data_editor(
        preset_frame(preset), key=f"depot_{preset}", num_rows="dynamic", hide_index=True,
        column_config={
            "Ticker": st.column_config.SelectboxColumn(options=list(CATALOG), required=True,
                                                       help="Yahoo-Ticker (Xetra, in Euro)"),
            "Bestand €": st.column_config.NumberColumn(min_value=0, step=100, format="%.0f"),
            "Einstand €": st.column_config.NumberColumn(
                min_value=0, step=100, format="%.0f",
                help="Was du insgesamt für die Position bezahlt hast (für die Steuerrechnung)"),
        })
    budget = float(st.slider("Monatliche Sparrate in €", 50, 2000, 500, 50))

depot = depot.dropna(subset=["Ticker"]).drop_duplicates("Ticker").fillna(0.0)
if depot.empty or depot["Bestand €"].sum() <= 0:
    theme.hero("Sparplan-Kompass", "Trage links mindestens eine Position mit Bestand ein.", [])
    st.stop()

# --------------------------------------------------------------------------- Daten
with st.sidebar:
    with st.expander("Annahmen & Daten"):
        years = st.slider("Kurshistorie in Jahren", 5, 15, 12)
        rf = st.slider("Risikofreier Zins in %", 0.0, 5.0, 2.5, 0.25) / 100
        allowance = float(st.number_input("Freier Sparerpauschbetrag in €", 0, 2000, 1000, 100))
        fee = float(st.number_input("Ordergebühr je Verkauf/Kauf in €", 0.0, 20.0, 1.0, 0.5))
        allow_live = st.toggle("Live-Kurse von Yahoo Finance", value=True)

prices, source, missing = cached_prices(tuple(depot["Ticker"]), years, allow_live)
if missing:
    st.warning("Keine Kursdaten für: " + ", ".join(missing) + ". Diese Positionen werden ignoriert.")
    depot = depot[~depot["Ticker"].isin(missing)]
if prices.empty or depot["Bestand €"].sum() <= 0:
    st.error("Für dieses Depot liegen keine Kursdaten vor.")
    st.stop()

tickers = list(prices.columns)
depot = depot.set_index("Ticker").loc[tickers]
names = [display_name(t) for t in tickers]
returns = to_returns(prices)
R = returns.values
h = depot["Bestand €"].to_numpy(dtype=float)
cost_basis = np.where(h > 0, depot["Einstand €"].to_numpy(dtype=float), 0.0)
tf = np.array([default_teilfreistellung(t) for t in tickers])
V = float(h.sum())
w_now = h / V
now = portfolio_stats(w_now, R, rf)
loss_now = now.cvar * V

# --------------------------------------------------------------------------- Seitenleiste 2
with st.sidebar:
    st.header("2 · Deine Schmerzschwelle")
    lo_s = max(10, int(round(loss_now * 0.35 / 10) * 10))
    hi_s = max(lo_s + 20, int(round(loss_now * 1.2 / 10) * 10))
    start_s = min(hi_s, max(lo_s, int(round(loss_now * 0.8 / 10) * 10)))
    limit_eur = float(st.slider(
        "Wie viel € darf dein Depot an einem Schrecktag verlieren?", lo_s, hi_s, start_s, 5,
        key=f"limit_{preset}_{int(V)}_{len(tickers)}",
        help="Schrecktag = Durchschnitt der 5 % schlechtesten Börsentage (95 %-CVaR), "
             "bezogen auf deinen heutigen Depotwert."))
    w_max = st.slider("Klumpengrenze: größte Position höchstens (%)", 20, 100, 40, 5) / 100
    st.header("3 · Dein Zeitplan")
    months = st.slider("In wie vielen Monaten willst du im Ziel sein?", 3, 48, 12)

limit = limit_eur / V
lo = diluted_weights(h, budget, months)
w_goal, status, cvar_min = cached_target(returns, rf, limit, w_max, tuple(np.round(lo, 6)))
w_free, status_free, _ = cached_target(returns, rf, limit, w_max, None)
first_month = cached_earliest(returns, tuple(h), budget, limit, w_max)
goal = portfolio_stats(w_goal, R, rf)
free = portfolio_stats(w_free, R, rf)
loss_goal = goal.cvar * V
plan = build_plan(h, w_goal, budget, months, names)
top = int(np.argmax(w_now))

# --------------------------------------------------------------------------- Kopf
theme.hero(
    "Sparplan-Kompass",
    "<b>Depot-Risiko senken, ohne einen einzigen Anteil zu verkaufen.</b> Die App verteilt nur "
    "deine Sparrate neu. So fällt keine Steuer auf Kursgewinne an.",
    [("#34d399" if source == "live" else "#fbbf24",
      "Live-Kurse von Yahoo Finance" if source == "live" else "Gespeicherter Kurs-Snapshot"),
     ("", f"{prices.index[0]:%m/%Y} – {prices.index[-1]:%m/%Y}"),
     ("", f"{len(returns):,} Börsentage".replace(",", ".")),
     ("", f"Depot {eur(V)}"), ("", f"Sparrate {eur(budget)} / Monat")])

tab_why, tab_check, tab_plan, tab_market, tab_stress, tab_arch = st.tabs(
    ["Problem & Idee", "Depot-Check", "Sparplan-Fahrplan", "Gegen den Markt", "Stresstest",
     "Architektur"])

# --------------------------------------------------------------------------- Tab: Problem
with tab_why:
    theme.kpi_row([
        {"label": "Dein Depot", "value": eur(V), "sub": f"{int((h > 0).sum())} Positionen"},
        {"label": "Verlust an einem Schrecktag", "value": eur(-loss_now),
         "sub": f"{pct(-now.cvar)} des Depots", "tone": "bad", "accent": theme.RED},
        {"label": "Größte Position", "value": pct(w_now[top], 0), "sub": names[top],
         "accent": theme.AMBER},
        {"label": "Nach dem Sparplan", "value": eur(-loss_goal),
         "sub": f"{eur(loss_now - loss_goal)} weniger Verlust am Schrecktag", "tone": "good",
         "accent": theme.GREEN},
    ])
    theme.callout(
        "<b>Der springende Punkt:</b> Fast jedes Tool sagt „verkaufe A, kaufe B“. Das kostet in "
        "Deutschland Steuern. Der Sparplan-Kompass erreicht ein ruhigeres Depot <b>nur über die "
        "Aufteilung der nächsten Sparraten</b> und zeigt das Risiko in Euro statt in Prozent.")
    left, right = st.columns(2)
    with left:
        st.subheader("Für wen?")
        st.markdown(
            "Privatanleger mit ETF-Sparplan bei Neobrokern: Studierende, Berufseinsteiger und "
            "langjährige Sparer, die 2 bis 6 ETFs besparen.")
        st.subheader("Welches Problem?")
        st.markdown(
            "- **Klumpen entstehen von selbst.** Was gut läuft, wird im Depot immer größer.\n"
            "- **Risiko in Prozent versteht niemand.** „1,8 % Tages-CVaR“ sagt einem Anleger nichts.\n"
            "- **Umschichten kostet Geld.** Wer verkauft, zahlt auf Kursgewinne 26,375 % Steuer "
            "plus Ordergebühren.")
    with right:
        st.subheader("So funktioniert es")
        theme.steps([
            ("Depot eingeben", "Bestand je ETF und die monatliche Sparrate."),
            ("Schmerzschwelle wählen", "Wie viel Euro darf ein schlechter Börsentag kosten?"),
            ("Fahrplan bekommen", "Feste Sparplan-Aufteilung, mit der das Depot ins Ziel wächst."),
            ("Gegenprobe ansehen", "Vergleich mit dem echten Markt über die letzten Jahre."),
        ])
    st.caption("Links Depot, Sparrate und Schmerzschwelle verändern – alle Tabs rechnen sofort neu.")

# --------------------------------------------------------------------------- Tab: Depot-Check
with tab_check:
    if status == "unreachable":
        hint = (f"frühestens nach <b>{first_month} Monaten</b>" if first_month
                else "auch nach 10 Jahren nicht")
        st.warning(f"Mit {months} Monaten Sparrate ist die Schmerzschwelle von {eur(limit_eur)} ohne "
                   f"Verkäufe noch nicht erreichbar ({hint.replace('<b>', '**').replace('</b>', '**')}). "
                   "Gezeigt wird das sicherste Depot, das bis dahin möglich ist. Tipp: Zeitplan "
                   "verlängern oder Sparrate erhöhen.")
    elif loss_now <= limit_eur:
        st.success("Dein Depot liegt schon unter der Schmerzschwelle. Das Ziel-Depot holt bei "
                   "gleichem Risiko-Budget mehr Ertrag pro Risiko heraus.")
    better = loss_goal < loss_now
    theme.kpi_row([
        {"label": "Schrecktag heute", "value": eur(-loss_now), "sub": "Ø der 5 % schlechtesten Tage",
         "accent": theme.SLATE},
        {"label": f"Schrecktag im Ziel (Monat {months})", "value": eur(-loss_goal),
         "sub": f"{eur(abs(loss_now - loss_goal))} {'weniger' if better else 'mehr'}",
         "tone": "good" if better else "bad", "accent": theme.BLUE},
        {"label": "Schwankung p.a.", "value": pct(goal.vol),
         "sub": f"{num((goal.vol - now.vol) * 100, 1, True)} Prozentpunkte",
         "tone": "good" if goal.vol < now.vol else "bad", "accent": theme.BLUE},
        {"label": "Sharpe Ratio", "value": num(goal.sharpe, 2),
         "sub": f"{num(goal.sharpe - now.sharpe, 2, True)} gegenüber heute",
         "tone": "good" if goal.sharpe >= now.sharpe else "bad", "accent": theme.BLUE},
    ])
    left, right = st.columns(2)
    with left:
        st.subheader("Gewichte: heute und Ziel")
        st.plotly_chart(charts.weights_bar(names, w_now, w_goal, w_max), width="stretch")
    with right:
        st.subheader("So sieht ein Börsentag in Euro aus")
        st.plotly_chart(charts.loss_histogram(R @ w_now * V, R @ w_goal * V, loss_now, loss_goal,
                                              limit_eur), width="stretch")
    st.caption("Gestrichelte Linien: durchschnittlicher Verlust an den 5 % schlechtesten Tagen "
               "(grau = heute, blau = Ziel). Rote Linie: deine Schmerzschwelle. Alle Euro-Werte "
               "beziehen sich auf den heutigen Depotwert.")
    with st.expander("Wie hängen die Positionen zusammen? (Korrelationen)"):
        st.plotly_chart(charts.correlation_heatmap(returns, names), width="stretch")

# --------------------------------------------------------------------------- Tab: Fahrplan
with tab_plan:
    st.subheader(f"Verkaufe nichts. Stelle deinen Sparplan für {months} Monate so ein:")
    table = pd.DataFrame({
        "Position": names,
        "Ticker": tickers,
        "Sparrate €/Monat": plan.monthly.round(0).to_numpy(),
        "Anteil an der Sparrate %": (plan.monthly / budget * 100).round(1).to_numpy(),
        "Gewicht heute %": (w_now * 100).round(1),
        "Gewicht im Ziel %": (w_goal * 100).round(1),
    }).sort_values("Sparrate €/Monat", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Anteil an der Sparrate %": st.column_config.ProgressColumn(format="%.1f %%", min_value=0,
                                                                    max_value=100)})
    paused = [n for n, m, w in zip(names, plan.monthly, w_now) if m < 0.5 and w > 0]
    if paused:
        st.markdown("⏸️ **Pausieren:** " + ", ".join(paused) + " – diese Positionen sind zu groß "
                    "und werden nur durch die Zukäufe der anderen verwässert.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Der Weg ins Ziel")
        st.plotly_chart(charts.plan_animation(plan.weights, w_goal), width="stretch")
        st.caption("▶ drücken: Die Balken wandern Monat für Monat zur Zielmarke (schwarzer Strich).")
    with right:
        st.subheader("Risiko sinkt mit jeder Sparrate")
        path = [cvar(w, R) * V for w in plan.weights.to_numpy()]
        st.plotly_chart(charts.risk_path(list(plan.weights.index), path, limit_eur), width="stretch")
        if first_month == 0:
            st.caption("Die Schmerzschwelle ist schon heute eingehalten.")
        elif first_month:
            st.caption(f"Ohne Verkäufe ist die Schmerzschwelle frühestens in Monat {first_month} "
                       "erreichbar (wenn nur auf Sicherheit optimiert wird).")
        else:
            st.caption("Diese Schmerzschwelle ist ohne Verkäufe in 10 Jahren nicht erreichbar.")

    st.divider()
    st.subheader("Und wenn du stattdessen heute verkaufst?")
    trades = sell_rebalance(h, w_free)
    cost = tax.sell_cost(trades, h, cost_basis, tf, allowance, fee)
    a, b = st.columns(2)
    with a:
        st.markdown("#### Sofort umschichten")
        theme.kpi_row([
            {"label": "Verkäufe", "value": eur(cost.sold), "accent": theme.RED},
            {"label": "Steuer + Gebühren", "value": eur(cost.total), "accent": theme.RED,
             "sub": f"{num(cost.total / budget, 1)} Monatssparraten", "tone": "bad"},
            {"label": "Schrecktag danach", "value": eur(-free.cvar * V), "accent": theme.RED,
             "sub": "sofort"},
        ])
        st.caption(f"Realisierter Gewinn {eur(cost.gain)}, davon steuerpflichtig {eur(cost.taxable)} "
                   f"(nach Teilfreistellung und Pauschbetrag). Steuer {eur(cost.tax)}, "
                   f"Gebühren {eur(cost.fees)}.")
    with b:
        st.markdown("#### Über den Sparplan")
        theme.kpi_row([
            {"label": "Verkäufe", "value": eur(0), "accent": theme.GREEN},
            {"label": "Steuer + Gebühren", "value": eur(0), "accent": theme.GREEN,
             "sub": "nichts verkauft", "tone": "good"},
            {"label": f"Schrecktag in Monat {months}", "value": eur(-loss_goal),
             "accent": theme.GREEN, "sub": "Preis: Wartezeit"},
        ])
        st.caption("Keine Verkäufe, also keine Steuer auf Kursgewinne. Sparplan-Ausführungen sind "
                   "bei den meisten Neobrokern gebührenfrei.")
    if cost.total > 0:
        theme.callout(f"Sofort umschichten kostet <b>{eur(cost.total)}</b>. Das entspricht "
                      f"<b>{num(cost.total / budget, 1)} Monatssparraten</b>, die nie wieder für "
                      "dich mitarbeiten.")
    else:
        st.info("Bei diesem Depot fiele beim Verkaufen keine Steuer an (Gewinne unter dem "
                "Pauschbetrag). Dann ist sofortiges Umschichten der schnellere Weg.")
    st.caption("Vereinfachte Rechnung, keine Steuerberatung: ohne Kirchensteuer, Vorabpauschale und "
               "FIFO-Reihenfolge; Teilfreistellung 30 % für Aktienfonds, 0 % für Anleihen und Gold.")

# --------------------------------------------------------------------------- Tab: Marktvergleich
with tab_market:
    st.subheader("Sparplan-Kompass gegen den echten Markt")
    st.markdown("Drei Anleger starten mit **demselben Depot** und zahlen **dieselbe Sparrate** ein. "
                "Gerechnet wird mit echten Tageskursen. Der Kompass darf an jedem Stichtag nur Kurse "
                "nutzen, die es damals schon gab, und er verkauft nie.")
    c1, c2 = st.columns([2, 3])
    with c1:
        sim_years = st.radio("Zeitraum", [3, 5, 7], horizontal=True,
                             format_func=lambda y: f"letzte {y} Jahre")
    with c2:
        bench_name = st.selectbox("Markt-Vergleich (alles in einen ETF)", list(BENCHMARKS))
    bench_prices, _, bench_missing = cached_prices((BENCHMARKS[bench_name],), years, allow_live)
    sim = None if bench_missing or bench_prices.empty else cached_walk_forward(
        prices, bench_prices.iloc[:, 0], tuple(h), budget, sim_years, rf, limit, w_max, months)
    if sim is None:
        st.info("Für diesen Zeitraum reicht die gemeinsame Kurshistorie des Depots nicht aus. "
                "Wähle einen kürzeren Zeitraum oder mehr Kurshistorie unter „Annahmen & Daten“.")
    else:
        values, rets, log = sim
        market_label = f"Markt ({bench_name})"
        stats = analytics.compare_stats(values, rets, rf)
        k, s, m = stats.loc["Sparplan-Kompass"], stats.loc["Weiter wie bisher"], stats.loc["Markt"]
        dd = analytics.underwater(rets)

        def versus(mine: float, other: float, unit: str, lower_is_better: bool, digits: int = 1):
            diff = mine - other
            good = diff < 0 if lower_is_better else diff > 0
            if unit == "€":
                text = f"{eur(abs(diff))} {'weniger' if diff < 0 else 'mehr'} als der Markt"
            else:
                text = f"{num(abs(diff), digits)} {unit} {'weniger' if diff < 0 else 'mehr'} als der Markt"
            return {"sub": text, "tone": "good" if good else "bad"}

        theme.kpi_row([
            {"label": "Größter Einbruch", "value": f"{num(k['Größter Einbruch %'])} %",
             "accent": theme.BLUE,
             **versus(-k["Größter Einbruch %"], -m["Größter Einbruch %"], "Punkte", True)},
            {"label": "Verlust am Schrecktag", "value": f"{num(-k['Schrecktag %'], 2)} %",
             "accent": theme.BLUE, **versus(k["Schrecktag %"], m["Schrecktag %"], "Punkte", True, 2)},
            {"label": "Ertrag pro Risiko (Sharpe)", "value": num(k["Sharpe"], 2),
             "accent": theme.BLUE, **versus(k["Sharpe"], m["Sharpe"], "", False, 2)},
            {"label": "Endwert", "value": eur(k["Endwert €"]), "accent": theme.BLUE,
             **versus(k["Endwert €"], m["Endwert €"], "€", False)},
        ])

        calmer = k["Größter Einbruch %"] > m["Größter Einbruch %"]
        richer = k["Endwert €"] >= m["Endwert €"]
        gap = abs(k["Endwert €"] - m["Endwert €"])
        risk_text = (f"Der größte Einbruch lag bei <b>{num(k['Größter Einbruch %'])} %</b> statt "
                     f"{num(m['Größter Einbruch %'])} % im Markt und {num(s['Größter Einbruch %'])} % "
                     "bei „weiter wie bisher“." if calmer else
                     f"In diesem Zeitraum war der Kompass <b>nicht</b> ruhiger als der Markt "
                     f"({num(k['Größter Einbruch %'])} % gegen {num(m['Größter Einbruch %'])} %).")
        money_text = (f"Am Ende stehen sogar <b>{eur(gap)} mehr</b> auf dem Depot." if richer else
                      f"Der Preis dafür: <b>{eur(gap)} weniger Endwert</b> als der Markt.")
        theme.callout(f"{risk_text} {money_text}")

        st.plotly_chart(charts.compare_lines(values, market_label, analytics.CRISES), width="stretch")
        st.caption("Depotwert in Euro inklusive aller Sparraten. Rot hinterlegt: echte Krisenphasen. "
                   "Unten im Schieber lässt sich der Zeitraum eingrenzen.")
        st.subheader("Wie tief ging es zwischendurch runter?")
        st.plotly_chart(charts.underwater_lines(dd, market_label, analytics.CRISES), width="stretch")
        st.caption("Abstand zum bisherigen Höchststand, ohne den Effekt der Einzahlungen. "
                   "Je flacher die Kurve, desto ruhiger schläft der Anleger.")

        show = stats.rename(index={"Markt": market_label})
        st.dataframe(show, width="stretch", column_config={
            "Endwert €": st.column_config.NumberColumn(format="%.0f"),
            "Gewinn €": st.column_config.NumberColumn(format="%.0f"),
            "Rendite p.a. %": st.column_config.NumberColumn(format="%.1f"),
            "Schwankung p.a. %": st.column_config.NumberColumn(format="%.1f"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Schrecktag %": st.column_config.NumberColumn(format="%.2f"),
            "Größter Einbruch %": st.column_config.NumberColumn(format="%.1f")})
        with st.expander("Was hat der Kompass wann gekauft?"):
            st.plotly_chart(charts.purchase_log_bar(log.rename(columns=dict(zip(names, names)))),
                            width="stretch")
            st.caption("Alle 6 Monate wird neu gerechnet. Jede Säule zeigt die Sparplan-Aufteilung, "
                       "die ab diesem Stichtag galt.")
        st.caption("Ehrliche Einordnung: Das Ziel der Methode ist weniger Verlust in schlechten "
                   "Phasen, nicht die höchste Rendite. In langen Aufwärtsphasen kann ein reines "
                   "Aktien-Depot mehr Endwert liefern. Vergangene Kurse sind keine Garantie.")

# --------------------------------------------------------------------------- Tab: Stresstest
with tab_stress:
    st.subheader("Was wäre in echten Krisen passiert?")
    crises = analytics.crisis_table(returns, {"Dein Depot": w_now, "Ziel-Depot": w_goal}, V)
    if crises.empty:
        st.info("Die gemeinsame Kurshistorie dieses Depots deckt keine der hinterlegten Krisen ab.")
    else:
        st.plotly_chart(charts.crisis_bar(crises), width="stretch")
        st.caption("Größter Einbruch vom Hoch zum Tief innerhalb der Krise, auf den heutigen "
                   "Depotwert gerechnet.")

    st.divider()
    st.subheader("Blindtest: Hätte die Methode funktioniert?")
    test_years = st.radio("Blindtest über die letzten …", [1, 2, 3, 5], index=2, horizontal=True,
                          format_func=lambda y: f"{y} Jahre")
    result = cached_blind_test(returns, tuple(w_now), rf, limit, w_max, test_years, V)
    if result is None:
        st.info("Zu wenig Kurshistorie für diesen Blindtest.")
    else:
        curves, split = result
        st.markdown(f"Das Ziel-Depot wird **nur mit Daten bis {split:%d.%m.%Y}** berechnet und dann "
                    "auf dem Zeitraum danach geprüft, den die Optimierung nie gesehen hat. "
                    "Hier ohne Sparraten: einmal umgestellt und liegen gelassen.")
        st.plotly_chart(charts.blind_test_lines(curves), width="stretch")
        st.dataframe(analytics.curve_stats(curves, rf), width="stretch", column_config={
            "Endwert €": st.column_config.NumberColumn(format="%.0f"),
            "Rendite p.a. %": st.column_config.NumberColumn(format="%.1f"),
            "Schwankung p.a. %": st.column_config.NumberColumn(format="%.1f"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Schrecktag %": st.column_config.NumberColumn(format="%.2f"),
            "Größter Einbruch %": st.column_config.NumberColumn(format="%.1f")})

# --------------------------------------------------------------------------- Tab: Architektur
with tab_arch:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Aufbau in drei Schichten")
        st.graphviz_chart("""
        digraph {
          rankdir=LR; bgcolor="transparent";
          node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#eff6ff", color="#bfdbfe"];
          edge [color="#94a3b8"];
          user [label="Nutzer\\n(Browser)", fillcolor="#fef3c7", color="#fde68a"];
          yahoo [label="Yahoo Finance", fillcolor="#f1f5f9", color="#e2e8f0"];
          snap [label="Kurs-Snapshot\\n(CSV im Repo)", fillcolor="#f1f5f9", color="#e2e8f0"];
          subgraph cluster_ui { label="Oberfläche"; color="#c7d2fe"; fontname="Helvetica";
            app [label="app.py\\nStreamlit"]; charts [label="ui/charts.py\\nPlotly"];
            theme [label="ui/theme.py\\nDesign"]; }
          subgraph cluster_core { label="Rechenkern (ohne Streamlit, getestet)"; color="#c7d2fe"; fontname="Helvetica";
            opt [label="optimizer.py\\nCVaR-LP + SLSQP"]; reb [label="rebalancer.py\\nSparplan-Fahrplan"];
            taxn [label="tax.py\\nSteuer & Gebühren"]; ana [label="analytics.py\\nMarktvergleich, Krisen"]; }
          data [label="data_service.py\\nLaden, Bereinigen, Cache"];
          user -> app; app -> charts; app -> theme; app -> opt; app -> reb; app -> taxn; app -> ana;
          app -> data; data -> yahoo; data -> snap [style=dashed, label="Notfall"];
        }""")
    with right:
        st.subheader("Rechenweg")
        st.markdown(
            "1. Tagesrenditen der gemeinsamen Kurshistorie.\n"
            "2. **Sicherstes Depot** per linearem Programm (CVaR nach Rockafellar/Uryasev).\n"
            "3. **Ziel-Depot**: höchste Sharpe Ratio, solange Schmerzschwelle und Klumpengrenze halten.\n"
            "4. **Nichts verkaufen** als Nebenbedingung: Jede Position behält mindestens ihren "
            "verwässerten Anteil.\n"
            "5. Daraus die feste Sparplan-Aufteilung und der Steuervergleich.")
        st.latex(r"\max_{w}\ \frac{w^{\top}\mu - r_f}{\sqrt{w^{\top}\Sigma\, w}}")
        st.latex(r"\mathrm{CVaR}_{95\,\%}(w)\le \frac{S}{V},\qquad \sum_i w_i=1")
        st.latex(r"\frac{h_i}{V+N\cdot B}\le w_i\le w_{\max}")
        st.caption("S = Schmerzschwelle in €, V = Depotwert, h = Bestand je Position, "
                   "B = Sparrate, N = Monate.")
        st.subheader("Werkzeuge")
        st.markdown("Python · Streamlit · Plotly · SciPy (HiGHS, SLSQP) · pandas · yfinance · "
                    "pytest · Git/GitHub · Streamlit Community Cloud")
        if REPO_URL:
            st.link_button("Zum GitHub-Repository", REPO_URL)
    st.subheader("Grenzen des Modells")
    st.markdown(
        "- Vergangene Kurse sagen die Zukunft nicht voraus; Renditen werden deshalb nur zur Hälfte geglaubt.\n"
        "- Kursbewegungen während der Sparplan-Laufzeit werden nicht prognostiziert; der Plan sollte "
        "alle paar Monate neu gerechnet werden.\n"
        "- Die Steuerrechnung ist vereinfacht. Die App ist ein Studienprojekt und keine Anlageberatung.")
