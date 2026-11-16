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
from ui import charts

REPO_URL = "https://github.com/turbom-cell/sparplan-kompass"

st.set_page_config(page_title="Sparplan-Kompass", page_icon="🧭", layout="wide")

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
    st.title("🧭 Sparplan-Kompass")
    st.info("Trage links mindestens eine Position mit Bestand ein.")
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
cost_basis = np.minimum(depot["Einstand €"].to_numpy(dtype=float), np.where(h > 0, np.inf, 0.0))
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

# --------------------------------------------------------------------------- Kopf
st.title("🧭 Sparplan-Kompass")
st.markdown("**Depot-Risiko senken, ohne einen einzigen Anteil zu verkaufen** – nur über die "
            "Aufteilung deiner Sparrate, also ohne Steuern auf Kursgewinne.")
span = f"{prices.index[0]:%d.%m.%Y} – {prices.index[-1]:%d.%m.%Y}"
st.caption(("🟢 Live-Kurse von Yahoo Finance" if source == "live"
            else "🟡 Gespeicherter Kurs-Snapshot (Yahoo nicht erreichbar oder Live-Kurse aus)")
           + f" · gemeinsame Historie {span} · {len(returns):,} Börsentage".replace(",", "."))

tab_why, tab_check, tab_plan, tab_stress, tab_arch = st.tabs(
    ["💡 Problem & Idee", "🩺 Depot-Check", "🗓️ Sparplan-Fahrplan", "🌪️ Stresstest & Blindtest",
     "🏗️ Architektur"])

# --------------------------------------------------------------------------- Tab: Problem
with tab_why:
    c1, c2, c3 = st.columns(3)
    c1.metric("Dein Depot", eur(V))
    c2.metric("Verlust an einem Schrecktag", eur(-loss_now), pct(-now.cvar), delta_color="off")
    top = int(np.argmax(w_now))
    c3.metric("Größte Position", pct(w_now[top], 0), names[top], delta_color="off")
    left, right = st.columns(2)
    with left:
        st.subheader("Für wen?")
        st.markdown(
            "Privatanleger mit ETF-Sparplan bei Neobrokern: Studierende und Berufseinsteiger, "
            "die seit ein paar Jahren 2 bis 6 ETFs besparen.")
        st.subheader("Welches Problem?")
        st.markdown(
            "- **Klumpen entstehen von selbst.** Was gut läuft, wird im Depot immer größer.\n"
            "- **Risiko in Prozent versteht niemand.** „1,8 % Tages-CVaR“ sagt einem Anleger nichts.\n"
            "- **Umschichten kostet Geld.** Wer in Deutschland verkauft, zahlt auf Kursgewinne "
            "26,375 % Steuer plus Ordergebühren.")
    with right:
        st.subheader("Was macht die App anders?")
        st.markdown(
            "1. Sie startet beim **bestehenden Depot**, nicht bei einem Wunschportfolio.\n"
            "2. Risiko wird als **Schmerzschwelle in Euro** eingestellt.\n"
            "3. Das Ziel-Depot wird so berechnet, dass es **allein durch Zukäufe** erreichbar ist.\n"
            "4. Ergebnis ist eine **feste Sparplan-Aufteilung**, die man beim Broker einstellt.\n"
            "5. Daneben steht, was **Verkaufen heute an Steuern kosten** würde.")
        st.info("Links Depot, Sparrate und Schmerzschwelle verändern – alle Tabs rechnen sofort neu.")

# --------------------------------------------------------------------------- Tab: Depot-Check
with tab_check:
    if status == "unreachable":
        hint = (f"frühestens nach **{first_month} Monaten**" if first_month
                else "auch nach 10 Jahren nicht")
        st.warning(f"Mit {months} Monaten Sparrate ist die Schmerzschwelle von {eur(limit_eur)} ohne "
                   f"Verkäufe noch nicht erreichbar ({hint}). Gezeigt wird das sicherste Depot, das "
                   f"bis dahin möglich ist. Tipp: Zeitplan verlängern oder Sparrate erhöhen.")
    elif loss_now <= limit_eur:
        st.success("Dein Depot liegt schon unter der Schmerzschwelle. Das Ziel-Depot holt bei "
                   "gleichem Risiko-Budget mehr Ertrag pro Risiko heraus.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Schrecktag heute", eur(-loss_now))
    c2.metric(f"Schrecktag im Ziel (Monat {months})", eur(-loss_goal), eur(loss_now - loss_goal),
              help="Bezogen auf den heutigen Depotwert, damit die Zahlen vergleichbar bleiben.")
    c3.metric("Schwankung p.a.", pct(goal.vol), pct(goal.vol - now.vol), delta_color="inverse")
    c4.metric("Sharpe Ratio", f"{goal.sharpe:.2f}".replace(".", ","),
              f"{goal.sharpe - now.sharpe:+.2f}".replace(".", ","))
    left, right = st.columns(2)
    with left:
        st.subheader("Gewichte: heute und Ziel")
        st.plotly_chart(charts.weights_bar(names, w_now, w_goal, w_max), width="stretch")
    with right:
        st.subheader("So sieht ein Börsentag in Euro aus")
        st.plotly_chart(charts.loss_histogram(R @ w_now * V, R @ w_goal * V, loss_now, loss_goal,
                                              limit_eur), width="stretch")
    st.caption("Gestrichelte Linien: durchschnittlicher Verlust an den 5 % schlechtesten Tagen "
               "(grau = heute, blau = Ziel). Rote Linie: deine Schmerzschwelle.")
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
        st.markdown("#### 🔴 Sofort umschichten")
        x1, x2, x3 = st.columns(3)
        x1.metric("Verkäufe", eur(cost.sold))
        x2.metric("Steuer + Gebühren", eur(cost.total))
        x3.metric("Schrecktag danach", eur(-free.cvar * V))
        st.caption(f"Realisierter Gewinn {eur(cost.gain)}, davon steuerpflichtig {eur(cost.taxable)} "
                   f"(nach Teilfreistellung und Pauschbetrag). Steuer {eur(cost.tax)}, "
                   f"Gebühren {eur(cost.fees)}.")
    with b:
        st.markdown("#### 🟢 Über den Sparplan")
        y1, y2, y3 = st.columns(3)
        y1.metric("Verkäufe", eur(0))
        y2.metric("Steuer + Gebühren", eur(0))
        y3.metric(f"Schrecktag in Monat {months}", eur(-loss_goal))
        st.caption("Keine Verkäufe, also keine Steuer auf Kursgewinne. Sparplan-Ausführungen sind "
                   "bei den meisten Neobrokern gebührenfrei. Der Preis ist die Wartezeit.")
    if cost.total > 0:
        st.info(f"Sofort umschichten kostet **{eur(cost.total)}**. Das entspricht "
                f"**{str(round(cost.total / budget, 1)).replace('.', ',')} Monatssparraten**, "
                f"die nie wieder für dich mitarbeiten.")
    else:
        st.info("Bei diesem Depot fiele beim Verkaufen keine Steuer an (Gewinne unter dem "
                "Pauschbetrag). Dann ist sofortiges Umschichten der schnellere Weg.")
    st.caption("Vereinfachte Rechnung, keine Steuerberatung: ohne Kirchensteuer, Vorabpauschale und "
               "FIFO-Reihenfolge; Teilfreistellung 30 % für Aktienfonds, 0 % für Anleihen und Gold.")

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
    test_years = st.radio("Blindtest über die letzten …", [1, 2, 3, 5], index=2, horizontal=True,
                          format_func=lambda y: f"{y} Jahre")
    st.subheader("Blindtest: Hätte die Methode funktioniert?")
    result = cached_blind_test(returns, tuple(w_now), rf, limit, w_max, test_years, V)
    if result is None:
        st.info("Zu wenig Kurshistorie für diesen Blindtest.")
    else:
        curves, split = result
        st.markdown(f"Das Ziel-Depot wird **nur mit Daten bis {split:%d.%m.%Y}** berechnet und dann "
                    "auf dem Zeitraum danach geprüft, den die Optimierung nie gesehen hat.")
        st.plotly_chart(charts.blind_test_lines(curves), width="stretch")
        stats = analytics.curve_stats(curves, rf)
        st.dataframe(stats, width="stretch", column_config={
            "Endwert €": st.column_config.NumberColumn(format="%.0f"),
            "Rendite p.a. %": st.column_config.NumberColumn(format="%.1f"),
            "Schwankung p.a. %": st.column_config.NumberColumn(format="%.1f"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Schrecktag %": st.column_config.NumberColumn(format="%.2f"),
            "Größter Einbruch %": st.column_config.NumberColumn(format="%.1f")})
        st.caption("Ein Blindtest ist kein Versprechen: Er zeigt einen einzigen Zeitraum. "
                   "Ziel der Methode ist weniger Verlust an schlechten Tagen, nicht die höchste Rendite.")

# --------------------------------------------------------------------------- Tab: Architektur
with tab_arch:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Aufbau in drei Schichten")
        st.graphviz_chart("""
        digraph {
          rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#eef2ff"];
          user [label="Nutzer\\n(Browser)", fillcolor="#fef3c7"];
          yahoo [label="Yahoo Finance", fillcolor="#f3f4f6"];
          snap [label="Kurs-Snapshot\\n(CSV im Repo)", fillcolor="#f3f4f6"];
          subgraph cluster_ui { label="Oberfläche"; color="#c7d2fe";
            app [label="app.py\\nStreamlit"]; charts [label="ui/charts.py\\nPlotly"]; }
          subgraph cluster_core { label="Rechenkern (ohne Streamlit, getestet)"; color="#c7d2fe";
            opt [label="optimizer.py\\nCVaR-LP + SLSQP"]; reb [label="rebalancer.py\\nSparplan-Fahrplan"];
            taxn [label="tax.py\\nSteuer & Gebühren"]; ana [label="analytics.py\\nKrisen & Blindtest"]; }
          data [label="data_service.py\\nLaden, Bereinigen, Cache"];
          user -> app; app -> charts; app -> opt; app -> reb; app -> taxn; app -> ana;
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
