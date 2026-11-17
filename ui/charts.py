"""Darstellungsschicht: alle Plotly-Grafiken der App. Keine Rechenlogik."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

C_NOW = "#94a3b8"     # heutiges Depot / weiter wie bisher
C_GOAL = "#2563eb"    # Ziel-Depot / Sparplan-Kompass
C_BAD = "#e11d48"
C_GOOD = "#059669"
C_ALT = "#f59e0b"     # Markt
C_GRID = "#eef1f6"
C_INK = "#0f172a"
PALETTE = ["#2563eb", "#7c3aed", "#0891b2", "#f59e0b", "#059669", "#e11d48", "#64748b", "#db2777"]
FONT = "Inter, -apple-system, Segoe UI, sans-serif"


def _base(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=44, b=8), separators=",.",
        font=dict(family=FONT, size=13, color="#334155"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(namelength=-1, bgcolor="#0f172a", font=dict(color="#fff", family=FONT),
                        bordercolor="#0f172a"),
        bargap=0.28)
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0", ticks="outside", tickcolor="#e2e8f0")
    fig.update_yaxes(gridcolor=C_GRID, zeroline=False)
    return fig


def weights_bar(names: list[str], w_now: np.ndarray, w_goal: np.ndarray, cap: float) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=names, y=w_now * 100, name="Heute", marker_color=C_NOW, marker_cornerradius=6,
                hovertemplate="%{x}: %{y:.1f} %<extra>Heute</extra>")
    fig.add_bar(x=names, y=w_goal * 100, name="Ziel", marker_color=C_GOAL, marker_cornerradius=6,
                hovertemplate="%{x}: %{y:.1f} %<extra>Ziel</extra>")
    fig.add_hline(y=cap * 100, line_dash="dot", line_color=C_BAD,
                  annotation_text="Klumpengrenze", annotation_position="top right")
    fig.update_layout(barmode="group", yaxis_title="Anteil am Depot in %")
    return _base(fig)


def loss_histogram(pnl_now: np.ndarray, pnl_goal: np.ndarray, cvar_now: float, cvar_goal: float,
                   limit: float) -> go.Figure:
    lo = float(min(np.percentile(pnl_now, 0.3), np.percentile(pnl_goal, 0.3)))
    hi = float(max(np.percentile(pnl_now, 99.7), np.percentile(pnl_goal, 99.7)))
    bins = dict(start=lo, end=hi, size=(hi - lo) / 70)
    fig = go.Figure()
    fig.add_histogram(x=np.clip(pnl_now, lo, hi), name="Heute", marker_color=C_NOW, opacity=0.7,
                      xbins=bins, hovertemplate="%{x} €: %{y} Tage<extra>Heute</extra>")
    fig.add_histogram(x=np.clip(pnl_goal, lo, hi), name="Ziel", marker_color=C_GOAL, opacity=0.55,
                      xbins=bins, hovertemplate="%{x} €: %{y} Tage<extra>Ziel</extra>")
    fig.add_vline(x=-limit, line_color=C_BAD, line_width=2,
                  annotation_text="Schmerzgrenze", annotation_position="top")
    fig.add_vline(x=-cvar_now, line_dash="dash", line_color="#64748b")
    fig.add_vline(x=-cvar_goal, line_dash="dash", line_color=C_GOAL)
    fig.update_layout(barmode="overlay", bargap=0,
                      xaxis_title="Gewinn / Verlust an einem Börsentag in €",
                      yaxis_title="Anzahl Tage")
    return _base(fig)


def correlation_heatmap(returns: pd.DataFrame, names: list[str]) -> go.Figure:
    corr = returns.corr().values
    fig = go.Figure(go.Heatmap(z=corr, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu_r",
                               text=np.round(corr, 2), texttemplate="%{text}",
                               hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
    return _base(fig, 360)


def plan_animation(weights: pd.DataFrame, w_goal: np.ndarray) -> go.Figure:
    """Balken wandern Monat für Monat vom heutigen Depot zum Ziel (Play-Knopf)."""
    names = list(weights.columns)
    long = (weights * 100).reset_index().melt(id_vars="Monat", var_name="Position", value_name="Anteil")
    fig = px.bar(long, x="Position", y="Anteil", animation_frame="Monat", color="Position",
                 color_discrete_sequence=PALETTE, range_y=[0, max(60, float(long["Anteil"].max()) + 5)])
    fig.add_scatter(x=names, y=w_goal * 100, mode="markers", name="Ziel",
                    marker=dict(symbol="line-ew-open", size=46, color=C_INK, line_width=3))
    fig.update_traces(hovertemplate="%{x}: %{y:.1f} %<extra></extra>", marker_cornerradius=6,
                      selector=dict(type="bar"))
    fig.update_layout(yaxis_title="Anteil am Depot in %", xaxis_title=None, showlegend=False)
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 350
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 250
    return _base(fig, 460)


def risk_path(months: list[int], loss_eur: list[float], limit: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=months, y=loss_eur, mode="lines+markers", line=dict(color=C_GOAL, width=3),
                    fill="tozeroy", fillcolor="rgba(37,99,235,.08)",
                    name="Verlust an einem Schrecktag",
                    hovertemplate="Monat %{x}: %{y:.2f} %<extra></extra>")
    fig.add_hline(y=limit, line_color=C_BAD, line_dash="dot",
                  annotation_text="Schmerzgrenze", annotation_position="bottom right")
    lo = min(min(loss_eur), limit) * 0.9
    fig.update_layout(xaxis_title="Monat", yaxis_title="Verlust am Schrecktag in % des Depots",
                      showlegend=False, yaxis_range=[lo, max(loss_eur) * 1.05])
    return _base(fig, 340)


def crisis_bar(table: pd.DataFrame) -> go.Figure:
    fig = px.bar(table, x="Krise", y="Einbruch €", color="Depot", barmode="group",
                 color_discrete_map={"Dein Depot": C_NOW, "Ziel-Depot": C_GOAL},
                 custom_data=["Einbruch %"])
    fig.update_traces(hovertemplate="%{x}: %{y:,.0f} € (%{customdata[0]:.1f} %)<extra></extra>",
                      marker_cornerradius=6)
    fig.update_layout(xaxis_title=None, yaxis_title="Größter Einbruch in €")
    return _base(fig)


def blind_test_lines(curves: pd.DataFrame) -> go.Figure:
    colors = {"Dein Depot": C_NOW, "Ziel-Depot": C_GOAL, "Gleichgewichtet": C_ALT}
    fig = go.Figure()
    for name in curves.columns:
        fig.add_scatter(x=curves.index, y=curves[name], name=name,
                        line=dict(color=colors.get(name), width=3 if name == "Ziel-Depot" else 2),
                        hovertemplate="%{y:,.0f} €<extra>" + name + "</extra>")
    fig.update_layout(yaxis_title="Depotwert in €", xaxis_title=None, hovermode="x unified")
    return _base(fig, 420)


# --------------------------------------------------------------------------- Marktvergleich
_COMPARE_STYLE = {
    "Sparplan-Kompass": dict(color=C_GOAL, width=3.5),
    "Einfach halten": dict(color=C_NOW, width=2.2),
    "Markt": dict(color=C_ALT, width=2.2),
}


def _shade_crises(fig: go.Figure, index: pd.DatetimeIndex, crises: dict[str, tuple[str, str]]) -> None:
    for name, (start, end) in crises.items():
        if pd.Timestamp(end) < index[0] or pd.Timestamp(start) > index[-1]:
            continue
        fig.add_vrect(x0=max(pd.Timestamp(start), index[0]), x1=pd.Timestamp(end),
                      fillcolor="#e11d48", opacity=0.07, line_width=0,
                      annotation_text=name, annotation_position="top left",
                      annotation_textangle=-90, annotation_font=dict(size=10, color="#9f1239"))


def compare_lines(values: pd.DataFrame, market_label: str,
                  crises: dict[str, tuple[str, str]]) -> go.Figure:
    """Depotwert in Euro: Sparplan-Kompass gegen 'weiter wie bisher' und den Markt."""
    fig = go.Figure()
    fig.add_scatter(x=values.index, y=values["Eingezahlt"], name="Eingezahlt",
                    line=dict(color="#cbd5e1", width=1.5, dash="dot"),
                    hovertemplate="%{y:,.0f} €<extra>Eingezahlt</extra>")
    for col in ["Markt", "Einfach halten", "Sparplan-Kompass"]:
        label = market_label if col == "Markt" else col
        fig.add_scatter(x=values.index, y=values[col], name=label, line=_COMPARE_STYLE[col],
                        hovertemplate="%{y:,.0f} €<extra>" + label + "</extra>")
    _shade_crises(fig, values.index, crises)
    fig.update_layout(yaxis_title="Depotwert in €", xaxis_title=None, hovermode="x unified")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    return _base(fig, 500)


def underwater_lines(dd: pd.DataFrame, market_label: str,
                     crises: dict[str, tuple[str, str]]) -> go.Figure:
    """Abstand zum Höchststand: je flacher die Kurve, desto ruhiger das Depot."""
    fig = go.Figure()
    for col in ["Markt", "Einfach halten", "Sparplan-Kompass"]:
        label = market_label if col == "Markt" else col
        style = _COMPARE_STYLE[col]
        fig.add_scatter(x=dd.index, y=dd[col], name=label, line=dict(color=style["color"],
                        width=style["width"] - 0.7),
                        fill="tozeroy" if col == "Sparplan-Kompass" else None,
                        fillcolor="rgba(37,99,235,.10)",
                        hovertemplate="%{y:.1f} %<extra>" + label + "</extra>")
    _shade_crises(fig, dd.index, crises)
    fig.update_layout(yaxis_title="Abstand zum Höchststand in %", xaxis_title=None,
                      hovermode="x unified")
    return _base(fig, 340)


def purchase_log_bar(log: pd.DataFrame) -> go.Figure:
    long = log.reset_index().melt(id_vars="Datum", var_name="Position", value_name="Sparrate")
    long["Datum"] = long["Datum"].dt.strftime("%m/%Y")
    fig = px.bar(long, x="Datum", y="Sparrate", color="Position", color_discrete_sequence=PALETTE)
    fig.update_traces(hovertemplate="%{x}: %{y:.0f} €<extra>%{fullData.name}</extra>")
    fig.update_layout(yaxis_title="Sparrate in € je Monat", xaxis_title="Neuberechnung am",
                      barmode="stack")
    return _base(fig, 340)


def crisis_compare_bar(table: pd.DataFrame, market_label: str) -> go.Figure:
    """Größter Einbruch je Krise aus dem Backtest: Kompass, ohne Kompass, Markt."""
    fig = go.Figure()
    for col in [c for c in ["Markt", "Einfach halten", "Sparplan-Kompass"] if c in table.columns]:
        label = market_label if col == "Markt" else col
        fig.add_bar(x=table["Krise"], y=table[col], name=label,
                    marker_color=_COMPARE_STYLE[col]["color"], marker_cornerradius=6,
                    hovertemplate="%{x}: %{y:.1f} %<extra>" + label + "</extra>")
    fig.update_layout(barmode="group", yaxis_title="Größter Absturz in %", xaxis_title=None)
    return _base(fig, 380)


def strategy_scatter(stats: pd.DataFrame) -> go.Figure:
    """Rendite gegen größten Einbruch: oben rechts ist besser."""
    colors = {"Einfach halten": C_NOW, "Markt": C_ALT}
    fig = go.Figure()
    for name, row in stats.iterrows():
        is_ref = name in colors or name.startswith("Markt")
        color = C_ALT if name.startswith("Markt") else colors.get(name, C_GOAL)
        fig.add_scatter(x=[row["Größter Einbruch %"]], y=[row["Rendite p.a. %"]], mode="markers+text",
                        name=name, text=[name], textposition="top center",
                        marker=dict(size=18 if not is_ref else 14, color=color,
                                    symbol="diamond" if is_ref else "circle",
                                    line=dict(color="#fff", width=2)),
                        hovertemplate=(f"{name}<br>Rendite %{{y:.1f}} % p.a.<br>"
                                       "Größter Einbruch %{x:.1f} %<extra></extra>"))
    fig.update_layout(showlegend=False, xaxis_title="Größter Einbruch in % (weiter rechts = weniger Absturz)",
                      yaxis_title="Rendite p.a. in %")
    return _base(fig, 420)


# --------------------------------------------------------------------------- Kernbotschaft
def money_and_calm(values: pd.DataFrame, rets: pd.DataFrame, market_label: str,
                   top_title: str, bottom_title: str,
                   crises: dict[str, tuple[str, str]]) -> go.Figure:
    """Oben: Depotwert in € (gleich viel Geld?). Unten: Abstand zum Höchststand (ruhiger?)."""
    from plotly.subplots import make_subplots

    dd = (1.0 + rets).cumprod()
    dd = (dd / dd.cummax() - 1.0) * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        row_heights=[0.56, 0.44], subplot_titles=(top_title, bottom_title))
    fig.add_scatter(x=values.index, y=values["Eingezahlt"], name="Eingezahlt", row=1, col=1,
                    line=dict(color="#cbd5e1", width=1.4, dash="dot"),
                    hovertemplate="%{y:,.0f} €<extra>Eingezahlt</extra>")
    if market_label:
        fig.add_scatter(x=values.index, y=values["Markt"], name=market_label, row=1, col=1,
                        line=dict(color=C_ALT, width=1.4), opacity=0.55,
                        hovertemplate="%{y:,.0f} €<extra>" + market_label + "</extra>")
    fig.add_scatter(x=values.index, y=values["Einfach halten"], name="Einfach halten", row=1, col=1,
                    line=dict(color=C_NOW, width=2.6),
                    hovertemplate="%{y:,.0f} €<extra>Ohne Kompass</extra>")
    fig.add_scatter(x=values.index, y=values["Sparplan-Kompass"], name="Sparplan-Kompass", row=1, col=1,
                    line=dict(color=C_GOAL, width=3.2),
                    hovertemplate="%{y:,.0f} €<extra>Sparplan-Kompass</extra>")
    fig.add_scatter(x=dd.index, y=dd["Einfach halten"], showlegend=False, row=2, col=1,
                    line=dict(color=C_NOW, width=1.6), fill="tozeroy", fillcolor="rgba(148,163,184,.30)",
                    hovertemplate="%{y:.1f} %<extra>Ohne Kompass</extra>")
    fig.add_scatter(x=dd.index, y=dd["Sparplan-Kompass"], showlegend=False, row=2, col=1,
                    line=dict(color=C_GOAL, width=2.2), fill="tozeroy", fillcolor="rgba(37,99,235,.18)",
                    hovertemplate="%{y:.1f} %<extra>Sparplan-Kompass</extra>")
    for name, (start, end) in crises.items():
        if pd.Timestamp(end) < values.index[0] or pd.Timestamp(start) > values.index[-1]:
            continue
        fig.add_vrect(x0=max(pd.Timestamp(start), values.index[0]), x1=pd.Timestamp(end),
                      fillcolor="#e11d48", opacity=0.06, line_width=0, row="all", col=1)
    fig.update_yaxes(title_text="Depotwert in €", row=1, col=1)
    fig.update_yaxes(title_text="unter Höchststand in %", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    fig.update_annotations(font=dict(size=15, color=C_INK, family=FONT), xanchor="left", x=0)
    _base(fig, 640)
    fig.update_layout(margin=dict(l=8, r=8, t=70, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0))
    return fig


def calm_bars(stats: pd.DataFrame) -> go.Figure:
    """Ruhe-Bilanz: je kürzer der Balken, desto ruhiger. Kompass gegen ohne Kompass."""
    rows = [("Größter Absturz", "Größter Einbruch %", "%"),
            ("Schlechtestes Jahr", "Schlechtestes Jahr %", "%"),
            ("Schwankung pro Jahr", "Schwankung p.a. %", "%"),
            ("Verlust am Schrecktag", "Schrecktag %", "%"),
            ("Längste Durststrecke", "Längste Durststrecke Monate", "Monate")]
    labels = [r[0] for r in rows]
    fig = go.Figure()
    for name, color in (("Einfach halten", C_NOW), ("Sparplan-Kompass", C_GOAL)):
        vals = [abs(float(stats.loc[name, col])) for _, col, _ in rows]
        text = [f"{v:.0f} Mon." if unit == "Monate" else f"{v:.1f} %".replace(".", ",")
                for v, (_, _, unit) in zip(vals, rows)]
        # Balken relativ zu "ohne Kompass" = 100, damit verschiedene Einheiten vergleichbar sind
        base = [abs(float(stats.loc["Einfach halten", col])) or 1.0 for _, col, _ in rows]
        rel = [v / b * 100 for v, b in zip(vals, base)]
        fig.add_bar(y=labels, x=rel, orientation="h", name=name, marker_color=color,
                    marker_cornerradius=6, text=text, textposition="outside",
                    hovertemplate="%{y}: %{text}<extra>" + name + "</extra>")
    fig.update_layout(barmode="group", xaxis=dict(visible=False, range=[0, 135]),
                      yaxis=dict(autorange="reversed"), bargap=0.3)
    return _base(fig, 380)
