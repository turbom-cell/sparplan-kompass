"""Darstellungsschicht: alle Plotly-Grafiken der App. Keine Rechenlogik."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

C_NOW = "#8b93a1"     # heutiges Depot
C_GOAL = "#2563eb"    # Ziel-Depot
C_BAD = "#dc2626"
C_GOOD = "#16a34a"
C_ALT = "#f59e0b"
PALETTE = px.colors.qualitative.Safe


def _base(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), separators=",.",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      hoverlabel=dict(namelength=-1))
    return fig


def weights_bar(names: list[str], w_now: np.ndarray, w_goal: np.ndarray, cap: float) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=names, y=w_now * 100, name="Heute", marker_color=C_NOW,
                hovertemplate="%{x}: %{y:.1f} %<extra>Heute</extra>")
    fig.add_bar(x=names, y=w_goal * 100, name="Ziel", marker_color=C_GOAL,
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
    fig.add_histogram(x=np.clip(pnl_now, lo, hi), name="Heute", marker_color=C_NOW, opacity=0.65,
                      xbins=bins, hovertemplate="%{x} €: %{y} Tage<extra>Heute</extra>")
    fig.add_histogram(x=np.clip(pnl_goal, lo, hi), name="Ziel", marker_color=C_GOAL, opacity=0.55,
                      xbins=bins, hovertemplate="%{x} €: %{y} Tage<extra>Ziel</extra>")
    fig.add_vline(x=-limit, line_color=C_BAD, line_width=2,
                  annotation_text="Schmerzschwelle", annotation_position="top")
    fig.add_vline(x=-cvar_now, line_dash="dash", line_color=C_NOW)
    fig.add_vline(x=-cvar_goal, line_dash="dash", line_color=C_GOAL)
    fig.update_layout(barmode="overlay", xaxis_title="Gewinn / Verlust an einem Börsentag in €",
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
                    marker=dict(symbol="line-ew-open", size=46, color="#111827", line_width=3))
    fig.update_traces(hovertemplate="%{x}: %{y:.1f} %<extra></extra>", selector=dict(type="bar"))
    fig.update_layout(yaxis_title="Anteil am Depot in %", xaxis_title=None, showlegend=False)
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 350
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 250
    return _base(fig, 460)


def risk_path(months: list[int], loss_eur: list[float], limit: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=months, y=loss_eur, mode="lines+markers", line=dict(color=C_GOAL, width=3),
                    name="Verlust an einem Schrecktag",
                    hovertemplate="Monat %{x}: %{y:.0f} €<extra></extra>")
    fig.add_hline(y=limit, line_color=C_BAD, line_dash="dot",
                  annotation_text="Schmerzschwelle", annotation_position="bottom right")
    fig.update_layout(xaxis_title="Monat", yaxis_title="€ (bezogen auf den heutigen Depotwert)",
                      showlegend=False)
    return _base(fig, 320)


def crisis_bar(table: pd.DataFrame) -> go.Figure:
    fig = px.bar(table, x="Krise", y="Einbruch €", color="Depot", barmode="group",
                 color_discrete_map={"Dein Depot": C_NOW, "Ziel-Depot": C_GOAL},
                 custom_data=["Einbruch %"])
    fig.update_traces(hovertemplate="%{x}: %{y:,.0f} € (%{customdata[0]:.1f} %)<extra></extra>")
    fig.update_layout(xaxis_title=None, yaxis_title="Größter Einbruch in €", legend_title=None)
    return _base(fig)


def blind_test_lines(curves: pd.DataFrame) -> go.Figure:
    colors = {"Dein Depot": C_NOW, "Ziel-Depot": C_GOAL, "Gleichgewichtet": C_ALT}
    fig = go.Figure()
    for name in curves.columns:
        fig.add_scatter(x=curves.index, y=curves[name], name=name,
                        line=dict(color=colors.get(name), width=3 if name == "Ziel-Depot" else 2),
                        hovertemplate="%{x|%d.%m.%Y}: %{y:,.0f} €<extra>" + name + "</extra>")
    fig.update_layout(yaxis_title="Depotwert in €", xaxis_title=None)
    return _base(fig, 420)
