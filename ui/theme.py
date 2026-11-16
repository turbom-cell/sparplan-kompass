"""Aussehen der App: Farben, CSS und kleine HTML-Bausteine (Kopfbereich, Kennzahl-Karten)."""
from __future__ import annotations

import streamlit as st

INK = "#0f172a"
MUTED = "#64748b"
BLUE = "#2563eb"
SLATE = "#94a3b8"
AMBER = "#f59e0b"
RED = "#e11d48"
GREEN = "#059669"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, .stMarkdown, p, li, label, h1, h2, h3, h4, button, input, textarea,
[data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"],
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] div,
[data-testid="stMarkdownContainer"] span, [data-testid="stMetric"] div {
  font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important;
}
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
.block-container { padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1280px; }
[data-testid="stHeader"] { background: transparent; }
h2, h3 { letter-spacing: -0.02em; font-weight: 700 !important; color: #0f172a; }

/* Kopfbereich */
.hero {
  background: radial-gradient(1200px 400px at 85% -40%, rgba(96,165,250,.55), transparent 60%),
              linear-gradient(135deg, #0b1220 0%, #172554 55%, #1d4ed8 130%);
  border-radius: 22px; padding: 30px 34px 26px; color: #fff; margin-bottom: 18px;
  box-shadow: 0 18px 40px -18px rgba(30,58,138,.55);
}
.hero .eyebrow { font-size: .74rem; letter-spacing: .14em; text-transform: uppercase;
  color: #93c5fd; font-weight: 600; }
.hero h1 { font-size: 2.35rem; font-weight: 800; letter-spacing: -0.035em; margin: 6px 0 8px;
  color: #fff; padding: 0; line-height: 1.1; }
.hero p { color: #cbd5e1; font-size: 1.02rem; margin: 0; max-width: 760px; line-height: 1.5; }
.hero p b { color: #fff; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.chip { background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.16);
  border-radius: 999px; padding: 5px 12px; font-size: .8rem; color: #e2e8f0; }
.chip .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }

/* Kennzahl-Karten */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px;
  margin: 4px 0 18px; }
.kpi { background: #fff; border: 1px solid #e6e9f0; border-radius: 16px; padding: 16px 18px;
  box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -16px rgba(15,23,42,.18);
  border-top: 3px solid var(--accent, #2563eb); }
.kpi .label { font-size: .78rem; color: #64748b; font-weight: 500; }
.kpi .value { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.03em; color: #0f172a;
  margin-top: 2px; font-variant-numeric: tabular-nums; }
.kpi .sub { font-size: .8rem; color: #64748b; margin-top: 2px; }
.kpi .sub.good { color: #059669; font-weight: 600; }
.kpi .sub.bad { color: #e11d48; font-weight: 600; }

/* Streamlit-eigene Kennzahlen im selben Stil */
[data-testid="stMetric"] { background: #fff; border: 1px solid #e6e9f0; border-radius: 16px;
  padding: 14px 18px; box-shadow: 0 8px 24px -18px rgba(15,23,42,.2); }
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.03em; }

/* Grafiken und Tabellen als Karten */
[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #e6e9f0; border-radius: 18px;
  padding: 10px 12px 4px; box-shadow: 0 8px 24px -18px rgba(15,23,42,.2); }

/* Tabs als Pillen (alte und neue Streamlit-Struktur) */
[role="tablist"], [data-baseweb="tab-list"] { gap: 6px; background: #e9edf5; padding: 5px;
  border-radius: 14px; width: fit-content; max-width: 100%; border: none !important;
  box-shadow: none !important; }
[data-testid="stTab"], [data-baseweb="tab"] { border-radius: 10px !important;
  padding: 8px 16px !important; height: auto !important; font-weight: 600; color: #475569;
  border: none !important; margin: 0 !important; }
[data-testid="stTab"] p, [data-baseweb="tab"] p { font-weight: 600; font-size: .93rem; }
[data-testid="stTab"][aria-selected="true"], [data-baseweb="tab"][aria-selected="true"] {
  background: #fff; color: #0f172a; box-shadow: 0 1px 3px rgba(15,23,42,.14); }
[data-testid="stTab"][aria-selected="true"] p { color: #1d4ed8; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"],
[data-testid="stTabs"] [role="presentation"], [data-testid="stTabs"] [class*="SelectionIndicator"] {
  display: none !important; }

/* Seitenleiste */
[data-testid="stSidebar"] { border-right: 1px solid #e6e9f0; }
[data-testid="stSidebar"] h2 { font-size: .78rem !important; letter-spacing: .12em;
  text-transform: uppercase; color: #2563eb; font-weight: 700 !important; margin-top: .6rem; }

/* Hervorgehobener Kernsatz */
.callout { background: linear-gradient(135deg, #eff6ff, #f5f3ff); border: 1px solid #dbeafe;
  border-radius: 16px; padding: 16px 20px; font-size: 1.05rem; line-height: 1.55; color: #0f172a;
  margin: 6px 0 16px; }
.callout b { color: #1d4ed8; }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }
.step { background: #fff; border: 1px solid #e6e9f0; border-radius: 16px; padding: 18px; }
.step .n { width: 30px; height: 30px; border-radius: 10px; background: #2563eb; color: #fff;
  font-weight: 700; display: flex; align-items: center; justify-content: center; margin-bottom: 10px; }
.step h4 { margin: 0 0 4px; font-size: 1rem; font-weight: 700; color: #0f172a; padding: 0; }
.step p { margin: 0; color: #475569; font-size: .92rem; line-height: 1.5; }
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, text: str, chips: list[tuple[str, str]]) -> None:
    """Dunkler Kopfbereich. chips = [(Punktfarbe oder "", Text)]."""
    chip_html = "".join(
        f'<span class="chip">{f"<span class=dot style=background:{c}></span>" if c else ""}{t}</span>'
        for c, t in chips)
    st.markdown(f'<div class="hero"><div class="eyebrow">ETF-Sparplan · Risiko · Steuern</div>'
                f'<h1>{title}</h1><p>{text}</p><div class="chips">{chip_html}</div></div>',
                unsafe_allow_html=True)


def kpi_row(items: list[dict]) -> None:
    """items: label, value, sub (optional), tone ("good"/"bad"/""), accent (Farbe)."""
    cards = []
    for it in items:
        sub = f'<div class="sub {it.get("tone", "")}">{it["sub"]}</div>' if it.get("sub") else ""
        cards.append(f'<div class="kpi" style="--accent:{it.get("accent", BLUE)}">'
                     f'<div class="label">{it["label"]}</div>'
                     f'<div class="value">{it["value"]}</div>{sub}</div>')
    st.markdown(f'<div class="kpis">{"".join(cards)}</div>', unsafe_allow_html=True)


def callout(html: str) -> None:
    st.markdown(f'<div class="callout">{html}</div>', unsafe_allow_html=True)


def steps(items: list[tuple[str, str]]) -> None:
    cards = "".join(f'<div class="step"><div class="n">{i}</div><h4>{h}</h4><p>{p}</p></div>'
                    for i, (h, p) in enumerate(items, 1))
    st.markdown(f'<div class="steps">{cards}</div>', unsafe_allow_html=True)
