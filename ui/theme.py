"""Aussehen der App: Farben, CSS und kleine HTML-Bausteine."""
from __future__ import annotations

import html

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

/* keine Seitenleiste, kein Kopfbalken */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { padding-top: 1.4rem; padding-bottom: 5rem; max-width: 1180px; }
h2, h3 { letter-spacing: -0.02em; font-weight: 700 !important; color: #0f172a; }

/* Kopfbereich */
.hero {
  background: radial-gradient(1100px 420px at 88% -30%, rgba(96,165,250,.55), transparent 60%),
              linear-gradient(135deg, #0b1220 0%, #172554 55%, #1d4ed8 130%);
  border-radius: 24px; padding: 40px 42px 34px; color: #fff; margin-bottom: 20px;
  box-shadow: 0 22px 48px -22px rgba(30,58,138,.6);
}
.hero .eyebrow { font-size: .74rem; letter-spacing: .16em; text-transform: uppercase;
  color: #93c5fd; font-weight: 700; }
.hero h1 { font-size: 2.7rem; font-weight: 800; letter-spacing: -0.04em; margin: 8px 0 10px;
  color: #fff; padding: 0; line-height: 1.08; }
.hero p { color: #cbd5e1; font-size: 1.08rem; margin: 0; max-width: 720px; line-height: 1.55; }
.hero p b { color: #fff; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; }
.chip { background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.16);
  border-radius: 999px; padding: 5px 12px; font-size: .8rem; color: #e2e8f0; }
.chip .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.hero-proofs { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 14px; margin-top: 26px; }
.hero-proof { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14);
  border-radius: 16px; padding: 16px 18px; }
.hero-proof .big { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em; color: #fff;
  font-variant-numeric: tabular-nums; }
.hero-proof .small { font-size: .84rem; color: #cbd5e1; margin-top: 2px; line-height: 1.4; }

/* Abschnitts-Überschriften */
.section { margin: 30px 0 12px; }
.section .kicker { font-size: .74rem; letter-spacing: .14em; text-transform: uppercase;
  color: #2563eb; font-weight: 700; }
.section h2 { font-size: 1.65rem; margin: 4px 0 4px; padding: 0; }
.section p { color: #475569; margin: 0; max-width: 760px; }

/* Kennzahl-Karten */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px;
  margin: 6px 0 18px; }
.kpi { background: #fff; border: 1px solid #e6e9f0; border-radius: 16px; padding: 16px 18px;
  box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -16px rgba(15,23,42,.18);
  border-top: 3px solid var(--accent, #2563eb); }
.kpi .label { font-size: .78rem; color: #64748b; font-weight: 500; }
.kpi .value { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.03em; color: #0f172a;
  margin-top: 2px; font-variant-numeric: tabular-nums; }
.kpi .sub { font-size: .8rem; color: #64748b; margin-top: 2px; }
.kpi .sub.good { color: #059669; font-weight: 600; }
.kpi .sub.bad { color: #e11d48; font-weight: 600; }

/* Karten-Raster (Problem, Belege, Schritte) */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }
.card { background: #fff; border: 1px solid #e6e9f0; border-radius: 18px; padding: 20px;
  box-shadow: 0 8px 24px -18px rgba(15,23,42,.2); }
.card .icon { width: 36px; height: 36px; border-radius: 11px; display: flex; align-items: center;
  justify-content: center; font-size: 1.1rem; margin-bottom: 12px; background: #eff6ff; }
.card h4 { margin: 0 0 6px; font-size: 1.02rem; font-weight: 700; color: #0f172a; padding: 0; }
.card p { margin: 0; color: #475569; font-size: .93rem; line-height: 1.55; }
.card .num { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.03em; color: #1d4ed8;
  margin-bottom: 4px; font-variant-numeric: tabular-nums; }
.card .src { margin-top: 10px; font-size: .78rem; color: #94a3b8; }

/* Vergleichstabelle */
.vs { width: 100%; border-collapse: separate; border-spacing: 0; background: #fff;
  border: 1px solid #e6e9f0; border-radius: 18px; overflow: hidden;
  box-shadow: 0 8px 24px -18px rgba(15,23,42,.2); font-size: .93rem; }
.vs th, .vs td { padding: 13px 16px; text-align: left; border-bottom: 1px solid #eef1f6;
  vertical-align: top; }
.vs tr:last-child td { border-bottom: none; }
.vs th { background: #f8fafc; color: #475569; font-weight: 600; font-size: .82rem; }
.vs th.us, .vs td.us { background: #eff6ff; color: #0f172a; }
.vs th.us { color: #1d4ed8; }
.vs td:first-child { color: #334155; font-weight: 600; width: 30%; }
.yes { color: #059669; font-weight: 700; }
.no { color: #e11d48; font-weight: 700; }
.mid { color: #b45309; font-weight: 700; }

/* Hervorgehobener Kernsatz */
.callout { background: linear-gradient(135deg, #eff6ff, #f5f3ff); border: 1px solid #dbeafe;
  border-radius: 16px; padding: 16px 20px; font-size: 1.04rem; line-height: 1.55; color: #0f172a;
  margin: 6px 0 16px; }
.callout b { color: #1d4ed8; }

/* Eingabe-Bereich */
.panel-title { font-size: .74rem; letter-spacing: .14em; text-transform: uppercase; color: #2563eb;
  font-weight: 700; margin: 4px 0 2px; }

/* Streamlit-Elemente im selben Stil */
[data-testid="stMetric"] { background: #fff; border: 1px solid #e6e9f0; border-radius: 16px;
  padding: 14px 18px; box-shadow: 0 8px 24px -18px rgba(15,23,42,.2); }
[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #e6e9f0; border-radius: 18px;
  padding: 10px 12px 4px; box-shadow: 0 8px 24px -18px rgba(15,23,42,.2); }

/* Tabs als Pillen */
[role="tablist"], [data-baseweb="tab-list"] { gap: 6px; background: #e9edf5; padding: 5px;
  border-radius: 14px; width: fit-content; max-width: 100%; border: none !important;
  box-shadow: none !important; }
[data-testid="stTab"], [data-baseweb="tab"] { border-radius: 10px !important;
  padding: 9px 20px !important; height: auto !important; font-weight: 600; color: #475569;
  border: none !important; margin: 0 !important; }
[data-testid="stTab"] p, [data-baseweb="tab"] p { font-weight: 600; font-size: .95rem; }
[data-testid="stTab"][aria-selected="true"], [data-baseweb="tab"][aria-selected="true"] {
  background: #fff; color: #0f172a; box-shadow: 0 1px 3px rgba(15,23,42,.14); }
[data-testid="stTab"][aria-selected="true"] p { color: #1d4ed8; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"],
[data-testid="stTabs"] [role="presentation"], [data-testid="stTabs"] [class*="SelectionIndicator"] {
  display: none !important; }
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def esc(text: str) -> str:
    return html.escape(str(text))


def hero(title: str, text: str, chips: list[tuple[str, str]],
         proofs: list[tuple[str, str]] | None = None, eyebrow: str = "") -> None:
    """Dunkler Kopfbereich. chips = [(Punktfarbe oder "", Text)], proofs = [(Zahl, Erklärung)]."""
    chip_html = "".join(
        f'<span class="chip">{f"<span class=dot style=background:{c}></span>" if c else ""}{esc(t)}</span>'
        for c, t in chips)
    proof_html = ""
    if proofs:
        proof_html = '<div class="hero-proofs">' + "".join(
            f'<div class="hero-proof"><div class="big">{esc(b)}</div><div class="small">{s}</div></div>'
            for b, s in proofs) + "</div>"
    st.markdown(f'<div class="hero"><div class="eyebrow">{esc(eyebrow)}</div>'
                f'<h1>{title}</h1><p>{text}</p><div class="chips">{chip_html}</div>{proof_html}</div>',
                unsafe_allow_html=True)


def section(kicker: str, title: str, text: str = "") -> None:
    p = f"<p>{text}</p>" if text else ""
    st.markdown(f'<div class="section"><div class="kicker">{esc(kicker)}</div><h2>{title}</h2>{p}</div>',
                unsafe_allow_html=True)


def kpi_row(items: list[dict]) -> None:
    """items: label, value, sub (optional), tone ("good"/"bad"/""), accent (Farbe)."""
    out = []
    for it in items:
        sub = f'<div class="sub {it.get("tone", "")}">{it["sub"]}</div>' if it.get("sub") else ""
        out.append(f'<div class="kpi" style="--accent:{it.get("accent", BLUE)}">'
                   f'<div class="label">{it["label"]}</div>'
                   f'<div class="value">{it["value"]}</div>{sub}</div>')
    st.markdown(f'<div class="kpis">{"".join(out)}</div>', unsafe_allow_html=True)


def cards(items: list[dict]) -> None:
    """items: title, text, icon (optional), num (optional), src (optional)."""
    out = []
    for it in items:
        icon = f'<div class="icon">{it["icon"]}</div>' if it.get("icon") else ""
        num = f'<div class="num">{it["num"]}</div>' if it.get("num") else ""
        src = f'<div class="src">{it["src"]}</div>' if it.get("src") else ""
        out.append(f'<div class="card">{icon}{num}<h4>{it["title"]}</h4><p>{it["text"]}</p>{src}</div>')
    st.markdown(f'<div class="cards">{"".join(out)}</div>', unsafe_allow_html=True)


def compare_table(columns: list[str], rows: list[list[str]], highlight: int) -> None:
    """Vergleichstabelle; Spalte `highlight` (ab 0, ohne Kriterienspalte) wird hervorgehoben."""
    head = "<th></th>" + "".join(
        f'<th class="{"us" if i == highlight else ""}">{c}</th>' for i, c in enumerate(columns))
    body = ""
    for row in rows:
        cells = f"<td>{row[0]}</td>" + "".join(
            f'<td class="{"us" if i == highlight else ""}">{c}</td>' for i, c in enumerate(row[1:]))
        body += f"<tr>{cells}</tr>"
    st.markdown(f'<table class="vs"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>',
                unsafe_allow_html=True)


def callout(html_text: str) -> None:
    st.markdown(f'<div class="callout">{html_text}</div>', unsafe_allow_html=True)


def panel_title(text: str) -> None:
    st.markdown(f'<div class="panel-title">{esc(text)}</div>', unsafe_allow_html=True)
