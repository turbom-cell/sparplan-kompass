"""Stresstest in historischen Krisen und Blindtest (Out-of-Sample-Backtest)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .optimizer import target_weights

CRISES: dict[str, tuple[str, str]] = {
    "Ausverkauf Ende 2018": ("2018-10-03", "2018-12-27"),
    "Corona-Crash 2020": ("2020-02-19", "2020-03-23"),
    "Zinswende 2022": ("2022-01-03", "2022-10-12"),
    "Zoll-Schock 2025": ("2025-02-19", "2025-04-08"),
}


def drawdown(values: pd.Series) -> float:
    return float((values / values.cummax() - 1.0).min())


def crisis_table(returns: pd.DataFrame, portfolios: dict[str, np.ndarray], depot_value: float) -> pd.DataFrame:
    """Größter Einbruch in Euro je Krise und Depot. Krisen ohne Daten entfallen."""
    rows = []
    for name, (start, end) in CRISES.items():
        window = returns.loc[start:end]
        if len(window) < 10 or returns.index[0] > pd.Timestamp(start):
            continue
        for label, w in portfolios.items():
            curve = (1.0 + window @ w).cumprod()
            curve = pd.concat([pd.Series([1.0]), curve.reset_index(drop=True)], ignore_index=True)
            rows.append({"Krise": name, "Depot": label,
                         "Einbruch %": drawdown(curve) * 100,
                         "Einbruch €": drawdown(curve) * depot_value})
    return pd.DataFrame(rows)


def blind_test(returns: pd.DataFrame, current_w: np.ndarray, rf: float, cvar_limit: float,
               w_max: float, test_years: int, depot_value: float) -> tuple[pd.DataFrame, pd.Timestamp] | None:
    """Ziel-Depot nur aus alten Daten berechnen, dann auf den letzten Jahren prüfen."""
    split = returns.index[-1] - pd.DateOffset(years=test_years)
    train, test = returns.loc[:split], returns.loc[split:].iloc[1:]
    if len(train) < 250 or len(test) < 60:
        return None
    w_target = target_weights(train.values, rf, cvar_limit, w_max).weights
    n = returns.shape[1]
    curves = pd.DataFrame({
        "Dein Depot": depot_value * (1.0 + test @ current_w).cumprod(),
        "Ziel-Depot": depot_value * (1.0 + test @ w_target).cumprod(),
        "Gleichgewichtet": depot_value * (1.0 + test @ np.full(n, 1.0 / n)).cumprod(),
    })
    return curves, split


def curve_stats(curves: pd.DataFrame, rf: float) -> pd.DataFrame:
    rows = []
    for name in curves.columns:
        v = curves[name]
        daily = v.pct_change().dropna()
        years = len(v) / 252
        cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1
        vol = daily.std() * np.sqrt(252)
        k = max(1, int(np.ceil(0.05 * len(daily))))
        rows.append({"Depot": name, "Endwert €": v.iloc[-1], "Rendite p.a. %": cagr * 100,
                     "Schwankung p.a. %": vol * 100,
                     "Sharpe": (cagr - rf) / vol if vol > 0 else 0.0,
                     "Schrecktag %": -daily.nsmallest(k).mean() * 100,
                     "Größter Einbruch %": drawdown(v) * 100})
    return pd.DataFrame(rows).set_index("Depot")
