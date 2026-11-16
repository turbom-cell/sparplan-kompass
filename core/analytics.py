"""Stresstest in historischen Krisen und Blindtest (Out-of-Sample-Backtest)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .optimizer import target_weights

CRISES: dict[str, tuple[str, str]] = {
    "Dotcom-Crash": ("2000-09-04", "2003-03-12"),
    "Finanzkrise": ("2007-10-09", "2009-03-09"),
    "Euro-Krise 2011": ("2011-05-02", "2011-10-04"),
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


# --------------------------------------------------------------------------- Marktvergleich
def walk_forward(prices: pd.DataFrame, bench: pd.Series, start_holdings: np.ndarray, budget: float,
                 years: int, rf: float, limit_pct: float, w_max: float, horizon: int,
                 reopt_months: int = 6, min_train_days: int = 500):
    """Spielt die letzten `years` Jahre mit echten Kursen nach, Monat für Monat.

    Drei Anleger starten mit demselben Geld und zahlen dieselbe Sparrate ein:
      - "Sparplan-Kompass": rechnet alle `reopt_months` Monate neu, nur mit Kursen,
        die es an dem Tag schon gab, und verteilt die Sparrate entsprechend. Verkauft nie.
      - "Weiter wie bisher": Sparrate nach den Start-Gewichten des Depots.
      - "Markt": alles in den Vergleichs-ETF.
    Rückgabe: (Depotwerte in €, Tagesrenditen ohne Einzahlungseffekt, Kauf-Protokoll) oder None.
    """
    from .optimizer import diluted_weights
    from .rebalancer import build_plan

    common = prices.index.intersection(bench.index)
    prices, bench = prices.loc[common], bench.loc[common]
    returns = prices.pct_change().dropna()
    bench_ret = bench.pct_change().dropna()
    start = returns.index[-1] - pd.DateOffset(years=years)
    sim_days = returns.index[returns.index > start]
    if (returns.index <= start).sum() < min_train_days or len(sim_days) < 120:
        return None

    names = list(prices.columns)
    h0 = np.asarray(start_holdings, dtype=float)
    v0 = h0.sum()
    static_alloc = h0 / v0 * budget
    h_kompass, h_static, v_market, paid = h0.copy(), h0.copy(), v0, v0
    alloc = static_alloc.copy()
    month_count, last_month = 0, None
    values, rets, log = [], [], []

    for day in sim_days:
        r = returns.loc[day].to_numpy()
        day_ret = [float(h_kompass @ r / h_kompass.sum()), float(h_static @ r / h_static.sum()),
                   float(bench_ret.loc[day])]
        h_kompass = h_kompass * (1.0 + r)
        h_static = h_static * (1.0 + r)
        v_market *= 1.0 + day_ret[2]

        if last_month is None or day.month != last_month:      # erster Börsentag des Monats
            if month_count % reopt_months == 0:
                train = returns.loc[:day].iloc[:-1].to_numpy()
                lo = diluted_weights(h_kompass, budget, horizon)
                target = target_weights(train, rf, limit_pct, w_max, lo).weights
                alloc = build_plan(h_kompass, target, budget, horizon, names).monthly.to_numpy()
                log.append({"Datum": day, **{n: a for n, a in zip(names, alloc)}})
            h_kompass = h_kompass + alloc
            h_static = h_static + static_alloc
            v_market += budget
            paid += budget
            month_count += 1
            last_month = day.month

        values.append([h_kompass.sum(), h_static.sum(), v_market, paid])
        rets.append(day_ret)

    cols = ["Sparplan-Kompass", "Weiter wie bisher", "Markt"]
    values_df = pd.DataFrame(values, index=sim_days, columns=cols + ["Eingezahlt"])
    rets_df = pd.DataFrame(rets, index=sim_days, columns=cols)
    return values_df, rets_df, pd.DataFrame(log).set_index("Datum")


def underwater(rets: pd.DataFrame) -> pd.DataFrame:
    """Abstand zum bisherigen Höchststand in % (ohne Einzahlungseffekt)."""
    index = (1.0 + rets).cumprod()
    return (index / index.cummax() - 1.0) * 100


def compare_stats(values: pd.DataFrame, rets: pd.DataFrame, rf: float) -> pd.DataFrame:
    rows = []
    paid = float(values["Eingezahlt"].iloc[-1])
    dd = underwater(rets)
    for name in rets.columns:
        daily = rets[name]
        k = max(1, int(np.ceil(0.05 * len(daily))))
        years = len(daily) / 252
        twr = float((1.0 + daily).prod() ** (1 / years) - 1)
        vol = float(daily.std() * np.sqrt(252))
        rows.append({"Depot": name, "Endwert €": float(values[name].iloc[-1]),
                     "Gewinn €": float(values[name].iloc[-1]) - paid,
                     "Rendite p.a. %": twr * 100, "Schwankung p.a. %": vol * 100,
                     "Sharpe": (twr - rf) / vol if vol > 0 else 0.0,
                     "Schrecktag %": float(-daily.nsmallest(k).mean() * 100),
                     "Größter Einbruch %": float(dd[name].min())})
    return pd.DataFrame(rows).set_index("Depot")
