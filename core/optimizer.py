"""Rechenkern: Risikokennzahlen und Ziel-Depot.

Für jedes Ziel-Depot gelten dieselben drei Regeln:
  1. Der Verlust an einem Schrecktag (95 %-CVaR) bleibt unter der Schmerzgrenze.
  2. Keine Position ist größer als die maximale Positionsgröße.
  3. Mindestgewichte je Position bilden "nichts verkaufen" ab: Was schon im Depot liegt,
     kann nur verwässert, nicht abgebaut werden.

Innerhalb dieser Regeln gibt es vier Bausteine, die sich mischen lassen:
  Rendite  – höchste erwartete Rendite                  (lineares Programm)
  Sharpe   – höchste Rendite pro Schwankung             (SLSQP)
  CVaR     – kleinster Verlust an Schrecktagen          (lineares Programm)
  Varianz  – kleinste Schwankung                        (SLSQP)
Jeder Baustein hält die Regeln ein. Weil alle Regeln konvex sind, hält sie auch jede
Mischung der Bausteine ein.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize

TRADING_DAYS = 252
ALPHA = 0.05        # die 5 % schlechtesten Börsentage
MEAN_SHRINK = 0.5   # vergangene Renditen werden nur zur Hälfte geglaubt


def _tail_size(n_days: int, alpha: float = ALPHA) -> int:
    return max(1, ceil(alpha * n_days))


def cvar(weights: np.ndarray, returns: np.ndarray, alpha: float = ALPHA) -> float:
    """Durchschnittlicher Tagesverlust an den alpha schlechtesten Tagen (positiv = Verlust)."""
    port = returns @ weights
    k = _tail_size(len(port), alpha)
    worst = np.partition(port, k - 1)[:k]
    return float(-worst.mean())


def _cvar_grad(weights: np.ndarray, returns: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    port = returns @ weights
    k = _tail_size(len(port), alpha)
    idx = np.argpartition(port, k - 1)[:k]
    return -returns[idx].mean(axis=0)


def expected_returns(returns: np.ndarray) -> np.ndarray:
    """Erwartete Jahresrenditen: halb Historie, halb "jede Anlage wird für ihr Risiko gleich bezahlt".

    Historische Mittelwerte sind sehr unsicher; allein genutzt jagt die Optimierung den
    Gewinnern der Vergangenheit hinterher. Die zweite Hälfte ist eine Ausgangsannahme mit
    gleicher Sharpe Ratio für alle Positionen: Rendite proportional zur Schwankung. So bleiben
    Aktien ertragreicher als Anleihen, aber kein Baustein wird nur wegen seiner Vergangenheit
    bevorzugt. (Im Backtest mit echten Kursen besser als ein Ausgleich zum einfachen Durchschnitt.)
    """
    mu = returns.mean(axis=0) * TRADING_DAYS
    vol = returns.std(axis=0) * np.sqrt(TRADING_DAYS)
    prior = vol * (mu.sum() / vol.sum()) if vol.sum() > 0 else np.full_like(mu, mu.mean())
    return (1.0 - MEAN_SHRINK) * mu + MEAN_SHRINK * prior


def covariance(returns: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.cov(returns, rowvar=False) * TRADING_DAYS)


@dataclass
class Stats:
    ret: float      # erwartete Rendite p.a.
    vol: float      # Volatilität p.a.
    sharpe: float
    cvar: float     # Tages-CVaR als Anteil des Depots


def portfolio_stats(weights: np.ndarray, returns: np.ndarray, rf: float) -> Stats:
    w = np.asarray(weights, dtype=float)
    ret = float(w @ expected_returns(returns))
    vol = float(np.sqrt(w @ covariance(returns) @ w))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return Stats(ret, vol, sharpe, cvar(w, returns))


def _bounds(n: int, w_max: float, w_min: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    lo = np.zeros(n) if w_min is None else np.clip(np.asarray(w_min, dtype=float), 0.0, 1.0)
    hi = np.maximum(max(w_max, 1.0 / n + 1e-9), lo)  # Bestand über der Klumpengrenze bleibt erlaubt
    return lo, hi


def min_cvar_weights(returns: np.ndarray, w_max: float = 1.0, w_min: np.ndarray | None = None,
                     alpha: float = ALPHA) -> np.ndarray:
    """Depot mit dem kleinstmöglichen CVaR (lineares Programm nach Rockafellar/Uryasev)."""
    T, n = returns.shape
    lo, hi = _bounds(n, w_max, w_min)
    k = _tail_size(T, alpha)
    # Variablen: [w_1..w_n, t, u_1..u_T]; minimiere t + (1/k) * Summe(u)
    c = np.concatenate([np.zeros(n), [1.0], np.full(T, 1.0 / k)])
    # u_j >= -r_j·w - t   <=>   -r_j·w - t - u_j <= 0
    A_ub = sparse.hstack([sparse.csr_matrix(-returns), sparse.csr_matrix(-np.ones((T, 1))),
                          -sparse.identity(T, format="csr")], format="csr")
    A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(T)]).reshape(1, -1)
    bounds = list(zip(lo, hi)) + [(None, None)] + [(0.0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(T), A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if not res.success:
        w = lo + (1.0 - lo.sum()) / n
        return w / w.sum()
    w = np.clip(res.x[:n], lo, hi)
    return w / w.sum()


@dataclass
class Target:
    weights: np.ndarray
    status: str        # "ok" | "unreachable" (Schmerzschwelle tiefer als das Minimum)
    cvar_min: float    # kleinster erreichbarer CVaR unter diesen Nebenbedingungen


def target_weights(returns: np.ndarray, rf: float, cvar_limit: float, w_max: float = 1.0,
                   w_min: np.ndarray | None = None, alpha: float = ALPHA) -> Target:
    """Höchste erwartete Rendite, solange Schmerzschwelle, Klumpengrenze und Mindestgewichte halten.

    Lineares Programm (Rockafellar/Uryasev): exakt lösbar, kein Startwert nötig.
    `rf` wird für die Lösung nicht gebraucht und bleibt nur für eine einheitliche Schnittstelle.
    """
    T, n = returns.shape
    if n == 1:
        return Target(np.array([1.0]), "ok", cvar(np.array([1.0]), returns))
    lo, hi = _bounds(n, w_max, w_min)

    w_safe = min_cvar_weights(returns, w_max, w_min)
    cvar_min = cvar(w_safe, returns)
    if cvar_limit <= cvar_min * 1.001:
        return Target(w_safe, "unreachable", cvar_min)

    k = _tail_size(T, alpha)
    mu = expected_returns(returns)
    # Variablen: [w_1..w_n, t, u_1..u_T]; maximiere mu·w
    c = np.concatenate([-mu, [0.0], np.zeros(T)])
    tail_rows = sparse.hstack([sparse.csr_matrix(-returns), sparse.csr_matrix(-np.ones((T, 1))),
                               -sparse.identity(T, format="csr")], format="csr")
    limit_row = sparse.csr_matrix(np.concatenate([np.zeros(n), [1.0], np.full(T, 1.0 / k)]))
    A_ub = sparse.vstack([tail_rows, limit_row], format="csr")
    b_ub = np.concatenate([np.zeros(T), [cvar_limit]])
    A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(T)]).reshape(1, -1)
    bounds = list(zip(lo, hi)) + [(None, None)] + [(0.0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if not res.success:
        return Target(w_safe, "ok", cvar_min)
    w = np.clip(res.x[:n], lo, hi)
    w = np.where(w < 1e-5, 0.0, w)
    return Target(w / w.sum(), "ok", cvar_min)


def diluted_weights(holdings: np.ndarray, budget: float, months: int,
                    core: np.ndarray | None = None) -> np.ndarray:
    """Mindestgewichte, wenn nie verkauft und `months` Sparraten zugekauft werden.

    Kernpositionen (`core` = True) werden nicht verwässert: Sie behalten mindestens ihren
    heutigen Anteil und bekommen dafür weiter ihren Teil der Sparrate.
    """
    holdings = np.asarray(holdings, dtype=float)
    lo = holdings / (holdings.sum() + budget * months)
    if core is not None and holdings.sum() > 0:
        core = np.asarray(core, dtype=bool)
        lo = np.where(core, holdings / holdings.sum(), lo)
    return lo


def earliest_month(returns: np.ndarray, holdings: np.ndarray, budget: float, cvar_limit: float,
                   w_max: float, max_months: int = 120) -> int | None:
    """Frühester Monat, in dem die Schmerzschwelle ohne Verkäufe erreichbar ist."""
    def reachable(months: int) -> bool:
        lo = diluted_weights(holdings, budget, months)
        return cvar(min_cvar_weights(returns, w_max, lo), returns) <= cvar_limit

    if reachable(0):
        return 0
    if budget <= 0 or not reachable(max_months):
        return None
    low, high = 0, max_months          # low: nicht erreichbar, high: erreichbar
    while high - low > 1:
        mid = (low + high) // 2
        low, high = (low, mid) if reachable(mid) else (mid, high)
    return high


# --------------------------------------------------------------------------- Strategie-Mix
BLOCKS = ("Rendite", "Sharpe", "CVaR", "Varianz")


def _slsqp_under_limit(objective, returns: np.ndarray, cvar_limit: float, lo: np.ndarray,
                       hi: np.ndarray, starts: list[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "ineq", "fun": lambda w: cvar_limit - cvar(w, returns),
             "jac": lambda w: -_cvar_grad(w, returns)}]
    best, best_val = fallback, objective(fallback)
    for w0 in starts:
        res = minimize(objective, w0, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-11})
        w = np.clip(res.x, lo, hi)
        w = w / w.sum()
        if cvar(w, returns) <= cvar_limit * 1.002 and objective(w) < best_val:
            best, best_val = w, objective(w)
    return best


def block_weights(block: str, returns: np.ndarray, rf: float, cvar_limit: float, w_max: float,
                  w_min: np.ndarray | None) -> Target:
    """Ein Baustein des Mix. Alle halten Schmerzgrenze, Positionsgröße und "nichts verkaufen" ein."""
    if block == "Rendite":
        return target_weights(returns, rf, cvar_limit, w_max, w_min)
    n = returns.shape[1]
    lo, hi = _bounds(n, w_max, w_min)
    w_safe = min_cvar_weights(returns, w_max, w_min)
    cvar_min = cvar(w_safe, returns)
    unreachable = cvar_limit <= cvar_min * 1.001
    if block == "CVaR" or n == 1 or unreachable:
        return Target(w_safe, "unreachable" if unreachable else "ok", cvar_min)
    mu, cov = expected_returns(returns), covariance(returns)
    w_ret = target_weights(returns, rf, cvar_limit, w_max, w_min).weights
    starts = [w_safe, 0.5 * w_safe + 0.5 * w_ret, w_ret]
    if block == "Sharpe":
        w = _slsqp_under_limit(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), returns, cvar_limit,
                               lo, hi, starts, w_safe)
    elif block == "Varianz":
        w = _slsqp_under_limit(lambda w: float(w @ cov @ w) * 1e4, returns, cvar_limit,
                               lo, hi, starts, w_safe)
    else:
        raise ValueError(f"Unbekannter Baustein: {block}")
    w = np.where(w < 1e-5, 0.0, w)
    return Target(w / w.sum(), "ok", cvar_min)


def mix_weights(returns: np.ndarray, rf: float, cvar_limit: float, w_max: float = 1.0,
                w_min: np.ndarray | None = None, mix: dict[str, float] | None = None) -> Target:
    """Gewichtete Mischung der Bausteine, z. B. {"Sharpe": .5, "CVaR": .35, "Varianz": .15}."""
    mix = {k: float(v) for k, v in (mix or {"Rendite": 1.0}).items() if v and v > 0}
    if not mix:
        mix = {"Rendite": 1.0}
    total = sum(mix.values())
    parts = {b: block_weights(b, returns, rf, cvar_limit, w_max, w_min) for b in mix}
    w = sum(parts[b].weights * (a / total) for b, a in mix.items())
    status = "unreachable" if any(p.status == "unreachable" for p in parts.values()) else "ok"
    cvar_min = min(p.cvar_min for p in parts.values())
    return Target(w / w.sum(), status, cvar_min)
