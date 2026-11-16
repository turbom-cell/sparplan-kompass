"""Rechenkern: Risikokennzahlen und Ziel-Depot.

Ziel-Depot = höchste Sharpe Ratio unter drei Nebenbedingungen:
  1. Der Verlust an einem Schrecktag (95 %-CVaR) bleibt unter der Schmerzschwelle.
  2. Keine Position ist größer als die Klumpengrenze.
  3. Optional Mindestgewichte je Position. Damit lässt sich "nichts verkaufen"
     abbilden: Was schon im Depot liegt, kann nur verwässert, nicht abgebaut werden.
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
    """Jahresrenditen, zur Hälfte auf den Durchschnitt aller Positionen gezogen.

    Historische Mittelwerte sind sehr unsicher. Ohne diese Dämpfung würde die
    Optimierung alles auf den Gewinner der letzten Jahre setzen.
    """
    mu = returns.mean(axis=0) * TRADING_DAYS
    return (1.0 - MEAN_SHRINK) * mu + MEAN_SHRINK * mu.mean()


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
                   w_min: np.ndarray | None = None) -> Target:
    """Maximiert die Sharpe Ratio unter CVaR-Grenze, Klumpengrenze und Mindestgewichten."""
    T, n = returns.shape
    if n == 1:
        return Target(np.array([1.0]), "ok", cvar(np.array([1.0]), returns))
    lo, hi = _bounds(n, w_max, w_min)

    w_safe = min_cvar_weights(returns, w_max, w_min)
    cvar_min = cvar(w_safe, returns)
    if cvar_limit <= cvar_min * 1.001:
        return Target(w_safe, "unreachable", cvar_min)

    mu, cov = expected_returns(returns), covariance(returns)

    def neg_sharpe(w):
        return -(w @ mu - rf) / np.sqrt(w @ cov @ w)

    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq", "fun": lambda w: cvar_limit - cvar(w, returns),
         "jac": lambda w: -_cvar_grad(w, returns)},
    ]
    spread = np.clip(lo + (1.0 - lo.sum()) / n, lo, hi)
    spread = spread / spread.sum()
    starts = [w_safe, 0.5 * w_safe + 0.5 * spread, 0.85 * w_safe + 0.15 * spread]

    best, best_val = w_safe, neg_sharpe(w_safe)
    for w0 in starts:
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-10})
        w = np.clip(res.x, lo, hi)
        w = w / w.sum()
        feasible = (cvar(w, returns) <= cvar_limit * 1.002
                    and (w >= lo - 1e-4).all() and (w <= hi + 1e-4).all())
        if feasible and neg_sharpe(w) < best_val:
            best, best_val = w, neg_sharpe(w)
    best = np.where(best < 1e-4, 0.0, best)
    return Target(best / best.sum(), "ok", cvar_min)


def diluted_weights(holdings: np.ndarray, budget: float, months: int) -> np.ndarray:
    """Mindestgewichte, wenn nie verkauft und `months` Sparraten zugekauft werden."""
    holdings = np.asarray(holdings, dtype=float)
    return holdings / (holdings.sum() + budget * months)


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
