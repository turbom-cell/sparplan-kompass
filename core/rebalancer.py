"""Sparplan-Rebalancing ohne Verkäufe.

Das Ziel-Depot ist so berechnet, dass es allein durch Zukäufe erreichbar ist.
Der Plan verteilt die nötigen Zukäufe gleichmäßig auf die Laufzeit: eine feste
Sparplan-Aufteilung, die man einmal beim Broker einstellt.
Kursänderungen während der Laufzeit werden nicht prognostiziert.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Plan:
    monthly: pd.Series        # Kauf je Position und Monat in Euro
    holdings: pd.DataFrame    # Zeile = Monat (0..N), Bestand in Euro

    @property
    def weights(self) -> pd.DataFrame:
        return self.holdings.div(self.holdings.sum(axis=1), axis=0)


def build_plan(holdings: np.ndarray, target_w: np.ndarray, budget: float, months: int,
               names: list[str]) -> Plan:
    """Feste Monatsaufteilung, die das Depot in `months` Monaten ins Ziel bringt."""
    h = np.asarray(holdings, dtype=float)
    need = np.clip(target_w * (h.sum() + budget * months) - h, 0.0, None)
    monthly = need / need.sum() * budget if need.sum() > 0 else np.asarray(target_w) * budget
    path = [h + monthly * m for m in range(months + 1)]
    holdings_df = pd.DataFrame(path, columns=names, index=range(months + 1))
    holdings_df.index.name = "Monat"
    return Plan(pd.Series(monthly, index=names), holdings_df)


def max_deviation(holdings: np.ndarray, target_w: np.ndarray) -> float:
    total = np.sum(holdings)
    return float(np.abs(np.asarray(holdings) / total - target_w).max()) if total > 0 else 1.0


def sell_rebalance(holdings: np.ndarray, target_w: np.ndarray) -> np.ndarray:
    """Klassisches Umschichten auf einen Schlag: positiv = kaufen, negativ = verkaufen."""
    holdings = np.asarray(holdings, dtype=float)
    return target_w * holdings.sum() - holdings
