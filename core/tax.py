"""Vereinfachte deutsche Steuerrechnung für den Verkauf von Fondsanteilen.

Berücksichtigt: Abgeltungsteuer inkl. Soli (26,375 %), Teilfreistellung,
Sparerpauschbetrag und Ordergebühren. Nicht berücksichtigt: Kirchensteuer,
Vorabpauschale, FIFO-Reihenfolge einzelner Käufe, Sonderfall Xetra-Gold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TAX_RATE = 0.26375


@dataclass
class SellCost:
    sold: float            # Verkaufsvolumen in Euro
    gain: float            # realisierter Kursgewinn
    taxable: float         # nach Teilfreistellung und Freibetrag
    tax: float
    fees: float

    @property
    def total(self) -> float:
        return self.tax + self.fees


def sell_cost(trades: np.ndarray, holdings: np.ndarray, cost_basis: np.ndarray,
              teilfreistellung: np.ndarray, allowance: float, fee_per_order: float) -> SellCost:
    """Kosten, wenn die negativen Einträge in `trades` sofort verkauft werden."""
    trades = np.asarray(trades, dtype=float)
    holdings = np.asarray(holdings, dtype=float)
    sells = np.where(trades < -0.5, -trades, 0.0)
    safe_h = np.where(holdings > 0, holdings, 1.0)
    gain_share = np.where(holdings > 0, 1.0 - np.asarray(cost_basis, dtype=float) / safe_h, 0.0)
    gains = sells * gain_share
    after_tf = gains * (1.0 - np.asarray(teilfreistellung, dtype=float) / 100.0)
    taxable = max(0.0, float(after_tf.sum()) - allowance)
    n_orders = int((np.abs(trades) > 0.5).sum())
    return SellCost(float(sells.sum()), float(gains.sum()), taxable, taxable * TAX_RATE,
                    n_orders * fee_per_order)
