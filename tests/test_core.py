import numpy as np
import pandas as pd
import pytest

from core.analytics import blind_test, crisis_table
from core.data_service import clean_prices, load_prices, to_returns
from core.optimizer import (cvar, diluted_weights, earliest_month, min_cvar_weights,
                            portfolio_stats, target_weights)
from core.rebalancer import build_plan, max_deviation, sell_rebalance
from core.tax import TAX_RATE, sell_cost


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(7)
    vols = np.array([0.012, 0.018, 0.006, 0.010])
    means = np.array([0.0004, 0.0006, 0.0001, 0.0003])
    market = rng.standard_t(4, size=(1500, 1)) * 0.6
    noise = rng.standard_normal((1500, 4))
    return means + vols * (0.6 * market + 0.8 * noise)


def test_cvar_is_mean_of_worst_days():
    r = np.arange(-50, 50).reshape(-1, 1) / 1000.0
    assert cvar(np.array([1.0]), r) == pytest.approx(0.048)  # Mittel der 5 schlechtesten Tage


def test_min_cvar_beats_other_portfolios(returns):
    w = min_cvar_weights(returns)
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
    rng = np.random.default_rng(1)
    for _ in range(200):
        other = rng.dirichlet(np.ones(4))
        assert cvar(w, returns) <= cvar(other, returns) + 1e-9


def test_target_respects_limit_and_cap(returns):
    safe = cvar(min_cvar_weights(returns, 0.5), returns)
    limit = safe * 1.4
    t = target_weights(returns, 0.02, limit, 0.5)
    assert t.status == "ok"
    assert t.weights.sum() == pytest.approx(1.0)
    assert t.weights.max() <= 0.5 + 1e-6
    assert cvar(t.weights, returns) <= limit * 1.003
    # mehr Risiko-Budget darf die Sharpe Ratio nicht verschlechtern
    loose = target_weights(returns, 0.02, safe * 3, 0.5)
    assert portfolio_stats(loose.weights, returns, 0.02).sharpe >= \
        portfolio_stats(t.weights, returns, 0.02).sharpe - 1e-6


def test_target_unreachable_returns_safest(returns):
    t = target_weights(returns, 0.02, 1e-6, 1.0)
    assert t.status == "unreachable"
    assert cvar(t.weights, returns) == pytest.approx(t.cvar_min)


def test_no_sell_target_and_plan(returns):
    h = np.array([500.0, 6000.0, 500.0, 1000.0])       # Klumpen in der riskantesten Position
    budget, months = 400.0, 12
    lo = diluted_weights(h, budget, months)
    limit = cvar(min_cvar_weights(returns, 0.5, lo), returns) * 1.1
    t = target_weights(returns, 0.02, limit, 0.5, lo)
    assert t.status == "ok" and (t.weights >= lo - 1e-4).all()
    assert cvar(t.weights, returns) <= limit * 1.003
    plan = build_plan(h, t.weights, budget, months, ["A", "B", "C", "D"])
    assert (plan.monthly >= 0).all() and plan.monthly.sum() == pytest.approx(budget)
    assert max_deviation(plan.holdings.loc[months].values, t.weights) < 2e-3
    assert (plan.holdings.diff().dropna().values >= -1e-9).all()   # nie verkauft


def test_earliest_month(returns):
    h = np.array([500.0, 6000.0, 500.0, 1000.0])
    now = cvar(h / h.sum(), returns)
    assert earliest_month(returns, h, 400.0, now * 1.01, 1.0) == 0
    m = earliest_month(returns, h, 400.0, now * 0.8, 1.0)
    assert m is not None and m > 0
    ok = cvar(min_cvar_weights(returns, 1.0, diluted_weights(h, 400.0, m)), returns)
    before = cvar(min_cvar_weights(returns, 1.0, diluted_weights(h, 400.0, m - 1)), returns)
    assert ok <= now * 0.8 < before
    assert earliest_month(returns, h, 400.0, 1e-6, 1.0) is None


def test_sell_cost():
    h = np.array([6000.0, 2000.0])
    trades = sell_rebalance(h, np.array([0.5, 0.5]))          # verkaufe 2000 von A
    c = sell_cost(trades, h, cost_basis=np.array([3000.0, 2000.0]),
                  teilfreistellung=np.array([30, 30]), allowance=200.0, fee_per_order=1.0)
    assert c.sold == pytest.approx(2000.0)
    assert c.gain == pytest.approx(1000.0)                    # 50 % Gewinnanteil
    assert c.taxable == pytest.approx(500.0)                  # 1000 * 0,7 - 200
    assert c.tax == pytest.approx(500.0 * TAX_RATE)
    assert c.fees == pytest.approx(2.0)


def test_snapshot_and_analytics():
    data = load_prices(["EUNL.DE", "SXRV.DE", "4GLD.DE"], years=15, allow_live=False)
    assert data.source == "snapshot" and not data.missing
    r = to_returns(data.prices)
    assert not r.isna().any().any()
    w = np.array([0.3, 0.6, 0.1])
    crises = crisis_table(r, {"Dein Depot": w}, 10000.0)
    assert "Corona-Crash 2020" in set(crises["Krise"]) and (crises["Einbruch €"] < 0).all()
    out = blind_test(r, w, 0.025, 0.03, 0.6, 3, 10000.0)
    assert out is not None and list(out[0].columns) == ["Dein Depot", "Ziel-Depot", "Gleichgewichtet"]


def test_clean_prices_reports_missing():
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    raw = pd.DataFrame({"A": np.linspace(1, 2, 100)}, index=idx)
    prices, missing = clean_prices(raw, ["A", "B"])
    assert list(prices.columns) == ["A"] and missing == ["B"]
