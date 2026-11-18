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
    data = load_prices(["EUNL.DE", "SXRV.DE", "4GLD.DE"], start_year=2014, allow_live=False)
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


def test_walk_forward_is_fair_and_never_sells():
    from core.analytics import compare_stats, underwater, walk_forward
    tickers = ["SXRV.DE", "EUNL.DE", "EUNH.DE", "4GLD.DE"]
    prices = load_prices(tickers, 2014, allow_live=False).prices
    bench = load_prices(["EUNL.DE"], 2014, allow_live=False).prices["EUNL.DE"]
    h = np.array([7000.0, 3000.0, 0.0, 0.0])
    out = walk_forward(prices, bench, h, 500.0, 3, 0.025, 0.02, 0.4, 12)
    assert out is not None
    values, rets, log = out
    # alle drei zahlen gleich viel ein
    months = int(round((values["Eingezahlt"].iloc[-1] - 10000.0) / 500.0))
    assert 35 <= months <= 37
    # der Kompass kauft nur: jede Aufteilung ist >= 0 und verteilt genau die Sparrate
    assert (log.values >= -1e-9).all()
    assert np.allclose(log.sum(axis=1), 500.0)
    stats = compare_stats(values, rets, 0.025)
    assert set(stats.index) == {"Sparplan-Kompass", "Einfach halten", "Markt"}
    assert (underwater(rets).values <= 1e-9).all()
    # zu wenig Historie -> kein Ergebnis statt falscher Zahlen
    assert walk_forward(prices.tail(400), bench, h, 500.0, 3, 0.025, 0.02, 0.4, 12) is None


def test_repair_and_splice():
    from core.data_service import repair_etf, splice
    idx = pd.date_range("2020-01-01", periods=12, freq="B")
    etf = pd.Series([140, 141, 139, 100, 101, 130, 101, 102, 103, 104, 105, 106.0], index=idx)
    fixed = repair_etf(etf)
    assert fixed.index[0] == idx[3]                 # Niveausprung 139 -> 100: alles davor verworfen
    assert fixed.loc[idx[5]] == 101                 # Ausreißer 130 ersetzt
    assert fixed.pct_change().abs().max() < 0.02
    proxy = pd.Series([50.0, 55.0, 60.0, 66.0], index=pd.date_range("2019-12-27", periods=4, freq="B"))
    joined = splice(fixed, proxy.reindex(proxy.index.union(fixed.index[:1])).ffill())
    assert joined.loc[fixed.index[0]] == 100        # ETF-Kurs bleibt unverändert
    assert joined.iloc[0] == pytest.approx(100 * 50 / 66)   # Ersatz-Reihe auf den ETF skaliert
    assert joined.index.is_monotonic_increasing and not joined.index.has_duplicates


def test_long_history_reaches_2000():
    data = load_prices(["SXR8.DE", "EXS1.DE", "EUNL.DE"], 2000, allow_live=False)
    assert data.prices.index[0].year == 2000
    assert data.etf_start["SXR8.DE"].year >= 2010   # davor Ersatz-Reihe
    assert to_returns(data.prices).abs().max().max() < 0.15


def test_mix_respects_all_rules(returns):
    from core.optimizer import BLOCKS, block_weights, mix_weights
    h = np.array([500.0, 6000.0, 500.0, 1000.0])
    lo = diluted_weights(h, 400.0, 24)
    limit = cvar(min_cvar_weights(returns, 0.6, lo), returns) * 1.3
    for block in BLOCKS:
        w = block_weights(block, returns, 0.02, limit, 0.6, lo).weights
        assert w.sum() == pytest.approx(1.0)
        assert (w >= lo - 1e-4).all() and w.max() <= max(0.6, lo.max()) + 1e-4
        assert cvar(w, returns) <= limit * 1.003, block
    mixed = mix_weights(returns, 0.02, limit, 0.6, lo, {"Sharpe": .5, "CVaR": .35, "Varianz": .15})
    assert mixed.status == "ok"
    assert cvar(mixed.weights, returns) <= limit * 1.003          # Mischung hält die Grenze (konvex)
    assert (mixed.weights >= lo - 1e-4).all()
    # Varianz-Baustein hat die kleinste Schwankung, CVaR-Baustein den kleinsten CVaR
    cov = np.cov(returns, rowvar=False)
    ws = {b: block_weights(b, returns, 0.02, limit, 0.6, lo).weights for b in BLOCKS}
    assert min(ws, key=lambda b: ws[b] @ cov @ ws[b]) == "Varianz"
    assert min(ws, key=lambda b: cvar(ws[b], returns)) == "CVaR"


def test_core_position_keeps_its_share():
    h = np.array([6000.0, 3000.0, 1000.0])
    core = np.array([True, False, False])
    lo = diluted_weights(h, 500.0, 36, core)
    assert lo[0] == pytest.approx(0.6)                      # Kern behält 60 %
    assert lo[1] == pytest.approx(3000.0 / (10000.0 + 500.0 * 36))   # Rest wird verwässert


def test_custom_ticker_names():
    from core.data_service import currency_of, display_name, group_of
    assert currency_of("AAPL") == "USD" and currency_of("SAP.DE") == "EUR"
    assert display_name("AAPL") == "AAPL" and group_of("AAPL") == "Eigene Titel"


def test_walk_forward_starts_not_before_benchmark():
    from core.analytics import walk_forward
    prices = load_prices(["SXRV.DE", "EUNH.DE", "4GLD.DE"], 2000, allow_live=False).prices
    bench_data = load_prices(["EUNL.DE"], 2000, allow_live=False)
    since = bench_data.etf_start["EUNL.DE"]
    assert since.year == 2009                                    # MSCI-World-ETF erst seit 2009
    out = walk_forward(prices, bench_data.prices["EUNL.DE"], np.array([5000.0, 0, 0]), 500.0, 25,
                       0.025, 0.03, 0.6, 36, not_before=since)
    assert out is not None and out[0].index[0] > since


def test_dip_buying_redistributes_same_budget():
    from core.analytics import apply_dip, dip_factors
    alloc = np.array([200.0, 200.0, 100.0])                # Aktie, Aktie, Anleihe
    mask = np.array([True, True, False])
    out = apply_dip(alloc, np.array([2.0, 1.0, 2.0]), mask, 500.0)
    assert out.sum() == pytest.approx(500.0)                # gleiche Sparrate
    assert out[0] > 200.0 and out[1] < 200.0 and out[2] < 100.0   # Anleihe bekommt keinen Aufschlag
    assert apply_dip(alloc, np.ones(3), mask, 500.0) == pytest.approx(alloc)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    falling = pd.DataFrame({"A": np.linspace(200, 100, 300)}, index=idx)
    f = dip_factors(falling)
    assert f["A"].iloc[-1] == pytest.approx(2.0)            # unter allen drei Linien
    assert f["A"].iloc[30] == pytest.approx(1.0)            # Linien noch nicht berechenbar


def test_search_kind_erkennt_isin_wkn_und_tippfehler():
    from core.data_service import isin_valid, search_kind
    assert isin_valid("IE00B4L5Y983") and isin_valid("DE000EWG2LD7") and isin_valid("US0378331005")
    assert not isin_valid("IE00B4L5Y984")
    assert search_kind("IE00BKT2S34") == "bad_isin"      # 11 Zeichen
    assert search_kind("A0S9GB") == "wkn" and search_kind("716460") == "wkn"
    assert search_kind("Rheinmetall") == "text"
