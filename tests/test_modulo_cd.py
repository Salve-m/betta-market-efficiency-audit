"""
Unit and integrity tests for Moduli C & D — Market-Anchored Blending, Walk-Forward Backtester,
Fractional Kelly with Commission, and Monte Carlo Risk Simulation (tests/test_modulo_cd.py).

Verifies:
1. Zero leakage in blending: weights are fitted strictly on Train 1920-2223 (altering Validation 2324-2425
   does not change estimated weights by more than 1e-12).
2. Accounting accuracy of backtester (net profit with 5% commission on wins vs losses, max 1 bet per match/market).
3. Invariants of Kelly with commission (EV <= 0 => f* = 0; f*_capped <= 0.02).
4. Exact reproducibility of Monte Carlo at fixed seed and total absence of 2526 in calculations.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modulo_cd_backtest import (
    fit_1x2_market_blend,
    fit_ou25_market_blend,
    fractional_kelly_capped,
    kelly_fraction_with_commission,
    load_evaluation_dataset,
    run_monte_carlo_risk_simulation,
    run_walk_forward_backtest,
)

LEVEL2A_PREDS_CSV = ROOT_DIR / "data" / "processed" / "level2a_predictions.csv"
LEVEL1_CSV = ROOT_DIR / "data" / "processed" / "level1_top5_1617_2526.csv"


@pytest.fixture(scope="module")
def df_eval() -> pd.DataFrame:
    return load_evaluation_dataset(level2a_preds_path=LEVEL2A_PREDS_CSV, level1_csv_path=LEVEL1_CSV)


def test_zero_leakage_in_market_anchored_blending(df_eval: pd.DataFrame):
    """
    Verify that Market-Anchored Blending weights for both 1X2 and O/U 2.5 are fitted
    strictly on Train (1920-2223): drastically corrupting all Validation (2324-2425) rows
    leaves the estimated weights identical within 1e-12.
    """
    m_1x2_clean = fit_1x2_market_blend(df_eval, constrained=False, use_residual=True)
    m_ou_clean = fit_ou25_market_blend(df_eval, constrained=False)

    df_corrupted = df_eval.copy()
    val_mask = df_corrupted["Season"].isin(["2324", "2425"])
    assert val_mask.sum() > 0

    df_corrupted.loc[val_mask, "FTHG"] = 9
    df_corrupted.loc[val_mask, "FTAG"] = 0
    df_corrupted.loc[val_mask, "FTR"] = "H"
    df_corrupted.loc[val_mask, "P_H"] = 0.98
    df_corrupted.loc[val_mask, "P_D"] = 0.01
    df_corrupted.loc[val_mask, "P_A"] = 0.01
    df_corrupted.loc[val_mask, "P_Over25"] = 0.99

    m_1x2_corr = fit_1x2_market_blend(df_corrupted, constrained=False, use_residual=True)
    m_ou_corr = fit_ou25_market_blend(df_corrupted, constrained=False)

    assert abs(m_1x2_clean.w_mkt - m_1x2_corr.w_mkt) < 1e-12
    assert abs(m_1x2_clean.w_mod - m_1x2_corr.w_mod) < 1e-12
    assert abs(m_1x2_clean.w_res - m_1x2_corr.w_res) < 1e-12

    assert abs(m_ou_clean.w0 - m_ou_corr.w0) < 1e-12
    assert abs(m_ou_clean.w_mkt - m_ou_corr.w_mkt) < 1e-12
    assert abs(m_ou_clean.w_mod - m_ou_corr.w_mod) < 1e-12


def test_backtester_accounting_and_max_one_bet_per_match():
    """
    Verify exact accounting of run_walk_forward_backtest:
    - Net profit on win with c = 5%: (Q - 1) * 0.95
    - Net profit on loss: -1.0
    - At most 1 bet per match/market (picks outcome with highest EV > threshold).
    """
    # Match 0: both H and A have EV > 0, but H has higher EV -> must pick only H (index 0) and win!
    # Match 1: D has EV > 0 -> picks D (index 1) and loses!
    # Match 2: no outcome has EV > 0 -> skipped!
    model_probs = np.array(
        [
            [0.60, 0.15, 0.25],
            [0.30, 0.40, 0.30],
            [0.50, 0.25, 0.25],
        ]
    )
    exec_odds = np.array(
        [
            [2.00, 4.00, 4.50],  # Match 0: EV_5%(H) = 0.6*0.95 - 0.4 = +0.17; EV_5%(A) = 0.25*3.5*0.95 - 0.75 = +0.08125
            [3.00, 3.00, 3.00],  # Match 1: EV_5%(D) = 0.4*2.0*0.95 - 0.6 = +0.16
            [1.90, 3.50, 3.50],  # Match 2: all EV_5% < 0
        ]
    )
    close_odds = np.array(
        [
            [1.80, 3.80, 4.40],
            [2.90, 3.10, 2.90],
            [1.95, 3.50, 3.90],
        ]
    )
    outcomes = ["H", "A", "H"]

    res = run_walk_forward_backtest(
        strategy_name="Synthetic Accounting Test",
        market="1X2",
        model_probs=model_probs,
        exec_odds=exec_odds,
        close_benchmark_odds=close_odds,
        actual_outcomes=outcomes,
        commission=0.05,
        ev_threshold=0.0,
    )

    # Only Match 0 (H, won at Q=2.00) and Match 1 (D, lost at Q=3.00) should be bet
    assert res.n_bets == 2
    assert list(res.bets_df["match_idx"]) == [0, 1]
    assert list(res.bets_df["outcome_idx"]) == [0, 1]

    expected_profit_0 = (2.00 - 1.0) * (1.0 - 0.05)  # +0.95
    expected_profit_1 = -1.0                         # -1.00
    assert abs(res.bets_df.iloc[0]["net_profit"] - expected_profit_0) < 1e-12
    assert abs(res.bets_df.iloc[1]["net_profit"] - expected_profit_1) < 1e-12
    assert abs(res.net_profit - (expected_profit_0 + expected_profit_1)) < 1e-12
    assert abs(res.roi_pct - ((-0.05 / 2.0) * 100.0)) < 1e-12


def test_kelly_with_commission_invariants():
    """
    Verify Kelly invariants:
    - EV <= 0 => f* == 0.0
    - EV > 0 => f* > 0.0
    - Fractional Kelly capped is always in [0.0, 0.02].
    """
    # At p = 0.50, c = 0.05, break-even odds Q_min = 1 + 1/0.95 = 2.0526315789...
    q_breakeven = 1.0 + 1.0 / 0.95
    assert kelly_fraction_with_commission(0.50, q_breakeven, commission=0.05) == 0.0
    assert kelly_fraction_with_commission(0.50, 2.00, commission=0.05) == 0.0
    assert kelly_fraction_with_commission(0.50, 1.80, commission=0.05) == 0.0

    # Positive EV case
    f_pos = kelly_fraction_with_commission(0.55, 2.10, commission=0.05)
    b = (2.10 - 1.0) * 0.95
    expected_f = (0.55 * b - 0.45) / b
    assert abs(f_pos - expected_f) < 1e-12

    # Capped check across grid of probabilities and odds
    p_arr = np.linspace(0.05, 0.95, 50)
    q_arr = np.linspace(1.10, 10.0, 50)
    f_cap = fractional_kelly_capped(p_arr, q_arr, commission=0.05, kelly_mult=0.20, max_cap=0.02)
    assert np.all(f_cap >= 0.0)
    assert np.all(f_cap <= 0.02 + 1e-15)
    # Huge edge must hit the 0.02 cap
    assert abs(fractional_kelly_capped(0.80, 3.00, commission=0.05, kelly_mult=0.20, max_cap=0.02) - 0.02) < 1e-12


def test_monte_carlo_reproducibility_and_2526_absence(df_eval: pd.DataFrame):
    """
    Verify exact reproducibility of Monte Carlo simulation with fixed seed
    and total absence of Blind Test season 2526 in the loaded evaluation dataset.
    """
    assert "2526" not in set(df_eval["Season"].unique()), "CRITICAL: 2526 found in Modulo C/D evaluation dataset!"
    assert "2627" not in set(df_eval["Season"].unique()), "CRITICAL: 2627 found in Modulo C/D evaluation dataset!"

    mc1 = run_monte_carlo_risk_simulation(n_trajectories=1000, n_bets=200, seed=12345)
    mc2 = run_monte_carlo_risk_simulation(n_trajectories=1000, n_bets=200, seed=12345)
    pd.testing.assert_frame_equal(mc1, mc2)

    # Higher net edge must have lower loss probability after 200 bets
    assert mc1.loc[2, "Flat_Loss_Prob_pct"] < mc1.loc[0, "Flat_Loss_Prob_pct"]
