"""
Unit test suite for Modulo A (src/modulo_a_market.py).
Tests devigging normalization, NaN resilience, favorite-longshot bias properties,
probabilistic metrics, and Exchange break-even mathematical invariants.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modulo_a_market import (
    closing_line_value,
    compute_brier_score,
    compute_calibration_table,
    compute_log_loss,
    devig_multiplicative,
    devig_power,
    devig_shin,
    expected_value_net,
    min_breakeven_odds,
)


@pytest.mark.parametrize("method_fn", [devig_multiplicative, devig_power, devig_shin])
def test_devig_probabilities_sum_to_one_1x2(method_fn):
    """Verify sum of devigged probabilities == 1.0 (within 1e-9) on 1D and 2D 1X2 odds."""
    odds_1d = np.array([2.10, 3.40, 3.60])
    p_1d = method_fn(odds_1d)
    assert p_1d.shape == (3,)
    assert np.all(p_1d > 0.0) and np.all(p_1d < 1.0)
    assert abs(np.sum(p_1d) - 1.0) < 1e-9

    odds_2d = np.array(
        [
            [1.25, 6.00, 12.00],
            [2.50, 3.25, 2.95],
            [4.50, 3.75, 1.80],
            [1.12, 9.50, 23.00],
        ]
    )
    p_2d = method_fn(odds_2d)
    assert p_2d.shape == (4, 3)
    np.testing.assert_allclose(np.sum(p_2d, axis=1), np.ones(4), rtol=0, atol=1e-9)


@pytest.mark.parametrize("method_fn", [devig_multiplicative, devig_power, devig_shin])
def test_devig_probabilities_sum_to_one_ou25(method_fn):
    """Verify sum of devigged probabilities == 1.0 (within 1e-9) on 1D and 2D O/U 2.5 odds."""
    odds_1d = np.array([1.85, 2.02])
    p_1d = method_fn(odds_1d)
    assert p_1d.shape == (2,)
    assert abs(np.sum(p_1d) - 1.0) < 1e-9

    odds_2d = np.array(
        [
            [1.55, 2.50],
            [2.15, 1.74],
            [1.92, 1.96],
        ]
    )
    p_2d = method_fn(odds_2d)
    assert p_2d.shape == (3, 2)
    np.testing.assert_allclose(np.sum(p_2d, axis=1), np.ones(3), rtol=0, atol=1e-9)


@pytest.mark.parametrize("method_fn", [devig_multiplicative, devig_power, devig_shin])
def test_devig_nan_and_invalid_odds_handling(method_fn):
    """Verify proper handling of NaN or impossible (<= 1.01) odds without crashing."""
    odds_1d_nan = np.array([1.90, np.nan, 4.00])
    p_1d = method_fn(odds_1d_nan)
    assert np.all(np.isnan(p_1d))

    odds_2d_mixed = np.array(
        [
            [2.00, 3.40, 3.90],
            [np.nan, 3.20, 2.50],
            [1.90, 0.00, 4.20],
            [1.80, 3.60, 4.60],
        ]
    )
    p_2d = method_fn(odds_2d_mixed)
    assert p_2d.shape == (4, 3)
    assert abs(np.sum(p_2d[0]) - 1.0) < 1e-9
    assert np.all(np.isnan(p_2d[1]))
    assert np.all(np.isnan(p_2d[2]))
    assert abs(np.sum(p_2d[3]) - 1.0) < 1e-9


def test_favorite_longshot_bias_property():
    """
    On unbalanced odds (e.g. [1.25, 6.00, 12.00]), Power and Shin must assign
    strictly lower probability to the longshot (and higher to the strong favorite)
    compared to the Multiplicative method.
    """
    odds = np.array([1.25, 6.00, 12.00])
    p_mult = devig_multiplicative(odds)
    p_pow = devig_power(odds)
    p_shin = devig_shin(odds)

    # Longshot (index 2, odd = 12.00)
    assert p_pow[2] < p_mult[2], f"Expected Power longshot prob ({p_pow[2]}) < Multiplicative ({p_mult[2]})"
    assert p_shin[2] < p_mult[2], f"Expected Shin longshot prob ({p_shin[2]}) < Multiplicative ({p_mult[2]})"

    # Strong favorite (index 0, odd = 1.25)
    assert p_pow[0] > p_mult[0], f"Expected Power favorite prob ({p_pow[0]}) > Multiplicative ({p_mult[0]})"
    assert p_shin[0] > p_mult[0], f"Expected Shin favorite prob ({p_shin[0]}) > Multiplicative ({p_mult[0]})"


def test_exchange_breakeven_and_ev_invariants():
    """
    Verify mathematical invariants for EV, Q_min, and CLV:
    With p = 0.50 and c = 0.05, Q_min == 2.0526315789... and EV(p=0.50, Q=Q_min, c=0.05) == 0.
    """
    p = 0.50
    c = 0.05
    q_min = min_breakeven_odds(p, commission=c)
    assert abs(q_min - 2.0526315789473686) < 1e-9
    assert round(q_min, 5) == 2.05263

    ev_at_breakeven = expected_value_net(p=p, odds=q_min, commission=c)
    assert abs(ev_at_breakeven) < 1e-12

    # Array vectorized check across multiple probabilities
    p_vec = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    for comm in [0.0, 0.045, 0.05]:
        q_min_vec = min_breakeven_odds(p_vec, commission=comm)
        ev_vec = expected_value_net(p_vec, q_min_vec, commission=comm)
        np.testing.assert_allclose(ev_vec, np.zeros_like(p_vec), atol=1e-12)
        assert np.all(expected_value_net(p_vec, q_min_vec + 0.10, commission=comm) > 0.0)
        assert np.all(expected_value_net(p_vec, q_min_vec - 0.05, commission=comm) < 0.0)

    # CLV invariant: when odds_taken == 1 / p_fair_closing, CLV == 0
    assert abs(closing_line_value(2.50, 0.40)) < 1e-12
    assert abs(closing_line_value(2.60, 0.40) - 0.04) < 1e-12


def test_probabilistic_metrics_and_calibration():
    """Verify Log-Loss, Brier Score, and calibration table calculations."""
    probs_1x2 = np.array(
        [
            [0.60, 0.25, 0.15],
            [0.20, 0.30, 0.50],
        ]
    )
    outcomes_1x2 = ["H", "A"]
    ll = compute_log_loss(probs_1x2, outcomes_1x2)
    expected_ll = -0.5 * (np.log(0.60) + np.log(0.50))
    assert abs(ll - expected_ll) < 1e-12

    brier = compute_brier_score(probs_1x2, outcomes_1x2)
    expected_brier = 0.5 * (((0.60 - 1.0) ** 2 + 0.25**2 + 0.15**2) + (0.20**2 + 0.30**2 + (0.50 - 1.0) ** 2))
    assert abs(brier - expected_brier) < 1e-12

    cal = compute_calibration_table(probs_1x2, outcomes_1x2)
    assert cal["Count"].sum() == 6
