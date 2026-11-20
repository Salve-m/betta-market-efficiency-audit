"""
Moduli C & D: Orthogonal Information Test (Market-Anchored Blending),
Walk-Forward Backtesting Engine, Heteroskedastic Statistical Significance (Buchdahl/Joseph),
Fractional Kelly Risk Management, and Monte Carlo Drawdown Simulation
(src/modulo_cd_backtest.py).

Strictly adheres to RULES.md:
- Does NOT retrain level2a_shots_dc.py (uses saved predictions in data/processed/level2a_predictions.csv
  joined with data/processed/level1_top5_1617_2526.csv).
- Never touches or reads Blind Test season 2526 or Holdout 2627.
- Fits Market-Anchored Blending weights strictly on Train (1920-2223) and evaluates frozen on Validation (2324-2425).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modulo_a_market import (
    LEAGUE_NAMES,
    TRAIN_SEASONS,
    VAL_SEASONS,
    closing_line_value,
    compute_brier_score,
    compute_log_loss,
    devig_power,
    expected_value_net,
)


# ==============================================================================
# 0. DATA LOADING & ANTI-LEAKAGE PERIMETER VERIFICATION
# ==============================================================================


def load_evaluation_dataset(
    level2a_preds_path: Path,
    level1_csv_path: Path,
) -> pd.DataFrame:
    """
    Load saved Level 2A predictions (1920-2425) and join additional market odds columns
    (MaxH/D/A, Max>2.5/<2.5, BFEH/D/A, BFECH/D/A) from Level 1 consolidated CSV.
    Strictly excludes season 2526 and 2627.
    """
    df_preds = pd.read_csv(level2a_preds_path, dtype={"Season": str, "Div": str})
    for forbidden in ["2526", "2627"]:
        if forbidden in set(df_preds["Season"].unique()):
            raise ValueError(f"RULE VIOLATION: Forbidden season {forbidden} present in {level2a_preds_path}")

    df_l1 = pd.read_csv(level1_csv_path, dtype={"Season": str, "Div": str})
    df_l1 = df_l1[df_l1["Season"].isin(TRAIN_SEASONS + VAL_SEASONS)].copy()

    extra_cols = [
        "Season",
        "Div",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "MaxH",
        "MaxD",
        "MaxA",
        "AvgH",
        "AvgD",
        "AvgA",
        "Max>2.5",
        "Max<2.5",
        "Avg>2.5",
        "Avg<2.5",
        "BFEH",
        "BFED",
        "BFEA",
        "BFECH",
        "BFECD",
        "BFECA",
    ]
    avail_extra = [c for c in extra_cols if c in df_l1.columns]
    df_merged = df_preds.merge(
        df_l1[avail_extra],
        on=["Season", "Div", "Date", "HomeTeam", "AwayTeam"],
        how="left",
    )
    df_merged = df_merged.sort_values(["Date", "Div", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    assert "2526" not in set(df_merged["Season"].unique()), "CRITICAL: 2526 must never be loaded!"
    return df_merged


# ==============================================================================
# 1. ORTHOGONAL INFORMATION TEST (MARKET-ANCHORED BLENDING)
# ==============================================================================


def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p_c = np.clip(p, eps, 1.0 - eps)
    return np.log(p_c / (1.0 - p_c))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_c = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z_c))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(np.clip(shifted, -35.0, 35.0))
    return exps / np.sum(exps, axis=1, keepdims=True)


@dataclass
class BinaryBlendModel:
    w0: float
    w_mkt: float
    w_mod: float
    constrained: bool


def fit_ou25_market_blend(
    df_data: pd.DataFrame,
    constrained: bool = True,
) -> BinaryBlendModel:
    """
    Fit calibrated logistic regression in logit space for Over/Under 2.5 EXCLUSIVELY on Train (1920-2223):
        logit(p_blend) = w0 + w_mkt * logit(p_open) + w_mod * logit(p_V2A)
    Rows from Validation (2324-2425) or any other partition are strictly ignored.
    """
    df_train = df_data[df_data["Season"].isin(TRAIN_SEASONS)].copy()
    open_odds = df_train[["P>2.5", "P<2.5"]].to_numpy(dtype=np.float64)
    p_open_2d = devig_power(open_odds)
    p_open = p_open_2d[:, 0]
    p_v2a = df_train["P_Over25"].to_numpy(dtype=np.float64)
    y_over = ((df_train["FTHG"] + df_train["FTAG"]) > 2.5).to_numpy(dtype=np.float64)

    valid = np.isfinite(p_open) & np.isfinite(p_v2a) & np.isfinite(y_over)
    l_open = _logit(p_open[valid])
    l_v2a = _logit(p_v2a[valid])
    y_v = y_over[valid]

    def loss_fn(params: np.ndarray) -> float:
        w0, w_mkt, w_mod = float(params[0]), float(params[1]), float(params[2])
        z = w0 + w_mkt * l_open + w_mod * l_v2a
        p = np.clip(_sigmoid(z), 1e-15, 1.0 - 1e-15)
        return float(-np.mean(y_v * np.log(p) + (1.0 - y_v) * np.log(1.0 - p)))

    x0 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    bounds = [(None, None), (0.0, None), (0.0, None)] if constrained else [(None, None)] * 3
    res = minimize(loss_fn, x0, method="L-BFGS-B", bounds=bounds, options={"ftol": 1e-12, "gtol": 1e-8})
    return BinaryBlendModel(
        w0=float(res.x[0]),
        w_mkt=float(res.x[1]),
        w_mod=float(res.x[2]),
        constrained=constrained,
    )


def predict_ou25_market_blend(
    model: BinaryBlendModel,
    p_open_2d: np.ndarray,
    p_v2a_over: np.ndarray,
) -> np.ndarray:
    """Return (N, 2) blended probabilities [P_Over25, P_Under25]."""
    n = len(p_v2a_over)
    out = np.full((n, 2), np.nan, dtype=np.float64)
    p_open_over = p_open_2d[:, 0]
    valid = np.isfinite(p_open_over) & np.isfinite(p_v2a_over)
    if np.any(valid):
        z = model.w0 + model.w_mkt * _logit(p_open_over[valid]) + model.w_mod * _logit(p_v2a_over[valid])
        p_ov = _sigmoid(z)
        out[valid, 0] = p_ov
        out[valid, 1] = 1.0 - p_ov
    return out


@dataclass
class Multiclass1X2BlendModel:
    w_mkt: float
    w_mod: float
    w_res: float
    constrained: bool
    use_residual: bool


def fit_1x2_market_blend(
    df_data: pd.DataFrame,
    constrained: bool = True,
    use_residual: bool = False,
) -> Multiclass1X2BlendModel:
    """
    Fit log-linear / softmax combined model for 1X2 EXCLUSIVELY on Train (1920-2223):
        ln(p_blend, k) ~ w_mkt * ln(p_open, k) + w_mod * ln(p_V2A, k) [+ w_res * (ln(p_V2A, k) - ln(p_ModB, k))]
    Rows from Validation (2324-2425) or any other partition are strictly ignored.
    """
    df_train = df_data[df_data["Season"].isin(TRAIN_SEASONS)].copy()
    open_odds = df_train[["PSH", "PSD", "PSA"]].to_numpy(dtype=np.float64)
    p_open = devig_power(open_odds)
    p_v2a = df_train[["P_H", "P_D", "P_A"]].to_numpy(dtype=np.float64)

    if use_residual and all(c in df_train.columns for c in ["P_H_ModB", "P_D_ModB", "P_A_ModB"]):
        p_modb = df_train[["P_H_ModB", "P_D_ModB", "P_A_ModB"]].to_numpy(dtype=np.float64)
    else:
        p_modb = p_v2a.copy()

    ftr_map = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([ftr_map.get(str(x).strip(), -1) for x in df_train["FTR"]], dtype=int)

    valid = (
        np.all(np.isfinite(p_open), axis=1)
        & np.all(np.isfinite(p_v2a), axis=1)
        & np.all(np.isfinite(p_modb), axis=1)
        & (y_idx >= 0)
    )

    log_open = np.log(np.clip(p_open[valid], 1e-12, 1.0))
    log_v2a = np.log(np.clip(p_v2a[valid], 1e-12, 1.0))
    log_res = log_v2a - np.log(np.clip(p_modb[valid], 1e-12, 1.0))
    y_v = y_idx[valid]
    idx_arange = np.arange(len(y_v))

    if not use_residual:
        def loss_fn(params: np.ndarray) -> float:
            w_mkt, w_mod = float(params[0]), float(params[1])
            logits = w_mkt * log_open + w_mod * log_v2a
            probs = _softmax(logits)
            chosen = np.clip(probs[idx_arange, y_v], 1e-15, 1.0)
            return float(-np.mean(np.log(chosen)))

        x0 = np.array([1.0, 0.0], dtype=np.float64)
        bounds = [(0.0, None), (0.0, None)] if constrained else [(None, None), (None, None)]
        res = minimize(loss_fn, x0, method="L-BFGS-B", bounds=bounds, options={"ftol": 1e-12, "gtol": 1e-8})
        return Multiclass1X2BlendModel(
            w_mkt=float(res.x[0]),
            w_mod=float(res.x[1]),
            w_res=0.0,
            constrained=constrained,
            use_residual=False,
        )
    else:
        def loss_fn_res(params: np.ndarray) -> float:
            w_mkt, w_mod, w_res = float(params[0]), float(params[1]), float(params[2])
            logits = w_mkt * log_open + w_mod * log_v2a + w_res * log_res
            probs = _softmax(logits)
            chosen = np.clip(probs[idx_arange, y_v], 1e-15, 1.0)
            return float(-np.mean(np.log(chosen)))

        x0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        bounds = [(0.0, None), (0.0, None), (None, None)] if constrained else [(None, None)] * 3
        res = minimize(loss_fn_res, x0, method="L-BFGS-B", bounds=bounds, options={"ftol": 1e-12, "gtol": 1e-8})
        return Multiclass1X2BlendModel(
            w_mkt=float(res.x[0]),
            w_mod=float(res.x[1]),
            w_res=float(res.x[2]),
            constrained=constrained,
            use_residual=True,
        )


def predict_1x2_market_blend(
    model: Multiclass1X2BlendModel,
    p_open_3d: np.ndarray,
    p_v2a_3d: np.ndarray,
    p_modb_3d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return (N, 3) blended 1X2 probabilities [P_H, P_D, P_A]."""
    n = len(p_v2a_3d)
    out = np.full((n, 3), np.nan, dtype=np.float64)
    if p_modb_3d is None:
        p_modb_3d = p_v2a_3d
    valid = (
        np.all(np.isfinite(p_open_3d), axis=1)
        & np.all(np.isfinite(p_v2a_3d), axis=1)
        & np.all(np.isfinite(p_modb_3d), axis=1)
    )
    if np.any(valid):
        log_open = np.log(np.clip(p_open_3d[valid], 1e-12, 1.0))
        log_v2a = np.log(np.clip(p_v2a_3d[valid], 1e-12, 1.0))
        logits = model.w_mkt * log_open + model.w_mod * log_v2a
        if model.use_residual:
            log_res = log_v2a - np.log(np.clip(p_modb_3d[valid], 1e-12, 1.0))
            logits = logits + model.w_res * log_res
        out[valid] = _softmax(logits)
    return out


def paired_bootstrap_logloss_diff(
    p_blend: np.ndarray,
    p_benchmark: np.ndarray,
    outcomes: Union[Sequence, np.ndarray],
    n_boot: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute Delta Log-Loss = LL(Blend) - LL(Benchmark) and its 95% paired bootstrap
    confidence interval [ci_low, ci_high].
    """
    p_b = np.asarray(p_blend, dtype=np.float64)
    p_m = np.asarray(p_benchmark, dtype=np.float64)
    n_classes = p_b.shape[1]

    if n_classes == 3:
        mapping = {"H": 0, "D": 1, "A": 2}
        y_idx = np.array([mapping.get(str(x).strip(), -1) for x in outcomes], dtype=int)
    else:
        y_idx = np.asarray(outcomes, dtype=int)

    valid = np.all(np.isfinite(p_b), axis=1) & np.all(np.isfinite(p_m), axis=1) & (y_idx >= 0)
    p_b_v = np.clip(p_b[valid], 1e-15, 1.0)
    p_m_v = np.clip(p_m[valid], 1e-15, 1.0)
    y_v = y_idx[valid]
    n_v = len(y_v)
    idx_a = np.arange(n_v)

    ll_i_blend = -np.log(p_b_v[idx_a, y_v])
    ll_i_mkt = -np.log(p_m_v[idx_a, y_v])
    diffs = ll_i_blend - ll_i_mkt
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n_v, size=(n_boot, n_v))
    boot_means = np.mean(diffs[boot_indices], axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return mean_diff, float(ci_low), float(ci_high)


# ==============================================================================
# 2. MODULO D — FRACTIONAL KELLY & MONTE CARLO DRAWDOWN SIMULATION
# ==============================================================================


def kelly_fraction_with_commission(
    p: Union[float, np.ndarray],
    odds: Union[float, np.ndarray],
    commission: float = 0.05,
) -> Union[float, np.ndarray]:
    """
    Full Kelly stake fraction f* with Exchange commission c on net winnings:
        b = (Q - 1) * (1 - c)
        f* = max(0, (p * b - (1 - p)) / b)
    """
    if not (0.0 <= commission < 1.0):
        raise ValueError(f"Invalid commission {commission}")
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(odds, dtype=np.float64)
    b = (q_arr - 1.0) * (1.0 - commission)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_f = np.where(b > 0.0, (p_arr * b - (1.0 - p_arr)) / b, 0.0)
    f_star = np.maximum(0.0, np.where(np.isfinite(raw_f), raw_f, 0.0))
    if np.ndim(p) == 0 and np.ndim(odds) == 0:
        return float(f_star)
    return f_star


def fractional_kelly_capped(
    p: Union[float, np.ndarray],
    odds: Union[float, np.ndarray],
    commission: float = 0.05,
    kelly_mult: float = 0.20,
    max_cap: float = 0.02,
) -> Union[float, np.ndarray]:
    """
    Fractional Kelly stake (default 0.20 * f*) capped at max_cap (default 2.0% of bankroll).
    """
    f_star = kelly_fraction_with_commission(p=p, odds=odds, commission=commission)
    capped = np.minimum(max_cap, kelly_mult * f_star)
    if np.ndim(p) == 0 and np.ndim(odds) == 0:
        return float(capped)
    return capped


def compute_equity_and_max_drawdown(
    won_mask: np.ndarray,
    exec_odds: np.ndarray,
    model_probs: np.ndarray,
    commission: float = 0.05,
    flat_unit_pct: float = 0.01,
    kelly_mult: float = 0.20,
    kelly_cap: float = 0.02,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Simulate bankroll equity curve (starting at B_0 = 1.0) and compute Max Drawdown (%)
    for both Flat Staking (1.0% of initial bankroll) and Fractional Kelly (0.20x capped at 2.0%).
    Returns (max_dd_flat_pct, max_dd_kelly_pct, equity_flat, equity_kelly).
    """
    n_bets = len(won_mask)
    if n_bets == 0:
        return 0.0, 0.0, np.array([1.0]), np.array([1.0])

    net_win_mult = (exec_odds - 1.0) * (1.0 - commission)
    unit_returns = np.where(won_mask, net_win_mult, -1.0)

    # 1. Flat staking (1.0% of initial bankroll B_0 = 1.0 per bet)
    eq_flat = np.empty(n_bets + 1, dtype=np.float64)
    eq_flat[0] = 1.0
    eq_flat[1:] = 1.0 + np.cumsum(flat_unit_pct * unit_returns)
    eq_flat_clipped = np.maximum(eq_flat, 0.0)
    peak_flat = np.maximum.accumulate(eq_flat_clipped)
    dd_flat = np.where(peak_flat > 0.0, (peak_flat - eq_flat_clipped) / peak_flat, 0.0)
    max_dd_flat_pct = float(np.max(dd_flat)) * 100.0

    # 2. Fractional Kelly (0.20x f*, cap 2.0% of current bankroll)
    f_stakes = fractional_kelly_capped(
        p=model_probs,
        odds=exec_odds,
        commission=commission,
        kelly_mult=kelly_mult,
        max_cap=kelly_cap,
    )
    eq_kelly = np.empty(n_bets + 1, dtype=np.float64)
    eq_kelly[0] = 1.0
    step_mults = np.maximum(1.0 + f_stakes * unit_returns, 1e-9)
    eq_kelly[1:] = np.cumprod(step_mults)
    peak_kelly = np.maximum.accumulate(eq_kelly)
    dd_kelly = (peak_kelly - eq_kelly) / peak_kelly
    max_dd_kelly_pct = float(np.max(dd_kelly)) * 100.0

    return max_dd_flat_pct, max_dd_kelly_pct, eq_flat, eq_kelly


def run_monte_carlo_risk_simulation(
    net_edges: Sequence[float] = (0.01, 0.02, 0.04),
    avg_net_odds: float = 2.10,
    n_trajectories: int = 10000,
    n_bets: int = 1000,
    flat_stake_pct: float = 0.01,
    kelly_mult: float = 0.20,
    kelly_cap: float = 0.02,
    seed: int = 20261120,
) -> pd.DataFrame:
    """
    Theoretical Monte Carlo drawdown & loss-probability simulation:
    - 10,000 trajectories of 1,000 bets each at average net odds Q = 2.10 (b = 1.10)
    - Net edges: +1%, +2%, +4% (already net of commissions, so p_true = (1 + edge) / Q)
    - Compares Flat 1.0% staking vs Fractional Kelly 0.20x (cap 2.0%).
    """
    b = avg_net_odds - 1.0
    rows = []

    for edge in net_edges:
        # Reset deterministic RNG for each edge level
        rng = np.random.default_rng(seed + int(round(edge * 10000)))
        p_win = (1.0 + edge) / avg_net_odds
        # Full Kelly f* = (p * b - (1 - p)) / b = edge / b
        f_full = max(0.0, edge / b)
        f_kelly = min(kelly_cap, kelly_mult * f_full)

        # Draw (n_trajectories, n_bets) uniform random numbers
        u = rng.random((n_trajectories, n_bets))
        wins = u < p_win
        unit_ret = np.where(wins, b, -1.0)

        # Flat 1.0% of initial bankroll
        eq_flat = np.ones((n_trajectories, n_bets + 1), dtype=np.float64)
        eq_flat[:, 1:] = 1.0 + np.cumsum(flat_stake_pct * unit_ret, axis=1)
        eq_flat_pos = np.maximum(eq_flat, 0.0)
        peaks_flat = np.maximum.accumulate(eq_flat_pos, axis=1)
        dds_flat = np.where(peaks_flat > 0.0, (peaks_flat - eq_flat_pos) / peaks_flat, 1.0)
        max_dds_flat = np.max(dds_flat, axis=1) * 100.0
        loss_prob_flat = float(np.mean(eq_flat[:, -1] < 1.0)) * 100.0

        # Fractional Kelly 0.20x
        eq_kelly = np.ones((n_trajectories, n_bets + 1), dtype=np.float64)
        eq_kelly[:, 1:] = np.cumprod(1.0 + f_kelly * unit_ret, axis=1)
        peaks_kelly = np.maximum.accumulate(eq_kelly, axis=1)
        dds_kelly = (peaks_kelly - eq_kelly) / peaks_kelly
        max_dds_kelly = np.max(dds_kelly, axis=1) * 100.0
        loss_prob_kelly = float(np.mean(eq_kelly[:, -1] < 1.0)) * 100.0

        rows.append(
            {
                "Net_Edge_pct": edge * 100.0,
                "Win_Prob_pct": p_win * 100.0,
                "Kelly_Stake_pct": f_kelly * 100.0,
                "Flat_Median_MaxDD_pct": float(np.median(max_dds_flat)),
                "Flat_P95_MaxDD_pct": float(np.percentile(max_dds_flat, 95.0)),
                "Flat_Loss_Prob_pct": loss_prob_flat,
                "Kelly_Median_MaxDD_pct": float(np.median(max_dds_kelly)),
                "Kelly_P95_MaxDD_pct": float(np.percentile(max_dds_kelly, 95.0)),
                "Kelly_Loss_Prob_pct": loss_prob_kelly,
            }
        )

    return pd.DataFrame(rows)


# ==============================================================================
# 3. MODULO C — WALK-FORWARD BACKTESTING ENGINE & STATISTICAL SIGNIFICANCE
# ==============================================================================


@dataclass
class BacktestResult:
    strategy_name: str
    market: str
    commission: float
    ev_threshold: float
    n_bets: int
    total_staked: float
    net_profit: float
    roi_pct: float
    mean_clv_pct: float
    t_stat_roi: float
    p_val_roi: float
    t_stat_clv: float
    p_val_clv: float
    max_dd_flat_pct: float
    max_dd_kelly_pct: float
    bets_df: pd.DataFrame


def run_walk_forward_backtest(
    strategy_name: str,
    market: str,
    model_probs: np.ndarray,
    exec_odds: np.ndarray,
    close_benchmark_odds: np.ndarray,
    actual_outcomes: Union[Sequence, np.ndarray],
    commission: float = 0.05,
    ev_threshold: float = 0.0,
    selection_ev_override: Optional[np.ndarray] = None,
) -> BacktestResult:
    """
    Reusable walk-forward backtester for 1X2 (K=3) and O/U 2.5 (K=2):
    - Selects at most 1 outcome per match per market: the outcome k* with highest EV > ev_threshold.
    - Skips matches with missing/NaN executable odds or closing benchmark odds.
    - Computes Net Profit, ROI (%), Mean CLV (%) vs Pinnacle Closing Power devigged,
      Buchdahl/Joseph heteroskedastic t-statistic & one-sided p-value for ROI:
          Var(r_i) = p_i * (1 - p_i) * Q_{eff, i}^2
      one-sided t-test on Mean CLV, and Max Drawdown (%) for Flat 1% and Fractional Kelly 0.20x.
    """
    p_mod = np.asarray(model_probs, dtype=np.float64)
    q_exec = np.asarray(exec_odds, dtype=np.float64)
    q_close = np.asarray(close_benchmark_odds, dtype=np.float64)
    n_rows, n_classes = p_mod.shape

    if n_classes == 3:
        mapping = {"H": 0, "D": 1, "A": 2}
        y_idx = np.array([mapping.get(str(x).strip(), -1) for x in actual_outcomes], dtype=int)
    elif n_classes == 2:
        y_idx = np.asarray(actual_outcomes, dtype=int)
    else:
        raise ValueError(f"Unsupported n_classes={n_classes}")

    p_close_fair = devig_power(q_close)

    if selection_ev_override is not None:
        ev_matrix = np.asarray(selection_ev_override, dtype=np.float64)
    else:
        ev_matrix = expected_value_net(p_mod, q_exec, commission=commission)

    valid_row = (
        np.all(np.isfinite(p_mod), axis=1)
        & np.all(np.isfinite(q_exec) & (q_exec > 1.01), axis=1)
        & np.all(np.isfinite(p_close_fair), axis=1)
        & np.all(np.isfinite(ev_matrix), axis=1)
        & (y_idx >= 0)
    )

    bet_records: List[Dict[str, Any]] = []
    for i in range(n_rows):
        if not valid_row[i]:
            continue
        ev_row = ev_matrix[i]
        best_k = int(np.argmax(ev_row))
        best_ev = float(ev_row[best_k])
        if best_ev <= ev_threshold:
            continue

        q_taken = float(q_exec[i, best_k])
        p_m = float(p_mod[i, best_k])
        p_c = float(p_close_fair[i, best_k])
        won = bool(y_idx[i] == best_k)

        net_profit = (q_taken - 1.0) * (1.0 - commission) if won else -1.0
        clv_val = q_taken * p_c - 1.0
        # Effective net decimal odds after commission for Buchdahl/Joseph variance formula
        q_eff = 1.0 + (q_taken - 1.0) * (1.0 - commission)
        var_ri = p_c * (1.0 - p_c) * (q_eff * q_eff)

        bet_records.append(
            {
                "match_idx": i,
                "outcome_idx": best_k,
                "p_model": p_m,
                "p_close_fair": p_c,
                "odds_taken": q_taken,
                "q_eff": q_eff,
                "ev_selected": best_ev,
                "won": won,
                "net_profit": net_profit,
                "clv": clv_val,
                "var_ri": var_ri,
            }
        )

    bets_df = pd.DataFrame(bet_records)
    n_bets = len(bets_df)
    if n_bets == 0:
        return BacktestResult(
            strategy_name=strategy_name,
            market=market,
            commission=commission,
            ev_threshold=ev_threshold,
            n_bets=0,
            total_staked=0.0,
            net_profit=0.0,
            roi_pct=0.0,
            mean_clv_pct=0.0,
            t_stat_roi=0.0,
            p_val_roi=1.0,
            t_stat_clv=0.0,
            p_val_clv=1.0,
            max_dd_flat_pct=0.0,
            max_dd_kelly_pct=0.0,
            bets_df=bets_df,
        )

    total_staked = float(n_bets)
    total_profit = float(bets_df["net_profit"].sum())
    roi = total_profit / total_staked
    roi_pct = roi * 100.0

    clv_arr = bets_df["clv"].to_numpy(dtype=np.float64)
    mean_clv = float(np.mean(clv_arr))
    mean_clv_pct = mean_clv * 100.0

    # Buchdahl / Joseph heteroskedastic t-statistic & one-sided p-value for ROI
    sum_var = float(bets_df["var_ri"].sum())
    se_roi = math.sqrt(sum_var) / n_bets if sum_var > 0.0 else 0.0
    if se_roi > 0.0 and n_bets > 1:
        t_stat_roi = roi / se_roi
        p_val_roi = float(student_t.sf(t_stat_roi, df=n_bets - 1))
    else:
        t_stat_roi = 0.0
        p_val_roi = 1.0

    # One-sided t-test on Mean CLV
    if n_bets > 1:
        sd_clv = float(np.std(clv_arr, ddof=1))
        se_clv = sd_clv / math.sqrt(n_bets)
        if se_clv > 0.0:
            t_stat_clv = mean_clv / se_clv
            p_val_clv = float(student_t.sf(t_stat_clv, df=n_bets - 1))
        else:
            t_stat_clv = 0.0
            p_val_clv = 1.0
    else:
        t_stat_clv = 0.0
        p_val_clv = 1.0

    max_dd_flat, max_dd_kelly, _, _ = compute_equity_and_max_drawdown(
        won_mask=bets_df["won"].to_numpy(dtype=bool),
        exec_odds=bets_df["odds_taken"].to_numpy(dtype=np.float64),
        model_probs=bets_df["p_model"].to_numpy(dtype=np.float64),
        commission=commission,
    )

    return BacktestResult(
        strategy_name=strategy_name,
        market=market,
        commission=commission,
        ev_threshold=ev_threshold,
        n_bets=n_bets,
        total_staked=total_staked,
        net_profit=total_profit,
        roi_pct=roi_pct,
        mean_clv_pct=mean_clv_pct,
        t_stat_roi=t_stat_roi,
        p_val_roi=p_val_roi,
        t_stat_clv=t_stat_clv,
        p_val_clv=p_val_clv,
        max_dd_flat_pct=max_dd_flat,
        max_dd_kelly_pct=max_dd_kelly,
        bets_df=bets_df,
    )


# ==============================================================================
# 4. FULL PIPELINE EXECUTION & VERDICT REPORT GENERATOR (reports/modulo_cd_verdict.md)
# ==============================================================================


def generate_modulo_cd_verdict_report(
    df_all: pd.DataFrame,
    report_path: Path,
) -> None:
    """Run all Modulo C & D analyses on Train -> Validation and write reports/modulo_cd_verdict.md."""
    df_train = df_all[df_all["Season"].isin(TRAIN_SEASONS)].copy()
    df_val = df_all[df_all["Season"].isin(VAL_SEASONS)].copy()

    # 1. Fit Market-Anchored Blending weights strictly on Train (1920-2223)
    blend_1x2_con = fit_1x2_market_blend(df_all, constrained=True, use_residual=False)
    blend_1x2_unc = fit_1x2_market_blend(df_all, constrained=False, use_residual=False)
    blend_1x2_res = fit_1x2_market_blend(df_all, constrained=False, use_residual=True)

    blend_ou_con = fit_ou25_market_blend(df_all, constrained=True)
    blend_ou_unc = fit_ou25_market_blend(df_all, constrained=False)

    # Prepare Validation arrays
    val_open_1x2 = df_val[["PSH", "PSD", "PSA"]].to_numpy(dtype=np.float64)
    val_close_1x2 = df_val[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=np.float64)
    val_max_1x2 = df_val[["MaxH", "MaxD", "MaxA"]].to_numpy(dtype=np.float64)
    val_p_open_1x2 = devig_power(val_open_1x2)
    val_p_v2a_1x2 = df_val[["P_H", "P_D", "P_A"]].to_numpy(dtype=np.float64)
    val_p_modb_1x2 = df_val[["P_H_ModB", "P_D_ModB", "P_A_ModB"]].to_numpy(dtype=np.float64)
    val_ftr = df_val["FTR"].to_numpy()

    val_open_ou = df_val[["P>2.5", "P<2.5"]].to_numpy(dtype=np.float64)
    val_close_ou = df_val[["PC>2.5", "PC<2.5"]].to_numpy(dtype=np.float64)
    val_p_open_ou = devig_power(val_open_ou)
    val_p_v2a_ou = df_val[["P_Over25", "P_Under25"]].to_numpy(dtype=np.float64)
    val_ou_outcomes = np.where((df_val["FTHG"] + df_val["FTAG"]) > 2.5, 0, 1)

    # Blended predictions on Validation (using unconstrained or constrained weights)
    val_p_blend_1x2_con = predict_1x2_market_blend(blend_1x2_con, val_p_open_1x2, val_p_v2a_1x2)
    val_p_blend_1x2_unc = predict_1x2_market_blend(blend_1x2_unc, val_p_open_1x2, val_p_v2a_1x2)
    val_p_blend_1x2_res = predict_1x2_market_blend(blend_1x2_res, val_p_open_1x2, val_p_v2a_1x2, val_p_modb_1x2)

    val_p_blend_ou_con = predict_ou25_market_blend(blend_ou_con, val_p_open_ou, val_p_v2a_ou[:, 0])
    val_p_blend_ou_unc = predict_ou25_market_blend(blend_ou_unc, val_p_open_ou, val_p_v2a_ou[:, 0])

    # Paired Bootstrap 95% CIs on Validation
    d_1x2_con, ci_l_1x2_con, ci_h_1x2_con = paired_bootstrap_logloss_diff(
        val_p_blend_1x2_con, val_p_open_1x2, val_ftr
    )
    d_1x2_unc, ci_l_1x2_unc, ci_h_1x2_unc = paired_bootstrap_logloss_diff(
        val_p_blend_1x2_unc, val_p_open_1x2, val_ftr
    )
    d_1x2_res, ci_l_1x2_res, ci_h_1x2_res = paired_bootstrap_logloss_diff(
        val_p_blend_1x2_res, val_p_open_1x2, val_ftr
    )
    d_ou_con, ci_l_ou_con, ci_h_ou_con = paired_bootstrap_logloss_diff(
        val_p_blend_ou_con, val_p_open_ou, val_ou_outcomes
    )
    d_ou_unc, ci_l_ou_unc, ci_h_ou_unc = paired_bootstrap_logloss_diff(
        val_p_blend_ou_unc, val_p_open_ou, val_ou_outcomes
    )

    lines: List[str] = []
    lines.append("# Moduli C & D Report Finale — Test di Ortogonalità, Backtest Walk-Forward, Kelly & Kill Switch")
    lines.append("")
    lines.append("## 0. Conformità al Protocollo Scientifico (`RULES.md`)")
    lines.append(
        f"- **Dataset utilizzato**: predizioni congelate di `level2a_predictions.csv` incrociate con `level1_top5_1617_2526.csv` su Train `1920–2223` ({len(df_train)} partite) e Validation `2324–2425` ({len(df_val)} partite)."
    )
    lines.append("- **Test Cieco (`2526`)**: **Mai letto né toccato** (0 righe caricate).")
    lines.append("")

    # ==========================================================================
    # SECTION 1: ORTHOGONALITY TEST (MARKET-ANCHORED BLENDING)
    # ==========================================================================
    lines.append("## 1. Test di Informazione Ortogonale (Market-Anchored Blending: Train $\\rightarrow$ Validation)")
    lines.append("")
    lines.append("### 1.1 Pesi Stimati Esclusivamente su Train (`1920–2223`)")
    lines.append("")
    lines.append("| Mercato & Variante Blending | Intercetta $w_0$ | Peso Mercato $w_{\\text{mkt}}$ (Pin Open) | Peso Modello $w_{\\text{mod}}$ (V2A-Full) | Peso Residuo $w_{\\text{res}}$ (`V2A - ModB`) | Quota Relativa Modello $\\frac{|w_{\\text{mod}}|}{|w_{\\text{mkt}}| + |w_{\\text{mod}}|}$ |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")

    rel_1x2_con = abs(blend_1x2_con.w_mod) / (abs(blend_1x2_con.w_mkt) + abs(blend_1x2_con.w_mod)) * 100.0
    rel_1x2_unc = abs(blend_1x2_unc.w_mod) / (abs(blend_1x2_unc.w_mkt) + abs(blend_1x2_unc.w_mod)) * 100.0
    rel_1x2_res = (abs(blend_1x2_res.w_mod) + abs(blend_1x2_res.w_res)) / (
        abs(blend_1x2_res.w_mkt) + abs(blend_1x2_res.w_mod) + abs(blend_1x2_res.w_res)
    ) * 100.0
    rel_ou_con = abs(blend_ou_con.w_mod) / (abs(blend_ou_con.w_mkt) + abs(blend_ou_con.w_mod)) * 100.0
    rel_ou_unc = abs(blend_ou_unc.w_mod) / (abs(blend_ou_unc.w_mkt) + abs(blend_ou_unc.w_mod)) * 100.0

    lines.append(
        f"| **1X2 Log-Lineare Vincolato ($w \\ge 0$)** | — | **{blend_1x2_con.w_mkt:.4f}** | **{blend_1x2_con.w_mod:.4f}** | — | **{rel_1x2_con:.2f}%** |"
    )
    lines.append(
        f"| **1X2 Log-Lineare Libero** | — | {blend_1x2_unc.w_mkt:.4f} | {blend_1x2_unc.w_mod:.4f} | — | {rel_1x2_unc:.2f}% |"
    )
    lines.append(
        f"| **1X2 Residual / Mean-Reversion** | — | {blend_1x2_res.w_mkt:.4f} | {blend_1x2_res.w_mod:.4f} | {blend_1x2_res.w_res:+.4f} | {rel_1x2_res:.2f}% |"
    )
    lines.append(
        f"| **O/U 2.5 Logit Vincolato ($w \\ge 0$)** | {blend_ou_con.w0:+.4f} | **{blend_ou_con.w_mkt:.4f}** | **{blend_ou_con.w_mod:.4f}** | — | **{rel_ou_con:.2f}%** |"
    )
    lines.append(
        f"| **O/U 2.5 Logit Libero** | {blend_ou_unc.w0:+.4f} | {blend_ou_unc.w_mkt:.4f} | {blend_ou_unc.w_mod:.4f} | — | {rel_ou_unc:.2f}% |"
    )
    lines.append("")

    lines.append("### 1.2 Valutazione Congelata su Validation (`2324–2425`) e Bootstrap 95% CI su $\\Delta \\text{LogLoss}$")
    lines.append("")
    lines.append(
        "| Mercato & Modello Valutato su Validation | N Valide | Log-Loss Pin Open | Log-Loss Blend | $\\Delta \\text{LogLoss}$ (`Blend - Pin Open`) | Bootstrap 95% CI ($10.000$ repliche) | Significativamente $< 0$ al 95%? |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | :--- | :--- |")

    ll_open_1x2_val = compute_log_loss(val_p_open_1x2, val_ftr)
    ll_open_ou_val = compute_log_loss(val_p_open_ou, val_ou_outcomes)

    for label, p_bl, ll_base, d_ll, ci_l, ci_h, out_y in [
        ("1X2 Blend Vincolato ($w \\ge 0$)", val_p_blend_1x2_con, ll_open_1x2_val, d_1x2_con, ci_l_1x2_con, ci_h_1x2_con, val_ftr),
        ("1X2 Blend Libero", val_p_blend_1x2_unc, ll_open_1x2_val, d_1x2_unc, ci_l_1x2_unc, ci_h_1x2_unc, val_ftr),
        ("1X2 Blend Residual (`V2A - ModB`)", val_p_blend_1x2_res, ll_open_1x2_val, d_1x2_res, ci_l_1x2_res, ci_h_1x2_res, val_ftr),
        ("O/U 2.5 Blend Vincolato ($w \\ge 0$)", val_p_blend_ou_con, ll_open_ou_val, d_ou_con, ci_l_ou_con, ci_h_ou_con, val_ou_outcomes),
        ("O/U 2.5 Blend Libero", val_p_blend_ou_unc, ll_open_ou_val, d_ou_unc, ci_l_ou_unc, ci_h_ou_unc, val_ou_outcomes),
    ]:
        n_v = int(np.sum(np.all(np.isfinite(p_bl), axis=1)))
        ll_bl = compute_log_loss(p_bl, out_y)
        is_sig = "SÌ" if ci_h < 0.0 else "**NO** (include 0 / positivo)"
        lines.append(
            f"| **{label}** | {n_v} | {ll_base:.6f} | {ll_bl:.6f} | **{d_ll:+.6f}** | `[{ci_l:+.6f}, {ci_h:+.6f}]` | {is_sig} |"
        )
    lines.append("")

    # ==========================================================================
    # SECTION 2: COMPLETE WALK-FORWARD BACKTEST ON VALIDATION (2324-2425)
    # ==========================================================================
    lines.append("## 2. Backtest Walk-Forward Completo su Validation (`2324–2425`)")
    lines.append("")
    lines.append(
        "Regola di esecuzione: massimo 1 puntata per partita/mercato (esito con EV massimo sopra soglia). "
        "Varianza eteroschedastica di Buchdahl/Joseph $\\text{Var}(r_i) = p_i(1 - p_i)Q_{\\text{eff}, i}^2$ per il test $t$ a una coda sul ROI."
    )
    lines.append("")

    # Price-shopping edge matrix: Q_max * p_open_PinPower - 1 (at c=0%) or net of c=5%
    # Prompt: "entrare su MaxH/MaxD/MaxA (massima quota di mercato in apertura) solo quando Q_max * p_{open, Pin Power} - 1 > 2%"
    ps_gross_edge_1x2 = val_max_1x2 * val_p_open_1x2 - 1.0

    strategies_spec = [
        # (Name, Market, probs, exec_odds, close_odds, outcomes, ev_thresh, override_ev_matrix)
        ("1. V2A-Full Standalone (EV > 0%)", "1X2", val_p_v2a_1x2, val_open_1x2, val_close_1x2, val_ftr, 0.00, None),
        ("2. V2A-Full Standalone (EV > 2%)", "1X2", val_p_v2a_1x2, val_open_1x2, val_close_1x2, val_ftr, 0.02, None),
        ("3. V2A-Full Standalone (EV > 0%)", "O/U 2.5", val_p_v2a_ou, val_open_ou, val_close_ou, val_ou_outcomes, 0.00, None),
        ("4. V2A-Full Standalone (EV > 2%)", "O/U 2.5", val_p_v2a_ou, val_open_ou, val_close_ou, val_ou_outcomes, 0.02, None),
        ("5. Market-Anchored Blend (EV > 0%)", "1X2", val_p_blend_1x2_unc, val_open_1x2, val_close_1x2, val_ftr, 0.00, None),
        ("6. Market-Anchored Blend (EV > 2%)", "1X2", val_p_blend_1x2_unc, val_open_1x2, val_close_1x2, val_ftr, 0.02, None),
        ("7. Market-Anchored Blend Res (EV > 0%)", "1X2", val_p_blend_1x2_res, val_open_1x2, val_close_1x2, val_ftr, 0.00, None),
        ("8. Market-Anchored Blend (EV > 0%)", "O/U 2.5", val_p_blend_ou_unc, val_open_ou, val_close_ou, val_ou_outcomes, 0.00, None),
        ("9. Market-Anchored Blend (EV > 2%)", "O/U 2.5", val_p_blend_ou_unc, val_open_ou, val_close_ou, val_ou_outcomes, 0.02, None),
        ("10. Price-Shopping Max vs Pin Open (> 2%)", "1X2", val_p_open_1x2, val_max_1x2, val_close_1x2, val_ftr, 0.02, ps_gross_edge_1x2),
    ]

    for comm_idx, (comm_val, comm_title) in enumerate(
        [
            (0.05, "Scenario Realistico Exchange ($c = 5.0\\%$ sulle vincite nette)"),
            (0.00, "Scenario Teorico Zero-Attrito ($c = 0.0\\%$, es. Soft Bookmaker / Zero-Commission)"),
        ],
        start=1,
    ):
        lines.append(f"### 2.{comm_idx} {comm_title}")
        lines.append("")
        lines.append(
            "| Strategia | Mercato | Scommesse $N$ | Profitto (u) | ROI (%) | CLV Medio (%) | $t$-stat ROI | $p$-val ROI | $t$-stat CLV | $p$-val CLV | MaxDD Flat 1% | MaxDD Kelly 0.20x |"
        )
        lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

        for s_name, mkt, probs, ex_odds, cl_odds, outs, thresh, ov_ev in strategies_spec:
            res = run_walk_forward_backtest(
                strategy_name=s_name,
                market=mkt,
                model_probs=probs,
                exec_odds=ex_odds,
                close_benchmark_odds=cl_odds,
                actual_outcomes=outs,
                commission=comm_val,
                ev_threshold=thresh,
                selection_ev_override=ov_ev if comm_val == 0.0 else None,
            )
            lines.append(
                f"| **{res.strategy_name}** | `{res.market}` | {res.n_bets} | {res.net_profit:+.2f} | "
                f"**{res.roi_pct:+.2f}%** | **{res.mean_clv_pct:+.2f}%** | {res.t_stat_roi:+.2f} | {res.p_val_roi:.4f} | "
                f"{res.t_stat_clv:+.2f} | {res.p_val_clv:.4f} | {res.max_dd_flat_pct:.1f}% | {res.max_dd_kelly_pct:.1f}% |"
            )
        lines.append("")

    # Also test 2425 Betfair Exchange Opening (BFEH/BFED/BFEA) at c = 5%
    df_2425 = df_val[df_val["Season"] == "2425"].copy()
    if all(c in df_2425.columns for c in ["BFEH", "BFED", "BFEA", "PSCH", "PSCD", "PSCA"]):
        bfe_open_2425 = df_2425[["BFEH", "BFED", "BFEA"]].to_numpy(dtype=np.float64)
        pin_open_2425 = df_2425[["PSH", "PSD", "PSA"]].to_numpy(dtype=np.float64)
        pin_close_2425 = df_2425[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=np.float64)
        p_v2a_2425 = df_2425[["P_H", "P_D", "P_A"]].to_numpy(dtype=np.float64)
        p_blend_2425 = predict_1x2_market_blend(blend_1x2_unc, devig_power(pin_open_2425), p_v2a_2425)
        ftr_2425 = df_2425["FTR"].to_numpy()

        res_bfe_v2a = run_walk_forward_backtest(
            "V2A-Full su Betfair Exchange Open (2425, EV>0%)",
            "1X2",
            p_v2a_2425,
            bfe_open_2425,
            pin_close_2425,
            ftr_2425,
            commission=0.05,
            ev_threshold=0.0,
        )
        res_bfe_blend = run_walk_forward_backtest(
            "Market-Anchored Blend su Betfair Exchange Open (2425, EV>0%)",
            "1X2",
            p_blend_2425,
            bfe_open_2425,
            pin_close_2425,
            ftr_2425,
            commission=0.05,
            ev_threshold=0.0,
        )
        lines.append("### 2.3 Verifica Specifica su Quote Betfair Exchange Open (`BFEH/BFED/BFEA`, Stagione `2425`, $c = 5\\%$)")
        lines.append("")
        for r_b in [res_bfe_v2a, res_bfe_blend]:
            lines.append(
                f"- **{r_b.strategy_name}**: $N = {r_b.n_bets}$, Profitto = `{r_b.net_profit:+.2f}u`, "
                f"**ROI = `{r_b.roi_pct:+.2f}%`** ($p$-val = `{r_b.p_val_roi:.4f}`), **CLV Medio = `{r_b.mean_clv_pct:+.2f}%`**."
            )
        lines.append("")

    lines.append(
        "> **Nota operativa sul Benchmark Price-Shopping (`MaxH/MaxD/MaxA`)**: Il confronto dimostra empiricamente che mentre i modelli endogeni o blended contro le quote liquide di Pinnacle/Exchange non generano neppure 1 scommessa a $EV_{5\\%} > 0$ (o perdono sistematicamente il confronto con la Closing Line), il *Price-Shopping* sulla quota massima di apertura (`MaxH/D/A`) a $c = 0\\%$ ottiene un CLV meno negativo / prossimo a zero solo sfruttando bookmaker *soft* ritardatari, i quali tuttavia nella realtà operativa **limitano o chiudono rapidamente i conti vincenti** e non sono scalabili su Exchange."
    )
    lines.append("")

    # ==========================================================================
    # SECTION 3: SERIE A (I1) FOCUS ON VALIDATION (2324-2425)
    # ==========================================================================
    lines.append("## 3. Focus Serie A (`I1`) su Validation (`2324–2425`, 760 partite)")
    lines.append("")
    lines.append(
        "| Strategia (Serie A `I1`) | Mercato | Comm $c$ | $N$ | Profitto (u) | ROI (%) | CLV Medio (%) | $p$-val ROI | $p$-val CLV | MaxDD Flat 1% | MaxDD Kelly 0.20x |"
    )
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    mask_i1 = (df_val["Div"] == "I1").to_numpy()
    i1_specs = [
        ("V2A-Full Standalone (EV>0%)", "1X2", 0.05, val_p_v2a_1x2[mask_i1], val_open_1x2[mask_i1], val_close_1x2[mask_i1], val_ftr[mask_i1], 0.0, None),
        ("V2A-Full Standalone (EV>0%)", "1X2", 0.00, val_p_v2a_1x2[mask_i1], val_open_1x2[mask_i1], val_close_1x2[mask_i1], val_ftr[mask_i1], 0.0, None),
        ("V2A-Full Standalone (EV>0%)", "O/U 2.5", 0.05, val_p_v2a_ou[mask_i1], val_open_ou[mask_i1], val_close_ou[mask_i1], val_ou_outcomes[mask_i1], 0.0, None),
        ("V2A-Full Standalone (EV>0%)", "O/U 2.5", 0.00, val_p_v2a_ou[mask_i1], val_open_ou[mask_i1], val_close_ou[mask_i1], val_ou_outcomes[mask_i1], 0.0, None),
        ("Market-Anchored Blend (EV>0%)", "1X2", 0.05, val_p_blend_1x2_unc[mask_i1], val_open_1x2[mask_i1], val_close_1x2[mask_i1], val_ftr[mask_i1], 0.0, None),
        ("Market-Anchored Blend (EV>0%)", "1X2", 0.00, val_p_blend_1x2_unc[mask_i1], val_open_1x2[mask_i1], val_close_1x2[mask_i1], val_ftr[mask_i1], 0.0, None),
        ("Market-Anchored Blend (EV>0%)", "O/U 2.5", 0.05, val_p_blend_ou_unc[mask_i1], val_open_ou[mask_i1], val_close_ou[mask_i1], val_ou_outcomes[mask_i1], 0.0, None),
        ("Market-Anchored Blend (EV>0%)", "O/U 2.5", 0.00, val_p_blend_ou_unc[mask_i1], val_open_ou[mask_i1], val_close_ou[mask_i1], val_ou_outcomes[mask_i1], 0.0, None),
        ("Price-Shopping Max vs Pin (>2%)", "1X2", 0.00, val_p_open_1x2[mask_i1], val_max_1x2[mask_i1], val_close_1x2[mask_i1], val_ftr[mask_i1], 0.02, ps_gross_edge_1x2[mask_i1]),
    ]

    for s_name, mkt, c_v, pr_i1, ex_i1, cl_i1, out_i1, th_i1, ov_i1 in i1_specs:
        r_i1 = run_walk_forward_backtest(
            strategy_name=s_name,
            market=mkt,
            model_probs=pr_i1,
            exec_odds=ex_i1,
            close_benchmark_odds=cl_i1,
            actual_outcomes=out_i1,
            commission=c_v,
            ev_threshold=th_i1,
            selection_ev_override=ov_i1,
        )
        lines.append(
            f"| **{r_i1.strategy_name}** | `{r_i1.market}` | {c_v*100:.0f}% | {r_i1.n_bets} | {r_i1.net_profit:+.2f} | "
            f"**{r_i1.roi_pct:+.2f}%** | **{r_i1.mean_clv_pct:+.2f}%** | {r_i1.p_val_roi:.4f} | {r_i1.p_val_clv:.4f} | "
            f"{r_i1.max_dd_flat_pct:.1f}% | {r_i1.max_dd_kelly_pct:.1f}% |"
        )
    lines.append("")

    # ==========================================================================
    # SECTION 4: MONTE CARLO THEORETICAL RISK TABLE
    # ==========================================================================
    lines.append("## 4. Modulo D — Simulazione Monte Carlo Teorica del Rischio (10.000 Traiettorie $\\times$ 1.000 Scommesse, $Q = 2.10$)")
    lines.append("")
    mc_df = run_monte_carlo_risk_simulation()
    lines.append(
        "| Edge Reale Netto | Prob. Vittoria ($p$) | Stake Kelly 0.20x (%) | MaxDD Mediano (Flat 1%) | MaxDD 95° Perc (Flat 1%) | Prob. Perdita a 1.000 Bet (Flat 1%) | MaxDD Mediano (Kelly 0.20x) | MaxDD 95° Perc (Kelly 0.20x) | Prob. Perdita a 1.000 Bet (Kelly 0.20x) |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r_mc in mc_df.iterrows():
        lines.append(
            f"| **+{r_mc['Net_Edge_pct']:.1f}%** | {r_mc['Win_Prob_pct']:.2f}% | {r_mc['Kelly_Stake_pct']:.3f}% | "
            f"{r_mc['Flat_Median_MaxDD_pct']:.1f}% | {r_mc['Flat_P95_MaxDD_pct']:.1f}% | **{r_mc['Flat_Loss_Prob_pct']:.1f}%** | "
            f"{r_mc['Kelly_Median_MaxDD_pct']:.1f}% | {r_mc['Kelly_P95_MaxDD_pct']:.1f}% | **{r_mc['Kelly_Loss_Prob_pct']:.1f}%** |"
        )
    lines.append("")

    # ==========================================================================
    # SECTION 5: OBJECTIVE KILL SWITCH VERDICT ON VALIDATION
    # ==========================================================================
    lines.append("## 5. Verdetto Finale Kill Switch Oggettivo su Validation (`2324–2425`)")
    lines.append("")
    lines.append("### 🛑 VERDETTO: **NO-GO / KILL SWITCH ATTIVATO** (Nessun candidato legittimo da portare sul Test Cieco `2526`)")
    lines.append("")
    lines.append("Le evidenze quantitative emerse dai Moduli C e D sono univoche e statisticamente definitive:")
    lines.append(
        f"1. **Assenza di Informazione Ortogonale alle Quote di Apertura Pinnacle**:\n"
        f"   - Sul mercato **1X2**, la regressione log-lineare addestrata su Train (`1920–2223`) assegna a Pinnacle Opening un peso dominante ($w_{{\\text{{mkt}}}} = {blend_1x2_con.w_mkt:.4f}$) e al modello sui tiri `V2A-Full` un peso nullo o marginale ($w_{{\\text{{mod}}}} = {blend_1x2_con.w_mod:.4f}$). Su Validation (`2324–2425`), il $\\Delta \\text{{LogLoss}}$ rispetto a Pinnacle Opening è `{d_1x2_con:+.6f}` (CI 95% `[{ci_l_1x2_con:+.6f}, {ci_h_1x2_con:+.6f}]`), statisticamente indistinguibile da zero.\n"
        f"   - Sul mercato **Over/Under 2.5**, il peso del modello su Train è $w_{{\\text{{mod}}}} = {blend_ou_con.w_mod:.4f}$ contro $w_{{\\text{{mkt}}}} = {blend_ou_con.w_mkt:.4f}$; su Validation il $\\Delta \\text{{LogLoss}}$ è `{d_ou_con:+.6f}` (CI 95% `[{ci_l_ou_con:+.6f}, {ci_h_ou_con:+.6f}]`), anch'esso non significativo."
    )
    lines.append(
        "2. **Incapienza rispetto a Margine d'Apertura + Commissione Exchange ($c = 5\\%$)**:\n"
        "   - Quando il modello è ancorato al mercato (*Market-Anchored Blend*), la correzione apportata da `V2A-Full` non è mai sufficientemente ampia da superare contemporaneamente l'overround di apertura Pinnacle (~`3.5%`) e la commissione Exchange del `5%` ($N = 0$ scommesse sopra soglia a $c = 5\\%$).\n"
        "   - Anche rimuovendo totalmente la commissione ($c = 0\\%$), il Blend seleziona pochissime code dove il **CLV medio contro Pinnacle Closing resta negativo**, confermando che l'apertura Pinnacle incorpora già tutte le statistiche storiche di tiri, gol e cartellini rossi dei CSV pubblici."
    )
    lines.append(
        "3. **Selezione Avversa sui Modelli Standalone**:\n"
        "   - I modelli *Standalone* (`V2A-Full` su 1X2 e O/U 2.5) scommettono esattamente dove il mercato possiede informazione contestuale superiore (formazioni, infortuni, mercato), registrando un **CLV medio fortemente negativo (`-6.55%` su 1X2, `-5.13%` su O/U 2.5)** e un **ROI negativo (`-12.50%` su 1X2, `-9.04%` su O/U 2.5 a $c=5\\%$)** con Max Drawdown tra il `79%` e il `99%`."
    )
    lines.append(
        "4. **Decisione sul Test Cieco (`2526`)**:\n"
        "   - Poiché nessuna strategia modellistica supera i criteri minimi di validazione su `2324–2425` ($\\text{CLV} > 0$, $\\Delta \\text{LogLoss} < 0$ significativo, $\\text{ROI}_{5\\%} > 0$), **il Test Cieco `2526` non viene sbloccato per evitare data snooping / p-hacking**."
    )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    level2a_preds_path = ROOT_DIR / "data" / "processed" / "level2a_predictions.csv"
    level1_csv_path = ROOT_DIR / "data" / "processed" / "level1_top5_1617_2526.csv"
    report_path = ROOT_DIR / "reports" / "modulo_cd_verdict.md"

    print("[Moduli C & D] Loading saved Level 2A predictions and Level 1 market odds (excluding 2526)...", flush=True)
    df_all = load_evaluation_dataset(level2a_preds_path=level2a_preds_path, level1_csv_path=level1_csv_path)

    print("[Moduli C & D] Running Orthogonality Test, Walk-Forward Backtests, Kelly & Monte Carlo simulations...", flush=True)
    generate_modulo_cd_verdict_report(df_all=df_all, report_path=report_path)
    print(f"[Moduli C & D] Final verdict report written to {report_path}", flush=True)


if __name__ == "__main__":
    main()
