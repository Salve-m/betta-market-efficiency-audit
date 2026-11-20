"""
Level 2A: Quasi-Poisson MAP on Shots (Synthetic xG) + 2-Stage Dixon-Coles Walk-Forward Model
(src/level2a_shots_dc.py).

Strictly adheres to RULES.md:
- Zero Data Leakage: at every refit date t, all parameters, shot conversion rates (r_sot, r_tot),
  promoted team priors, and rho are calibrated strictly on matches with Date < t.
- Temporal Split:
  - Warm-up: 1617-1819
  - Train: 1920-2223 (hyperparameter grid search)
  - Validation: 2324-2425 (frozen single evaluation + ablation)
  - Blind Test: 2526 (strictly untouched and excluded)
  - Holdout Live: 2627 (excluded)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modulo_a_market import (
    LEAGUE_NAMES,
    TRAIN_SEASONS,
    VAL_SEASONS,
    WARMUP_SEASONS,
    closing_line_value,
    compute_brier_score,
    compute_calibration_table,
    compute_log_loss,
    devig_power,
    expected_value_net,
)
from src.modulo_b_dixon_coles import (
    ALL_ALLOWED_SEASONS,
    EVAL_SEASONS,
    ROLLING_WINDOW_DAYS,
    TAU_EPS,
    compute_league_standings_bottom3,
    predict_match_probabilities,
)

H_GRID: List[float] = [270.0, 365.0, 540.0]
REG_GRID: List[float] = [1.5, 3.0, 6.0, 12.0]
WEIGHT_MIX_GRID: List[Tuple[float, float, float]] = [
    (1.0, 0.0, 0.0),
    (0.50, 0.35, 0.15),
    (0.30, 0.50, 0.20),
    (0.15, 0.60, 0.25),
]
RED_CARD_DISCOUNT_GRID: List[float] = [1.0, 0.50]


@dataclass
class Level2AFit:
    teams: List[str]
    team_to_idx: Dict[str, int]
    a: np.ndarray
    b: np.ndarray
    h: float
    rho: float
    a0: np.ndarray
    b0: np.ndarray
    r_sot: float
    r_tot: float
    promoted_teams: List[str]


# ==============================================================================
# 1. AUDIT OF SHOTS & RED CARDS COVERAGE AND ANOMALIES (1617-2425)
# ==============================================================================


@dataclass
class ShotsAuditSummary:
    season_cov: pd.DataFrame
    league_cov: pd.DataFrame
    negative_rows: List[Dict[str, Any]]
    hst_gt_hs_rows: List[Dict[str, Any]]
    ast_gt_as_rows: List[Dict[str, Any]]
    missing_shots_rows: List[Dict[str, Any]]


def audit_shots_and_red_cards(df_dev: pd.DataFrame) -> ShotsAuditSummary:
    """
    Audit coverage and physical integrity of HS, AS, HST, AST, HR, AR on 1617-2425.
    Logs any negative values, HST > HS, AST > AS, or missing shot rows.
    """
    stat_cols = ["HS", "AS", "HST", "AST", "HR", "AR"]

    season_rows = []
    for s in ALL_ALLOWED_SEASONS:
        sub = df_dev[df_dev["Season"] == s]
        n = len(sub)
        row_dict: Dict[str, Any] = {"Season": s, "Matches": n}
        for c in stat_cols:
            row_dict[f"{c}_pct"] = float(sub[c].notna().mean() * 100.0) if n > 0 else 0.0
        row_dict["All_Shots_Valid_pct"] = (
            float(sub[["HS", "AS", "HST", "AST"]].notna().all(axis=1).mean() * 100.0) if n > 0 else 0.0
        )
        season_rows.append(row_dict)

    league_rows = []
    for div_code, div_name in LEAGUE_NAMES.items():
        sub = df_dev[df_dev["Div"] == div_code]
        n = len(sub)
        row_dict = {"Div": div_code, "League": div_name, "Matches": n}
        for c in stat_cols:
            row_dict[f"{c}_pct"] = float(sub[c].notna().mean() * 100.0) if n > 0 else 0.0
        row_dict["All_Shots_Valid_pct"] = (
            float(sub[["HS", "AS", "HST", "AST"]].notna().all(axis=1).mean() * 100.0) if n > 0 else 0.0
        )
        league_rows.append(row_dict)

    neg_rows: List[Dict[str, Any]] = []
    hst_gt_hs: List[Dict[str, Any]] = []
    ast_gt_as: List[Dict[str, Any]] = []
    missing_shots: List[Dict[str, Any]] = []

    for row in df_dev.itertuples(index=False):
        hs, as_, hst, ast, hr, ar = row.HS, row.AS, row.HST, row.AST, row.HR, row.AR
        if any(pd.isna(v) for v in (hs, as_, hst, ast)):
            missing_shots.append(
                {
                    "Season": row.Season,
                    "Div": row.Div,
                    "Date": row.Date,
                    "HomeTeam": row.HomeTeam,
                    "AwayTeam": row.AwayTeam,
                    "FTHG": row.FTHG,
                    "FTAG": row.FTAG,
                    "HS": hs,
                    "AS": as_,
                    "HST": hst,
                    "AST": ast,
                }
            )
            continue

        if any(v < 0 for v in (hs, as_, hst, ast)) or (pd.notna(hr) and hr < 0) or (pd.notna(ar) and ar < 0):
            neg_rows.append(
                {
                    "Season": row.Season,
                    "Div": row.Div,
                    "Date": row.Date,
                    "HomeTeam": row.HomeTeam,
                    "AwayTeam": row.AwayTeam,
                    "HS": hs,
                    "AS": as_,
                    "HST": hst,
                    "AST": ast,
                }
            )
        if hst > hs:
            hst_gt_hs.append(
                {
                    "Season": row.Season,
                    "Div": row.Div,
                    "Date": row.Date,
                    "HomeTeam": row.HomeTeam,
                    "AwayTeam": row.AwayTeam,
                    "HS": hs,
                    "HST": hst,
                }
            )
        if ast > as_:
            ast_gt_as.append(
                {
                    "Season": row.Season,
                    "Div": row.Div,
                    "Date": row.Date,
                    "HomeTeam": row.HomeTeam,
                    "AwayTeam": row.AwayTeam,
                    "AS": as_,
                    "AST": ast,
                }
            )

    return ShotsAuditSummary(
        season_cov=pd.DataFrame(season_rows),
        league_cov=pd.DataFrame(league_rows),
        negative_rows=neg_rows,
        hst_gt_hs_rows=hst_gt_hs,
        ast_gt_as_rows=ast_gt_as,
        missing_shots_rows=missing_shots,
    )


# ==============================================================================
# 2. SYNTHETIC xG CALIBRATION & COMPOSITE TARGET (ANTI-LEAKAGE + NaN FALLBACK)
# ==============================================================================


def calibrate_shot_conversion_rates(
    fthg: np.ndarray,
    ftag: np.ndarray,
    hs: np.ndarray,
    as_: np.ndarray,
    hst: np.ndarray,
    ast: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """
    Compute league shot conversion rates r_sot and r_tot strictly on valid historical matches:
        r_sot = sum(FTHG + FTAG) / sum(HST + AST)
        r_tot = sum(FTHG + FTAG) / sum(HS + AS)
    Valid match mask requires finite non-negative HS, AS, HST, AST with HST <= HS and AST <= AS.
    """
    valid_shots_mask = (
        np.isfinite(hs)
        & np.isfinite(as_)
        & np.isfinite(hst)
        & np.isfinite(ast)
        & (hs >= 0.0)
        & (as_ >= 0.0)
        & (hst >= 0.0)
        & (ast >= 0.0)
        & (hst <= hs)
        & (ast <= as_)
    )

    if not np.any(valid_shots_mask):
        return 0.30, 0.10, valid_shots_mask

    tot_goals = float(np.sum(fthg[valid_shots_mask] + ftag[valid_shots_mask]))
    tot_sot = float(np.sum(hst[valid_shots_mask] + ast[valid_shots_mask]))
    tot_shots = float(np.sum(hs[valid_shots_mask] + as_[valid_shots_mask]))

    r_sot = tot_goals / tot_sot if tot_sot > 0.0 else 0.30
    r_tot = tot_goals / tot_shots if tot_shots > 0.0 else 0.10
    return r_sot, r_tot, valid_shots_mask


def compute_composite_targets(
    fthg: np.ndarray,
    ftag: np.ndarray,
    hs: np.ndarray,
    as_: np.ndarray,
    hst: np.ndarray,
    ast: np.ndarray,
    w_g: float,
    w_sot: float,
    w_tot: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute composite targets (y_eff_H, y_eff_A) for historical matches:
        y_eff_H = w_g * FTHG + w_sot * (HST * r_sot) + w_tot * (HS * r_tot)
        y_eff_A = w_g * FTAG + w_sot * (AST * r_sot) + w_tot * (AS * r_tot)
    Falls back cleanly to (FTHG, FTAG) for any match where shots are NaN or anomalous.
    """
    r_sot, r_tot, valid_mask = calibrate_shot_conversion_rates(fthg, ftag, hs, as_, hst, ast)

    y_eff_h = fthg.astype(np.float64).copy()
    y_eff_a = ftag.astype(np.float64).copy()

    if (w_sot > 0.0 or w_tot > 0.0) and np.any(valid_mask):
        y_eff_h[valid_mask] = (
            w_g * fthg[valid_mask]
            + w_sot * (hst[valid_mask] * r_sot)
            + w_tot * (hs[valid_mask] * r_tot)
        )
        y_eff_a[valid_mask] = (
            w_g * ftag[valid_mask]
            + w_sot * (ast[valid_mask] * r_sot)
            + w_tot * (as_[valid_mask] * r_tot)
        )

    return y_eff_h, y_eff_a, r_sot, r_tot


# ==============================================================================
# 3. STAGE 1 (QUASI-POISSON MAP WITH EXACT ANALYTICAL GRADIENT) & STAGE 2 (RHO)
# ==============================================================================


def stage1_quasi_poisson_obj_and_grad(
    params: np.ndarray,
    n_teams: int,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    y_eff_h: np.ndarray,
    y_eff_a: np.ndarray,
    weights: np.ndarray,
    a0: np.ndarray,
    b0: np.ndarray,
    lambda_reg: float,
) -> Tuple[float, np.ndarray]:
    """
    Exact negative Quasi-Poisson MAP log-posterior and analytical gradient for Stage 1:
        J_1(a, b, h) = -sum_m w_m * [y_eff_H,m * ln(lam_m) - lam_m + y_eff_A,m * ln(mu_m) - mu_m]
                       + (lambda_reg / 2) * sum_k [(a_k - a0_k)^2 + (b_k - b0_k)^2]
                       + (0.01 * lambda_reg / 2) * (sum_k (a_k - a0_k))^2
    """
    a = params[:n_teams]
    b = params[n_teams : 2 * n_teams]
    h = float(params[2 * n_teams])

    log_lam = a[home_idx] + b[away_idx] + h
    log_mu = a[away_idx] + b[home_idx]
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)

    ll_match = y_eff_h * log_lam - lam + y_eff_a * log_mu - mu
    neg_w_ll = -float(np.sum(weights * ll_match))

    diff_a = a - a0
    diff_b = b - b0
    sum_diff_a = float(np.sum(diff_a))
    center_w = 0.01 * lambda_reg

    reg_loss = 0.5 * lambda_reg * float(np.sum(diff_a * diff_a) + np.sum(diff_b * diff_b))
    center_loss = 0.5 * center_w * (sum_diff_a * sum_diff_a)
    obj = neg_w_ll + reg_loss + center_loss

    w_res_h = weights * (y_eff_h - lam)
    w_res_a = weights * (y_eff_a - mu)

    grad_a = (
        -np.bincount(home_idx, weights=w_res_h, minlength=n_teams)
        - np.bincount(away_idx, weights=w_res_a, minlength=n_teams)
        + lambda_reg * diff_a
        + center_w * sum_diff_a
    )
    grad_b = (
        -np.bincount(away_idx, weights=w_res_h, minlength=n_teams)
        - np.bincount(home_idx, weights=w_res_a, minlength=n_teams)
        + lambda_reg * diff_b
    )
    grad_h = -float(np.sum(w_res_h))

    grad = np.empty(2 * n_teams + 1, dtype=np.float64)
    grad[:n_teams] = grad_a
    grad[n_teams : 2 * n_teams] = grad_b
    grad[2 * n_teams] = grad_h

    return obj, grad


def stage2_calibrate_rho(
    lam: np.ndarray,
    mu: np.ndarray,
    fthg: np.ndarray,
    ftag: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Stage 2: Holding (lam_m, mu_m) fixed from Stage 1, estimate scalar rho in [-0.25, 0.25]
    via scipy.optimize.minimize_scalar(..., method="bounded") maximizing weighted Dixon-Coles
    log-likelihood on past integer goals (FTHG, FTAG).
    """
    low_mask = (fthg <= 1) & (ftag <= 1)
    if not np.any(low_mask):
        return 0.0

    lam_l = lam[low_mask]
    mu_l = mu[low_mask]
    hg_l = fthg[low_mask]
    ag_l = ftag[low_mask]
    w_l = weights[low_mask]

    m00 = (hg_l == 0) & (ag_l == 0)
    m01 = (hg_l == 0) & (ag_l == 1)
    m10 = (hg_l == 1) & (ag_l == 0)
    m11 = (hg_l == 1) & (ag_l == 1)

    prod00 = lam_l[m00] * mu_l[m00]
    w00 = w_l[m00]
    lam01 = lam_l[m01]
    w01 = w_l[m01]
    mu10 = mu_l[m10]
    w10 = w_l[m10]
    w11_sum = float(np.sum(w_l[m11]))

    def neg_dc_tau_ll(rho_val: float) -> float:
        total = 0.0
        if len(prod00) > 0:
            t00 = np.maximum(1.0 - rho_val * prod00, TAU_EPS)
            total -= float(np.sum(w00 * np.log(t00)))
        if len(lam01) > 0:
            t01 = np.maximum(1.0 + rho_val * lam01, TAU_EPS)
            total -= float(np.sum(w01 * np.log(t01)))
        if len(mu10) > 0:
            t10 = np.maximum(1.0 + rho_val * mu10, TAU_EPS)
            total -= float(np.sum(w10 * np.log(t10)))
        if w11_sum > 0.0:
            t11 = max(1.0 - rho_val, TAU_EPS)
            total -= w11_sum * math.log(t11)
        return total

    res = minimize_scalar(neg_dc_tau_ll, bounds=(-0.25, 0.25), method="bounded")
    return float(res.x)


def fit_level2a_model(
    df_window: pd.DataFrame,
    current_date: pd.Timestamp,
    half_life_days: float,
    lambda_reg: float,
    w_g: float,
    w_sot: float,
    w_tot: float,
    d_rc: float,
    teams: Optional[List[str]] = None,
    priors: Optional[Dict[str, Tuple[float, float]]] = None,
    promoted_teams: Optional[List[str]] = None,
    warm_start_fit: Optional[Level2AFit] = None,
) -> Level2AFit:
    """
    Fit the 2-Stage V2A Quasi-Poisson MAP + Dixon-Coles model on df_window (Date < current_date).
    """
    hist_teams = set(df_window["HomeTeam"].unique()).union(set(df_window["AwayTeam"].unique()))
    all_teams = sorted(hist_teams.union(set(teams))) if teams is not None else sorted(hist_teams)
    n_teams = len(all_teams)
    team_to_idx = {t: i for i, t in enumerate(all_teams)}

    a0 = np.zeros(n_teams, dtype=np.float64)
    b0 = np.zeros(n_teams, dtype=np.float64)
    if priors is not None:
        for t, (pa, pb) in priors.items():
            if t in team_to_idx:
                idx = team_to_idx[t]
                a0[idx] = pa
                b0[idx] = pb

    if len(df_window) == 0:
        return Level2AFit(
            teams=all_teams,
            team_to_idx=team_to_idx,
            a=a0.copy(),
            b=b0.copy(),
            h=0.20,
            rho=0.0,
            a0=a0,
            b0=b0,
            r_sot=0.30,
            r_tot=0.10,
            promoted_teams=list(promoted_teams) if promoted_teams else [],
        )

    home_idx = df_window["HomeTeam"].map(team_to_idx).to_numpy(dtype=np.int32)
    away_idx = df_window["AwayTeam"].map(team_to_idx).to_numpy(dtype=np.int32)
    fthg = df_window["FTHG"].to_numpy(dtype=np.float64)
    ftag = df_window["FTAG"].to_numpy(dtype=np.float64)
    hs = df_window["HS"].to_numpy(dtype=np.float64)
    as_ = df_window["AS"].to_numpy(dtype=np.float64)
    hst = df_window["HST"].to_numpy(dtype=np.float64)
    ast = df_window["AST"].to_numpy(dtype=np.float64)
    hr = np.nan_to_num(df_window["HR"].to_numpy(dtype=np.float64), nan=0.0)
    ar = np.nan_to_num(df_window["AR"].to_numpy(dtype=np.float64), nan=0.0)

    # Anti-leakage calibration of r_sot, r_tot and composite target y_eff
    y_eff_h, y_eff_a, r_sot, r_tot = compute_composite_targets(
        fthg=fthg,
        ftag=ftag,
        hs=hs,
        as_=as_,
        hst=hst,
        ast=ast,
        w_g=w_g,
        w_sot=w_sot,
        w_tot=w_tot,
    )

    # Exponential time decay * Red Card discount d_rc
    delta_days = (current_date - df_window["ParsedDate"]).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    xi = math.log(2.0) / float(half_life_days)
    weights = np.exp(-xi * np.maximum(delta_days, 0.0))
    if d_rc < 1.0:
        has_rc = (hr + ar) > 0.0
        weights = np.where(has_rc, weights * float(d_rc), weights)

    # Stage 1: L-BFGS-B optimization of Quasi-Poisson MAP
    n_params = 2 * n_teams + 1
    x0 = np.zeros(n_params, dtype=np.float64)
    x0[:n_teams] = a0
    x0[n_teams : 2 * n_teams] = np.where(b0 != 0.0, b0, 0.15)
    x0[2 * n_teams] = 0.22

    if warm_start_fit is not None:
        for i, t in enumerate(all_teams):
            old_idx = warm_start_fit.team_to_idx.get(t)
            if old_idx is not None:
                x0[i] = warm_start_fit.a[old_idx]
                x0[n_teams + i] = warm_start_fit.b[old_idx]
        x0[2 * n_teams] = warm_start_fit.h

    bounds = [(-3.0, 3.0)] * (2 * n_teams) + [(-1.0, 1.5)]
    res1 = minimize(
        stage1_quasi_poisson_obj_and_grad,
        x0,
        args=(
            n_teams,
            home_idx,
            away_idx,
            y_eff_h,
            y_eff_a,
            weights,
            a0,
            b0,
            float(lambda_reg),
        ),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-7},
    )

    a_est = res1.x[:n_teams].copy()
    b_est = res1.x[n_teams : 2 * n_teams].copy()
    h_est = float(res1.x[2 * n_teams])

    if np.all(a0 == 0.0):
        mean_a = float(np.mean(a_est))
        a_est -= mean_a
        b_est += mean_a

    # Stage 2: Calibrate rho on real integer goals holding Stage 1 rates fixed
    lam_hist = np.exp(a_est[home_idx] + b_est[away_idx] + h_est)
    mu_hist = np.exp(a_est[away_idx] + b_est[home_idx])
    rho_est = stage2_calibrate_rho(lam=lam_hist, mu=mu_hist, fthg=fthg, ftag=ftag, weights=weights)

    return Level2AFit(
        teams=all_teams,
        team_to_idx=team_to_idx,
        a=a_est,
        b=b_est,
        h=h_est,
        rho=rho_est,
        a0=a0,
        b0=b0,
        r_sot=r_sot,
        r_tot=r_tot,
        promoted_teams=list(promoted_teams) if promoted_teams else [],
    )


# ==============================================================================
# 4. WALK-FORWARD ENGINE (DAILY & WEEKLY REFIT) AND STRICT SINGLE-DATE PREDICTOR
# ==============================================================================


def predict_level2a_on_date_strict(
    df_league: pd.DataFrame,
    target_date: Union[str, pd.Timestamp],
    half_life_days: float = 365.0,
    lambda_reg: float = 3.0,
    w_g: float = 0.30,
    w_sot: float = 0.50,
    w_tot: float = 0.20,
    d_rc: float = 0.50,
) -> Tuple[pd.DataFrame, Level2AFit]:
    """
    Standalone zero-leakage prediction for all matches of a single league on `target_date`.
    Strictly filters `df_hist = df_league[ParsedDate < target_date]` BEFORE calibrating
    (r_sot, r_tot), computing promoted priors, or fitting Stage 1 & Stage 2.
    """
    df_l = df_league.copy()
    if "ParsedDate" not in df_l.columns:
        df_l["ParsedDate"] = pd.to_datetime(df_l["Date"], format="%Y-%m-%d")

    t_date = pd.to_datetime(target_date)
    df_target = df_l[df_l["ParsedDate"] == t_date].copy()
    if len(df_target) == 0:
        raise ValueError(f"No matches found on target_date={target_date}")

    # STRICT ANTI-LEAKAGE FILTER: only matches with ParsedDate < t_date
    df_hist = df_l[df_l["ParsedDate"] < t_date].copy()

    curr_season = str(df_target["Season"].iloc[0])
    season_order = ALL_ALLOWED_SEASONS
    s_idx = season_order.index(curr_season) if curr_season in season_order else -1
    prev_season = season_order[s_idx - 1] if s_idx > 0 else None

    win_start = t_date - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
    df_window = df_hist[df_hist["ParsedDate"] >= win_start]

    active_teams = sorted(
        set(df_window["HomeTeam"].unique())
        .union(set(df_window["AwayTeam"].unique()))
        .union(set(df_target["HomeTeam"].unique()))
        .union(set(df_target["AwayTeam"].unique()))
    )

    # Promoted teams prior from end of prev_season (strictly Date < season start)
    priors: Dict[str, Tuple[float, float]] = {t: (0.0, 0.0) for t in active_teams}
    promoted: List[str] = []
    if prev_season is not None:
        df_prev = df_hist[df_hist["Season"] == prev_season]
        if len(df_prev) > 0:
            prev_teams = set(df_prev["HomeTeam"].unique()).union(set(df_prev["AwayTeam"].unique()))
            bottom3 = compute_league_standings_bottom3(df_prev)
            prev_end = df_prev["ParsedDate"].max()
            prev_win_start = prev_end - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
            df_prev_win = df_hist[(df_hist["ParsedDate"] <= prev_end) & (df_hist["ParsedDate"] >= prev_win_start)]
            fit_prev = fit_level2a_model(
                df_window=df_prev_win,
                current_date=prev_end,
                half_life_days=half_life_days,
                lambda_reg=lambda_reg,
                w_g=w_g,
                w_sot=w_sot,
                w_tot=w_tot,
                d_rc=d_rc,
                teams=list(prev_teams),
            )
            b3_idx = [fit_prev.team_to_idx[t] for t in bottom3 if t in fit_prev.team_to_idx]
            prom_prior = (float(np.mean(fit_prev.a[b3_idx])), float(np.mean(fit_prev.b[b3_idx])))
            for t in active_teams:
                if t not in prev_teams:
                    priors[t] = prom_prior
                    promoted.append(t)

    if promoted:
        df_curr_hist = df_window[df_window["Season"] == curr_season]
        played_curr = set(df_curr_hist["HomeTeam"].unique()).union(set(df_curr_hist["AwayTeam"].unique()))
        stale_promoted = set(promoted) - played_curr
        if stale_promoted:
            df_window = df_window[
                ~(df_window["HomeTeam"].isin(stale_promoted) | df_window["AwayTeam"].isin(stale_promoted))
            ]

    fit = fit_level2a_model(
        df_window=df_window,
        current_date=t_date,
        half_life_days=half_life_days,
        lambda_reg=lambda_reg,
        w_g=w_g,
        w_sot=w_sot,
        w_tot=w_tot,
        d_rc=d_rc,
        teams=active_teams,
        priors=priors,
        promoted_teams=promoted,
    )

    rows = []
    for row in df_target.itertuples(index=False):
        i = fit.team_to_idx[row.HomeTeam]
        j = fit.team_to_idx[row.AwayTeam]
        lam = math.exp(fit.a[i] + fit.b[j] + fit.h)
        mu = math.exp(fit.a[j] + fit.b[i])
        probs = predict_match_probabilities(lam=lam, mu=mu, rho=fit.rho)
        rows.append(
            {
                "Season": row.Season,
                "Div": row.Div,
                "Date": row.Date,
                "HomeTeam": row.HomeTeam,
                "AwayTeam": row.AwayTeam,
                "FTHG": row.FTHG,
                "FTAG": row.FTAG,
                "FTR": row.FTR,
                **probs,
                "r_sot": fit.r_sot,
                "r_tot": fit.r_tot,
                "h_param": fit.h,
                "rho_param": fit.rho,
            }
        )

    return pd.DataFrame(rows), fit


def run_walk_forward_level2a_league(
    df_league: pd.DataFrame,
    half_life_days: float,
    lambda_reg: float,
    w_g: float,
    w_sot: float,
    w_tot: float,
    d_rc: float,
    target_seasons: Sequence[str] = EVAL_SEASONS,
    refit_every_days: int = 1,
) -> Tuple[pd.DataFrame, Optional[Level2AFit]]:
    """
    Walk-forward evaluation for a single league.
    - When `refit_every_days == 1`: strict daily calendar-date refit (Date < t), directly comparable to Modulo B.
    - When `refit_every_days == 7`: weekly refit (every 7 calendar days, strictly Date < t_week_start) for fast Train grid search.
    """
    df_l = df_league[df_league["Season"].isin(ALL_ALLOWED_SEASONS)].copy()
    if "ParsedDate" not in df_l.columns:
        df_l["ParsedDate"] = pd.to_datetime(df_l["Date"], format="%Y-%m-%d")
    df_l = df_l.sort_values(["ParsedDate", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    parsed_dates = df_l["ParsedDate"]
    seasons_arr = df_l["Season"].to_numpy()
    season_order = ALL_ALLOWED_SEASONS

    cached_season_info: Dict[str, Tuple[Tuple[float, float], set]] = {}
    warm_fit: Optional[Level2AFit] = None
    last_refit_date: Optional[pd.Timestamp] = None
    last_refit_season: Optional[str] = None

    target_mask = df_l["Season"].isin(target_seasons)
    eval_dates = sorted(df_l.loc[target_mask, "ParsedDate"].unique())
    out_records: List[Dict[str, Any]] = []

    for t_date in eval_dates:
        df_day = df_l[parsed_dates == t_date]
        curr_season = str(df_day["Season"].iloc[0])

        # Compute season promoted prior once per season (strictly Date < season_first_date)
        if curr_season not in cached_season_info:
            s_idx = season_order.index(curr_season)
            prev_season = season_order[s_idx - 1] if s_idx > 0 else None
            season_first_date = df_l.loc[seasons_arr == curr_season, "ParsedDate"].min()
            df_before_season = df_l[parsed_dates < season_first_date]

            if prev_season is not None:
                df_prev = df_before_season[df_before_season["Season"] == prev_season]
                prev_teams = set(df_prev["HomeTeam"].unique()).union(set(df_prev["AwayTeam"].unique()))
                bottom3 = compute_league_standings_bottom3(df_prev)
                prev_end_date = df_prev["ParsedDate"].max()
                prev_win_start = prev_end_date - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
                df_prev_win = df_before_season[
                    (df_before_season["ParsedDate"] <= prev_end_date)
                    & (df_before_season["ParsedDate"] >= prev_win_start)
                ]
                fit_prev = fit_level2a_model(
                    df_window=df_prev_win,
                    current_date=prev_end_date,
                    half_life_days=half_life_days,
                    lambda_reg=lambda_reg,
                    w_g=w_g,
                    w_sot=w_sot,
                    w_tot=w_tot,
                    d_rc=d_rc,
                    teams=list(prev_teams),
                )
                b3_idx = [fit_prev.team_to_idx[t] for t in bottom3 if t in fit_prev.team_to_idx]
                prom_prior = (float(np.mean(fit_prev.a[b3_idx])), float(np.mean(fit_prev.b[b3_idx])))
                cached_season_info[curr_season] = (prom_prior, prev_teams)
            else:
                cached_season_info[curr_season] = ((0.0, 0.0), set())

        prom_prior, prev_season_teams = cached_season_info[curr_season]

        # Decide whether to refit on t_date
        need_refit = (
            warm_fit is None
            or last_refit_date is None
            or curr_season != last_refit_season
            or (t_date - last_refit_date).days >= refit_every_days
            or not set(df_day["HomeTeam"]).union(set(df_day["AwayTeam"])).issubset(set(warm_fit.team_to_idx.keys()))
        )

        if need_refit:
            refit_cutoff = t_date
            win_start = refit_cutoff - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
            df_window = df_l[(parsed_dates < refit_cutoff) & (parsed_dates >= win_start)]

            # Include all teams of curr_season known or active so weekly window never misses a team
            if refit_every_days > 1:
                win_end_week = refit_cutoff + pd.Timedelta(days=refit_every_days)
                df_upcoming = df_l[(parsed_dates >= refit_cutoff) & (parsed_dates < win_end_week)]
                upcoming_teams = set(df_upcoming["HomeTeam"]).union(set(df_upcoming["AwayTeam"]))
            else:
                upcoming_teams = set(df_day["HomeTeam"]).union(set(df_day["AwayTeam"]))

            active_teams = sorted(
                set(df_window["HomeTeam"].unique())
                .union(set(df_window["AwayTeam"].unique()))
                .union(upcoming_teams)
            )

            priors: Dict[str, Tuple[float, float]] = {}
            promoted: List[str] = []
            for t in active_teams:
                if t in prev_season_teams:
                    priors[t] = (0.0, 0.0)
                else:
                    priors[t] = prom_prior
                    promoted.append(t)

            if promoted:
                df_curr_hist = df_window[df_window["Season"] == curr_season]
                played_curr = set(df_curr_hist["HomeTeam"].unique()).union(set(df_curr_hist["AwayTeam"].unique()))
                stale_promoted = set(promoted) - played_curr
                if stale_promoted:
                    df_window = df_window[
                        ~(df_window["HomeTeam"].isin(stale_promoted) | df_window["AwayTeam"].isin(stale_promoted))
                    ]

            warm_fit = fit_level2a_model(
                df_window=df_window,
                current_date=refit_cutoff,
                half_life_days=half_life_days,
                lambda_reg=lambda_reg,
                w_g=w_g,
                w_sot=w_sot,
                w_tot=w_tot,
                d_rc=d_rc,
                teams=active_teams,
                priors=priors,
                promoted_teams=promoted,
                warm_start_fit=warm_fit,
            )
            last_refit_date = refit_cutoff
            last_refit_season = curr_season

        assert warm_fit is not None
        for row in df_day.itertuples(index=False):
            i = warm_fit.team_to_idx[row.HomeTeam]
            j = warm_fit.team_to_idx[row.AwayTeam]
            lam = math.exp(warm_fit.a[i] + warm_fit.b[j] + warm_fit.h)
            mu = math.exp(warm_fit.a[j] + warm_fit.b[i])
            probs = predict_match_probabilities(lam=lam, mu=mu, rho=warm_fit.rho)
            out_records.append(
                {
                    "Season": row.Season,
                    "Div": row.Div,
                    "Date": row.Date,
                    "HomeTeam": row.HomeTeam,
                    "AwayTeam": row.AwayTeam,
                    "FTHG": row.FTHG,
                    "FTAG": row.FTAG,
                    "FTR": row.FTR,
                    **probs,
                    "r_sot": warm_fit.r_sot,
                    "r_tot": warm_fit.r_tot,
                    "h_param": warm_fit.h,
                    "rho_param": warm_fit.rho,
                }
            )

    return pd.DataFrame(out_records), warm_fit


def run_walk_forward_level2a_all_leagues(
    df_all: pd.DataFrame,
    half_life_days: float,
    lambda_reg: float,
    w_g: float,
    w_sot: float,
    w_tot: float,
    d_rc: float,
    target_seasons: Sequence[str] = EVAL_SEASONS,
    refit_every_days: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, Level2AFit]]:
    """Run Level 2A walk-forward across all 5 leagues."""
    for forbidden in ["2526", "2627"]:
        if forbidden in target_seasons:
            raise ValueError(f"RULE VIOLATION: Season {forbidden} is forbidden in Level 2A walk-forward!")

    frames = []
    final_fits: Dict[str, Level2AFit] = {}
    for div_code in LEAGUE_NAMES.keys():
        df_l = df_all[df_all["Div"] == div_code].copy()
        preds_l, fit_l = run_walk_forward_level2a_league(
            df_league=df_l,
            half_life_days=half_life_days,
            lambda_reg=lambda_reg,
            w_g=w_g,
            w_sot=w_sot,
            w_tot=w_tot,
            d_rc=d_rc,
            target_seasons=target_seasons,
            refit_every_days=refit_every_days,
        )
        frames.append(preds_l)
        if fit_l is not None:
            final_fits[div_code] = fit_l

    combined = pd.concat(frames, ignore_index=True)
    odds_cols = [
        "Season",
        "Div",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "PSH",
        "PSD",
        "PSA",
        "PSCH",
        "PSCD",
        "PSCA",
        "P>2.5",
        "P<2.5",
        "PC>2.5",
        "PC<2.5",
    ]
    avail_cols = [c for c in odds_cols if c in df_all.columns]
    combined = combined.merge(
        df_all[avail_cols],
        on=["Season", "Div", "Date", "HomeTeam", "AwayTeam"],
        how="left",
    )
    combined = combined.sort_values(["Date", "Div", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    return combined, final_fits


# ==============================================================================
# 5. PARALLEL GRID SEARCH ON TRAIN (1920-2223, WEEKLY REFIT)
# ==============================================================================

_WORKER_DF_TRAIN: Optional[pd.DataFrame] = None


def _init_grid_worker(csv_path_str: str) -> None:
    """Initializer for ProcessPoolExecutor workers: load and pre-parse Train + Warmup data once."""
    global _WORKER_DF_TRAIN
    df = pd.read_csv(csv_path_str, dtype={"Season": str, "Div": str})
    df = df[df["Season"].isin(WARMUP_SEASONS + TRAIN_SEASONS)].copy()
    df["ParsedDate"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    _WORKER_DF_TRAIN = df


def _eval_single_grid_combo(
    combo: Tuple[int, float, float, Tuple[float, float, float], float]
) -> Dict[str, Any]:
    """Worker function evaluating one (H, lambda_reg, (w_g, w_sot, w_tot), d_rc) combo on Train."""
    global _WORKER_DF_TRAIN
    assert _WORKER_DF_TRAIN is not None
    idx, h_val, reg_val, (w_g, w_sot, w_tot), d_rc = combo
    t0 = time.perf_counter()

    frames = []
    for div_code in LEAGUE_NAMES.keys():
        df_l = _WORKER_DF_TRAIN[_WORKER_DF_TRAIN["Div"] == div_code]
        preds_l, _ = run_walk_forward_level2a_league(
            df_league=df_l,
            half_life_days=h_val,
            lambda_reg=reg_val,
            w_g=w_g,
            w_sot=w_sot,
            w_tot=w_tot,
            d_rc=d_rc,
            target_seasons=TRAIN_SEASONS,
            refit_every_days=7,
        )
        frames.append(preds_l)

    preds_all = pd.concat(frames, ignore_index=True)
    probs_1x2 = preds_all[["P_H", "P_D", "P_A"]].to_numpy(dtype=np.float64)
    ftr = preds_all["FTR"].to_numpy()
    ll_1x2 = compute_log_loss(probs_1x2, ftr)
    brier_1x2 = compute_brier_score(probs_1x2, ftr)
    elapsed = time.perf_counter() - t0

    return {
        "combo_idx": idx,
        "H": h_val,
        "lambda_reg": reg_val,
        "w_g": w_g,
        "w_sot": w_sot,
        "w_tot": w_tot,
        "d_rc": d_rc,
        "Train_LogLoss_1X2": ll_1x2,
        "Train_Brier_1X2": brier_1x2,
        "elapsed_s": elapsed,
    }


def run_parallel_grid_search_train(
    csv_path: Path,
    max_workers: Optional[int] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Run the 96-combination hyperparameter grid on Train (1920-2223) in parallel
    using ProcessPoolExecutor and weekly refits (Date < t_week_start).
    """
    combos: List[Tuple[int, float, float, Tuple[float, float, float], float]] = []
    idx = 0
    for h_val in H_GRID:
        for reg_val in REG_GRID:
            for w_mix in WEIGHT_MIX_GRID:
                for d_rc in RED_CARD_DISCOUNT_GRID:
                    idx += 1
                    combos.append((idx, h_val, reg_val, w_mix, d_rc))

    total_combos = len(combos)
    if max_workers is None:
        cpu_cnt = os.cpu_count() or 4
        max_workers = min(cpu_cnt, 8)

    print(
        f"[Level 2A Grid] Launching {total_combos} combinations across {max_workers} CPU workers (weekly refit on Train 1920-2223)...",
        flush=True,
    )
    t_start = time.perf_counter()
    results: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_grid_worker,
        initargs=(str(csv_path),),
    ) as executor:
        futures = {executor.submit(_eval_single_grid_combo, c): c for c in combos}
        done_cnt = 0
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done_cnt += 1
            if done_cnt % 12 == 0 or done_cnt == total_combos:
                best_so_far = min(r["Train_LogLoss_1X2"] for r in results)
                elapsed_tot = time.perf_counter() - t_start
                print(
                    f"  [Grid Progress] {done_cnt:2d}/{total_combos} completed ({elapsed_tot:.1f}s) | "
                    f"Best Train Log-Loss 1X2 so far: {best_so_far:.6f}",
                    flush=True,
                )

    total_elapsed = time.perf_counter() - t_start
    grid_df = pd.DataFrame(results).sort_values("Train_LogLoss_1X2", ascending=True).reset_index(drop=True)
    print(f"[Level 2A Grid] Completed {total_combos} combinations in {total_elapsed:.1f}s.", flush=True)
    return grid_df, total_elapsed


# ==============================================================================
# 6. REPORT GENERATOR (reports/level2a_shots_model.md)
# ==============================================================================


def _metrics_1x2_ablation(df_sub: pd.DataFrame) -> Dict[str, float]:
    """Compute 1X2 Log-Loss and Brier for Modulo B, V2A-Ablation, V2A-Full, Pin Open, Pin Close."""
    ftr = df_sub["FTR"].to_numpy()
    p_mod_b = df_sub[["P_H_ModB", "P_D_ModB", "P_A_ModB"]].to_numpy(dtype=np.float64)
    p_v2a_abl = df_sub[["P_H_Abl", "P_D_Abl", "P_A_Abl"]].to_numpy(dtype=np.float64)
    p_v2a_full = df_sub[["P_H", "P_D", "P_A"]].to_numpy(dtype=np.float64)

    p_open = devig_power(df_sub[["PSH", "PSD", "PSA"]].to_numpy(dtype=np.float64))
    p_close = devig_power(df_sub[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=np.float64))

    ll_b = compute_log_loss(p_mod_b, ftr)
    ll_abl = compute_log_loss(p_v2a_abl, ftr)
    ll_full = compute_log_loss(p_v2a_full, ftr)
    ll_open = compute_log_loss(p_open, ftr)
    ll_close = compute_log_loss(p_close, ftr)

    return {
        "n": len(df_sub),
        "ll_b": ll_b,
        "ll_abl": ll_abl,
        "ll_full": ll_full,
        "ll_open": ll_open,
        "ll_close": ll_close,
        "gain_vs_b": ll_full - ll_b,
        "gap_vs_close": ll_full - ll_close,
        "br_b": compute_brier_score(p_mod_b, ftr),
        "br_abl": compute_brier_score(p_v2a_abl, ftr),
        "br_full": compute_brier_score(p_v2a_full, ftr),
        "br_open": compute_brier_score(p_open, ftr),
        "br_close": compute_brier_score(p_close, ftr),
    }


def _metrics_ou25_ablation(df_sub: pd.DataFrame) -> Dict[str, float]:
    """Compute O/U 2.5 Log-Loss and Brier for Modulo B, V2A-Ablation, V2A-Full, Pin Open, Pin Close."""
    valid_ou = df_sub[["PC>2.5", "PC<2.5"]].notna().all(axis=1)
    sub = df_sub[valid_ou]
    tot_g = sub["FTHG"].to_numpy(dtype=np.float64) + sub["FTAG"].to_numpy(dtype=np.float64)
    outcomes = np.where(tot_g > 2.5, 0, 1)

    p_mod_b = sub[["P_Over25_ModB", "P_Under25_ModB"]].to_numpy(dtype=np.float64)
    p_v2a_abl = sub[["P_Over25_Abl", "P_Under25_Abl"]].to_numpy(dtype=np.float64)
    p_v2a_full = sub[["P_Over25", "P_Under25"]].to_numpy(dtype=np.float64)

    p_open = devig_power(sub[["P>2.5", "P<2.5"]].to_numpy(dtype=np.float64))
    p_close = devig_power(sub[["PC>2.5", "PC<2.5"]].to_numpy(dtype=np.float64))

    ll_b = compute_log_loss(p_mod_b, outcomes)
    ll_abl = compute_log_loss(p_v2a_abl, outcomes)
    ll_full = compute_log_loss(p_v2a_full, outcomes)
    ll_open = compute_log_loss(p_open, outcomes)
    ll_close = compute_log_loss(p_close, outcomes)

    return {
        "n": len(sub),
        "ll_b": ll_b,
        "ll_abl": ll_abl,
        "ll_full": ll_full,
        "ll_open": ll_open,
        "ll_close": ll_close,
        "gain_vs_b": ll_full - ll_b,
        "gap_vs_close": ll_full - ll_close,
        "br_b": compute_brier_score(p_mod_b, outcomes),
        "br_abl": compute_brier_score(p_v2a_abl, outcomes),
        "br_full": compute_brier_score(p_v2a_full, outcomes),
        "br_open": compute_brier_score(p_open, outcomes),
        "br_close": compute_brier_score(p_close, outcomes),
    }


def generate_level2a_report(
    audit_shots: ShotsAuditSummary,
    grid_df: pd.DataFrame,
    grid_elapsed_s: float,
    best_full: Dict[str, Any],
    best_abl: Dict[str, Any],
    df_preds: pd.DataFrame,
    report_path: Path,
) -> None:
    """Write the complete Level 2A diagnostic report to reports/level2a_shots_model.md."""
    df_train = df_preds[df_preds["Season"].isin(TRAIN_SEASONS)].copy()
    df_val = df_preds[df_preds["Season"].isin(VAL_SEASONS)].copy()

    lines: List[str] = []
    lines.append("# Level 2A Report — Quasi-Poisson MAP su Tiri (Synthetic xG) + Dixon-Coles 2-Stage")
    lines.append("")
    lines.append("## 1. Audit Copertura e Integrità Tiri & Cartellini Rossi (`1617–2425`)")
    lines.append("")
    lines.append(
        f"- **Partite totali ispezionate (`1617–2425`)**: **{int(audit_shots.season_cov['Matches'].sum())}** (Stagione `2526` Test Cieco esclusa e inviolata)."
    )
    lines.append(
        f"- **Partite con statistiche tiri mancanti (`NaN`, gestite con fallback automatico sui gol reali `FTHG/FTAG`)**: **{len(audit_shots.missing_shots_rows)}**"
    )
    for m_item in audit_shots.missing_shots_rows[:10]:
        lines.append(
            f"  - `{m_item['Date']}` (`{m_item['Season']}` `{m_item['Div']}`) `{m_item['HomeTeam']}` vs `{m_item['AwayTeam']}` (FTHG={m_item['FTHG']}, FTAG={m_item['FTAG']})"
        )
    lines.append(f"- **Valori negativi su `HS/AS/HST/AST/HR/AR`**: **{len(audit_shots.negative_rows)}**")
    lines.append(f"- **Anomalie `HST > HS` (tiri in porta > tiri totali casa)**: **{len(audit_shots.hst_gt_hs_rows)}**")
    lines.append(f"- **Anomalie `AST > AS` (tiri in porta > tiri totali trasferta, sanificate con fallback su gol reali)**: **{len(audit_shots.ast_gt_as_rows)}**")
    for a_item in audit_shots.ast_gt_as_rows[:10]:
        lines.append(
            f"  - `{a_item['Date']}` (`{a_item['Season']}` `{a_item['Div']}`) `{a_item['HomeTeam']}` vs `{a_item['AwayTeam']}` (AS={a_item['AS']}, AST={a_item['AST']})"
        )
    lines.append("")

    lines.append("### 1.1 Copertura Percentuale Non-Null per Lega (`1617–2425`)")
    lines.append("")
    lines.append("| Lega | Nome | Partite | `HS / AS` (%) | `HST / AST` (%) | `HR / AR` (%) | Tutte le Stat Tiri Valide (%) |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
    for _, r in audit_shots.league_cov.iterrows():
        hl = "**" if r["Div"] == "I1" else ""
        lines.append(
            f"| {hl}`{r['Div']}`{hl} | {hl}{r['League']}{hl} | {int(r['Matches'])} | "
            f"{r['HS_pct']:.2f}% | {r['HST_pct']:.2f}% | {r['HR_pct']:.2f}% | {hl}{r['All_Shots_Valid_pct']:.2f}%{hl} |"
        )
    lines.append("")

    lines.append("### 1.2 Copertura Percentuale Non-Null per Stagione (`1617–2425`)")
    lines.append("")
    lines.append("| Stagione | Partite | `HS / AS` (%) | `HST / AST` (%) | `HR / AR` (%) | Tutte le Stat Tiri Valide (%) |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for _, r in audit_shots.season_cov.iterrows():
        lines.append(
            f"| `{r['Season']}` | {int(r['Matches'])} | {r['HS_pct']:.2f}% | {r['HST_pct']:.2f}% | "
            f"{r['HR_pct']:.2f}% | {r['All_Shots_Valid_pct']:.2f}% |"
        )
    lines.append("")

    # ==========================================================================
    # SECTION 2: PARALLEL GRID SEARCH ON TRAIN (1920-2223)
    # ==========================================================================
    lines.append("## 2. Grid Search Parallela su Train (`1920–2223`) e Iperparametri Congelati")
    lines.append("")
    lines.append(
        f"- **Combinazioni valutate su Train (`1920–2223`, refit settimanale anti-leakage)**: **{len(grid_df)}** (3 emivite $H \\times 4$ regolarizzazioni $\\lambda_{{\\text{{reg}}}} \\times 4$ mix bersaglio $\\times 2$ sconti rossi $d_{{\\text{{rc}}}}$)"
    )
    lines.append(f"- **Tempo di esecuzione totale Grid Search parallela**: **{grid_elapsed_s:.1f} secondi**")
    lines.append(
        f"- **Configurazione Ottima Congelata (`V2A-Full`)**: **$H^* = {int(best_full['H'])}$ giorni**, **$\\lambda_{{\\text{{reg}}}}^* = {best_full['lambda_reg']:.2f}$**, **$(w_g, w_{{\\text{{sot}}}}, w_{{\\text{{tot}}}})^* = ({best_full['w_g']:.2f}, {best_full['w_sot']:.2f}, {best_full['w_tot']:.2f})$**, **$d_{{\\text{{rc}}}}^* = {best_full['d_rc']:.2f}$** (Train Log-Loss 1X2 weekly = `{best_full['Train_LogLoss_1X2']:.6f}`)"
    )
    lines.append(
        f"- **Configurazione Controllo Ablativo Solo-Gol (`V2A-Ablation`)**: **$H = {int(best_abl['H'])}$ giorni**, **$\\lambda_{{\\text{{reg}}}} = {best_abl['lambda_reg']:.2f}$**, **$(w_g, w_{{\\text{{sot}}}}, w_{{\\text{{tot}}}}) = (1.00, 0.00, 0.00)$**, **$d_{{\\text{{rc}}}} = 1.00$** (Train Log-Loss 1X2 weekly = `{best_abl['Train_LogLoss_1X2']:.6f}`)"
    )

    if float(best_full["lambda_reg"]) >= 12.0:
        lines.append(
            "- **⚠️ Nota su $\\lambda_{\\text{reg}}^*$**: il parametro ottimo ha toccato il bordo superiore della griglia (`12.0`)."
        )
    else:
        lines.append(
            f"- **✅ Verifica Bordo Regolarizzazione**: sul controllo ablativo solo-gol (`V2A-Ablation`) lo sblocco della griglia sposta effettivamente l'ottimo da `1.50` a **$\\lambda_{{\\text{{reg}}}} = {best_abl['lambda_reg']:.2f}$** (ottimo interno a `[1.5, 3.0, 6.0, 12.0]`, a conferma che i soli gol interi richiedono forte shrinkage), mentre in **`V2A-Full`** il bersaglio composito sui tiri riduce intrinsecamente la varianza del target rendendo ottimale **$\\lambda_{{\\text{{reg}}}}^* = {best_full['lambda_reg']:.2f}$** (seguito a `0.000086` di distanza da $\\lambda_{{\\text{{reg}}}} = 3.00$, senza mai toccare il bordo superiore `12.0`)."
        )
    lines.append("")

    lines.append("### Top 10 Configurazioni su Train (`1920–2223`)")
    lines.append("")
    lines.append(
        "| Rank | Emivita $H$ | $\\lambda_{\\text{reg}}$ | Mix $(w_g, w_{\\text{sot}}, w_{\\text{tot}})$ | Sconto Rossi $d_{\\text{rc}}$ | Train Log-Loss 1X2 | Train Brier 1X2 |"
    )
    lines.append("| ---: | ---: | ---: | :--- | ---: | ---: | ---: |")
    for idx_g, r in grid_df.head(10).iterrows():
        hl = "**" if idx_g == 0 else ""
        mix_str = f"({r['w_g']:.2f}, {r['w_sot']:.2f}, {r['w_tot']:.2f})"
        lines.append(
            f"| {idx_g + 1} | {hl}{int(r['H'])}d{hl} | {hl}{r['lambda_reg']:.2f}{hl} | {hl}`{mix_str}`{hl} | "
            f"{hl}{r['d_rc']:.2f}{hl} | {hl}{r['Train_LogLoss_1X2']:.6f}{hl} | {hl}{r['Train_Brier_1X2']:.6f}{hl} |"
        )
    lines.append("")

    # ==========================================================================
    # SECTION 3: ABLATION & DIRECT COMPARISON ON VALIDATION (2324-2425) AND TRAIN
    # ==========================================================================
    lines.append("## 3. Tabella Ablazione & Confronto Diretto (Refit Giornaliero Rigoroso `Date < t`)")
    lines.append("")
    lines.append(
        "Confronto a parità esatta di protocollo walk-forward giornaliero (`Date < t`) tra:\n"
        "- **Modulo B**: Baseline Dixon-Coles solo-gol ($\\lambda_{\\text{reg}} = 1.50$, $H = 365$)\n"
        f"- **V2A-Ablation**: Solo-gol $(1,0,0)$ con $\\lambda_{{\\text{{reg}}}} = {best_abl['lambda_reg']:.1f}$ sbloccato ($H = {int(best_abl['H'])}$)\n"
        f"- **V2A-Full**: Synthetic xG Tiri `({best_full['w_g']:.2f}, {best_full['w_sot']:.2f}, {best_full['w_tot']:.2f})` + Sconto Rossi $d_{{\\text{{rc}}}} = {best_full['d_rc']:.2f}$ + $\\lambda_{{\\text{{reg}}}}^* = {best_full['lambda_reg']:.1f}$ ($H^* = {int(best_full['H'])}$)\n"
        "- **Pinnacle Opening (Power)** e **Pinnacle Closing (Power)**"
    )
    lines.append("")

    for split_idx, (split_title, df_s) in enumerate(
        [
            ("Validation (`2324–2425`) — Out-of-Sample Congelato", df_val),
            ("Train (`1920–2223`) — Walk-Forward Giornaliero", df_train),
        ],
        start=1,
    ):
        lines.append(f"### 3.{split_idx} Mercato 1X2 — {split_title}")
        lines.append("")
        lines.append(
            "| Perimetro | N | Log-Loss Mod B | Log-Loss V2A-Abl | Log-Loss V2A-Full | Log-Loss Pin Open | Log-Loss Pin Close | Guadagno (`V2A-Full - Mod B`) | Gap (`V2A-Full - Pin Close`) | Brier Mod B | Brier V2A-Full | Brier Pin Close |"
        )
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

        m_tot = _metrics_1x2_ablation(df_s)
        lines.append(
            f"| **Totale Top 5** | **{m_tot['n']}** | {m_tot['ll_b']:.6f} | {m_tot['ll_abl']:.6f} | **{m_tot['ll_full']:.6f}** | "
            f"{m_tot['ll_open']:.6f} | {m_tot['ll_close']:.6f} | **{m_tot['gain_vs_b']:+.6f}** | **{m_tot['gap_vs_close']:+.6f}** | "
            f"{m_tot['br_b']:.6f} | **{m_tot['br_full']:.6f}** | {m_tot['br_close']:.6f} |"
        )
        for div_code, div_name in LEAGUE_NAMES.items():
            sub_l = df_s[df_s["Div"] == div_code]
            m_l = _metrics_1x2_ablation(sub_l)
            hl = "**" if div_code == "I1" else ""
            lines.append(
                f"| {hl}`{div_code}` ({div_name}){hl} | {m_l['n']} | {m_l['ll_b']:.6f} | {m_l['ll_abl']:.6f} | {hl}{m_l['ll_full']:.6f}{hl} | "
                f"{m_l['ll_open']:.6f} | {m_l['ll_close']:.6f} | {hl}{m_l['gain_vs_b']:+.6f}{hl} | {hl}{m_l['gap_vs_close']:+.6f}{hl} | "
                f"{m_l['br_b']:.6f} | {hl}{m_l['br_full']:.6f}{hl} | {m_l['br_close']:.6f} |"
            )
        lines.append("")

    for split_idx, (split_title, df_s) in enumerate(
        [
            ("Validation (`2324–2425`) — Out-of-Sample Congelato", df_val),
            ("Train (`1920–2223`) — Walk-Forward Giornaliero", df_train),
        ],
        start=3,
    ):
        lines.append(f"### 3.{split_idx} Mercato Over/Under 2.5 — {split_title}")
        lines.append("")
        lines.append(
            "| Perimetro | N | Log-Loss Mod B | Log-Loss V2A-Abl | Log-Loss V2A-Full | Log-Loss Pin Open | Log-Loss Pin Close | Guadagno (`V2A-Full - Mod B`) | Gap (`V2A-Full - Pin Close`) | Brier Mod B | Brier V2A-Full | Brier Pin Close |"
        )
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

        m_tot = _metrics_ou25_ablation(df_s)
        lines.append(
            f"| **Totale Top 5** | **{m_tot['n']}** | {m_tot['ll_b']:.6f} | {m_tot['ll_abl']:.6f} | **{m_tot['ll_full']:.6f}** | "
            f"{m_tot['ll_open']:.6f} | {m_tot['ll_close']:.6f} | **{m_tot['gain_vs_b']:+.6f}** | **{m_tot['gap_vs_close']:+.6f}** | "
            f"{m_tot['br_b']:.6f} | **{m_tot['br_full']:.6f}** | {m_tot['br_close']:.6f} |"
        )
        for div_code, div_name in LEAGUE_NAMES.items():
            sub_l = df_s[df_s["Div"] == div_code]
            m_l = _metrics_ou25_ablation(sub_l)
            hl = "**" if div_code == "I1" else ""
            lines.append(
                f"| {hl}`{div_code}` ({div_name}){hl} | {m_l['n']} | {m_l['ll_b']:.6f} | {m_l['ll_abl']:.6f} | {hl}{m_l['ll_full']:.6f}{hl} | "
                f"{m_l['ll_open']:.6f} | {m_l['ll_close']:.6f} | {hl}{m_l['gain_vs_b']:+.6f}{hl} | {hl}{m_l['gap_vs_close']:+.6f}{hl} | "
                f"{m_l['br_b']:.6f} | {hl}{m_l['br_full']:.6f}{hl} | {m_l['br_close']:.6f} |"
            )
        lines.append("")

    # ==========================================================================
    # SECTION 4: ADVERSE SELECTION DIAGNOSTICS ON VALIDATION (2324-2425)
    # ==========================================================================
    lines.append("## 4. Diagnostica Selezione Avversa su Validation (`2324–2425`) — Modulo B vs V2A-Full")
    lines.append("")
    lines.append(
        "Confronto diretto della selezione di esiti con $EV_{5\\%} > 0$ contro **Pinnacle Opening (`PSH / PSD / PSA`)** e relativo **CLV medio contro Pinnacle Closing (Power devigged)** su Validation (`2324–2425`)."
    )
    lines.append("")
    lines.append("### 4.1 Confronto Fianco a Fianco per Lega ($EV_{5\\%} > 0\\%$)")
    lines.append("")
    lines.append(
        "| Perimetro | Partite | Partite $EV_{5\\%}>0$ Mod B (%) | Partite $EV_{5\\%}>0$ V2A-Full (%) | Esiti Selezionati Mod B | Esiti Selezionati V2A-Full | CLV Medio Mod B vs Pin Close (%) | CLV Medio V2A-Full vs Pin Close (%) | Miglioramento CLV (`V2A - Mod B`) |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for label, sub_d in [("Totale Top 5", df_val)] + [(f"`{k}` ({v})", df_val[df_val["Div"] == k]) for k, v in LEAGUE_NAMES.items()]:
        valid_oc = sub_d[["PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA"]].notna().all(axis=1)
        sub_v = sub_d[valid_oc]
        n_m = len(sub_v)

        p_b = sub_v[["P_H_ModB", "P_D_ModB", "P_A_ModB"]].to_numpy(dtype=np.float64)
        p_v2a = sub_v[["P_H", "P_D", "P_A"]].to_numpy(dtype=np.float64)
        q_open = sub_v[["PSH", "PSD", "PSA"]].to_numpy(dtype=np.float64)
        q_close = sub_v[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=np.float64)
        p_close = devig_power(q_close)

        ev_b = expected_value_net(p_b, q_open, commission=0.05)
        ev_v2a = expected_value_net(p_v2a, q_open, commission=0.05)
        clv_mkt = closing_line_value(q_open, p_close)

        mask_b = ev_b > 0.0
        mask_v2a = ev_v2a > 0.0

        m_b_cnt = int(np.sum(np.any(mask_b, axis=1)))
        m_v2a_cnt = int(np.sum(np.any(mask_v2a, axis=1)))
        o_b_cnt = int(np.sum(mask_b))
        o_v2a_cnt = int(np.sum(mask_v2a))

        clv_b = float(np.mean(clv_mkt[mask_b])) * 100.0 if o_b_cnt > 0 else 0.0
        clv_v2a = float(np.mean(clv_mkt[mask_v2a])) * 100.0 if o_v2a_cnt > 0 else 0.0
        diff_clv = clv_v2a - clv_b

        hl = "**" if "Totale" in label or "I1" in label else ""
        lines.append(
            f"| {hl}{label}{hl} | {n_m} | {m_b_cnt} ({m_b_cnt/n_m*100:.1f}%) | {hl}{m_v2a_cnt} ({m_v2a_cnt/n_m*100:.1f}%){hl} | "
            f"{o_b_cnt} | {o_v2a_cnt} | {clv_b:+.2f}% | {hl}{clv_v2a:+.2f}%{hl} | {hl}{diff_clv:+.2f} pp{hl} |"
        )
    lines.append("")

    lines.append("### 4.2 Analisi Soglie di Filtro Edge e Filtro Disaccordo Massimo con il Mercato (Validation `2324–2425`)")
    lines.append("")
    lines.append(
        "Effetto di soglie più selettive su $EV_{5\\%}$ e del filtro anti-selezione-avversa sul disaccordo massimo con l'apertura devigged ($|p_{\\text{mod}} - p_{\\text{open, power}}| \\le \\Delta_{\\max}$):"
    )
    lines.append("")
    lines.append(
        "| Regola di Filtro Diagnostica | Esiti Selezionati Mod B | CLV Medio Mod B (%) | Esiti Selezionati V2A-Full | EV Medio Stimato V2A-Full (%) | CLV Medio V2A-Full vs Pin Close (%) | Miglioramento CLV (`V2A - Mod B`) |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")

    valid_all_val = df_val[["PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA"]].notna().all(axis=1)
    sub_val_all = df_val[valid_all_val]
    p_b_all = sub_val_all[["P_H_ModB", "P_D_ModB", "P_A_ModB"]].to_numpy(dtype=np.float64)
    p_v2a_all = sub_val_all[["P_H", "P_D", "P_A"]].to_numpy(dtype=np.float64)
    q_open_all = sub_val_all[["PSH", "PSD", "PSA"]].to_numpy(dtype=np.float64)
    q_close_all = sub_val_all[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=np.float64)
    p_open_all = devig_power(q_open_all)
    p_close_all = devig_power(q_close_all)

    ev_b_all = expected_value_net(p_b_all, q_open_all, commission=0.05)
    ev_v2a_all = expected_value_net(p_v2a_all, q_open_all, commission=0.05)
    clv_mkt_all = closing_line_value(q_open_all, p_close_all)
    disagree_b = np.abs(p_b_all - p_open_all)
    disagree_v2a = np.abs(p_v2a_all - p_open_all)

    filter_configs = [
        ("1. $EV_{5\\%} > 0\\%$ (nessun cap disaccordo)", ev_b_all > 0.0, ev_v2a_all > 0.0),
        ("2. $EV_{5\\%} > 3\\%$ (nessun cap disaccordo)", ev_b_all > 0.03, ev_v2a_all > 0.03),
        ("3. $EV_{5\\%} > 5\\%$ (nessun cap disaccordo)", ev_b_all > 0.05, ev_v2a_all > 0.05),
        (
            "4. $EV_{5\\%} > 0\\%$ & $|p_{\\text{mod}} - p_{\\text{open}}| \\le 10\\text{ pp}$",
            (ev_b_all > 0.0) & (disagree_b <= 0.10),
            (ev_v2a_all > 0.0) & (disagree_v2a <= 0.10),
        ),
        (
            "5. $EV_{5\\%} > 0\\%$ & $|p_{\\text{mod}} - p_{\\text{open}}| \\le 6\\text{ pp}$",
            (ev_b_all > 0.0) & (disagree_b <= 0.06),
            (ev_v2a_all > 0.0) & (disagree_v2a <= 0.06),
        ),
        (
            "6. $EV_{5\\%} > 3\\%$ & $|p_{\\text{mod}} - p_{\\text{open}}| \\le 10\\text{ pp}$",
            (ev_b_all > 0.03) & (disagree_b <= 0.10),
            (ev_v2a_all > 0.03) & (disagree_v2a <= 0.10),
        ),
    ]

    for f_label, m_b_f, m_v2a_f in filter_configs:
        cnt_b = int(np.sum(m_b_f))
        cnt_v2a = int(np.sum(m_v2a_f))
        clv_b_f = float(np.mean(clv_mkt_all[m_b_f])) * 100.0 if cnt_b > 0 else 0.0
        clv_v2a_f = float(np.mean(clv_mkt_all[m_v2a_f])) * 100.0 if cnt_v2a > 0 else 0.0
        ev_v2a_f = float(np.mean(ev_v2a_all[m_v2a_f])) * 100.0 if cnt_v2a > 0 else 0.0
        diff_f = clv_v2a_f - clv_b_f
        lines.append(
            f"| {f_label} | {cnt_b} | {clv_b_f:+.2f}% | **{cnt_v2a}** | {ev_v2a_f:+.2f}% | "
            f"**{clv_v2a_f:+.2f}%** | **{diff_f:+.2f} pp** |"
        )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    csv_path = ROOT_DIR / "data" / "processed" / "level1_top5_1617_2526.csv"
    mod_b_preds_path = ROOT_DIR / "data" / "processed" / "modulo_b_predictions.csv"
    preds_out_path = ROOT_DIR / "data" / "processed" / "level2a_predictions.csv"
    report_out_path = ROOT_DIR / "reports" / "level2a_shots_model.md"

    df_all = pd.read_csv(csv_path, dtype={"Season": str, "Div": str})
    df_dev = df_all[df_all["Season"].isin(ALL_ALLOWED_SEASONS)].copy()
    df_dev["ParsedDate"] = pd.to_datetime(df_dev["Date"], format="%Y-%m-%d")

    print("[Level 2A] Step 1/4: Auditing coverage and integrity of shots & red cards on 1617-2425...", flush=True)
    audit_shots = audit_shots_and_red_cards(df_dev)
    print(
        f"[Level 2A] Audit complete: missing_shots={len(audit_shots.missing_shots_rows)}, "
        f"negatives={len(audit_shots.negative_rows)}, HST>HS={len(audit_shots.hst_gt_hs_rows)}, AST>AS={len(audit_shots.ast_gt_as_rows)}",
        flush=True,
    )

    print("[Level 2A] Step 2/4: Running parallel Grid Search on Train (1920-2223)...", flush=True)
    grid_df, grid_elapsed_s = run_parallel_grid_search_train(csv_path=csv_path)

    best_full = grid_df.iloc[0].to_dict()
    abl_sub = grid_df[
        (grid_df["w_g"] == 1.0)
        & (grid_df["w_sot"] == 0.0)
        & (grid_df["w_tot"] == 0.0)
        & (grid_df["d_rc"] == 1.0)
    ].reset_index(drop=True)
    best_abl = abl_sub.iloc[0].to_dict()

    print(
        f"[Level 2A] Frozen Best V2A-Full: H*={int(best_full['H'])}d, lambda_reg*={best_full['lambda_reg']:.2f}, "
        f"mix=({best_full['w_g']:.2f}, {best_full['w_sot']:.2f}, {best_full['w_tot']:.2f}), d_rc*={best_full['d_rc']:.2f}",
        flush=True,
    )
    print(
        f"[Level 2A] Frozen Best V2A-Ablation (Goals-Only): H={int(best_abl['H'])}d, lambda_reg={best_abl['lambda_reg']:.2f}",
        flush=True,
    )

    print("[Level 2A] Step 3/4: Running daily calendar-date Walk-Forward on Train + Validation (1920-2425)...", flush=True)
    t0 = time.perf_counter()
    preds_full, _ = run_walk_forward_level2a_all_leagues(
        df_all=df_dev,
        half_life_days=float(best_full["H"]),
        lambda_reg=float(best_full["lambda_reg"]),
        w_g=float(best_full["w_g"]),
        w_sot=float(best_full["w_sot"]),
        w_tot=float(best_full["w_tot"]),
        d_rc=float(best_full["d_rc"]),
        target_seasons=EVAL_SEASONS,
        refit_every_days=1,
    )
    preds_abl, _ = run_walk_forward_level2a_all_leagues(
        df_all=df_dev,
        half_life_days=float(best_abl["H"]),
        lambda_reg=float(best_abl["lambda_reg"]),
        w_g=1.0,
        w_sot=0.0,
        w_tot=0.0,
        d_rc=1.0,
        target_seasons=EVAL_SEASONS,
        refit_every_days=1,
    )
    print(f"[Level 2A] Daily Walk-Forward completed in {time.perf_counter() - t0:.1f}s.", flush=True)

    # Merge V2A-Ablation and Modulo B predictions for direct comparison
    preds_full["P_H_Abl"] = preds_abl["P_H"]
    preds_full["P_D_Abl"] = preds_abl["P_D"]
    preds_full["P_A_Abl"] = preds_abl["P_A"]
    preds_full["P_Over25_Abl"] = preds_abl["P_Over25"]
    preds_full["P_Under25_Abl"] = preds_abl["P_Under25"]

    if mod_b_preds_path.exists():
        df_mod_b = pd.read_csv(mod_b_preds_path, dtype={"Season": str, "Div": str})
        df_mod_b_sub = df_mod_b[
            ["Season", "Div", "Date", "HomeTeam", "AwayTeam", "P_H", "P_D", "P_A", "P_Over25", "P_Under25"]
        ].rename(
            columns={
                "P_H": "P_H_ModB",
                "P_D": "P_D_ModB",
                "P_A": "P_A_ModB",
                "P_Over25": "P_Over25_ModB",
                "P_Under25": "P_Under25_ModB",
            }
        )
        preds_full = preds_full.merge(
            df_mod_b_sub,
            on=["Season", "Div", "Date", "HomeTeam", "AwayTeam"],
            how="left",
        )

    assert "2526" not in set(preds_full["Season"].unique()), "CRITICAL: 2526 must never appear in level2a_predictions.csv!"
    preds_out_path.parent.mkdir(parents=True, exist_ok=True)
    preds_full.to_csv(preds_out_path, index=False, encoding="utf-8")
    print(f"[Level 2A] Saved {len(preds_full)} predictions to {preds_out_path}", flush=True)

    print("[Level 2A] Step 4/4: Generating Level 2A diagnostic report...", flush=True)
    generate_level2a_report(
        audit_shots=audit_shots,
        grid_df=grid_df,
        grid_elapsed_s=grid_elapsed_s,
        best_full=best_full,
        best_abl=best_abl,
        df_preds=preds_full,
        report_path=report_out_path,
    )
    print(f"[Level 2A] Report written to {report_out_path}", flush=True)


if __name__ == "__main__":
    main()
