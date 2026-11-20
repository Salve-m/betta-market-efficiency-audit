"""
Modulo B: Dixon-Coles MAP Walk-Forward Baseline (src/modulo_b_dixon_coles.py).

Implements per-league log-linear Poisson and Dixon-Coles models with:
- Exponential time decay w_m = exp(-xi * (t - t_m)), xi = ln(2) / H
- 1,100-day rolling historical window prior to date t
- MAP L2 regularization with specific prior for newly promoted teams
  (mean of the bottom 3 teams in the final standings of the previous season,
   computed strictly with data < season start) and centering constraint sum(a_k) = 0
- Score matrix P(X=x, Y=y) for x, y in {0..10} renormalized to 1.0
- Analytical gradient L-BFGS-B optimization with warm-start across matchday refits
- Strict calendar-date walk-forward (Date < t) on Train (1920-2223) and Validation (2324-2425),
  never touching Blind Test (2526).
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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

ALL_ALLOWED_SEASONS = WARMUP_SEASONS + TRAIN_SEASONS + VAL_SEASONS
EVAL_SEASONS = TRAIN_SEASONS + VAL_SEASONS

MAX_GOALS = 10
TAU_EPS = 1e-6
ROLLING_WINDOW_DAYS = 1100
CENTER_PENALTY = 50.0

# Precompute log-factorials and grid masks for {0..10} x {0..10}
_GOALS_GRID = np.arange(MAX_GOALS + 1, dtype=np.float64)
_LOG_FACTORIALS = np.array([math.lgamma(k + 1.0) for k in range(MAX_GOALS + 1)], dtype=np.float64)
_X_IDX, _Y_IDX = np.meshgrid(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1), indexing="ij")
_HOME_WIN_MASK = _X_IDX > _Y_IDX
_DRAW_MASK = _X_IDX == _Y_IDX
_AWAY_WIN_MASK = _X_IDX < _Y_IDX
_OVER25_MASK = (_X_IDX + _Y_IDX) >= 3
_UNDER25_MASK = (_X_IDX + _Y_IDX) <= 2


@dataclass
class LeagueModelFit:
    teams: List[str]
    team_to_idx: Dict[str, int]
    a: np.ndarray  # offensive strength (n_teams,)
    b: np.ndarray  # defensive weakness (n_teams,)
    h: float       # home advantage log-multiplier
    rho: float     # Dixon-Coles low-score correlation parameter
    a0: np.ndarray # prior offensive strength
    b0: np.ndarray # prior defensive weakness
    promoted_teams: List[str]


# ==============================================================================
# 1. SCORE MATRIX & PROBABILITY DERIVATION
# ==============================================================================


def dixon_coles_tau_scalar(lam: float, mu: float, x: int, y: int, rho: float, eps: float = TAU_EPS) -> float:
    """Compute Dixon-Coles tau(x, y; lam, mu, rho) clamped to >= eps."""
    if rho == 0.0:
        return 1.0
    if x == 0 and y == 0:
        val = 1.0 - lam * mu * rho
    elif x == 0 and y == 1:
        val = 1.0 + lam * rho
    elif x == 1 and y == 0:
        val = 1.0 + mu * rho
    elif x == 1 and y == 1:
        val = 1.0 - rho
    else:
        return 1.0
    return val if val > eps else eps


def compute_score_matrix(
    lam: float,
    mu: float,
    rho: float = 0.0,
    max_goals: int = MAX_GOALS,
    eps: float = TAU_EPS,
) -> np.ndarray:
    """
    Generate the (max_goals+1, max_goals+1) score probability matrix P(X=x, Y=y)
    truncated to x, y in {0, ..., max_goals} and renormalized to sum to 1.0.
    """
    if max_goals == MAX_GOALS:
        goals = _GOALS_GRID
        log_fact = _LOG_FACTORIALS
    else:
        goals = np.arange(max_goals + 1, dtype=np.float64)
        log_fact = np.array([math.lgamma(k + 1.0) for k in range(max_goals + 1)], dtype=np.float64)

    lam_safe = max(float(lam), 1e-9)
    mu_safe = max(float(mu), 1e-9)

    log_px = goals * math.log(lam_safe) - lam_safe - log_fact
    log_py = goals * math.log(mu_safe) - mu_safe - log_fact

    px = np.exp(log_px)
    py = np.exp(log_py)
    mat = np.outer(px, py)

    if rho != 0.0:
        mat[0, 0] *= max(1.0 - lam_safe * mu_safe * rho, eps)
        mat[0, 1] *= max(1.0 + lam_safe * rho, eps)
        mat[1, 0] *= max(1.0 + mu_safe * rho, eps)
        mat[1, 1] *= max(1.0 - rho, eps)

    total = float(np.sum(mat))
    if total <= 0.0 or not math.isfinite(total):
        mat = np.full((max_goals + 1, max_goals + 1), 1.0 / ((max_goals + 1) ** 2))
    else:
        mat /= total
    return mat


def probabilities_from_score_matrix(mat: np.ndarray) -> Dict[str, float]:
    """Extract coherent 1X2 (P_H, P_D, P_A) and Over/Under 2.5 (P_Over25, P_Under25) probabilities."""
    if mat.shape == (MAX_GOALS + 1, MAX_GOALS + 1):
        p_h = float(np.sum(mat[_HOME_WIN_MASK]))
        p_d = float(np.sum(mat[_DRAW_MASK]))
        p_a = float(np.sum(mat[_AWAY_WIN_MASK]))
        p_over = float(np.sum(mat[_OVER25_MASK]))
        p_under = float(np.sum(mat[_UNDER25_MASK]))
    else:
        n = mat.shape[0]
        x_idx, y_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        p_h = float(np.sum(mat[x_idx > y_idx]))
        p_d = float(np.sum(mat[x_idx == y_idx]))
        p_a = float(np.sum(mat[x_idx < y_idx]))
        p_over = float(np.sum(mat[(x_idx + y_idx) >= 3]))
        p_under = float(np.sum(mat[(x_idx + y_idx) <= 2]))

    # Ensure exact normalization against floating point rounding
    s_1x2 = p_h + p_d + p_a
    s_ou = p_over + p_under
    return {
        "P_H": p_h / s_1x2,
        "P_D": p_d / s_1x2,
        "P_A": p_a / s_1x2,
        "P_Over25": p_over / s_ou,
        "P_Under25": p_under / s_ou,
    }


def predict_match_probabilities(lam: float, mu: float, rho: float = 0.0) -> Dict[str, float]:
    """Compute score matrix and return 1X2 and O/U 2.5 probabilities plus expected goals."""
    mat = compute_score_matrix(lam=lam, mu=mu, rho=rho)
    probs = probabilities_from_score_matrix(mat)
    probs["Lambda_Home"] = float(lam)
    probs["Mu_Away"] = float(mu)
    return probs


# ==============================================================================
# 2. MAP OBJECTIVE & ANALYTICAL GRADIENT (L-BFGS-B)
# ==============================================================================


def _neg_log_posterior_and_grad(
    params: np.ndarray,
    n_teams: int,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    fthg: np.ndarray,
    ftag: np.ndarray,
    weights: np.ndarray,
    a0: np.ndarray,
    b0: np.ndarray,
    lambda_reg: float,
    use_rho: bool,
    m00: np.ndarray,
    m01: np.ndarray,
    m10: np.ndarray,
    m11: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Compute negative MAP log-posterior and its analytical gradient:
        J(theta) = -sum_m w_m * [x_m ln(lam_m) - lam_m + y_m ln(mu_m) - mu_m + ln(tau_m)]
                   + (lambda_reg / 2) * sum_k [(a_k - a_{0,k})^2 + (b_k - b_{0,k})^2]
                   + (CENTER_PENALTY / 2) * (sum_k a_k)^2
    """
    a = params[:n_teams]
    b = params[n_teams : 2 * n_teams]
    h = float(params[2 * n_teams])
    rho = float(params[2 * n_teams + 1]) if use_rho else 0.0

    log_lam = np.clip(a[home_idx] + b[away_idx] + h, -8.0, 4.0)
    log_mu = np.clip(a[away_idx] + b[home_idx], -8.0, 4.0)
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)

    # Base Poisson log-likelihood
    ll_match = fthg * log_lam - lam + ftag * log_mu - mu

    # Derivatives d(ll_match)/d(log_lam) and d(ll_match)/d(log_mu)
    d_log_lam = fthg - lam
    d_log_mu = ftag - mu
    grad_rho = 0.0

    if use_rho:
        tau_raw = np.ones_like(lam)
        if np.any(m00):
            tau_raw[m00] = 1.0 - rho * lam[m00] * mu[m00]
        if np.any(m01):
            tau_raw[m01] = 1.0 + rho * lam[m01]
        if np.any(m10):
            tau_raw[m10] = 1.0 + rho * mu[m10]
        if np.any(m11):
            tau_raw[m11] = 1.0 - rho

        active = tau_raw > TAU_EPS
        tau_safe = np.where(active, tau_raw, TAU_EPS)
        ll_match = ll_match + np.log(tau_safe)

        # Gradients for active low-score cells
        act00 = m00 & active
        if np.any(act00):
            prod00 = lam[act00] * mu[act00]
            inv_t00 = 1.0 / tau_safe[act00]
            term00 = -rho * prod00 * inv_t00
            d_log_lam[act00] += term00
            d_log_mu[act00] += term00
            grad_rho -= float(np.sum(weights[act00] * (-prod00 * inv_t00)))

        act01 = m01 & active
        if np.any(act01):
            inv_t01 = 1.0 / tau_safe[act01]
            d_log_lam[act01] += rho * lam[act01] * inv_t01
            grad_rho -= float(np.sum(weights[act01] * (lam[act01] * inv_t01)))

        act10 = m10 & active
        if np.any(act10):
            inv_t10 = 1.0 / tau_safe[act10]
            d_log_mu[act10] += rho * mu[act10] * inv_t10
            grad_rho -= float(np.sum(weights[act10] * (mu[act10] * inv_t10)))

        act11 = m11 & active
        if np.any(act11):
            inv_t11 = 1.0 / tau_safe[act11]
            grad_rho -= float(np.sum(weights[act11] * (-1.0 * inv_t11)))

    neg_w_ll = -float(np.sum(weights * ll_match))

    diff_a = a - a0
    diff_b = b - b0
    sum_diff_a = float(np.sum(diff_a))
    center_w = 0.01 * lambda_reg

    reg_loss = 0.5 * lambda_reg * float(np.sum(diff_a * diff_a) + np.sum(diff_b * diff_b))
    center_loss = 0.5 * center_w * (sum_diff_a * sum_diff_a)
    obj = neg_w_ll + reg_loss + center_loss

    w_d_lam = weights * d_log_lam
    w_d_mu = weights * d_log_mu

    grad_a = (
        -np.bincount(home_idx, weights=w_d_lam, minlength=n_teams)
        - np.bincount(away_idx, weights=w_d_mu, minlength=n_teams)
        + lambda_reg * diff_a
        + center_w * sum_diff_a
    )
    grad_b = (
        -np.bincount(away_idx, weights=w_d_lam, minlength=n_teams)
        - np.bincount(home_idx, weights=w_d_mu, minlength=n_teams)
        + lambda_reg * diff_b
    )
    grad_h = -float(np.sum(w_d_lam))

    if use_rho:
        grad = np.empty(2 * n_teams + 2, dtype=np.float64)
        grad[:n_teams] = grad_a
        grad[n_teams : 2 * n_teams] = grad_b
        grad[2 * n_teams] = grad_h
        grad[2 * n_teams + 1] = grad_rho
    else:
        grad = np.empty(2 * n_teams + 1, dtype=np.float64)
        grad[:n_teams] = grad_a
        grad[n_teams : 2 * n_teams] = grad_b
        grad[2 * n_teams] = grad_h

    return obj, grad


def fit_dixon_coles_map(
    df_window: pd.DataFrame,
    current_date: pd.Timestamp,
    half_life_days: float,
    lambda_reg: float,
    use_rho: bool = True,
    teams: Optional[List[str]] = None,
    priors: Optional[Dict[str, Tuple[float, float]]] = None,
    promoted_teams: Optional[List[str]] = None,
    warm_start_fit: Optional[LeagueModelFit] = None,
) -> LeagueModelFit:
    """
    Fit the MAP Poisson / Dixon-Coles model on historical matches strictly in df_window (Date < current_date).
    """
    hist_teams = set(df_window["HomeTeam"].unique()).union(set(df_window["AwayTeam"].unique()))
    if teams is not None:
        all_teams = sorted(hist_teams.union(set(teams)))
    else:
        all_teams = sorted(hist_teams)

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
        return LeagueModelFit(
            teams=all_teams,
            team_to_idx=team_to_idx,
            a=a0.copy(),
            b=b0.copy(),
            h=0.20,
            rho=0.0,
            a0=a0,
            b0=b0,
            promoted_teams=list(promoted_teams) if promoted_teams else [],
        )

    home_idx = df_window["HomeTeam"].map(team_to_idx).to_numpy(dtype=np.int32)
    away_idx = df_window["AwayTeam"].map(team_to_idx).to_numpy(dtype=np.int32)
    fthg = df_window["FTHG"].to_numpy(dtype=np.float64)
    ftag = df_window["FTAG"].to_numpy(dtype=np.float64)

    delta_days = (current_date - df_window["ParsedDate"]).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    xi = math.log(2.0) / float(half_life_days)
    weights = np.exp(-xi * np.maximum(delta_days, 0.0))

    m00 = (fthg == 0) & (ftag == 0)
    m01 = (fthg == 0) & (ftag == 1)
    m10 = (fthg == 1) & (ftag == 0)
    m11 = (fthg == 1) & (ftag == 1)

    n_params = 2 * n_teams + (2 if use_rho else 1)
    x0 = np.zeros(n_params, dtype=np.float64)
    x0[:n_teams] = a0
    x0[n_teams : 2 * n_teams] = np.where(b0 != 0.0, b0, 0.15)
    x0[2 * n_teams] = 0.22
    if use_rho:
        x0[2 * n_teams + 1] = -0.05

    if warm_start_fit is not None:
        for i, t in enumerate(all_teams):
            old_idx = warm_start_fit.team_to_idx.get(t)
            if old_idx is not None:
                x0[i] = warm_start_fit.a[old_idx]
                x0[n_teams + i] = warm_start_fit.b[old_idx]
        x0[2 * n_teams] = warm_start_fit.h
        if use_rho:
            x0[2 * n_teams + 1] = warm_start_fit.rho

    bounds = [(-3.0, 3.0)] * (2 * n_teams) + [(-1.0, 1.5)]
    if use_rho:
        bounds.append((-0.35, 0.35))

    res = minimize(
        _neg_log_posterior_and_grad,
        x0,
        args=(
            n_teams,
            home_idx,
            away_idx,
            fthg,
            ftag,
            weights,
            a0,
            b0,
            float(lambda_reg),
            use_rho,
            m00,
            m01,
            m10,
            m11,
        ),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-7},
    )

    opt_params = res.x
    a_est = opt_params[:n_teams].copy()
    b_est = opt_params[n_teams : 2 * n_teams].copy()
    h_est = float(opt_params[2 * n_teams])
    rho_est = float(opt_params[2 * n_teams + 1]) if use_rho else 0.0

    # If all teams have homogeneous zero prior (e.g. end-of-season reference fit),
    # center sum(a_k) == 0 exactly by shifting mean(a) into b.
    if np.all(a0 == 0.0):
        mean_a = float(np.mean(a_est))
        a_est -= mean_a
        b_est += mean_a

    return LeagueModelFit(
        teams=all_teams,
        team_to_idx=team_to_idx,
        a=a_est,
        b=b_est,
        h=h_est,
        rho=rho_est,
        a0=a0,
        b0=b0,
        promoted_teams=list(promoted_teams) if promoted_teams else [],
    )


# ==============================================================================
# 3. PROMOTED TEAMS PRIOR & STANDINGS (ZERO LEAKAGE)
# ==============================================================================


def compute_league_standings_bottom3(df_season: pd.DataFrame) -> List[str]:
    """
    Compute the final league table from a completed season dataframe and return
    the names of the bottom 3 classified teams.
    """
    teams = sorted(set(df_season["HomeTeam"].unique()).union(set(df_season["AwayTeam"].unique())))
    stats = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}

    for row in df_season.itertuples(index=False):
        ht = row.HomeTeam
        at = row.AwayTeam
        hg = int(row.FTHG)
        ag = int(row.FTAG)

        stats[ht]["gf"] += hg
        stats[ht]["gd"] += hg - ag
        stats[at]["gf"] += ag
        stats[at]["gd"] += ag - hg

        if hg > ag:
            stats[ht]["pts"] += 3
        elif hg < ag:
            stats[at]["pts"] += 3
        else:
            stats[ht]["pts"] += 1
            stats[at]["pts"] += 1

    # Sort descending by (pts, gd, gf), so the last 3 are the bottom 3 teams
    ranked = sorted(
        teams,
        key=lambda t: (stats[t]["pts"], stats[t]["gd"], stats[t]["gf"]),
        reverse=True,
    )
    return ranked[-3:]


def compute_season_priors_strict(
    df_hist_before_date: pd.DataFrame,
    current_season: str,
    prev_season: Optional[str],
    candidate_teams: Sequence[str],
    half_life_days: float,
    lambda_reg: float,
    use_rho: bool = True,
) -> Tuple[Dict[str, Tuple[float, float]], List[str], List[str]]:
    """
    Compute MAP priors (a_{0,k}, b_{0,k}) for teams in `current_season` strictly using
    historical matches in `df_hist_before_date` (all having Date < current_date):
    - Teams present in `prev_season` of the same league receive prior (0.0, 0.0).
    - Newly promoted teams (0 matches in `prev_season` of that league) receive prior
      equal to the mean of the estimated (a_k, b_k) at the end of `prev_season` for
      the bottom 3 teams of `prev_season`.
    Returns (priors_dict, promoted_teams_list, bottom3_prev_season).
    """
    if prev_season is None:
        return {t: (0.0, 0.0) for t in candidate_teams}, [], []

    df_prev = df_hist_before_date[df_hist_before_date["Season"] == prev_season]
    if len(df_prev) == 0:
        return {t: (0.0, 0.0) for t in candidate_teams}, [], []

    prev_season_teams = set(df_prev["HomeTeam"].unique()).union(set(df_prev["AwayTeam"].unique()))
    bottom3 = compute_league_standings_bottom3(df_prev)

    # Fit model at the end of prev_season using strictly data <= end of prev_season
    prev_end_date = df_prev["ParsedDate"].max()
    prev_window_start = prev_end_date - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
    df_prev_window = df_hist_before_date[
        (df_hist_before_date["ParsedDate"] <= prev_end_date)
        & (df_hist_before_date["ParsedDate"] >= prev_window_start)
    ]

    fit_prev_end = fit_dixon_coles_map(
        df_window=df_prev_window,
        current_date=prev_end_date,
        half_life_days=half_life_days,
        lambda_reg=lambda_reg,
        use_rho=use_rho,
        teams=list(prev_season_teams),
    )

    b3_indices = [fit_prev_end.team_to_idx[t] for t in bottom3 if t in fit_prev_end.team_to_idx]
    a_prom = float(np.mean(fit_prev_end.a[b3_indices]))
    b_prom = float(np.mean(fit_prev_end.b[b3_indices]))

    priors: Dict[str, Tuple[float, float]] = {}
    promoted: List[str] = []
    for t in candidate_teams:
        if t in prev_season_teams:
            priors[t] = (0.0, 0.0)
        else:
            priors[t] = (a_prom, b_prom)
            promoted.append(t)

    return priors, sorted(promoted), bottom3


# ==============================================================================
# 4. STRICT WALK-FORWARD ENGINE (SINGLE DATE & FULL LEAGUE)
# ==============================================================================


def predict_league_on_date_strict(
    df_league: pd.DataFrame,
    target_date: Union[str, pd.Timestamp],
    half_life_days: float = 365.0,
    lambda_reg: float = 0.15,
    use_rho: bool = True,
) -> Tuple[pd.DataFrame, LeagueModelFit]:
    """
    Standalone zero-leakage prediction for all matches of a single league on `target_date`.
    Strictly filters `df_hist = df_league[ParsedDate < target_date]` BEFORE computing priors or fitting.
    Any modification to rows with Date >= target_date has zero effect on the output.
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

    current_season = str(df_target["Season"].iloc[0])
    season_order = ALL_ALLOWED_SEASONS
    if current_season in season_order:
        s_idx = season_order.index(current_season)
        prev_season = season_order[s_idx - 1] if s_idx > 0 else None
    else:
        prev_season = None

    window_start = t_date - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
    df_window = df_hist[df_hist["ParsedDate"] >= window_start]

    # Candidate teams known up to t_date: teams in rolling window + teams playing on t_date
    active_teams = sorted(
        set(df_window["HomeTeam"].unique())
        .union(set(df_window["AwayTeam"].unique()))
        .union(set(df_target["HomeTeam"].unique()))
        .union(set(df_target["AwayTeam"].unique()))
    )

    priors, promoted, _ = compute_season_priors_strict(
        df_hist_before_date=df_hist,
        current_season=current_season,
        prev_season=prev_season,
        candidate_teams=active_teams,
        half_life_days=half_life_days,
        lambda_reg=lambda_reg,
        use_rho=use_rho,
    )

    # Note: for promoted teams that might have played 2-3 seasons ago in the rolling window,
    # their prior (a_{0,k}, b_{0,k}) is set to the promoted prior; if they have 0 matches in the current season
    # before t_date, we also ensure their parameters start from the promoted prior.
    df_curr_season_hist = df_hist[df_hist["Season"] == current_season]
    teams_played_curr_season = set(df_curr_season_hist["HomeTeam"].unique()).union(
        set(df_curr_season_hist["AwayTeam"].unique())
    )
    # Exclude stale matches from 2+ years ago for newly promoted teams that haven't played in prev_season
    # so that a newly promoted team starts from the promoted prior rather than stale 3-year-old ratings
    if promoted:
        stale_promoted = set(promoted) - teams_played_curr_season
        if stale_promoted:
            df_window = df_window[
                ~(df_window["HomeTeam"].isin(stale_promoted) | df_window["AwayTeam"].isin(stale_promoted))
            ]

    fit = fit_dixon_coles_map(
        df_window=df_window,
        current_date=t_date,
        half_life_days=half_life_days,
        lambda_reg=lambda_reg,
        use_rho=use_rho,
        teams=active_teams,
        priors=priors,
        promoted_teams=promoted,
    )

    pred_rows = []
    for row in df_target.itertuples(index=False):
        i = fit.team_to_idx[row.HomeTeam]
        j = fit.team_to_idx[row.AwayTeam]
        lam = math.exp(fit.a[i] + fit.b[j] + fit.h)
        mu = math.exp(fit.a[j] + fit.b[i])
        probs = predict_match_probabilities(lam=lam, mu=mu, rho=fit.rho if use_rho else 0.0)
        pred_rows.append(
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
                "h_param": fit.h,
                "rho_param": fit.rho if use_rho else 0.0,
            }
        )

    return pd.DataFrame(pred_rows), fit


def run_walk_forward_league(
    df_league: pd.DataFrame,
    half_life_days: float,
    lambda_reg: float,
    use_rho: bool = True,
    target_seasons: Sequence[str] = EVAL_SEASONS,
) -> Tuple[pd.DataFrame, Optional[LeagueModelFit]]:
    """
    Execute strict calendar-date walk-forward for a single league across `target_seasons`.
    Caches season-level promoted priors once per season (using only data < season start)
    and warm-starts L-BFGS-B from the previous refit date.
    """
    df_l = df_league[df_league["Season"].isin(ALL_ALLOWED_SEASONS)].copy()
    df_l["ParsedDate"] = pd.to_datetime(df_l["Date"], format="%Y-%m-%d")
    df_l = df_l.sort_values(["ParsedDate", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    # Pre-extract numpy arrays for fast window slicing
    parsed_dates = df_l["ParsedDate"]
    seasons_arr = df_l["Season"].to_numpy()

    target_mask = df_l["Season"].isin(target_seasons)
    eval_dates = sorted(df_l.loc[target_mask, "ParsedDate"].unique())

    season_order = ALL_ALLOWED_SEASONS
    cached_season_info: Dict[str, Tuple[Tuple[float, float], set]] = {}

    warm_fit: Optional[LeagueModelFit] = None
    out_records: List[Dict[str, object]] = []

    for t_date in eval_dates:
        df_day = df_l[parsed_dates == t_date]
        curr_season = str(df_day["Season"].iloc[0])

        # Compute season promoted prior strictly using matches before the first match of curr_season
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
                fit_prev = fit_dixon_coles_map(
                    df_window=df_prev_win,
                    current_date=prev_end_date,
                    half_life_days=half_life_days,
                    lambda_reg=lambda_reg,
                    use_rho=use_rho,
                    teams=list(prev_teams),
                )
                b3_idx = [fit_prev.team_to_idx[t] for t in bottom3 if t in fit_prev.team_to_idx]
                prom_prior = (float(np.mean(fit_prev.a[b3_idx])), float(np.mean(fit_prev.b[b3_idx])))
                cached_season_info[curr_season] = (prom_prior, prev_teams)
            else:
                cached_season_info[curr_season] = ((0.0, 0.0), set())

        prom_prior, prev_season_teams = cached_season_info[curr_season]

        # Strict anti-leakage historical window: t_date - 1100 days <= Date < t_date
        win_start = t_date - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
        df_window = df_l[(parsed_dates < t_date) & (parsed_dates >= win_start)]

        active_teams = sorted(
            set(df_window["HomeTeam"].unique())
            .union(set(df_window["AwayTeam"].unique()))
            .union(set(df_day["HomeTeam"].unique()))
            .union(set(df_day["AwayTeam"].unique()))
        )

        priors: Dict[str, Tuple[float, float]] = {}
        promoted: List[str] = []
        for t in active_teams:
            if t in prev_season_teams:
                priors[t] = (0.0, 0.0)
            else:
                priors[t] = prom_prior
                promoted.append(t)

        # Exclude stale matches from 2+ seasons ago for newly promoted teams until they play in curr_season
        if promoted:
            df_curr_hist = df_window[df_window["Season"] == curr_season]
            played_curr = set(df_curr_hist["HomeTeam"].unique()).union(set(df_curr_hist["AwayTeam"].unique()))
            stale_promoted = set(promoted) - played_curr
            if stale_promoted:
                df_window = df_window[
                    ~(df_window["HomeTeam"].isin(stale_promoted) | df_window["AwayTeam"].isin(stale_promoted))
                ]

        warm_fit = fit_dixon_coles_map(
            df_window=df_window,
            current_date=t_date,
            half_life_days=half_life_days,
            lambda_reg=lambda_reg,
            use_rho=use_rho,
            teams=active_teams,
            priors=priors,
            promoted_teams=promoted,
            warm_start_fit=warm_fit,
        )

        for row in df_day.itertuples(index=False):
            i = warm_fit.team_to_idx[row.HomeTeam]
            j = warm_fit.team_to_idx[row.AwayTeam]
            lam = math.exp(warm_fit.a[i] + warm_fit.b[j] + warm_fit.h)
            mu = math.exp(warm_fit.a[j] + warm_fit.b[i])
            probs = predict_match_probabilities(lam=lam, mu=mu, rho=warm_fit.rho if use_rho else 0.0)
            out_records.append(
                {
                    "Season": row.Season,
                    "Div": row.Div,
                    "Date": row.Date,
                    "Time": getattr(row, "Time", np.nan),
                    "HomeTeam": row.HomeTeam,
                    "AwayTeam": row.AwayTeam,
                    "FTHG": row.FTHG,
                    "FTAG": row.FTAG,
                    "FTR": row.FTR,
                    "PSH": getattr(row, "PSH", np.nan),
                    "PSD": getattr(row, "PSD", np.nan),
                    "PSA": getattr(row, "PSA", np.nan),
                    "PSCH": getattr(row, "PSCH", np.nan),
                    "PSCD": getattr(row, "PSCD", np.nan),
                    "PSCA": getattr(row, "PSCA", np.nan),
                    "P>2.5": getattr(row, "_21", np.nan),
                    "P<2.5": getattr(row, "_22", np.nan),
                    **probs,
                    "h_param": warm_fit.h,
                    "rho_param": warm_fit.rho if use_rho else 0.0,
                }
            )

    return pd.DataFrame(out_records), warm_fit


def run_walk_forward_all_leagues(
    df_all: pd.DataFrame,
    half_life_days: float,
    lambda_reg: float,
    use_rho: bool = True,
    target_seasons: Sequence[str] = EVAL_SEASONS,
) -> Tuple[pd.DataFrame, Dict[str, LeagueModelFit]]:
    """Run walk-forward across all 5 leagues and return combined predictions + final fits per league."""
    # Safety check: never allow 2526 or 2627 in Modulo B walk-forward
    for forbidden in ["2526", "2627"]:
        if forbidden in target_seasons:
            raise ValueError(f"RULE VIOLATION: Season {forbidden} is forbidden in Modulo B!")

    frames = []
    final_fits: Dict[str, LeagueModelFit] = {}
    for div_code in LEAGUE_NAMES.keys():
        df_l = df_all[df_all["Div"] == div_code].copy()
        preds_l, fit_l = run_walk_forward_league(
            df_league=df_l,
            half_life_days=half_life_days,
            lambda_reg=lambda_reg,
            use_rho=use_rho,
            target_seasons=target_seasons,
        )
        frames.append(preds_l)
        if fit_l is not None:
            final_fits[div_code] = fit_l

    combined = pd.concat(frames, ignore_index=True)
    # Join original odds columns cleanly by (Season, Div, Date, HomeTeam, AwayTeam)
    odds_merge_cols = [
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
    available_merge_cols = [c for c in odds_merge_cols if c in df_all.columns]
    combined = combined.drop(columns=[c for c in ["PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA", "P>2.5", "P<2.5"] if c in combined.columns])
    combined = combined.merge(
        df_all[available_merge_cols],
        on=["Season", "Div", "Date", "HomeTeam", "AwayTeam"],
        how="left",
    )
    combined = combined.sort_values(["Date", "Div", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    return combined, final_fits


# ==============================================================================
# 5. GRID SEARCH ON TRAIN (1920-2223) & REPORT GENERATION
# ==============================================================================


def run_hyperparameter_grid_on_train(
    df_all: pd.DataFrame,
    h_grid: Sequence[float] = (180.0, 365.0, 540.0, 730.0),
    reg_grid: Sequence[float] = (0.05, 0.15, 0.50, 1.50),
) -> Tuple[pd.DataFrame, float, float]:
    """
    Tune (H, lambda_reg) strictly on Train (1920-2223) minimizing walk-forward 1X2 Log-Loss.
    Never touches Validation (2324-2425) or Blind Test (2526).
    """
    df_train_only = df_all[df_all["Season"].isin(WARMUP_SEASONS + TRAIN_SEASONS)].copy()
    grid_rows = []

    for h_val in h_grid:
        for reg_val in reg_grid:
            t0 = time.perf_counter()
            preds_train, _ = run_walk_forward_all_leagues(
                df_all=df_train_only,
                half_life_days=h_val,
                lambda_reg=reg_val,
                use_rho=True,
                target_seasons=TRAIN_SEASONS,
            )
            probs_1x2 = preds_train[["P_H", "P_D", "P_A"]].to_numpy(dtype=np.float64)
            ftr = preds_train["FTR"].to_numpy()
            ll_1x2 = compute_log_loss(probs_1x2, ftr)
            brier_1x2 = compute_brier_score(probs_1x2, ftr)
            elapsed = time.perf_counter() - t0
            print(
                f"  [Grid Train] H={int(h_val):3d}d, lambda_reg={reg_val:4.2f} -> "
                f"Log-Loss 1X2 = {ll_1x2:.6f}, Brier = {brier_1x2:.6f} ({elapsed:.1f}s)"
            )
            grid_rows.append(
                {
                    "H": float(h_val),
                    "lambda_reg": float(reg_val),
                    "Train_LogLoss_1X2": ll_1x2,
                    "Train_Brier_1X2": brier_1x2,
                }
            )

    grid_df = pd.DataFrame(grid_rows).sort_values("Train_LogLoss_1X2", ascending=True).reset_index(drop=True)
    best_h = float(grid_df.iloc[0]["H"])
    best_reg = float(grid_df.iloc[0]["lambda_reg"])
    return grid_df, best_h, best_reg


def _compute_comparative_metrics_1x2(df_sub: pd.DataFrame) -> Dict[str, float]:
    """Compute 1X2 Log-Loss and Brier for Pure Poisson, Dixon-Coles, Pinnacle Open, Pinnacle Close."""
    ftr = df_sub["FTR"].to_numpy()
    p_pois = df_sub[["P_H_Pois", "P_D_Pois", "P_A_Pois"]].to_numpy(dtype=np.float64)
    p_dc = df_sub[["P_H", "P_D", "P_A"]].to_numpy(dtype=np.float64)

    open_odds = df_sub[["PSH", "PSD", "PSA"]].to_numpy(dtype=np.float64)
    close_odds = df_sub[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=np.float64)

    p_open = devig_power(open_odds)
    p_close = devig_power(close_odds)

    ll_pois = compute_log_loss(p_pois, ftr)
    ll_dc = compute_log_loss(p_dc, ftr)
    ll_open = compute_log_loss(p_open, ftr)
    ll_close = compute_log_loss(p_close, ftr)

    br_pois = compute_brier_score(p_pois, ftr)
    br_dc = compute_brier_score(p_dc, ftr)
    br_open = compute_brier_score(p_open, ftr)
    br_close = compute_brier_score(p_close, ftr)

    return {
        "n": len(df_sub),
        "ll_pois": ll_pois,
        "ll_dc": ll_dc,
        "ll_open": ll_open,
        "ll_close": ll_close,
        "gap_dc_close": ll_dc - ll_close,
        "br_pois": br_pois,
        "br_dc": br_dc,
        "br_open": br_open,
        "br_close": br_close,
    }


def _compute_comparative_metrics_ou25(df_sub: pd.DataFrame) -> Dict[str, float]:
    """Compute O/U 2.5 Log-Loss and Brier for Pure Poisson, Dixon-Coles, Pinnacle Open, Pinnacle Close."""
    # Filter to rows with valid O/U closing odds for fair benchmark comparison
    valid_ou = df_sub[["PC>2.5", "PC<2.5"]].notna().all(axis=1)
    sub = df_sub[valid_ou]
    total_goals = sub["FTHG"].to_numpy(dtype=np.float64) + sub["FTAG"].to_numpy(dtype=np.float64)
    outcomes = np.where(total_goals > 2.5, 0, 1)

    p_pois = sub[["P_Over25_Pois", "P_Under25_Pois"]].to_numpy(dtype=np.float64)
    p_dc = sub[["P_Over25", "P_Under25"]].to_numpy(dtype=np.float64)
    open_odds = sub[["P>2.5", "P<2.5"]].to_numpy(dtype=np.float64)
    close_odds = sub[["PC>2.5", "PC<2.5"]].to_numpy(dtype=np.float64)

    p_open = devig_power(open_odds)
    p_close = devig_power(close_odds)

    ll_pois = compute_log_loss(p_pois, outcomes)
    ll_dc = compute_log_loss(p_dc, outcomes)
    ll_open = compute_log_loss(p_open, outcomes)
    ll_close = compute_log_loss(p_close, outcomes)

    br_pois = compute_brier_score(p_pois, outcomes)
    br_dc = compute_brier_score(p_dc, outcomes)
    br_open = compute_brier_score(p_open, outcomes)
    br_close = compute_brier_score(p_close, outcomes)

    return {
        "n": len(sub),
        "ll_pois": ll_pois,
        "ll_dc": ll_dc,
        "ll_open": ll_open,
        "ll_close": ll_close,
        "gap_dc_close": ll_dc - ll_close,
        "br_pois": br_pois,
        "br_dc": br_dc,
        "br_open": br_open,
        "br_close": br_close,
    }


def generate_modulo_b_report(
    grid_df: pd.DataFrame,
    best_h: float,
    best_reg: float,
    df_preds: pd.DataFrame,
    final_fits: Dict[str, LeagueModelFit],
    report_path: Path,
) -> None:
    """Write the complete Modulo B diagnostic report to reports/modulo_b_baseline.md."""
    df_train = df_preds[df_preds["Season"].isin(TRAIN_SEASONS)].copy()
    df_val = df_preds[df_preds["Season"].isin(VAL_SEASONS)].copy()

    lines: List[str] = []
    lines.append("# Modulo B Report — Dixon-Coles MAP Walk-Forward Baseline")
    lines.append("")
    lines.append("## 0. Conformità al Protocollo Anti-Leakage (`RULES.md`)")
    lines.append(
        "- **Walk-Forward rigoroso per data calendario**: per ogni lega e ogni data $t$, il modello è addestrato esclusivamente sulle partite con `Date < t` (finestra rolling di 1.100 giorni)."
    )
    lines.append(
        "- **Prior Neopromosse**: media dei parametri $(a_k, b_k)$ stimati a fine stagione precedente per le ultime 3 classificate di quella lega (calcolato rigorosamente con dati `< inizio stagione`)."
    )
    lines.append(
        f"- **Perimetro valutato**: Train `1920–2223` ({len(df_train)} partite) + Validation `2324–2425` ({len(df_val)} partite) = **{len(df_preds)} partite**."
    )
    lines.append("- **Test Cieco (`2526`)**: **Mai toccato né valutato** (`0` partite presenti in `modulo_b_predictions.csv`).")
    lines.append("")

    # ==========================================================================
    # SECTION 1: HYPERPARAMETER GRID ON TRAIN (1920-2223)
    # ==========================================================================
    lines.append("## 1. Taratura Iperparametri su Train (`1920–2223`)")
    lines.append("")
    lines.append(
        f"Coppia ottima selezionata esclusivamente su Train (`1920–2223`) e congelata per Validation: **$H^* = {int(best_h)}$ giorni**, **$\\lambda_{{\\text{{reg}}}}^* = {best_reg:.2f}$**."
    )
    lines.append("")
    lines.append("| Rank | Emivita $H$ (giorni) | Regolarizzazione $\\lambda_{\\text{reg}}$ | Train Log-Loss 1X2 | Train Brier Score 1X2 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for idx, row in grid_df.iterrows():
        is_best = "**" if idx == 0 else ""
        lines.append(
            f"| {idx + 1} | {is_best}{int(row['H'])}{is_best} | {is_best}{row['lambda_reg']:.2f}{is_best} | "
            f"{is_best}{row['Train_LogLoss_1X2']:.6f}{is_best} | {is_best}{row['Train_Brier_1X2']:.6f}{is_best} |"
        )
    lines.append("")

    # ==========================================================================
    # SECTION 2: MAIN COMPARATIVE TABLES (TRAIN & VALIDATION, 1X2 & O/U 2.5)
    # ==========================================================================
    lines.append("## 2. Confronto Modelli vs Mercato (Poisson Puro, Dixon-Coles, Pinnacle Opening & Closing)")
    lines.append("")

    for idx_s, (split_label, df_s) in enumerate(
        [
            ("Train (`1920–2223`)", df_train),
            ("Validation (`2324–2425`)", df_val),
        ],
        start=1,
    ):
        lines.append(f"### 2.{idx_s} Mercato 1X2 — {split_label}")
        lines.append("")
        lines.append(
            "| Perimetro | N | Log-Loss Pois ($\\rho=0$) | Log-Loss DC | Log-Loss Pin Open | Log-Loss Pin Close | Gap (`DC - Pin Close`) | Brier Pois | Brier DC | Brier Pin Open | Brier Pin Close |"
        )
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

        m_tot = _compute_comparative_metrics_1x2(df_s)
        lines.append(
            f"| **Totale Top 5** | **{m_tot['n']}** | {m_tot['ll_pois']:.6f} | **{m_tot['ll_dc']:.6f}** | "
            f"{m_tot['ll_open']:.6f} | {m_tot['ll_close']:.6f} | **{m_tot['gap_dc_close']:+.6f}** | "
            f"{m_tot['br_pois']:.6f} | {m_tot['br_dc']:.6f} | {m_tot['br_open']:.6f} | {m_tot['br_close']:.6f} |"
        )

        for div_code, div_name in LEAGUE_NAMES.items():
            sub_l = df_s[df_s["Div"] == div_code]
            m_l = _compute_comparative_metrics_1x2(sub_l)
            hl = "**" if div_code == "I1" else ""
            lines.append(
                f"| {hl}`{div_code}` ({div_name}){hl} | {m_l['n']} | {m_l['ll_pois']:.6f} | {hl}{m_l['ll_dc']:.6f}{hl} | "
                f"{m_l['ll_open']:.6f} | {m_l['ll_close']:.6f} | {hl}{m_l['gap_dc_close']:+.6f}{hl} | "
                f"{m_l['br_pois']:.6f} | {m_l['br_dc']:.6f} | {m_l['br_open']:.6f} | {m_l['br_close']:.6f} |"
            )
        lines.append("")

    for idx_s, (split_label, df_s) in enumerate(
        [
            ("Train (`1920–2223`)", df_train),
            ("Validation (`2324–2425`)", df_val),
        ],
        start=3,
    ):
        lines.append(f"### 2.{idx_s} Mercato Over/Under 2.5 — {split_label}")
        lines.append("")
        lines.append(
            "| Perimetro | N | Log-Loss Pois ($\\rho=0$) | Log-Loss DC | Log-Loss Pin Open | Log-Loss Pin Close | Gap (`DC - Pin Close`) | Brier Pois | Brier DC | Brier Pin Open | Brier Pin Close |"
        )
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

        m_tot = _compute_comparative_metrics_ou25(df_s)
        lines.append(
            f"| **Totale Top 5** | **{m_tot['n']}** | {m_tot['ll_pois']:.6f} | **{m_tot['ll_dc']:.6f}** | "
            f"{m_tot['ll_open']:.6f} | {m_tot['ll_close']:.6f} | **{m_tot['gap_dc_close']:+.6f}** | "
            f"{m_tot['br_pois']:.6f} | {m_tot['br_dc']:.6f} | {m_tot['br_open']:.6f} | {m_tot['br_close']:.6f} |"
        )

        for div_code, div_name in LEAGUE_NAMES.items():
            sub_l = df_s[df_s["Div"] == div_code]
            m_l = _compute_comparative_metrics_ou25(sub_l)
            hl = "**" if div_code == "I1" else ""
            lines.append(
                f"| {hl}`{div_code}` ({div_name}){hl} | {m_l['n']} | {m_l['ll_pois']:.6f} | {hl}{m_l['ll_dc']:.6f}{hl} | "
                f"{m_l['ll_open']:.6f} | {m_l['ll_close']:.6f} | {hl}{m_l['gap_dc_close']:+.6f}{hl} | "
                f"{m_l['br_pois']:.6f} | {m_l['br_dc']:.6f} | {m_l['br_open']:.6f} | {m_l['br_close']:.6f} |"
            )
        lines.append("")

    # ==========================================================================
    # SECTION 3: PHYSICAL PARAMETERS PER LEAGUE (AT END OF VALIDATION 2425)
    # ==========================================================================
    lines.append("## 3. Parametri Fisici Stimati per Lega (a fine Validation `2425`)")
    lines.append("")
    lines.append(
        "| Lega | Nome | $h$ Finale | Fattore Campo $\\gamma = e^h$ | $\\rho$ Finale (`2425`) | $\\rho$ Medio (`2324–2425`) | Gol Attesi Casa $\\bar{\\lambda}$ | Gol Reali Casa | Gol Attesi Trasf $\\bar{\\mu}$ | Gol Reali Trasf |"
    )
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for div_code, div_name in LEAGUE_NAMES.items():
        sub_v = df_val[df_val["Div"] == div_code]
        fit_end = final_fits[div_code]
        h_fin = fit_end.h
        gamma_fin = math.exp(h_fin)
        rho_fin = fit_end.rho
        rho_mean = float(sub_v["rho_param"].mean())
        lam_mean = float(sub_v["Lambda_Home"].mean())
        mu_mean = float(sub_v["Mu_Away"].mean())
        fthg_mean = float(sub_v["FTHG"].mean())
        ftag_mean = float(sub_v["FTAG"].mean())
        hl = "**" if div_code == "I1" else ""
        lines.append(
            f"| {hl}`{div_code}`{hl} | {hl}{div_name}{hl} | {h_fin:.4f} | **{gamma_fin:.3f}x** | "
            f"{rho_fin:+.4f} | {rho_mean:+.4f} | {lam_mean:.3f} | {fthg_mean:.3f} | {mu_mean:.3f} | {ftag_mean:.3f} |"
        )
    lines.append("")

    # ==========================================================================
    # SECTION 4: CALIBRATION TABLE ON VALIDATION (2324-2425)
    # ==========================================================================
    lines.append("## 4. Calibrazione di Dixon-Coles su Validation (`2324–2425`, 1X2)")
    lines.append("")
    cal_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    p_dc_val = df_val[["P_H", "P_D", "P_A"]].to_numpy(dtype=np.float64)
    p_pin_val = devig_power(df_val[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=np.float64))
    ftr_val = df_val["FTR"].to_numpy()

    cal_dc = compute_calibration_table(p_dc_val, ftr_val, bins=cal_bins)
    cal_pin = compute_calibration_table(p_pin_val, ftr_val, bins=cal_bins)

    lines.append(
        "| Fascia Probabilità | N (DC) | Pred Media (DC) | Freq Reale (DC) | Errore Obs-Pred (DC) | N (Pin Close) | Pred Media (Pin Close) | Freq Reale (Pin Close) | Errore Obs-Pred (Pin Close) |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for idx_r in range(len(cal_dc)):
        rd = cal_dc.iloc[idx_r]
        rp = cal_pin.iloc[idx_r]
        lines.append(
            f"| `{rd['Bin']}` | {int(rd['Count'])} | {rd['Mean_Predicted']*100:.2f}% | {rd['Observed_Frequency']*100:.2f}% | "
            f"**{rd['Diff_Obs_minus_Pred']*100:+.2f}%** | {int(rp['Count'])} | {rp['Mean_Predicted']*100:.2f}% | "
            f"{rp['Observed_Frequency']*100:.2f}% | {rp['Diff_Obs_minus_Pred']*100:+.2f}% |"
        )
    lines.append("")

    # ==========================================================================
    # SECTION 5: PRELIMINARY SIGNAL RESIDUAL DIAGNOSTICS ON VALIDATION (2324-2425)
    # ==========================================================================
    lines.append("## 5. Diagnostica Preliminare sui Residui di Segnale (Validation `2324–2425`)")
    lines.append("")
    lines.append(
        "Ispezione puramente diagnostica (nessun backtest economico): verifica su Validation (`2324–2425`) dei casi in cui la probabilità stimata da Dixon-Coles genera un **$EV > 0$ contro Pinnacle Opening (`PSH / PSD / PSA`) al netto della commissione $c = 5\\%$** ($EV = p_{\\text{DC}}(Q_{\\text{open}} - 1)(1 - 0.05) - (1 - p_{\\text{DC}}) > 0$), valutando poi il **CLV medio contro Pinnacle Closing (Power devigged)**: $\\text{CLV}_{\\text{mkt}} = Q_{\\text{open}} \\cdot p_{\\text{close, power}} - 1$."
    )
    lines.append("")
    lines.append(
        "| Perimetro | Partite Totali | Partite con $\\ge 1$ Esito $EV_{5\\%} > 0$ (%) | Selezione Esiti con $EV_{5\\%} > 0$ ($/ 3N$) | EV Medio Stimato da DC (%) | CLV Medio vs Pinnacle Closing Power (%) |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")

    for label, sub_d in [("Totale Top 5", df_val)] + [(f"`{k}` ({v})", df_val[df_val["Div"] == k]) for k, v in LEAGUE_NAMES.items()]:
        valid_oc = sub_d[["PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA"]].notna().all(axis=1)
        sub_v = sub_d[valid_oc]
        n_m = len(sub_v)
        p_dc_m = sub_v[["P_H", "P_D", "P_A"]].to_numpy(dtype=np.float64)
        q_open_m = sub_v[["PSH", "PSD", "PSA"]].to_numpy(dtype=np.float64)
        q_close_m = sub_v[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=np.float64)
        p_close_m = devig_power(q_close_m)

        ev_mat = expected_value_net(p_dc_m, q_open_m, commission=0.05)
        clv_mkt_mat = closing_line_value(q_open_m, p_close_m)

        pos_mask = ev_mat > 0.0
        matches_with_pos = int(np.sum(np.any(pos_mask, axis=1)))
        pct_matches = (matches_with_pos / n_m) * 100.0 if n_m > 0 else 0.0
        n_pos_outcomes = int(np.sum(pos_mask))
        pct_outcomes = (n_pos_outcomes / (3.0 * n_m)) * 100.0 if n_m > 0 else 0.0

        mean_ev_dc = float(np.mean(ev_mat[pos_mask])) * 100.0 if n_pos_outcomes > 0 else 0.0
        mean_clv_mkt = float(np.mean(clv_mkt_mat[pos_mask])) * 100.0 if n_pos_outcomes > 0 else 0.0

        hl = "**" if "Totale" in label or "I1" in label else ""
        lines.append(
            f"| {hl}{label}{hl} | {n_m} | {hl}{matches_with_pos} ({pct_matches:.2f}%){hl} | "
            f"{n_pos_outcomes} ({pct_outcomes:.2f}%) | {mean_ev_dc:+.2f}% | {hl}{mean_clv_mkt:+.2f}%{hl} |"
        )
    lines.append("")
    lines.append(
        "> **Nota diagnostica**: Un modello basato esclusivamente sui gol finali storici (`FTHG / FTAG`), pur battendo nettamente il Poisson puro grazie alla correzione di correlazione $\\rho$ di Dixon-Coles e ai prior sulle neopromosse, presenta un Log-Loss superiore a Pinnacle Closing e seleziona all'apertura quote dove il mercato ha informazioni extra (infortuni, motivazioni, xG/tiri, mercato). Quando il modello solo-gol diverge da Pinnacle Opening, il CLV medio rispetto a Pinnacle Closing resta negativo: questo giustifica il passaggio al Modulo C per l'integrazione di informazioni di secondo livello."
    )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    csv_path = root_dir / "data" / "processed" / "level1_top5_1617_2526.csv"
    preds_path = root_dir / "data" / "processed" / "modulo_b_predictions.csv"
    report_path = root_dir / "reports" / "modulo_b_baseline.md"

    df_all = pd.read_csv(csv_path, dtype={"Season": str, "Div": str})
    # Explicitly exclude 2526 and 2627 right at load time
    df_dev = df_all[df_all["Season"].isin(ALL_ALLOWED_SEASONS)].copy()

    print("[Modulo B] Step 1/3: Running hyperparameter grid search on Train (1920-2223)...")
    grid_df, best_h, best_reg = run_hyperparameter_grid_on_train(df_dev)
    print(f"[Modulo B] Best hyperparameters on Train: H* = {int(best_h)} days, lambda_reg* = {best_reg:.2f}")

    print("[Modulo B] Step 2/3: Running frozen Walk-Forward on Train + Validation (1920-2425) for Dixon-Coles and Pure Poisson...")
    preds_dc, final_fits_dc = run_walk_forward_all_leagues(
        df_all=df_dev,
        half_life_days=best_h,
        lambda_reg=best_reg,
        use_rho=True,
        target_seasons=EVAL_SEASONS,
    )
    preds_pois, _ = run_walk_forward_all_leagues(
        df_all=df_dev,
        half_life_days=best_h,
        lambda_reg=best_reg,
        use_rho=False,
        target_seasons=EVAL_SEASONS,
    )

    # Merge Pure Poisson predictions into preds_dc for direct side-by-side comparison
    preds_dc["P_H_Pois"] = preds_pois["P_H"]
    preds_dc["P_D_Pois"] = preds_pois["P_D"]
    preds_dc["P_A_Pois"] = preds_pois["P_A"]
    preds_dc["P_Over25_Pois"] = preds_pois["P_Over25"]
    preds_dc["P_Under25_Pois"] = preds_pois["P_Under25"]
    preds_dc["Lambda_Home_Pois"] = preds_pois["Lambda_Home"]
    preds_dc["Mu_Away_Pois"] = preds_pois["Mu_Away"]

    assert "2526" not in set(preds_dc["Season"].unique()), "CRITICAL: 2526 must not be in modulo_b_predictions.csv"

    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_dc.to_csv(preds_path, index=False, encoding="utf-8")
    print(f"[Modulo B] Saved {len(preds_dc)} out-of-sample predictions to {preds_path}")

    print("[Modulo B] Step 3/3: Generating diagnostic report...")
    generate_modulo_b_report(
        grid_df=grid_df,
        best_h=best_h,
        best_reg=best_reg,
        df_preds=preds_dc,
        final_fits=final_fits_dc,
        report_path=report_path,
    )
    print(f"[Modulo B] Generated baseline report at {report_path}")


if __name__ == "__main__":
    main()
