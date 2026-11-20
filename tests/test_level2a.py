"""
Unit and integrity tests for Level 2A — Quasi-Poisson MAP on Shots + Dixon-Coles 2-Stage
(tests/test_level2a.py).

Verifies:
1. Analytical gradient accuracy: scipy.optimize.check_grad on Stage 1 Quasi-Poisson MAP with error < 1e-5.
2. Explicit Anti-Leakage test: altering goals, shots (HS/HST), and red cards (HR/AR) for matches
   with Date >= t changes neither (r_sot, r_tot) nor predictions at date t by more than 1e-12.
3. Clean fallback on NaN shots: a historical match with HST = NaN produces no NaN in the objective
   or gradient and falls back cleanly to real goals (FTHG, FTAG).
4. Complete absence of Blind Test season 2526 in data/processed/level2a_predictions.csv.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.level2a_shots_dc import (
    compute_composite_targets,
    fit_level2a_model,
    predict_level2a_on_date_strict,
    stage1_quasi_poisson_obj_and_grad,
)

CONSOLIDATED_CSV = ROOT_DIR / "data" / "processed" / "level1_top5_1617_2526.csv"
LEVEL2A_PREDS_CSV = ROOT_DIR / "data" / "processed" / "level2a_predictions.csv"


@pytest.fixture(scope="module")
def df_serie_a() -> pd.DataFrame:
    assert CONSOLIDATED_CSV.exists(), f"Missing consolidated CSV: {CONSOLIDATED_CSV}"
    df = pd.read_csv(CONSOLIDATED_CSV, dtype={"Season": str, "Div": str})
    df_i1 = df[(df["Div"] == "I1") & (df["Season"] != "2526")].copy()
    df_i1["ParsedDate"] = pd.to_datetime(df_i1["Date"], format="%Y-%m-%d")
    return df_i1


def test_stage1_analytical_gradient_check_grad():
    """
    Verify Stage 1 Quasi-Poisson MAP analytical gradient against finite differences
    using scipy.optimize.check_grad with error < 1e-5.
    """
    rng = np.random.default_rng(42)
    n_teams = 6
    n_matches = 30

    home_idx = rng.integers(0, n_teams, size=n_matches, dtype=np.int32)
    away_idx = (home_idx + rng.integers(1, n_teams, size=n_matches, dtype=np.int32)) % n_teams
    y_eff_h = rng.uniform(0.3, 2.8, size=n_matches)
    y_eff_a = rng.uniform(0.2, 2.2, size=n_matches)
    weights = rng.uniform(0.2, 1.0, size=n_matches)
    a0 = rng.normal(0.0, 0.1, size=n_teams)
    b0 = rng.normal(0.1, 0.1, size=n_teams)
    lambda_reg = 3.0

    def func(x: np.ndarray) -> float:
        val, _ = stage1_quasi_poisson_obj_and_grad(
            x, n_teams, home_idx, away_idx, y_eff_h, y_eff_a, weights, a0, b0, lambda_reg
        )
        return val

    def grad(x: np.ndarray) -> np.ndarray:
        _, g = stage1_quasi_poisson_obj_and_grad(
            x, n_teams, home_idx, away_idx, y_eff_h, y_eff_a, weights, a0, b0, lambda_reg
        )
        return g

    for _ in range(5):
        x_test = rng.normal(0.0, 0.35, size=2 * n_teams + 1)
        err = check_grad(func, grad, x_test)
        assert err < 1e-5, f"Analytical gradient check_grad error {err} >= 1e-5"


def test_explicit_anti_leakage_shots_goals_red_cards(df_serie_a: pd.DataFrame):
    """
    Verify that drastically altering goals (FTHG/FTAG), shots (HS/AS/HST/AST),
    and red cards (HR/AR) on matches with Date >= t changes neither (r_sot, r_tot)
    nor predictions at date t by more than 1e-12.
    """
    s_2223 = df_serie_a[df_serie_a["Season"] == "2223"]
    date_counts = s_2223["Date"].value_counts()
    target_date = str(date_counts[date_counts >= 4].index[0])

    preds_clean, fit_clean = predict_level2a_on_date_strict(
        df_league=df_serie_a,
        target_date=target_date,
        half_life_days=365.0,
        lambda_reg=3.0,
        w_g=0.30,
        w_sot=0.50,
        w_tot=0.20,
        d_rc=0.50,
    )
    assert len(preds_clean) >= 4

    # Corrupt all rows with Date >= target_date
    df_corrupted = df_serie_a.copy()
    mask_future = df_corrupted["Date"] >= target_date
    assert mask_future.sum() > 0

    df_corrupted.loc[mask_future, "FTHG"] = 12
    df_corrupted.loc[mask_future, "FTAG"] = 0
    df_corrupted.loc[mask_future, "FTR"] = "H"
    df_corrupted.loc[mask_future, "HS"] = 45
    df_corrupted.loc[mask_future, "AS"] = 1
    df_corrupted.loc[mask_future, "HST"] = 30
    df_corrupted.loc[mask_future, "AST"] = 0
    df_corrupted.loc[mask_future, "HR"] = 3
    df_corrupted.loc[mask_future, "AR"] = 2

    preds_corrupted, fit_corrupted = predict_level2a_on_date_strict(
        df_league=df_corrupted,
        target_date=target_date,
        half_life_days=365.0,
        lambda_reg=3.0,
        w_g=0.30,
        w_sot=0.50,
        w_tot=0.20,
        d_rc=0.50,
    )

    assert abs(fit_clean.r_sot - fit_corrupted.r_sot) < 1e-12
    assert abs(fit_clean.r_tot - fit_corrupted.r_tot) < 1e-12
    assert abs(fit_clean.h - fit_corrupted.h) < 1e-12
    assert abs(fit_clean.rho - fit_corrupted.rho) < 1e-12
    np.testing.assert_allclose(fit_clean.a, fit_corrupted.a, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(fit_clean.b, fit_corrupted.b, rtol=0.0, atol=1e-12)

    for col in ["P_H", "P_D", "P_A", "P_Over25", "P_Under25", "Lambda_Home", "Mu_Away", "r_sot", "r_tot"]:
        np.testing.assert_allclose(
            preds_clean[col].to_numpy(),
            preds_corrupted[col].to_numpy(),
            rtol=0.0,
            atol=1e-12,
            err_msg=f"Leakage detected in column {col} at date {target_date}!",
        )


def test_clean_nan_fallback_on_missing_shots(df_serie_a: pd.DataFrame):
    """
    Verify that if a historical match has HST = NaN (or HS = NaN),
    compute_composite_targets and fit_level2a_model fall back cleanly to (FTHG, FTAG)
    for that match and produce finite, non-NaN objectives and parameter estimates.
    """
    sub = df_serie_a[df_serie_a["Season"] == "2122"].head(40).copy()
    # Inject NaNs in HST and HS on specific rows
    sub.iloc[0, sub.columns.get_loc("HST")] = np.nan
    sub.iloc[5, sub.columns.get_loc("HS")] = np.nan
    sub.iloc[10, sub.columns.get_loc("HR")] = np.nan

    fthg = sub["FTHG"].to_numpy(dtype=np.float64)
    ftag = sub["FTAG"].to_numpy(dtype=np.float64)
    hs = sub["HS"].to_numpy(dtype=np.float64)
    as_ = sub["AS"].to_numpy(dtype=np.float64)
    hst = sub["HST"].to_numpy(dtype=np.float64)
    ast = sub["AST"].to_numpy(dtype=np.float64)

    y_h, y_a, r_sot, r_tot = compute_composite_targets(
        fthg=fthg, ftag=ftag, hs=hs, as_=as_, hst=hst, ast=ast, w_g=0.30, w_sot=0.50, w_tot=0.20
    )
    assert np.all(np.isfinite(y_h)) and np.all(np.isfinite(y_a))
    assert np.isfinite(r_sot) and np.isfinite(r_tot)
    # Row 0 and Row 5 must have fallen back exactly to FTHG and FTAG
    assert abs(y_h[0] - fthg[0]) < 1e-12 and abs(y_a[0] - ftag[0]) < 1e-12
    assert abs(y_h[5] - fthg[5]) < 1e-12 and abs(y_a[5] - ftag[5]) < 1e-12

    ref_date = sub["ParsedDate"].max() + pd.Timedelta(days=1)
    fit = fit_level2a_model(
        df_window=sub,
        current_date=ref_date,
        half_life_days=365.0,
        lambda_reg=3.0,
        w_g=0.30,
        w_sot=0.50,
        w_tot=0.20,
        d_rc=0.50,
    )
    assert np.all(np.isfinite(fit.a))
    assert np.all(np.isfinite(fit.b))
    assert np.isfinite(fit.h) and np.isfinite(fit.rho)


def test_level2a_predictions_file_excludes_2526():
    """
    Verify that data/processed/level2a_predictions.csv exists, covers 1920-2425 (10,707 matches),
    and contains ZERO rows from Blind Test 2526 or Holdout 2627.
    """
    assert LEVEL2A_PREDS_CSV.exists(), f"Missing Level 2A predictions file: {LEVEL2A_PREDS_CSV}"
    df_preds = pd.read_csv(LEVEL2A_PREDS_CSV, dtype={"Season": str, "Div": str})
    assert len(df_preds) == 10707

    seasons = set(df_preds["Season"].unique())
    assert "2526" not in seasons, "CRITICAL VIOLATION: Blind Test 2526 found in level2a_predictions.csv!"
    assert "2627" not in seasons, "CRITICAL VIOLATION: Holdout 2627 found in level2a_predictions.csv!"
    assert seasons == {"1920", "2021", "2122", "2223", "2324", "2425"}

    for col in ["P_H", "P_D", "P_A", "P_Over25", "P_Under25"]:
        assert df_preds[col].notna().all()
        assert ((df_preds[col] > 0.0) & (df_preds[col] < 1.0)).all()
