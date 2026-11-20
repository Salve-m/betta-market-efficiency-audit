"""
Unit and integrity tests for Modulo B — Dixon-Coles MAP Walk-Forward Baseline (tests/test_modulo_b.py).
Verifies:
1. Score matrix normalization (sum == 1.0 within 1e-9) and coherent 1X2 & O/U 2.5 probabilities in (0, 1).
2. Explicit Anti-Leakage test on a real Serie A date t (altering results at Date >= t to 10-0 leaves predictions at Date t identical within 1e-12).
3. Newly Promoted Team Prior test on Matchday 1 of a new season (offensive prior < 0 and/or defensive weakness prior > 0 compared to league baseline (0, 0)).
4. Complete absence of Blind Test season 2526 in data/processed/modulo_b_predictions.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modulo_b_dixon_coles import (
     compute_score_matrix,
    predict_league_on_date_strict,
    probabilities_from_score_matrix,
)

CONSOLIDATED_CSV = ROOT_DIR / "data" / "processed" / "level1_top5_1617_2526.csv"
MODULO_B_PREDS_CSV = ROOT_DIR / "data" / "processed" / "modulo_b_predictions.csv"


@pytest.fixture(scope="module")
def df_serie_a() -> pd.DataFrame:
    assert CONSOLIDATED_CSV.exists(), f"Missing consolidated CSV: {CONSOLIDATED_CSV}"
    df = pd.read_csv(CONSOLIDATED_CSV, dtype={"Season": str, "Div": str})
    df_i1 = df[(df["Div"] == "I1") & (df["Season"] != "2526")].copy()
    df_i1["ParsedDate"] = pd.to_datetime(df_i1["Date"], format="%Y-%m-%d")
    return df_i1


@pytest.mark.parametrize(
    "lam,mu,rho",
    [
        (1.45, 1.15, 0.0),
        (1.60, 1.10, -0.08),
        (1.20, 1.30, 0.06),
        (2.40, 0.70, -0.12),
        (0.85, 0.90, -0.15),
    ],
)
def test_score_matrix_normalization_and_coherent_probabilities(lam: float, mu: float, rho: float):
    """
    Verify that the 11x11 Dixon-Coles score matrix sums to 1.0 (within 1e-9)
    and that 1X2 and O/U 2.5 probabilities are coherent and strictly in (0, 1).
    """
    mat = compute_score_matrix(lam=lam, mu=mu, rho=rho)
    assert mat.shape == (11, 11)
    assert np.all(mat >= 0.0)
    assert abs(float(np.sum(mat)) - 1.0) < 1e-9

    probs = probabilities_from_score_matrix(mat)
    for key in ["P_H", "P_D", "P_A", "P_Over25", "P_Under25"]:
        assert 0.0 < probs[key] < 1.0, f"Probability {key}={probs[key]} not in (0, 1)"

    assert abs((probs["P_H"] + probs["P_D"] + probs["P_A"]) - 1.0) < 1e-9
    assert abs((probs["P_Over25"] + probs["P_Under25"]) - 1.0) < 1e-9

    # Check consistency with direct matrix summation
    x_idx, y_idx = np.meshgrid(np.arange(11), np.arange(11), indexing="ij")
    assert abs(probs["P_H"] - float(np.sum(mat[x_idx > y_idx]))) < 1e-9
    assert abs(probs["P_D"] - float(np.sum(mat[x_idx == y_idx]))) < 1e-9
    assert abs(probs["P_A"] - float(np.sum(mat[x_idx < y_idx]))) < 1e-9
    assert abs(probs["P_Over25"] - float(np.sum(mat[(x_idx + y_idx) >= 3]))) < 1e-9
    assert abs(probs["P_Under25"] - float(np.sum(mat[(x_idx + y_idx) <= 2]))) < 1e-9


def test_explicit_anti_leakage_on_serie_a_date(df_serie_a: pd.DataFrame):
    """
    Take a real date t in Serie A, generate predictions for matches on date t,
    then drastically alter (e.g. 10-0) all match results on Date >= t in a copy of the dataframe,
    and verify that predictions at date t remain identical within 1e-12.
    """
    # Pick a mid-season Serie A matchday in 2122 with multiple matches
    s_2122 = df_serie_a[df_serie_a["Season"] == "2122"]
    date_counts = s_2122["Date"].value_counts()
    target_date = str(date_counts[date_counts >= 4].index[0])

    preds_clean, fit_clean = predict_league_on_date_strict(
        df_league=df_serie_a,
        target_date=target_date,
        half_life_days=365.0,
        lambda_reg=0.15,
        use_rho=True,
    )
    assert len(preds_clean) >= 4

    # Corrupt all matches with Date >= target_date to 10-0 home wins
    df_corrupted = df_serie_a.copy()
    future_or_today_mask = df_corrupted["Date"] >= target_date
    assert future_or_today_mask.sum() > 0
    df_corrupted.loc[future_or_today_mask, "FTHG"] = 10
    df_corrupted.loc[future_or_today_mask, "FTAG"] = 0
    df_corrupted.loc[future_or_today_mask, "FTR"] = "H"

    preds_corrupted, fit_corrupted = predict_league_on_date_strict(
        df_league=df_corrupted,
        target_date=target_date,
        half_life_days=365.0,
        lambda_reg=0.15,
        use_rho=True,
    )

    for col in ["P_H", "P_D", "P_A", "P_Over25", "P_Under25", "Lambda_Home", "Mu_Away"]:
        np.testing.assert_allclose(
            preds_clean[col].to_numpy(),
            preds_corrupted[col].to_numpy(),
            rtol=0.0,
            atol=1e-12,
            err_msg=f"Data leakage detected in column {col} on date {target_date}!",
        )

    np.testing.assert_allclose(fit_clean.a, fit_corrupted.a, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(fit_clean.b, fit_corrupted.b, rtol=0.0, atol=1e-12)
    assert abs(fit_clean.h - fit_corrupted.h) < 1e-12
    assert abs(fit_clean.rho - fit_corrupted.rho) < 1e-12


def test_newly_promoted_team_prior_matchday_1(df_serie_a: pd.DataFrame):
    """
    Verify that on Matchday 1 of a new season in Serie A, newly promoted teams receive
    an offensive prior < 0 and a defensive weakness prior > 0 compared to the (0, 0) baseline
    of established teams.
    """
    # Take Matchday 1 of Serie A 2022/23 (2223), where Monza, Cremonese, Lecce were newly promoted
    s_2223 = df_serie_a[df_serie_a["Season"] == "2223"]
    matchday_1_date = str(s_2223["Date"].min())

    _, fit_md1 = predict_league_on_date_strict(
        df_league=df_serie_a,
        target_date=matchday_1_date,
        half_life_days=365.0,
        lambda_reg=0.15,
        use_rho=True,
    )

    assert len(fit_md1.promoted_teams) > 0, "Expected newly promoted teams on Matchday 1 of 2223."
    for prom_team in fit_md1.promoted_teams:
        idx = fit_md1.team_to_idx[prom_team]
        a_prior = float(fit_md1.a0[idx])
        b_prior = float(fit_md1.b0[idx])
        # Promoted team must have weaker attack prior (< 0) and worse defense prior (> 0) than (0, 0)
        assert a_prior < 0.0, f"Expected promoted team {prom_team} offensive prior a0 < 0, got {a_prior}"
        assert b_prior > 0.0, f"Expected promoted team {prom_team} defensive weakness prior b0 > 0, got {b_prior}"
        # Since it has 0 matches in the active window on Matchday 1, its posterior MAP estimate also reflects the penalty
        assert float(fit_md1.a[idx]) < 0.0
        assert float(fit_md1.b[idx]) > 0.0


def test_predictions_file_excludes_2526():
    """
    Verify that data/processed/modulo_b_predictions.csv exists, is non-empty,
    covers Train (1920-2223) and Validation (2324-2425), and contains ZERO matches from 2526.
    """
    assert MODULO_B_PREDS_CSV.exists(), f"Missing predictions file: {MODULO_B_PREDS_CSV}"
    df_preds = pd.read_csv(MODULO_B_PREDS_CSV, dtype={"Season": str, "Div": str})
    assert len(df_preds) > 10000

    seasons = set(df_preds["Season"].unique())
    assert "2526" not in seasons, "CRITICAL VIOLATION: Blind Test season 2526 found in modulo_b_predictions.csv!"
    assert "2627" not in seasons, "CRITICAL VIOLATION: Holdout season 2627 found in modulo_b_predictions.csv!"
    assert seasons == {"1920", "2021", "2122", "2223", "2324", "2425"}

    # Verify probability columns are valid
    for col in ["P_H", "P_D", "P_A", "P_Over25", "P_Under25"]:
        assert df_preds[col].notna().all()
        assert ((df_preds[col] > 0.0) & (df_preds[col] < 1.0)).all()
