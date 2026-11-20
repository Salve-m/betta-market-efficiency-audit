"""
Pytest integrity suite for Level 1 consolidated dataset (data/processed/level1_top5_1617_2526.csv).
Verifies compliance with RULES.md and Level 1 specifications.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT_DIR / "data" / "processed" / "level1_top5_1617_2526.csv"
REPORT_MD = ROOT_DIR / "reports" / "level1_audit.md"

EXPECTED_SEASONS = {"1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"}
EXPECTED_LEAGUES = {"I1", "E0", "SP1", "D1", "F1"}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    assert PROCESSED_CSV.exists(), f"Consolidated file not found: {PROCESSED_CSV}"
    return pd.read_csv(PROCESSED_CSV, dtype={"Season": str, "Div": str})


def test_files_exist():
    """Verify that the consolidated CSV and audit report exist and are non-empty."""
    assert PROCESSED_CSV.exists() and PROCESSED_CSV.stat().st_size > 0
    assert REPORT_MD.exists() and REPORT_MD.stat().st_size > 0


def test_holdout_exclusion_and_perimeter(df: pd.DataFrame):
    """Verify Rule 2: Holdout season 2627 is strictly excluded, and all 10 seasons & 5 leagues exist."""
    seasons = set(df["Season"].astype(str).unique())
    assert "2627" not in seasons, "Holdout season 2627 must never be present in the historical dataset!"
    assert seasons == EXPECTED_SEASONS

    leagues = set(df["Div"].unique())
    assert leagues == EXPECTED_LEAGUES
    # 10 seasons across 5 leagues should yield roughly ~17,500 - 18,500 matches
    assert len(df) > 15000


def test_no_duplicates(df: pd.DataFrame):
    """Verify absence of duplicate matches on (Div, Date, HomeTeam, AwayTeam)."""
    dup_count = df.duplicated(subset=["Div", "Date", "HomeTeam", "AwayTeam"]).sum()
    assert dup_count == 0, f"Found {dup_count} duplicate matches in consolidated dataset."


def test_chronological_order(df: pd.DataFrame):
    """Verify strict chronological ordering (Rule 1: Zero Data Leakage foundation)."""
    parsed_dates = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    assert parsed_dates.notna().all(), "All dates in consolidated dataset must be valid YYYY-MM-DD."
    assert parsed_dates.is_monotonic_increasing, "Dataset must be strictly ordered by Date ascending."


def test_numeric_types_and_valid_results(df: pd.DataFrame):
    """Verify numeric types for goals, stats, odds, and overround columns."""
    required_numeric = [
        "FTHG",
        "FTAG",
        "HTHG",
        "HTAG",
        "HS",
        "AS",
        "HST",
        "AST",
        "HR",
        "AR",
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
        "Overround_Pinnacle_Close",
        "Overround_BFE_Close",
    ]
    for col in required_numeric:
        assert col in df.columns, f"Missing expected column: {col}"
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} is not numeric dtype."

    assert df["FTHG"].notna().all() and (df["FTHG"] >= 0).all()
    assert df["FTAG"].notna().all() and (df["FTAG"] >= 0).all()
    assert set(df["FTR"].dropna().unique()).issubset({"H", "D", "A"})


def test_overround_calculation_accuracy(df: pd.DataFrame):
    """Verify Pinnacle and Betfair Exchange closing overround formulas."""
    valid_psc = df[["PSCH", "PSCD", "PSCA"]].notna().all(axis=1)
    expected_or = (1.0 / df.loc[valid_psc, "PSCH"] + 1.0 / df.loc[valid_psc, "PSCD"] + 1.0 / df.loc[valid_psc, "PSCA"]) - 1.0
    np.testing.assert_allclose(
        df.loc[valid_psc, "Overround_Pinnacle_Close"],
        expected_or,
        rtol=1e-6,
        atol=1e-6,
    )


def test_no_impossible_odds_in_consolidated_csv(df: pd.DataFrame):
    """Fail if any valid (non-NaN) odds value in the consolidated CSV is <= 1.01."""
    non_odds_cols = {
        "Season",
        "Div",
        "Date",
        "Time",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HTHG",
        "HTAG",
        "HTR",
        "HS",
        "AS",
        "HST",
        "AST",
        "HR",
        "AR",
        "Overround_Pinnacle_Close",
        "Overround_BFE_Close",
    }
    odds_cols = [c for c in df.columns if c not in non_odds_cols]
    assert len(odds_cols) > 0, "Expected odds columns in dataset."

    for col in odds_cols:
        series = df[col].dropna()
        impossible = series[series <= 1.01]
        assert len(impossible) == 0, (
            f"Found {len(impossible)} impossible odds (<= 1.01) in column '{col}': {impossible.tolist()}"
        )

