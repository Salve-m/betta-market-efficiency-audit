"""
Level 1: Data Ingestion, Cleaning, Reconciliation & Audit for Betta.
Downloads historical CSVs from Football-Data.co.uk (seasons 1617 to 2526, Top 5 leagues),
reconciles schema, runs fail-fast integrity checks, saves consolidated dataset,
and generates the Level 1 Audit Report.
"""

from __future__ import annotations

import csv
import io
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Strictly 10 historical seasons (1617 to 2526). Holdout 2627 is strictly excluded (RULES.md).
SEASONS: List[str] = [
    "1617",
    "1718",
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
    "2526",
]

LEAGUES: Dict[str, str] = {
    "I1": "Serie A",
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# Unified schema definition
META_COLS = ["Season", "Div", "Date", "Time", "HomeTeam", "AwayTeam"]
SCORE_STATS_COLS = [
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
]
PINNACLE_1X2_OPEN_COLS = ["PSH", "PSD", "PSA"]
PINNACLE_1X2_CLOSE_COLS = ["PSCH", "PSCD", "PSCA"]
PINNACLE_OU_OPEN_COLS = ["P>2.5", "P<2.5"]
PINNACLE_OU_CLOSE_COLS = ["PC>2.5", "PC<2.5"]
MAX_AVG_COLS = [
    "MaxH",
    "MaxD",
    "MaxA",
    "AvgH",
    "AvgD",
    "AvgA",
    "MaxCH",
    "MaxCD",
    "MaxCA",
    "AvgCH",
    "AvgCD",
    "AvgCA",
    "Max>2.5",
    "Max<2.5",
    "Avg>2.5",
    "Avg<2.5",
    "MaxC>2.5",
    "MaxC<2.5",
    "AvgC>2.5",
    "AvgC<2.5",
    "BbMxH",
    "BbMxD",
    "BbMxA",
    "BbAvH",
    "BbAvD",
    "BbAvA",
    "BbMx>2.5",
    "BbAv>2.5",
    "BbMx<2.5",
    "BbAv<2.5",
]
CANONICAL_BFE_COLS = [
    "BFEH",
    "BFED",
    "BFEA",
    "BFE>2.5",
    "BFE<2.5",
    "BFEAHH",
    "BFEAHA",
    "BFECH",
    "BFECD",
    "BFECA",
    "BFEC>2.5",
    "BFEC<2.5",
    "BFECAHH",
    "BFECAHA",
]

NUMERIC_SCORE_STATS_COLS = [
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
]


@dataclass
class AuditResults:
    download_errors: List[str] = field(default_factory=list)
    files_downloaded: int = 0
    files_cached: int = 0
    raw_rows_read: int = 0
    empty_rows_dropped: int = 0
    unparsed_date_rows: List[Dict[str, object]] = field(default_factory=list)
    missing_result_rows: List[Dict[str, object]] = field(default_factory=list)
    duplicate_matches: List[Dict[str, object]] = field(default_factory=list)
    impossible_odds_rows: List[Dict[str, object]] = field(default_factory=list)
    anomalous_overround_pinnacle: List[Dict[str, object]] = field(default_factory=list)
    anomalous_overround_bfe: List[Dict[str, object]] = field(default_factory=list)


def download_raw_csvs(
    raw_dir: Path,
    seasons: List[str] = SEASONS,
    leagues: List[str] = list(LEAGUES.keys()),
    pause_seconds: float = 0.25,
) -> Tuple[List[Tuple[str, str, Path]], AuditResults]:
    """Download raw CSV files with local caching and polite delay."""
    if "2627" in seasons:
        raise ValueError("RULE VIOLATION: Holdout season 2627 must never be included in historical dataset.")

    raw_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditResults()
    downloaded_files: List[Tuple[str, str, Path]] = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BettaQuantAudit/1.0"
    }

    for season in seasons:
        for league in leagues:
            file_path = raw_dir / f"{season}_{league}.csv"
            if file_path.exists() and file_path.stat().st_size > 0:
                audit.files_cached += 1
                downloaded_files.append((season, league, file_path))
                continue

            url = BASE_URL.format(season=season, league=league)
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=20) as resp:
                    content = resp.read()
                file_path.write_bytes(content)
                audit.files_downloaded += 1
                downloaded_files.append((season, league, file_path))
                time.sleep(pause_seconds)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                audit.download_errors.append(f"{season}_{league} ({url}): {exc}")

    return downloaded_files, audit


def read_single_csv(file_path: Path, season: str, league: str, audit: AuditResults) -> pd.DataFrame:
    """Read a Football-Data.co.uk CSV handling mixed encodings, trailing commas, and blank tail rows."""
    raw_bytes = file_path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return pd.DataFrame()

    # Clean header names
    raw_header = [col.strip() for col in rows[0]]
    # Keep non-empty header column indices
    valid_indices = [idx for idx, col in enumerate(raw_header) if col != ""]
    header = [raw_header[idx] for idx in valid_indices]

    # Deduplicate any duplicate header names in raw file by keeping first occurrence
    seen_cols = set()
    dedup_indices = []
    dedup_header = []
    for idx, col_name in zip(valid_indices, header):
        if col_name not in seen_cols:
            seen_cols.add(col_name)
            dedup_indices.append(idx)
            dedup_header.append(col_name)

    data_rows = []
    for row in rows[1:]:
        audit.raw_rows_read += 1
        if not row or all(cell.strip() == "" for cell in row):
            audit.empty_rows_dropped += 1
            continue

        extracted = [row[idx].strip() if idx < len(row) else "" for idx in dedup_indices]
        if all(cell == "" for cell in extracted):
            audit.empty_rows_dropped += 1
            continue

        data_rows.append(extracted)

    df = pd.DataFrame(data_rows, columns=dedup_header)
    df = df.replace("", np.nan)

    # Filter rows where Date, HomeTeam, and AwayTeam are all NaN (trailing empty/corrupt lines)
    key_check_cols = [c for c in ["Date", "HomeTeam", "AwayTeam"] if c in df.columns]
    if key_check_cols:
        all_missing_mask = df[key_check_cols].isna().all(axis=1)
        dropped_count = int(all_missing_mask.sum())
        if dropped_count > 0:
            audit.empty_rows_dropped += dropped_count
            df = df[~all_missing_mask].copy()

    df["Season"] = season
    df["Div"] = league
    return df


def parse_mixed_dates(date_series: pd.Series) -> pd.Series:
    """Parse dd/mm/yyyy and dd/mm/yy dates consistently without ambiguity."""
    cleaned = date_series.astype(str).str.strip()
    parsed = pd.to_datetime(cleaned, format="%d/%m/%Y", errors="coerce")
    mask_na = parsed.isna()
    if mask_na.any():
        parsed_short = pd.to_datetime(cleaned[mask_na], format="%d/%m/%y", errors="coerce")
        parsed.loc[mask_na] = parsed_short
    mask_na = parsed.isna()
    if mask_na.any():
        parsed_fallback = pd.to_datetime(cleaned[mask_na], dayfirst=True, errors="coerce")
        parsed.loc[mask_na] = parsed_fallback
    return parsed


def build_level1_dataset(
    downloaded_files: List[Tuple[str, str, Path]],
    audit: AuditResults,
) -> pd.DataFrame:
    """Consolidate all 50 CSVs into a unified schema, sort chronologically, and run Reconciler checks."""
    frames: List[pd.DataFrame] = []
    discovered_bfe_cols: List[str] = list(CANONICAL_BFE_COLS)

    for season, league, path in downloaded_files:
        df_part = read_single_csv(path, season, league, audit)
        for col in df_part.columns:
            if col.startswith("BFE") and col not in discovered_bfe_cols:
                discovered_bfe_cols.append(col)
        frames.append(df_part)

    if not frames:
        raise RuntimeError("No CSV frames were loaded.")

    full_df = pd.concat(frames, ignore_index=True)

    # Coalesce historical Betbrain Max/Avg (1617-1819) into Max*/Avg* where Max*/Avg* are NaN,
    # while also keeping the raw BbMx*/BbAv* columns for completeness.
    bb_mapping = {
        "MaxH": "BbMxH",
        "MaxD": "BbMxD",
        "MaxA": "BbMxA",
        "AvgH": "BbAvH",
        "AvgD": "BbAvD",
        "AvgA": "BbAvA",
        "Max>2.5": "BbMx>2.5",
        "Avg>2.5": "BbAv>2.5",
        "Max<2.5": "BbMx<2.5",
        "Avg<2.5": "BbAv<2.5",
    }
    for modern_col, legacy_col in bb_mapping.items():
        if modern_col not in full_df.columns:
            full_df[modern_col] = np.nan
        if legacy_col in full_df.columns:
            full_df[modern_col] = full_df[modern_col].fillna(full_df[legacy_col])

    unified_cols = (
        META_COLS
        + SCORE_STATS_COLS
        + PINNACLE_1X2_OPEN_COLS
        + PINNACLE_1X2_CLOSE_COLS
        + PINNACLE_OU_OPEN_COLS
        + PINNACLE_OU_CLOSE_COLS
        + MAX_AVG_COLS
        + discovered_bfe_cols
    )
    # Remove any duplicates in unified_cols preserving order
    seen = set()
    ordered_cols = []
    for c in unified_cols:
        if c not in seen:
            seen.add(c)
            ordered_cols.append(c)

    for col in ordered_cols:
        if col not in full_df.columns:
            full_df[col] = np.nan

    full_df = full_df[ordered_cols].copy()

    # Parse dates
    raw_date_str = full_df["Date"].copy()
    parsed_dates = parse_mixed_dates(full_df["Date"])
    full_df["ParsedDate"] = parsed_dates

    # 1. Check unparsed dates
    unparsed_mask = full_df["ParsedDate"].isna()
    if unparsed_mask.any():
        for idx, row in full_df[unparsed_mask].iterrows():
            audit.unparsed_date_rows.append(
                {
                    "index": int(idx),
                    "Season": row["Season"],
                    "Div": row["Div"],
                    "RawDate": raw_date_str.loc[idx],
                    "HomeTeam": row["HomeTeam"],
                    "AwayTeam": row["AwayTeam"],
                }
            )
        full_df = full_df[~unparsed_mask].copy()

    # Clean team names whitespace
    full_df["HomeTeam"] = full_df["HomeTeam"].astype(str).str.strip()
    full_df["AwayTeam"] = full_df["AwayTeam"].astype(str).str.strip()

    # Convert numeric columns
    odds_cols = (
        PINNACLE_1X2_OPEN_COLS
        + PINNACLE_1X2_CLOSE_COLS
        + PINNACLE_OU_OPEN_COLS
        + PINNACLE_OU_CLOSE_COLS
        + MAX_AVG_COLS
        + discovered_bfe_cols
    )
    for col in NUMERIC_SCORE_STATS_COLS + odds_cols:
        full_df[col] = pd.to_numeric(full_df[col], errors="coerce")

    # 2. Check missing match results (FTHG, FTAG, FTR, HomeTeam, AwayTeam)
    missing_res_mask = (
        full_df["FTHG"].isna()
        | full_df["FTAG"].isna()
        | full_df["FTR"].isna()
        | full_df["HomeTeam"].isna()
        | full_df["AwayTeam"].isna()
        | (full_df["HomeTeam"] == "nan")
        | (full_df["AwayTeam"] == "nan")
    )
    if missing_res_mask.any():
        for idx, row in full_df[missing_res_mask].iterrows():
            audit.missing_result_rows.append(
                {
                    "Season": row["Season"],
                    "Div": row["Div"],
                    "Date": row["ParsedDate"].strftime("%Y-%m-%d") if pd.notna(row["ParsedDate"]) else str(row["Date"]),
                    "HomeTeam": row["HomeTeam"],
                    "AwayTeam": row["AwayTeam"],
                    "FTHG": row["FTHG"],
                    "FTAG": row["FTAG"],
                    "FTR": row["FTR"],
                }
            )
        full_df = full_df[~missing_res_mask].copy()

    # Format Date as YYYY-MM-DD and sort strictly by chronological order (Date, Time, Div, HomeTeam)
    full_df["Date"] = full_df["ParsedDate"].dt.strftime("%Y-%m-%d")
    full_df["SortTime"] = full_df["Time"].fillna("00:00").astype(str)
    full_df = full_df.sort_values(
        by=["ParsedDate", "SortTime", "Div", "HomeTeam", "AwayTeam"],
        ascending=[True, True, True, True, True],
    )
    full_df = full_df.drop(columns=["ParsedDate", "SortTime"]).reset_index(drop=True)

    # 3. Check duplicate matches (Div, Date, HomeTeam, AwayTeam) or (Season, Div, HomeTeam, AwayTeam)
    dup_mask = full_df.duplicated(subset=["Div", "Date", "HomeTeam", "AwayTeam"], keep=False)
    if dup_mask.any():
        for idx, row in full_df[dup_mask].iterrows():
            audit.duplicate_matches.append(
                {
                    "Season": row["Season"],
                    "Div": row["Div"],
                    "Date": row["Date"],
                    "HomeTeam": row["HomeTeam"],
                    "AwayTeam": row["AwayTeam"],
                }
            )
        full_df = full_df.drop_duplicates(subset=["Div", "Date", "HomeTeam", "AwayTeam"], keep="first").reset_index(drop=True)

    # 4. Check and sanitize impossible odds (<= 1.01) across all odds columns BEFORE calculating overrounds or saving
    for col in odds_cols:
        if col not in full_df.columns:
            continue
        imp_mask = full_df[col].notna() & (full_df[col] <= 1.01)
        if imp_mask.any():
            for idx, row in full_df[imp_mask].iterrows():
                audit.impossible_odds_rows.append(
                    {
                        "Season": row["Season"],
                        "Div": row["Div"],
                        "Date": row["Date"],
                        "HomeTeam": row["HomeTeam"],
                        "AwayTeam": row["AwayTeam"],
                        "Column": col,
                        "Value": float(row[col]),
                    }
                )
            # Sanitize impossible odds to NaN
            full_df.loc[imp_mask, col] = np.nan

    # 5. Calculate Overround for Pinnacle Closing (PSCH, PSCD, PSCA) and Betfair Exchange Closing (BFECH, BFECD, BFECA)
    # Overround = (1/OddH + 1/OddD + 1/OddA) - 1.0
    valid_psc = (full_df["PSCH"] > 1.01) & (full_df["PSCD"] > 1.01) & (full_df["PSCA"] > 1.01)
    full_df["Overround_Pinnacle_Close"] = np.where(
        valid_psc,
        (1.0 / full_df["PSCH"] + 1.0 / full_df["PSCD"] + 1.0 / full_df["PSCA"]) - 1.0,
        np.nan,
    )

    if all(c in full_df.columns for c in ["BFECH", "BFECD", "BFECA"]):
        valid_bfec = (full_df["BFECH"] > 1.01) & (full_df["BFECD"] > 1.01) & (full_df["BFECA"] > 1.01)
        full_df["Overround_BFE_Close"] = np.where(
            valid_bfec,
            (1.0 / full_df["BFECH"] + 1.0 / full_df["BFECD"] + 1.0 / full_df["BFECA"]) - 1.0,
            np.nan,
        )
    else:
        full_df["Overround_BFE_Close"] = np.nan

    # 6. Check anomalous overrounds (< -0.5% or > 12%, i.e., < -0.005 or > 0.12)
    psc_anom_mask = full_df["Overround_Pinnacle_Close"].notna() & (
        (full_df["Overround_Pinnacle_Close"] < -0.005) | (full_df["Overround_Pinnacle_Close"] > 0.12)
    )
    if psc_anom_mask.any():
        for idx, row in full_df[psc_anom_mask].iterrows():
            audit.anomalous_overround_pinnacle.append(
                {
                    "Season": row["Season"],
                    "Div": row["Div"],
                    "Date": row["Date"],
                    "HomeTeam": row["HomeTeam"],
                    "AwayTeam": row["AwayTeam"],
                    "PSCH": float(row["PSCH"]),
                    "PSCD": float(row["PSCD"]),
                    "PSCA": float(row["PSCA"]),
                    "Overround_pct": float(row["Overround_Pinnacle_Close"] * 100.0),
                }
            )

    bfe_anom_mask = full_df["Overround_BFE_Close"].notna() & (
        (full_df["Overround_BFE_Close"] < -0.005) | (full_df["Overround_BFE_Close"] > 0.12)
    )
    if bfe_anom_mask.any():
        for idx, row in full_df[bfe_anom_mask].iterrows():
            audit.anomalous_overround_bfe.append(
                {
                    "Season": row["Season"],
                    "Div": row["Div"],
                    "Date": row["Date"],
                    "HomeTeam": row["HomeTeam"],
                    "AwayTeam": row["AwayTeam"],
                    "BFECH": float(row["BFECH"]),
                    "BFECD": float(row["BFECD"]),
                    "BFECA": float(row["BFECA"]),
                    "Overround_pct": float(row["Overround_BFE_Close"] * 100.0),
                }
            )

    return full_df


def generate_audit_report(df: pd.DataFrame, audit: AuditResults, report_path: Path) -> None:
    """Generate the Markdown Level 1 Audit Report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    total_matches = len(df)

    lines: List[str] = []
    lines.append("# Level 1 Audit Report — Data Ingestion & Integrity (Top 5 Leagues, 2016/17 – 2025/26)")
    lines.append("")
    lines.append("## 1. Riepilogo Generale & Download")
    lines.append(f"- **File CSV totali previsti**: 50 (10 stagioni `1617`–`2526` × 5 leghe `I1`, `E0`, `SP1`, `D1`, `F1`)")
    lines.append(f"- **Stagione Holdout `2627`**: **Esclusa categoricamente** (in conformità a `RULES.md`)")
    lines.append(f"- **File scaricati ex-novo**: {audit.files_downloaded}")
    lines.append(f"- **File letti da cache locale (`data/raw`)**: {audit.files_cached}")
    lines.append(f"- **Errori di download**: {len(audit.download_errors)}")
    if audit.download_errors:
        for err in audit.download_errors:
            lines.append(f"  - `{err}`")
    lines.append(f"- **Righe grezze lette dai CSV**: {audit.raw_rows_read}")
    lines.append(f"- **Righe vuote di coda / padding rimosse**: {audit.empty_rows_dropped}")
    lines.append(f"- **Numero totale di partite valide nel dataset consolidato**: **{total_matches}**")
    lines.append(f"- **Range temporale**: `{df['Date'].min()}` → `{df['Date'].max()}`")
    lines.append("")

    # Section 2: Coverage table per season
    lines.append("## 2. Copertura Percentuale Non-Null per Stagione")
    lines.append("")
    lines.append("| Stagione | Partite | Pinnacle Open 1X2 (%) | Pinnacle Close 1X2 (%) | Pinnacle Close O/U 2.5 (%) | Betfair Exchange Close 1X2 (%) |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")

    for season in SEASONS:
        sub = df[df["Season"].astype(str) == season]
        n = len(sub)
        if n == 0:
            lines.append(f"| `{season}` | 0 | 0.00% | 0.00% | 0.00% | 0.00% |")
            continue
        p_open = sub[["PSH", "PSD", "PSA"]].notna().all(axis=1).mean() * 100.0
        p_close = sub[["PSCH", "PSCD", "PSCA"]].notna().all(axis=1).mean() * 100.0
        p_ou_close = sub[["PC>2.5", "PC<2.5"]].notna().all(axis=1).mean() * 100.0
        bfe_close = sub[["BFECH", "BFECD", "BFECA"]].notna().all(axis=1).mean() * 100.0
        lines.append(
            f"| `{season}` | {n} | {p_open:.2f}% | {p_close:.2f}% | {p_ou_close:.2f}% | {bfe_close:.2f}% |"
        )
    lines.append("")

    # Section 3: League summary table
    lines.append("## 3. Riepilogo per Lega")
    lines.append("")
    lines.append("| Lega | Nome | Numero Partite | NaN su Pinnacle Closing 1X2 | Overround Medio Pinnacle Closing (%) | % Partite con almeno 1 Cartellino Rosso |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: |")

    for div_code, div_name in LEAGUES.items():
        sub = df[df["Div"] == div_code]
        n = len(sub)
        if n == 0:
            continue
        nan_psc = int((~sub[["PSCH", "PSCD", "PSCA"]].notna().all(axis=1)).sum())
        mean_or = sub["Overround_Pinnacle_Close"].mean() * 100.0
        has_red = ((sub["HR"].fillna(0) + sub["AR"].fillna(0)) >= 1).mean() * 100.0
        lines.append(
            f"| `{div_code}` | {div_name} | {n} | {nan_psc} | {mean_or:.2f}% | {has_red:.2f}% |"
        )
    lines.append("")

    # Section 4: Reconciler anomalies
    lines.append("## 4. Riepilogo Anomalie Rilevate dal Reconciler (Fail-Fast & Data Integrity)")
    lines.append("")
    lines.append(f"- **Date non parsate / righe corrotte scartate**: **{len(audit.unparsed_date_rows)}**")
    if audit.unparsed_date_rows:
        for item in audit.unparsed_date_rows[:15]:
            lines.append(
                f"  - Stagione `{item['Season']}`, Lega `{item['Div']}`, RawDate=`{item['RawDate']}`, `{item['HomeTeam']}` vs `{item['AwayTeam']}`"
            )
    lines.append(f"- **Partite con risultato finale mancante (FTHG/FTAG/FTR) scartate**: **{len(audit.missing_result_rows)}**")
    if audit.missing_result_rows:
        for item in audit.missing_result_rows[:15]:
            lines.append(
                f"  - Stagione `{item['Season']}`, Lega `{item['Div']}`, Data `{item['Date']}`, `{item['HomeTeam']}` vs `{item['AwayTeam']}` (FTHG={item['FTHG']}, FTAG={item['FTAG']}, FTR={item['FTR']})"
            )
    lines.append(f"- **Partite duplicate rilevate (e deduplicate)**: **{len(audit.duplicate_matches)}**")
    if audit.duplicate_matches:
        for item in audit.duplicate_matches[:15]:
            lines.append(
                f"  - Stagione `{item['Season']}`, Lega `{item['Div']}`, Data `{item['Date']}`, `{item['HomeTeam']}` vs `{item['AwayTeam']}`"
            )
    lines.append(f"- **Quote impossibili ($\\le 1.01$) rilevate e bonificate in `NaN`**: **{len(audit.impossible_odds_rows)}**")
    if audit.impossible_odds_rows:
        for item in audit.impossible_odds_rows[:25]:
            lines.append(
                f"  - `{item['Date']}` (`{item['Season']}` `{item['Div']}`) `{item['HomeTeam']}` vs `{item['AwayTeam']}` — Colonna `{item['Column']}` = `{item['Value']}`"
            )
        if len(audit.impossible_odds_rows) > 25:
            lines.append(f"  - *(...e altre {len(audit.impossible_odds_rows) - 25} occorrenze)*")

    lines.append(
        f"- **Overround anomali Pinnacle Closing ($< -0.5\\%$ o $> 12\\%$)**: **{len(audit.anomalous_overround_pinnacle)}**"
    )
    if audit.anomalous_overround_pinnacle:
        for item in audit.anomalous_overround_pinnacle[:25]:
            lines.append(
                f"  - `{item['Date']}` (`{item['Season']}` `{item['Div']}`) `{item['HomeTeam']}` vs `{item['AwayTeam']}`: "
                f"PSCH={item['PSCH']}, PSCD={item['PSCD']}, PSCA={item['PSCA']} → Overround = **{item['Overround_pct']:.2f}%**"
            )
        if len(audit.anomalous_overround_pinnacle) > 25:
            lines.append(f"  - *(...e altre {len(audit.anomalous_overround_pinnacle) - 25} occorrenze)*")

    lines.append(
        f"- **Overround anomali Betfair Exchange Closing ($< -0.5\\%$ o $> 12\\%$)**: **{len(audit.anomalous_overround_bfe)}**"
    )
    if audit.anomalous_overround_bfe:
        for item in audit.anomalous_overround_bfe[:25]:
            lines.append(
                f"  - `{item['Date']}` (`{item['Season']}` `{item['Div']}`) `{item['HomeTeam']}` vs `{item['AwayTeam']}`: "
                f"BFECH={item['BFECH']}, BFECD={item['BFECD']}, BFECA={item['BFECA']} → Overround = **{item['Overround_pct']:.2f}%**"
            )
        if len(audit.anomalous_overround_bfe) > 25:
            lines.append(f"  - *(...e altre {len(audit.anomalous_overround_bfe) - 25} occorrenze)*")
    lines.append("")

    # Section 5: Serie A unique team names in alphabetical order
    serie_a = df[df["Div"] == "I1"]
    serie_a_teams = sorted(
        set(serie_a["HomeTeam"].dropna().unique()).union(set(serie_a["AwayTeam"].dropna().unique()))
    )
    lines.append("## 5. Elenco Completo Nomi Unici Squadre in Serie A (`I1`)")
    lines.append(f"Totale nomi squadra unici rilevati in Serie A (`1617`–`2526`): **{len(serie_a_teams)}**")
    lines.append("")
    for idx, team in enumerate(serie_a_teams, start=1):
        lines.append(f"{idx}. `{team}`")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    raw_dir = root_dir / "data" / "raw"
    processed_path = root_dir / "data" / "processed" / "level1_top5_1617_2526.csv"
    report_path = root_dir / "reports" / "level1_audit.md"

    print("[Level 1] Starting download / cache verification...")
    downloaded_files, audit = download_raw_csvs(raw_dir=raw_dir)
    print(
        f"[Level 1] Files ready: {len(downloaded_files)} "
        f"(downloaded={audit.files_downloaded}, cached={audit.files_cached}, errors={len(audit.download_errors)})"
    )

    print("[Level 1] Reconciling schema and running integrity checks...")
    df = build_level1_dataset(downloaded_files, audit)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False, encoding="utf-8")
    print(f"[Level 1] Saved consolidated dataset ({len(df)} matches) to {processed_path}")

    generate_audit_report(df, audit, report_path)
    print(f"[Level 1] Generated audit report at {report_path}")


if __name__ == "__main__":
    main()
