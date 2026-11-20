"""
Level 2: Understat xG / npxG Data Ingestion & Local Caching (src/level2_xg_ingest.py).

Acquires match-level xG, npxG, goals, PPDA, and Deep Completions from Understat
for the Top 5 European leagues (Serie_A, EPL, La_liga, Bundesliga, Ligue_1)
across the 10 historical seasons 2016..2025 (corresponding to 1617..2526).
Strictly excludes season 2026 (2627) in compliance with RULES.md.
Caches raw JSON responses in data/raw/understat/{season}_{div}.json.
"""

from __future__ import annotations

import codecs
import http.cookiejar
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Mapping from Football-Data season code (1617..2526) to Understat season start year (2016..2025)
SEASON_TO_UNDERSTAT_YEAR: Dict[str, int] = {
    "1617": 2016,
    "1718": 2017,
    "1819": 2018,
    "1920": 2019,
    "2021": 2020,
    "2122": 2021,
    "2223": 2022,
    "2324": 2023,
    "2425": 2024,
    "2526": 2025,
}

# Mapping from Football-Data division code to Understat league slug
DIV_TO_UNDERSTAT_LEAGUE: Dict[str, str] = {
    "I1": "Serie_A",
    "E0": "EPL",
    "SP1": "La_liga",
    "D1": "Bundesliga",
    "F1": "Ligue_1",
}

LEAGUE_PAGE_URL = "https://understat.com/league/{u_league}/{year}"
LEAGUE_AJAX_URL = "https://understat.com/getLeagueData/{u_league}/{year}"

DATES_REGEX = re.compile(r"datesData\s*=\s*JSON\.parse\('([^']+)'\)")
TEAMS_REGEX = re.compile(r"teamsData\s*=\s*JSON\.parse\('([^']+)'\)")


def _decode_escaped_json(escaped_str: str) -> Any:
    """Decode Understat's hex/unicode-escaped JSON string from JSON.parse('...')."""
    decoded = codecs.decode(escaped_str, "unicode_escape")
    return json.loads(decoded)


def fetch_understat_league_season(
    u_league: str,
    year: int,
    timeout: int = 25,
) -> Dict[str, Any]:
    """
    Fetch datesData and teamsData for a given Understat league and year.
    Supports both embedded HTML JSON.parse('...') and Understat's getLeagueData AJAX endpoint.
    Fails fast with a clear diagnostic error if blocked or if structure is unrecognizable.
    """
    if year >= 2026:
        raise ValueError(f"RULE VIOLATION: Holdout year {year} (season 2627) is strictly forbidden.")

    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    page_url = LEAGUE_PAGE_URL.format(u_league=u_league, year=year)
    req_page = urllib.request.Request(
        page_url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )

    try:
        with opener.open(req_page, timeout=timeout) as resp:
            status = resp.status
            html_bytes = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Fail-Fast: HTTP error fetching {page_url}: {exc}") from exc

    if status != 200:
        raise RuntimeError(f"Fail-Fast: Unexpected HTTP status {status} from {page_url}")

    html_text = html_bytes.decode("utf-8", errors="replace")

    if "We will come back soon" in html_text or "/css/errors.css" in html_text:
        # Also verify if the internal AJAX endpoint getLeagueData is likewise returning the maintenance page
        ajax_url = LEAGUE_AJAX_URL.format(u_league=u_league, year=year)
        req_ajax = urllib.request.Request(
            ajax_url,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": page_url,
            },
        )
        try:
            with opener.open(req_ajax, timeout=timeout) as resp_ajax:
                ajax_text = resp_ajax.read().decode("utf-8", errors="replace")
        except Exception as exc:
            raise RuntimeError(
                f"FAIL-FAST [Understat Offline]: {page_url} returned maintenance page "
                f"('We will come back soon', /css/errors.css) and AJAX endpoint {ajax_url} failed: {exc}"
            ) from exc

        if "We will come back soon" in ajax_text or "/css/errors.css" in ajax_text or not ajax_text.strip().startswith("{"):
            raise RuntimeError(
                f"FAIL-FAST [Understat Offline / Blocco Strutturale]: Sia la pagina HTML ({page_url}) "
                f"sia l'endpoint XHR ({ajax_url}) restituiscono la splash page di manutenzione di Understat "
                f"(26.719 byte, <link href='/css/errors.css'>, '<p>We will come back soon</p>'). "
                f"Nessun JSON datesData/teamsData disponibile. Esecuzione interrotta per evitare dati finti o vuoti."
            )
        payload = json.loads(ajax_text)
        dates_data = payload.get("dates") or payload.get("datesData")
        teams_data = payload.get("teams") or payload.get("teamsData")
        return {"datesData": dates_data, "teamsData": teams_data, "source": "ajax_getLeagueData"}

    # Path A: Check if datesData and teamsData are directly embedded in HTML
    m_dates = DATES_REGEX.search(html_text)
    m_teams = TEAMS_REGEX.search(html_text)
    if m_dates and m_teams:
        dates_data = _decode_escaped_json(m_dates.group(1))
        teams_data = _decode_escaped_json(m_teams.group(1))
        if not dates_data or not teams_data:
            raise RuntimeError(f"Fail-Fast: Empty embedded datesData/teamsData for {u_league} {year}")
        return {"datesData": dates_data, "teamsData": teams_data, "source": "embedded_html"}

    # Path B: Fetch from Understat's AJAX endpoint getLeagueData/{u_league}/{year}
    ajax_url = LEAGUE_AJAX_URL.format(u_league=u_league, year=year)
    req_ajax = urllib.request.Request(
        ajax_url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": page_url,
        },
    )
    try:
        with opener.open(req_ajax, timeout=timeout) as resp_ajax:
            ajax_bytes = resp_ajax.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Fail-Fast: Neither embedded JSON nor AJAX endpoint succeeded for {u_league} {year}: {exc}"
        ) from exc

    try:
        payload = json.loads(ajax_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        snippet = ajax_bytes[:200].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Fail-Fast: Understat returned non-JSON payload on {ajax_url} (snippet: {snippet!r})"
        ) from exc

    dates_data = payload.get("dates") or payload.get("datesData")
    teams_data = payload.get("teams") or payload.get("teamsData")
    if not dates_data or not teams_data:
        raise RuntimeError(
            f"Fail-Fast: Missing 'dates'/'teams' keys in Understat JSON response for {u_league} {year} "
            f"(keys={list(payload.keys())})"
        )

    return {"datesData": dates_data, "teamsData": teams_data, "source": "ajax_getLeagueData"}


def download_all_understat_raw(
    raw_understat_dir: Path,
    pause_seconds: float = 0.6,
) -> Tuple[List[Tuple[str, str, Path]], int, int]:
    """
    Download or load from local cache all 50 league-seasons (1617..2526 x 5 leagues).
    Returns (list of (season, div, json_path), downloaded_count, cached_count).
    """
    raw_understat_dir.mkdir(parents=True, exist_ok=True)
    files_list: List[Tuple[str, str, Path]] = []
    downloaded = 0
    cached = 0

    for season, year in SEASON_TO_UNDERSTAT_YEAR.items():
        if season == "2627" or year >= 2026:
            raise ValueError("RULE VIOLATION: Holdout season 2627 must never be downloaded.")
        for div, u_league in DIV_TO_UNDERSTAT_LEAGUE.items():
            cache_file = raw_understat_dir / f"{season}_{div}.json"
            if cache_file.exists() and cache_file.stat().st_size > 100:
                cached += 1
                files_list.append((season, div, cache_file))
                continue

            print(f"  [Understat Fetch] {season} {div} ({u_league}/{year})...")
            data = fetch_understat_league_season(u_league=u_league, year=year)
            cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            downloaded += 1
            files_list.append((season, div, cache_file))
            time.sleep(pause_seconds)

    return files_list, downloaded, cached


def _compute_ppda_ratio(ppda_obj: Any) -> float:
    """Compute PPDA = att / def from Understat ppda dict {'att': int, 'def': int}."""
    if not isinstance(ppda_obj, dict):
        return float("nan")
    att = float(ppda_obj.get("att", 0.0))
    dfn = float(ppda_obj.get("def", 0.0))
    if dfn <= 0.0:
        return float("nan")
    return att / dfn


def parse_understat_league_season_json(
    season: str,
    div: str,
    json_path: Path,
) -> pd.DataFrame:
    """
    Parse a cached Understat league-season JSON file (containing datesData and teamsData)
    into a flat match-level DataFrame with:
      Season, Div, understat_match_id, understat_datetime, understat_date,
      understat_home_team, understat_away_team,
      understat_home_goals, understat_away_goals,
      home_xg, away_xg, home_npxg, away_npxg,
      home_ppda, away_ppda, home_deep, away_deep
    """
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    dates_data = payload["datesData"]
    teams_data = payload["teamsData"]

    # Index team history by (team_id, h_a, datetime) and fallback by (team_id, h_a, date_only)
    hist_by_exact_dt: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    hist_by_date_only: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    hist_by_team_ha: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    if isinstance(teams_data, dict):
        team_entries = teams_data.values()
    elif isinstance(teams_data, list):
        team_entries = teams_data
    else:
        raise RuntimeError(f"Unexpected teamsData type in {json_path}: {type(teams_data)}")

    for t_obj in team_entries:
        t_id = str(t_obj.get("id", ""))
        t_title = str(t_obj.get("title", "")).strip()
        for h_item in t_obj.get("history", []):
            ha = str(h_item.get("h_a", "")).strip()
            dt_str = str(h_item.get("date", "")).strip()
            d_only = dt_str[:10]
            hist_by_exact_dt[(t_id, ha, dt_str)] = h_item
            hist_by_exact_dt[(t_title, ha, dt_str)] = h_item
            hist_by_date_only.setdefault((t_id, ha, d_only), []).append(h_item)
            hist_by_date_only.setdefault((t_title, ha, d_only), []).append(h_item)
            hist_by_team_ha.setdefault((t_id, ha), []).append(h_item)

    records: List[Dict[str, Any]] = []
    for m in dates_data:
        # Consider only completed matches
        is_res = m.get("isResult")
        goals_obj = m.get("goals", {})
        xg_obj = m.get("xG", {})
        if not is_res and (goals_obj.get("h") is None or xg_obj.get("h") is None):
            continue
        if goals_obj.get("h") is None or goals_obj.get("a") is None:
            continue
        if xg_obj.get("h") is None or xg_obj.get("a") is None:
            continue

        m_id = str(m.get("id", ""))
        dt_str = str(m.get("datetime", "")).strip()
        d_only = dt_str[:10]

        h_info = m.get("h", {})
        a_info = m.get("a", {})
        h_id = str(h_info.get("id", ""))
        a_id = str(a_info.get("id", ""))
        h_title = str(h_info.get("title", "")).strip()
        a_title = str(a_info.get("title", "")).strip()

        u_hg = int(goals_obj["h"])
        u_ag = int(goals_obj["a"])
        h_xg = float(xg_obj["h"])
        a_xg = float(xg_obj["a"])

        # Match with home team's history entry (h_a == 'h')
        h_hist = (
            hist_by_exact_dt.get((h_id, "h", dt_str))
            or hist_by_exact_dt.get((h_title, "h", dt_str))
        )
        if h_hist is None:
            cands = hist_by_date_only.get((h_id, "h", d_only), [])
            if len(cands) == 1:
                h_hist = cands[0]
            elif len(cands) > 1:
                for c in cands:
                    if int(c.get("scored", -1)) == u_hg and int(c.get("missed", -1)) == u_ag:
                        h_hist = c
                        break
        if h_hist is None:
            # Fallback: search by (scored, missed, closest xG) in hist_by_team_ha
            cands = [
                c
                for c in hist_by_team_ha.get((h_id, "h"), [])
                if int(c.get("scored", -1)) == u_hg
                and int(c.get("missed", -1)) == u_ag
                and abs(float(c.get("xG", -99.0)) - h_xg) < 1e-2
                and abs(float(c.get("xGA", -99.0)) - a_xg) < 1e-2
            ]
            if len(cands) >= 1:
                h_hist = cands[0]

        # Also lookup away team's history entry (h_a == 'a') as verification / fallback
        a_hist = (
            hist_by_exact_dt.get((a_id, "a", dt_str))
            or hist_by_exact_dt.get((a_title, "a", dt_str))
        )
        if a_hist is None:
            cands_a = hist_by_date_only.get((a_id, "a", d_only), [])
            if len(cands_a) == 1:
                a_hist = cands_a[0]

        if h_hist is not None:
            h_npxg = float(h_hist.get("npxG", np.nan))
            a_npxg = float(h_hist.get("npxGA", np.nan))
            h_ppda = _compute_ppda_ratio(h_hist.get("ppda"))
            a_ppda = _compute_ppda_ratio(h_hist.get("ppda_allowed"))
            h_deep = float(h_hist.get("deep", np.nan))
            a_deep = float(h_hist.get("deep_allowed", np.nan))
        elif a_hist is not None:
            h_npxg = float(a_hist.get("npxGA", np.nan))
            a_npxg = float(a_hist.get("npxG", np.nan))
            h_ppda = _compute_ppda_ratio(a_hist.get("ppda_allowed"))
            a_ppda = _compute_ppda_ratio(a_hist.get("ppda"))
            h_deep = float(a_hist.get("deep_allowed", np.nan))
            a_deep = float(a_hist.get("deep", np.nan))
        else:
            h_npxg = float("nan")
            a_npxg = float("nan")
            h_ppda = float("nan")
            a_ppda = float("nan")
            h_deep = float("nan")
            a_deep = float("nan")

        records.append(
            {
                "Season": season,
                "Div": div,
                "understat_match_id": m_id,
                "understat_datetime": dt_str,
                "understat_date": d_only,
                "understat_home_team": h_title,
                "understat_away_team": a_title,
                "understat_home_goals": u_hg,
                "understat_away_goals": u_ag,
                "home_xg": h_xg,
                "away_xg": a_xg,
                "home_npxg": h_npxg,
                "away_npxg": a_npxg,
                "home_ppda": h_ppda,
                "away_ppda": a_ppda,
                "home_deep": h_deep,
                "away_deep": a_deep,
            }
        )

    return pd.DataFrame(records)


def load_all_understat_matches(
    raw_understat_dir: Path,
    pause_seconds: float = 0.6,
) -> Tuple[pd.DataFrame, int, int]:
    """Download/load all 50 Understat JSON files and return the combined match DataFrame."""
    files_list, downloaded, cached = download_all_understat_raw(
        raw_understat_dir=raw_understat_dir,
        pause_seconds=pause_seconds,
    )
    frames: List[pd.DataFrame] = []
    for season, div, json_path in files_list:
        df_part = parse_understat_league_season_json(season=season, div=div, json_path=json_path)
        frames.append(df_part)

    df_understat = pd.concat(frames, ignore_index=True)
    return df_understat, downloaded, cached


def main() -> None:
    raw_understat_dir = ROOT_DIR / "data" / "raw" / "understat"
    print("[Level 2 Ingest] Downloading / verifying Understat raw JSON cache...")
    df_u, downloaded, cached = load_all_understat_matches(raw_understat_dir=raw_understat_dir)
    print(
        f"[Level 2 Ingest] Understat ready: {len(df_u)} matches across 50 league-seasons "
        f"(downloaded={downloaded}, cached={cached})."
    )


if __name__ == "__main__":
    main()
