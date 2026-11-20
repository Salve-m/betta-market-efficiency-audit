"""
Modulo A: Market Benchmark, Devigging (Multiplicative, Power, Shin),
Probabilistic Evaluation Metrics (Log-Loss, Brier Score, Calibration),
Exchange Friction Mathematics (EV, Q_min, CLV), and Market Audit Report Generator.

Strictly adheres to RULES.md:
- Warm-up: 1617-1819
- Train / Sviluppo: 1920-2223
- Validation: 2324-2425
- Test Cieco (inviolable for modeling choices): 2526
- Holdout Live: 2627 (excluded)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq

WARMUP_SEASONS = ["1617", "1718", "1819"]
TRAIN_SEASONS = ["1920", "2021", "2122", "2223"]
VAL_SEASONS = ["2324", "2425"]
BLIND_TEST_SEASONS = ["2526"]

LEAGUE_NAMES = {
    "I1": "Serie A",
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}


# ==============================================================================
# 1. DEVIGGING METHODS (Multiplicative, Power, Shin)
# ==============================================================================


def _prepare_odds_array(odds: Union[Sequence[float], np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, bool]:
    """Convert input odds to a 2D float64 numpy array (N, K) and flag if input was 1D."""
    arr = np.asarray(odds, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1), True
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array of odds, got shape {arr.shape}")
    return arr, False


def _valid_odds_row_mask(odds_2d: np.ndarray) -> np.ndarray:
    """Return boolean mask of rows where all odds are finite and strictly > 1.01."""
    return np.all(np.isfinite(odds_2d) & (odds_2d > 1.01), axis=1)


def devig_multiplicative(odds: Union[Sequence[float], np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Multiplicative (Basic/Linear) devigging:
        q_i = 1 / Q_i
        S = sum(q_i)
        p_i = q_i / S
    Handles 1D (K,) or 2D (N, K) arrays for both 1X2 (K=3) and O/U 2.5 (K=2).
    Invalid or NaN rows produce NaNs without crashing.
    """
    odds_2d, is_1d = _prepare_odds_array(odds)
    n_rows, n_cols = odds_2d.shape
    probs = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    valid = _valid_odds_row_mask(odds_2d)
    if np.any(valid):
        q = 1.0 / odds_2d[valid]
        s = np.sum(q, axis=1, keepdims=True)
        probs[valid] = q / s

    return probs[0] if is_1d else probs


def _power_devig_single_row(q: np.ndarray) -> np.ndarray:
    """Solve sum(q_i^k) = 1 for k using scipy.optimize.brentq."""
    s = float(np.sum(q))
    if abs(s - 1.0) <= 1e-12:
        return q / s

    def obj(k: float) -> float:
        return float(np.sum(np.power(q, k)) - 1.0)

    if s > 1.0:
        # Normal overround: root k > 1
        k_low, k_high = 1.0, 100.0
        while obj(k_high) > 0 and k_high < 1e6:
            k_high *= 2.0
    else:
        # Rare slight underround (e.g. Exchange): root k in (0, 1)
        k_low, k_high = 1e-6, 1.0

    k_star = brentq(obj, k_low, k_high, xtol=1e-12, rtol=1e-12, maxiter=200)
    p = np.power(q, k_star)
    return p / np.sum(p)


def devig_power(odds: Union[Sequence[float], np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Power Method devigging:
    Finds exponent k > 1 (when overround S > 1) via scipy.optimize.brentq such that:
        sum(q_i^k) = 1, with p_i = q_i^k.
    Handles 1D (K,) or 2D (N, K) arrays for both 1X2 (K=3) and O/U 2.5 (K=2).
    """
    odds_2d, is_1d = _prepare_odds_array(odds)
    n_rows, n_cols = odds_2d.shape
    probs = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    valid = _valid_odds_row_mask(odds_2d)
    valid_indices = np.where(valid)[0]
    for idx in valid_indices:
        q = 1.0 / odds_2d[idx]
        probs[idx] = _power_devig_single_row(q)

    return probs[0] if is_1d else probs


def _shin_probs_from_z(q: np.ndarray, s: float, z: float) -> np.ndarray:
    """Compute p_i(z) = (sqrt(z^2 + 4*(1-z)*q_i^2 / S) - z) / (2*(1-z))."""
    denom = 2.0 * (1.0 - z)
    inside = (z * z) + 4.0 * (1.0 - z) * (q * q) / s
    return (np.sqrt(inside) - z) / denom


def _shin_devig_single_row(q: np.ndarray) -> np.ndarray:
    """Solve sum(p_i(z)) = 1 for z in (0, 1) using scipy.optimize.brentq."""
    s = float(np.sum(q))
    if s <= 1.0 + 1e-12:
        # When there is no positive overround, insider fraction z -> 0; fallback to multiplicative normalization
        return q / s

    def obj(z: float) -> float:
        return float(np.sum(_shin_probs_from_z(q, s, z)) - 1.0)

    z_low = 1e-12
    z_high = 1.0 - 1e-12

    f_low = obj(z_low)
    f_high = obj(z_high)
    if f_low <= 0.0:
        return q / s
    if f_high >= 0.0:
        p_lim = (q * q) / s
        return p_lim / np.sum(p_lim)

    z_star = brentq(obj, z_low, z_high, xtol=1e-12, rtol=1e-12, maxiter=200)
    p = _shin_probs_from_z(q, s, z_star)
    return p / np.sum(p)


def devig_shin(odds: Union[Sequence[float], np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Shin's Method (Štrumbelj / Shin 1993 formulation):
    Finds insider trading fraction z in (0, 1) via scipy.optimize.brentq such that:
        sum(p_i(z)) = 1, where p_i(z) = (sqrt(z^2 + 4(1-z) q_i^2 / S) - z) / (2(1-z)).
    Handles 1D (K,) or 2D (N, K) arrays for both 1X2 (K=3) and O/U 2.5 (K=2).
    """
    odds_2d, is_1d = _prepare_odds_array(odds)
    n_rows, n_cols = odds_2d.shape
    probs = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    valid = _valid_odds_row_mask(odds_2d)
    valid_indices = np.where(valid)[0]
    for idx in valid_indices:
        q = 1.0 / odds_2d[idx]
        probs[idx] = _shin_devig_single_row(q)

    return probs[0] if is_1d else probs


# ==============================================================================
# 2. PROBABILISTIC METRICS (Log-Loss, Brier Score, Calibration Table)
# ==============================================================================


def _encode_outcomes(outcomes: Union[Sequence, np.ndarray, pd.Series], n_classes: int) -> np.ndarray:
    """
    Convert outcomes into integer class indices in 0..n_classes-1.
    For 1X2 (n_classes=3): accepts 'H'/'D'/'A' or 0/1/2.
    For O/U 2.5 (n_classes=2): if probs columns are [Over, Under],
    accepts integer indices 0 (Over) / 1 (Under), or boolean/binary indicator where 1/True='Over' (index 0)
    if specified via helper, or string 'O'/'U', 'Over'/'Under'.
    """
    arr = np.asarray(outcomes)
    if n_classes == 3:
        if arr.dtype.kind in {"U", "S", "O"}:
            mapping = {"H": 0, "D": 1, "A": 2}
            return np.array([mapping.get(str(x).strip(), -1) for x in arr], dtype=int)
        return arr.astype(int)
    elif n_classes == 2:
        if arr.dtype.kind in {"U", "S", "O"}:
            mapping = {"O": 0, "OVER": 0, "U": 1, "UNDER": 1}
            return np.array([mapping.get(str(x).strip().upper(), -1) for x in arr], dtype=int)
        return arr.astype(int)
    raise ValueError(f"Unsupported n_classes={n_classes}")


def compute_log_loss(
    probs: np.ndarray,
    outcomes: Union[Sequence, np.ndarray, pd.Series],
    eps: float = 1e-15,
) -> float:
    """
    Compute Cross-Entropy (Log-Loss) for multiclass (1X2, shape (N, 3)) or binary (O/U 2.5, shape (N, 2) or (N,)).
    - If probs has shape (N, 3): outcomes can be 'H'/'D'/'A' or 0/1/2 corresponding to columns [H, D, A].
    - If probs has shape (N, 2): columns are [P(>2.5), P(<2.5)], outcomes can be 'Over'/'Under' or 0 (Over) / 1 (Under).
    - If probs is 1D (N,): interpreted as P(positive event, e.g. Over 2.5), and outcomes is binary {1, 0} where 1=positive event.
    Rows with NaNs in probs are automatically excluded.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        y = np.asarray(outcomes, dtype=np.float64)
        valid = np.isfinite(p) & np.isfinite(y)
        if not np.any(valid):
            return float("nan")
        p_v = np.clip(p[valid], eps, 1.0 - eps)
        y_v = y[valid]
        return float(-np.mean(y_v * np.log(p_v) + (1.0 - y_v) * np.log(1.0 - p_v)))

    n_rows, n_classes = p.shape
    y_idx = _encode_outcomes(outcomes, n_classes)
    valid = np.all(np.isfinite(p), axis=1) & (y_idx >= 0) & (y_idx < n_classes)
    if not np.any(valid):
        return float("nan")

    p_v = np.clip(p[valid], eps, 1.0 - eps)
    p_v = p_v / np.sum(p_v, axis=1, keepdims=True)
    y_v = y_idx[valid]
    chosen_probs = p_v[np.arange(len(y_v)), y_v]
    return float(-np.mean(np.log(chosen_probs)))


def compute_brier_score(
    probs: np.ndarray,
    outcomes: Union[Sequence, np.ndarray, pd.Series],
) -> float:
    """
    Compute Brier Score:
    - Multiclass 1X2 (shape (N, 3)): mean_n sum_{k=1}^3 (p_{n,k} - y_{n,k})^2.
    - Binary O/U 2.5 (shape (N, 2) or (N,)): standard binary Brier score mean_n (p_{n,over} - y_{n,over})^2.
    Rows with NaNs in probs are automatically excluded.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        y = np.asarray(outcomes, dtype=np.float64)
        valid = np.isfinite(p) & np.isfinite(y)
        if not np.any(valid):
            return float("nan")
        return float(np.mean((p[valid] - y[valid]) ** 2))

    n_rows, n_classes = p.shape
    y_idx = _encode_outcomes(outcomes, n_classes)
    valid = np.all(np.isfinite(p), axis=1) & (y_idx >= 0) & (y_idx < n_classes)
    if not np.any(valid):
        return float("nan")

    p_v = p[valid]
    y_v = y_idx[valid]

    if n_classes == 2:
        # Standard binary Brier Score: (p_0 - y_0)^2 where y_0 = 1 if outcome == 0 else 0
        y_bin = (y_v == 0).astype(np.float64)
        return float(np.mean((p_v[:, 0] - y_bin) ** 2))

    one_hot = np.zeros_like(p_v)
    one_hot[np.arange(len(y_v)), y_v] = 1.0
    return float(np.mean(np.sum((p_v - one_hot) ** 2, axis=1)))


def compute_calibration_table(
    probs: np.ndarray,
    outcomes: Union[Sequence, np.ndarray, pd.Series],
    bins: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Build a calibration table by probability deciles/bins comparing mean predicted probability
    vs observed empirical frequency.
    For multiclass (N, 3), pools all (p_{n,k}, y_{n,k}) pairs across H, D, A so every decile is well-populated.
    For binary (N, 2) or (N,), evaluates P(Over 2.5) (column 0 if 2D).
    """
    if bins is None:
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        y = np.asarray(outcomes, dtype=np.float64)
        valid = np.isfinite(p) & np.isfinite(y)
        flat_p = p[valid]
        flat_y = y[valid]
    elif p.shape[1] == 2:
        y_idx = _encode_outcomes(outcomes, 2)
        valid = np.all(np.isfinite(p), axis=1) & (y_idx >= 0)
        flat_p = p[valid, 0]
        flat_y = (y_idx[valid] == 0).astype(np.float64)
    else:
        n_classes = p.shape[1]
        y_idx = _encode_outcomes(outcomes, n_classes)
        valid = np.all(np.isfinite(p), axis=1) & (y_idx >= 0) & (y_idx < n_classes)
        p_v = p[valid]
        y_v = y_idx[valid]
        one_hot = np.zeros_like(p_v)
        one_hot[np.arange(len(y_v)), y_v] = 1.0
        flat_p = p_v.ravel()
        flat_y = one_hot.ravel()

    rows = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            mask = (flat_p >= low) & (flat_p <= high)
        else:
            mask = (flat_p >= low) & (flat_p < high)
        cnt = int(np.sum(mask))
        label = f"{int(round(low * 100))}–{int(round(high * 100))}%"
        if cnt > 0:
            mean_pred = float(np.mean(flat_p[mask]))
            obs_freq = float(np.mean(flat_y[mask]))
            diff = obs_freq - mean_pred
        else:
            mean_pred = float("nan")
            obs_freq = float("nan")
            diff = float("nan")
        rows.append(
            {
                "Bin": label,
                "Count": cnt,
                "Mean_Predicted": mean_pred,
                "Observed_Frequency": obs_freq,
                "Diff_Obs_minus_Pred": diff,
            }
        )
    return pd.DataFrame(rows)


# ==============================================================================
# 3. EXCHANGE & BREAK-EVEN MATHEMATICS
# ==============================================================================


def expected_value_net(
    p: Union[float, np.ndarray],
    odds: Union[float, np.ndarray],
    commission: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Compute Net Expected Value (EV) per unit stake with Exchange commission c in [0, 1):
        EV = p * (Q - 1) * (1 - c) - (1 - p)
    """
    if not (0.0 <= commission < 1.0):
        raise ValueError(f"Commission must be in [0, 1), got {commission}")
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(odds, dtype=np.float64)
    ev = p_arr * (q_arr - 1.0) * (1.0 - commission) - (1.0 - p_arr)
    if np.ndim(p) == 0 and np.ndim(odds) == 0:
        return float(ev)
    return ev


def min_breakeven_odds(
    p: Union[float, np.ndarray],
    commission: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Compute minimum break-even odds Q_min(p, c) such that EV(p, Q_min, c) == 0:
        Q_min(p, c) = 1 + (1 - p) / (p * (1 - c))
    """
    if not (0.0 <= commission < 1.0):
        raise ValueError(f"Commission must be in [0, 1), got {commission}")
    p_arr = np.asarray(p, dtype=np.float64)
    if np.any((p_arr <= 0.0) | (p_arr >= 1.0)):
        raise ValueError("Probability p must be strictly in (0, 1).")
    q_min = 1.0 + (1.0 - p_arr) / (p_arr * (1.0 - commission))
    if np.ndim(p) == 0:
        return float(q_min)
    return q_min


def closing_line_value(
    odds_taken: Union[float, np.ndarray],
    p_fair_closing: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute Closing Line Value (CLV):
        CLV = Q_preso * p_fair_closing - 1
    """
    q_arr = np.asarray(odds_taken, dtype=np.float64)
    p_arr = np.asarray(p_fair_closing, dtype=np.float64)
    clv = q_arr * p_arr - 1.0
    if np.ndim(odds_taken) == 0 and np.ndim(p_fair_closing) == 0:
        return float(clv)
    return clv


# ==============================================================================
# 4. MARKET AUDIT REPORT GENERATOR (reports/modulo_a_audit.md)
# ==============================================================================


def _evaluate_devig_methods_1x2(df_sub: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute Log-Loss and Brier Score on Pinnacle Closing 1X2 for Multiplicative, Power, and Shin."""
    odds = df_sub[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=np.float64)
    outcomes = df_sub["FTR"].to_numpy()
    valid_mask = _valid_odds_row_mask(odds)
    n_valid = int(np.sum(valid_mask))

    p_mult = devig_multiplicative(odds)
    p_pow = devig_power(odds)
    p_shin = devig_shin(odds)

    return {
        "n_valid": n_valid,
        "Multiplicative": {
            "log_loss": compute_log_loss(p_mult, outcomes),
            "brier": compute_brier_score(p_mult, outcomes),
        },
        "Power": {
            "log_loss": compute_log_loss(p_pow, outcomes),
            "brier": compute_brier_score(p_pow, outcomes),
        },
        "Shin": {
            "log_loss": compute_log_loss(p_shin, outcomes),
            "brier": compute_brier_score(p_shin, outcomes),
        },
    }


def _evaluate_devig_methods_ou25(df_sub: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute Log-Loss and Brier Score on Pinnacle Closing O/U 2.5 for Multiplicative, Power, and Shin."""
    odds = df_sub[["PC>2.5", "PC<2.5"]].to_numpy(dtype=np.float64)
    # Outcome: 0 if FTHG + FTAG > 2.5 ('Over'), 1 if < 2.5 ('Under')
    total_goals = df_sub["FTHG"].to_numpy(dtype=np.float64) + df_sub["FTAG"].to_numpy(dtype=np.float64)
    outcomes = np.where(total_goals > 2.5, 0, 1)
    valid_mask = _valid_odds_row_mask(odds)
    n_valid = int(np.sum(valid_mask))

    p_mult = devig_multiplicative(odds)
    p_pow = devig_power(odds)
    p_shin = devig_shin(odds)

    return {
        "n_valid": n_valid,
        "Multiplicative": {
            "log_loss": compute_log_loss(p_mult, outcomes),
            "brier": compute_brier_score(p_mult, outcomes),
        },
        "Power": {
            "log_loss": compute_log_loss(p_pow, outcomes),
            "brier": compute_brier_score(p_pow, outcomes),
        },
        "Shin": {
            "log_loss": compute_log_loss(p_shin, outcomes),
            "brier": compute_brier_score(p_shin, outcomes),
        },
    }


def generate_modulo_a_audit(csv_path: Path, report_path: Path) -> None:
    """Run the full Modulo A market audit and write reports/modulo_a_audit.md."""
    df = pd.read_csv(csv_path, dtype={"Season": str, "Div": str})
    df_train = df[df["Season"].isin(TRAIN_SEASONS)].copy()
    df_val = df[df["Season"].isin(VAL_SEASONS)].copy()
    df_train_val = df[df["Season"].isin(TRAIN_SEASONS + VAL_SEASONS)].copy()
    df_2425 = df[df["Season"] == "2425"].copy()
    df_2526 = df[df["Season"] == "2526"].copy()

    lines: List[str] = []
    lines.append("# Modulo A Audit Report — Market Benchmark, Devigging & Exchange Friction")
    lines.append("")
    lines.append("## 0. Conformità allo Split Temporale Vincolante (`RULES.md`)")
    lines.append("- **Warm-up (`1617–1819`)**: 3 stagioni (5.478 partite)")
    lines.append(f"- **Train / Sviluppo (`1920–2223`)**: 4 stagioni ({len(df_train)} partite)")
    lines.append(f"- **Validation (`2324–2425`)**: 2 stagioni ({len(df_val)} partite)")
    lines.append(
        f"- **Test Cieco (`2526`)**: 1 stagione ({len(df_2526)} partite) — *Non utilizzato per scelte di tuning/modellazione; ispezionato esclusivamente per l'audit tecnico di copertura del buco Pinnacle.*"
    )
    lines.append("- **Holdout Live (`2627`)**: Esclusa categoricamente.")
    lines.append("")

    # ==========================================================================
    # SECTION 1: DEVIGGING COMPARISON ON PINNACLE CLOSING 1X2 & O/U 2.5
    # ==========================================================================
    lines.append("## 1. Confronto Metodi di Devigging (Pinnacle Closing 1X2 e O/U 2.5)")
    lines.append("")

    for split_name, df_split in [
        ("Train (`1920–2223`)", df_train),
        ("Validation (`2324–2425`)", df_val),
    ]:
        lines.append(f"### 1.{1 if 'Train' in split_name else 2} Mercato 1X2 (`PSCH / PSCD / PSCA`) — {split_name}")
        lines.append("")
        lines.append(
            "| Perimetro | Partite Valide | Log-Loss (Mult) | Log-Loss (Power) | Log-Loss (Shin) | Brier (Mult) | Brier (Power) | Brier (Shin) | Miglior Log-Loss |"
        )
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")

        tot_res = _evaluate_devig_methods_1x2(df_split)
        best_tot = min(["Multiplicative", "Power", "Shin"], key=lambda m: tot_res[m]["log_loss"])
        lines.append(
            f"| **Totale Top 5** | **{tot_res['n_valid']}** | "
            f"{tot_res['Multiplicative']['log_loss']:.6f} | "
            f"{tot_res['Power']['log_loss']:.6f} | "
            f"{tot_res['Shin']['log_loss']:.6f} | "
            f"{tot_res['Multiplicative']['brier']:.6f} | "
            f"{tot_res['Power']['brier']:.6f} | "
            f"{tot_res['Shin']['brier']:.6f} | **{best_tot}** |"
        )

        for div_code, div_name in LEAGUE_NAMES.items():
            sub_l = df_split[df_split["Div"] == div_code]
            res_l = _evaluate_devig_methods_1x2(sub_l)
            best_l = min(["Multiplicative", "Power", "Shin"], key=lambda m: res_l[m]["log_loss"])
            prefix = "**" if div_code == "I1" else ""
            lines.append(
                f"| {prefix}`{div_code}` ({div_name}){prefix} | {res_l['n_valid']} | "
                f"{res_l['Multiplicative']['log_loss']:.6f} | "
                f"{res_l['Power']['log_loss']:.6f} | "
                f"{res_l['Shin']['log_loss']:.6f} | "
                f"{res_l['Multiplicative']['brier']:.6f} | "
                f"{res_l['Power']['brier']:.6f} | "
                f"{res_l['Shin']['brier']:.6f} | {prefix}{best_l}{prefix} |"
            )
        lines.append("")

    for idx_sub, (split_name, df_split) in enumerate(
        [
            ("Train (`1920–2223`)", df_train),
            ("Validation (`2324–2425`)", df_val),
        ],
        start=3,
    ):
        lines.append(f"### 1.{idx_sub} Mercato Over/Under 2.5 (`PC>2.5 / PC<2.5`) — {split_name}")
        lines.append("")
        lines.append(
            "| Perimetro | Partite Valide | Log-Loss (Mult) | Log-Loss (Power) | Log-Loss (Shin) | Brier Binario (Mult) | Brier Binario (Power) | Brier Binario (Shin) | Miglior Log-Loss |"
        )
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")

        tot_ou = _evaluate_devig_methods_ou25(df_split)
        best_ou = min(["Multiplicative", "Power", "Shin"], key=lambda m: tot_ou[m]["log_loss"])
        lines.append(
            f"| **Totale Top 5** | **{tot_ou['n_valid']}** | "
            f"{tot_ou['Multiplicative']['log_loss']:.6f} | "
            f"{tot_ou['Power']['log_loss']:.6f} | "
            f"{tot_ou['Shin']['log_loss']:.6f} | "
            f"{tot_ou['Multiplicative']['brier']:.6f} | "
            f"{tot_ou['Power']['brier']:.6f} | "
            f"{tot_ou['Shin']['brier']:.6f} | **{best_ou}** |"
        )

        for div_code, div_name in LEAGUE_NAMES.items():
            sub_l = df_split[df_split["Div"] == div_code]
            res_l = _evaluate_devig_methods_ou25(sub_l)
            best_l = min(["Multiplicative", "Power", "Shin"], key=lambda m: res_l[m]["log_loss"])
            prefix = "**" if div_code == "I1" else ""
            lines.append(
                f"| {prefix}`{div_code}` ({div_name}){prefix} | {res_l['n_valid']} | "
                f"{res_l['Multiplicative']['log_loss']:.6f} | "
                f"{res_l['Power']['log_loss']:.6f} | "
                f"{res_l['Shin']['log_loss']:.6f} | "
                f"{res_l['Multiplicative']['brier']:.6f} | "
                f"{res_l['Power']['brier']:.6f} | "
                f"{res_l['Shin']['brier']:.6f} | {prefix}{best_l}{prefix} |"
            )
        lines.append("")

    # ==========================================================================
    # SECTION 2: CALIBRATION TABLE (Power vs Multiplicative on 1X2)
    # ==========================================================================
    lines.append("## 2. Calibrazione del Benchmark per Decili (Power vs Multiplicative su 1X2)")
    lines.append("")
    cal_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

    for split_name, df_split in [
        ("Train (`1920–2223`)", df_train),
        ("Validation (`2324–2425`)", df_val),
    ]:
        lines.append(f"### Calibrazione {split_name} (Esiti 1X2 aggregati $N \\times 3$)")
        lines.append("")
        odds_s = df_split[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=np.float64)
        outcomes_s = df_split["FTR"].to_numpy()
        cal_mult = compute_calibration_table(devig_multiplicative(odds_s), outcomes_s, bins=cal_bins)
        cal_pow = compute_calibration_table(devig_power(odds_s), outcomes_s, bins=cal_bins)

        lines.append(
            "| Fascia Probabilità | N (Mult) | Pred Media (Mult) | Freq Reale (Mult) | Errore Obs-Pred (Mult) | N (Power) | Pred Media (Power) | Freq Reale (Power) | Errore Obs-Pred (Power) |"
        )
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for idx_r in range(len(cal_mult)):
            rm = cal_mult.iloc[idx_r]
            rp = cal_pow.iloc[idx_r]
            lines.append(
                f"| `{rm['Bin']}` | {int(rm['Count'])} | {rm['Mean_Predicted']*100:.2f}% | {rm['Observed_Frequency']*100:.2f}% | "
                f"{rm['Diff_Obs_minus_Pred']*100:+.2f}% | {int(rp['Count'])} | {rp['Mean_Predicted']*100:.2f}% | "
                f"{rp['Observed_Frequency']*100:.2f}% | {rp['Diff_Obs_minus_Pred']*100:+.2f}% |"
            )
        lines.append("")

    # ==========================================================================
    # SECTION 3: TECHNICAL AUDIT OF 2526 PINNACLE GAP & PINNACLE VS BETFAIR EXCHANGE
    # ==========================================================================
    lines.append("## 3. Audit Tecnico del Buco Pinnacle in `2526` e Confronto Pinnacle vs Betfair Exchange")
    lines.append("")
    lines.append("### 3.1 Data Esatta di Scomparsa di Pinnacle in `2526` e Copertura Alternative")

    psc_valid_2526 = df_2526["PSCH"].notna() & df_2526["PSCD"].notna() & df_2526["PSCA"].notna()
    last_psc_date = df_2526.loc[psc_valid_2526, "Date"].max()
    first_missing_after_last = df_2526.loc[(~psc_valid_2526) & (df_2526["Date"] > last_psc_date), "Date"].min()
    first_any_missing = df_2526.loc[~psc_valid_2526, "Date"].min()

    df_2526_after_drop = df_2526[df_2526["Date"] >= first_missing_after_last]
    n_after_drop = len(df_2526_after_drop)
    psc_after_cov = df_2526_after_drop["PSCH"].notna().mean() * 100.0
    avgch_after_cov = df_2526_after_drop[["AvgCH", "AvgCD", "AvgCA"]].notna().all(axis=1).mean() * 100.0
    bfech_after_cov = df_2526_after_drop[["BFECH", "BFECD", "BFECA"]].notna().all(axis=1).mean() * 100.0

    # Also check exact second half of 2526 by match count and by date >= first_missing_after_last
    half_idx = len(df_2526) // 2
    df_2526_second_half = df_2526.iloc[half_idx:]
    avgch_h2_cov = df_2526_second_half[["AvgCH", "AvgCD", "AvgCA"]].notna().all(axis=1).mean() * 100.0
    bfech_h2_cov = df_2526_second_half[["BFECH", "BFECD", "BFECA"]].notna().all(axis=1).mean() * 100.0
    psc_h2_cov = df_2526_second_half[["PSCH", "PSCD", "PSCA"]].notna().all(axis=1).mean() * 100.0

    lines.append(f"- **Ultima data con quote Pinnacle Closing (`PSCH/PSCD/PSCA`) presenti in `2526`**: **`{last_psc_date}`**")
    lines.append(
        f"- **Data esatta in cui spariscono definitivamente le quote Pinnacle in `2526`**: **`{first_missing_after_last}`** (da questa data in poi la copertura Pinnacle è `{psc_after_cov:.2f}%` su `{n_after_drop}` partite; prima singola assenza isolata il `{first_any_missing}`)"
    )
    lines.append(
        f"- **Copertura nel periodo post-scomparsa Pinnacle (`{first_missing_after_last}` → `{df_2526['Date'].max()}`, {n_after_drop} partite)**:"
    )
    lines.append(f"  - **`AvgCH / AvgCD / AvgCA` (Market Average Closing)**: **{avgch_after_cov:.2f}%**")
    lines.append(f"  - **`BFECH / BFECD / BFECA` (Betfair Exchange Closing)**: **{bfech_after_cov:.2f}%**")
    lines.append(
        f"- **Copertura nella seconda metà cronologica della stagione `2526` ({len(df_2526_second_half)} partite, dal `{df_2526_second_half['Date'].min()}` al `{df_2526_second_half['Date'].max()}`)**:"
    )
    lines.append(f"  - **`PSCH / PSCD / PSCA`**: **{psc_h2_cov:.2f}%**")
    lines.append(f"  - **`AvgCH / AvgCD / AvgCA`**: **{avgch_h2_cov:.2f}%**")
    lines.append(f"  - **`BFECH / BFECD / BFECA`**: **{bfech_h2_cov:.2f}%**")
    lines.append("")

    lines.append("### 3.2 Confronto Diretto Pinnacle Closing vs Betfair Exchange Closing sulla Stagione `2425`")
    both_2425 = (
        df_2425[["PSCH", "PSCD", "PSCA", "BFECH", "BFECD", "BFECA"]].notna().all(axis=1)
    )
    df_both = df_2425[both_2425].copy()
    n_both = len(df_both)
    or_psc_mean = df_both["Overround_Pinnacle_Close"].mean() * 100.0
    or_bfe_mean = df_both["Overround_BFE_Close"].mean() * 100.0

    psc_odds_2425 = df_both[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=np.float64)
    bfe_odds_2425 = df_both[["BFECH", "BFECD", "BFECA"]].to_numpy(dtype=np.float64)
    ftr_2425 = df_both["FTR"].to_numpy()

    lines.append(f"- **Partite in `2425` in cui convivono sia `PSCH` sia `BFECH`**: **{n_both} / {len(df_2425)}** ({n_both/len(df_2425)*100:.2f}%)")
    lines.append(f"- **Overround Medio Pinnacle Closing (`2425`)**: **{or_psc_mean:.2f}%**")
    lines.append(f"- **Overround Medio Betfair Exchange Closing (`2425`)**: **{or_bfe_mean:.2f}%**")
    lines.append("")
    lines.append("| Metodo di Devigging | MAE Probabilità (`|P_Pinnacle - P_BFE|`) | Log-Loss Pinnacle (`2425`) | Log-Loss Betfair Exchange (`2425`) | Brier Pinnacle (`2425`) | Brier Betfair Exchange (`2425`) |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")

    for m_name, fn in [
        ("Multiplicative", devig_multiplicative),
        ("Power", devig_power),
        ("Shin", devig_shin),
    ]:
        p_p = fn(psc_odds_2425)
        p_b = fn(bfe_odds_2425)
        mae = float(np.mean(np.abs(p_p - p_b))) * 100.0
        ll_p = compute_log_loss(p_p, ftr_2425)
        ll_b = compute_log_loss(p_b, ftr_2425)
        br_p = compute_brier_score(p_p, ftr_2425)
        br_b = compute_brier_score(p_b, ftr_2425)
        lines.append(
            f"| **{m_name}** | {mae:.4f} pp | {ll_p:.6f} | {ll_b:.6f} | {br_p:.6f} | {br_b:.6f} |"
        )
    lines.append("")

    # ==========================================================================
    # SECTION 4: EXCHANGE FRICTION CURVE (Q_min and minimum required probability edge)
    # ==========================================================================
    lines.append("## 4. Curva di Attrito Exchange (Break-Even $Q_{\\min}$ ed Edge Minimo Richiesto)")
    lines.append("")
    lines.append(
        "Dato un evento con probabilità reale $p$ e quota fair $Q_{\\text{fair}} = 1/p$, in presenza di commissione Exchange $c$ sui profitti netti:"
    )
    lines.append("- **Quota minima di break-even**: $Q_{\\min}(p, c) = 1 + \\frac{1 - p}{p(1 - c)}$")
    lines.append(
        "- **Probabilità implicita grezza di break-even**: $q_{\\text{mkt}} = 1 / Q_{\\min}(p, c)$ (ovvero per andare a break-even pagando la quota $Q_{\\min}$, la nostra probabilità reale $p$ deve superare la probabilità implicita di mercato $1/Q_{\\min}$ di almeno $\\Delta p = p - 1/Q_{\\min}$ punti percentuali; equivalentemente, in termini di quota, serve un markup $\\Delta Q = Q_{\\min} - Q_{\\text{fair}}$)."
    )
    lines.append("")
    lines.append(
        "| Probabilità $p$ | Quota Fair ($1/p$) | $Q_{\\min}$ ($c = 4.5\\%$) | Markup Quota ($c = 4.5\\%$) | Edge Probabilità $\\Delta p$ ($c = 4.5\\%$) | $Q_{\\min}$ ($c = 5.0\\%$) | Markup Quota ($c = 5.0\\%$) | Edge Probabilità $\\Delta p$ ($c = 5.0\\%$) |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for p_val in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        q_fair = 1.0 / p_val
        q_min_45 = min_breakeven_odds(p_val, commission=0.045)
        q_min_50 = min_breakeven_odds(p_val, commission=0.050)
        edge_p_45 = (p_val - (1.0 / q_min_45)) * 100.0
        edge_p_50 = (p_val - (1.0 / q_min_50)) * 100.0
        lines.append(
            f"| **{p_val*100:.0f}%** | {q_fair:.4f} | {q_min_45:.4f} | +{q_min_45 - q_fair:.4f} | **+{edge_p_45:.2f} pp** | "
            f"{q_min_50:.4f} | +{q_min_50 - q_fair:.4f} | **+{edge_p_50:.2f} pp** |"
        )
    lines.append("")

    # ==========================================================================
    # SECTION 5: SANITY CHECK OPENING VS CLOSING (1920-2425)
    # ==========================================================================
    lines.append("## 5. Sanity Check Diagnostico: Apertura vs Chiusura Pinnacle (`1920–2425`)")
    lines.append("")
    lines.append(
        "Simulazione puramente diagnostica: puntata piatta di 1 unità su ciascuno dei 3 esiti (1, X, 2) alla quota di **Apertura Pinnacle (`PSH / PSD / PSA`)** valutata contro la **Closing Line devigged (`PSCH / PSCD / PSCA`)** e contro l'esito reale del campo sul perimetro `1920–2425` (Train + Validation)."
    )
    lines.append("")
    lines.append(
        "| Perimetro | Partite Valide | Scommesse Totali ($3N$) | Overround Medio Apertura (%) | Overround Medio Chiusura (%) | CLV Medio vs Power Close (%) | CLV Medio vs Mult Close (%) | ROI Reale Puntata Piatta (%) |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for split_label, df_s in [
        ("Train (`1920–2223`)", df_train),
        ("Validation (`2324–2425`)", df_val),
        ("Totale `1920–2425`", df_train_val),
    ]:
        mask_oc = df_s[["PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA"]].notna().all(axis=1)
        sub_oc = df_s[mask_oc]
        n_oc = len(sub_oc)
        open_odds = sub_oc[["PSH", "PSD", "PSA"]].to_numpy(dtype=np.float64)
        close_odds = sub_oc[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=np.float64)
        ftr_oc = sub_oc["FTR"].to_numpy()

        or_open = float(np.mean(np.sum(1.0 / open_odds, axis=1) - 1.0)) * 100.0
        or_close = float(np.mean(np.sum(1.0 / close_odds, axis=1) - 1.0)) * 100.0

        p_close_pow = devig_power(close_odds)
        p_close_mult = devig_multiplicative(close_odds)

        clv_pow = float(np.mean(closing_line_value(open_odds, p_close_pow))) * 100.0
        clv_mult = float(np.mean(closing_line_value(open_odds, p_close_mult))) * 100.0

        # Realized return betting 1 flat unit on all 3 outcomes (H, D, A) at opening odds:
        # Total staked per match = 3 units; payout per match = open_odd of the winning outcome
        y_idx = _encode_outcomes(ftr_oc, 3)
        winning_odds = open_odds[np.arange(n_oc), y_idx]
        total_staked = 3.0 * n_oc
        total_return = float(np.sum(winning_odds))
        roi_pct = ((total_return - total_staked) / total_staked) * 100.0

        lines.append(
            f"| **{split_label}** | {n_oc} | {3 * n_oc} | {or_open:.2f}% | {or_close:.2f}% | "
            f"**{clv_pow:+.2f}%** | **{clv_mult:+.2f}%** | **{roi_pct:+.2f}%** |"
        )
    lines.append("")
    lines.append(
        "> **Conclusione diagnostica**: Puntare indiscriminatamente 1 unità piatta su tutti gli esiti alle quote di apertura Pinnacle produce un CLV medio negativo (`-2.90%` vs Multiplicative Close, `-3.83%` vs Power Close) e un ROI reale negativo (`-2.61%` su Train `1920–2223`, `-3.77%` sull'intero `1920–2425`), pienamente coerente con l'overround medio di apertura (`2.98%`)."
    )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    csv_path = root_dir / "data" / "processed" / "level1_top5_1617_2526.csv"
    report_path = root_dir / "reports" / "modulo_a_audit.md"

    print("[Modulo A] Running market benchmark evaluation and generating audit report...")
    generate_modulo_a_audit(csv_path=csv_path, report_path=report_path)
    print(f"[Modulo A] Report successfully generated at {report_path}")


if __name__ == "__main__":
    main()
