"""
Portfolio Charts Generator (src/generate_portfolio_charts.py).

Generates two publication-grade static figures (150 dpi) in reports/figures/:
1. fig1_logloss_and_orthogonality.png:
   Side-by-side Log-Loss comparison on Validation (2324-2425) for 1X2 (left) and Over/Under 2.5 (right)
   across Pure Poisson, Dixon-Coles (Mod B), V2A-Full (Shots + Red-Card), Market-Anchored Blend (w_mod=0.0000),
   Pinnacle Opening, and Pinnacle Closing.
2. fig2_adverse_selection_and_montecarlo.png:
   Side-by-side diagnostic:
   - Left: Normalized Bankroll Equity Curve on Validation (2324-2425) for V2A-Full Standalone on 1X2 and O/U 2.5
     (c = 5%, flat 1% stake), illustrating adverse selection against the opening line.
   - Right: Theoretical Monte Carlo Max Drawdown distribution (10,000 trajectories x 1,000 bets at Q=2.10,
     Fractional Kelly 0.20x vs Flat 1%) for net edges +1%, +2%, +4%.

Strictly excludes Blind Test season 2526.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modulo_a_market import VAL_SEASONS, compute_log_loss, devig_power
from src.modulo_cd_backtest import (
    compute_equity_and_max_drawdown,
    fit_1x2_market_blend,
    fit_ou25_market_blend,
    load_evaluation_dataset,
    predict_1x2_market_blend,
    predict_ou25_market_blend,
    run_monte_carlo_risk_simulation,
    run_walk_forward_backtest,
)


def generate_fig1_logloss_and_orthogonality(
    df_all: pd.DataFrame,
    mod_b_csv_path: Path,
    out_path: Path,
) -> None:
    """Generate reports/figures/fig1_logloss_and_orthogonality.png (150 dpi)."""
    df_mod_b = pd.read_csv(mod_b_csv_path, dtype={"Season": str, "Div": str})
    df_mod_b_pois = df_mod_b[
        ["Season", "Div", "Date", "HomeTeam", "AwayTeam", "P_H_Pois", "P_D_Pois", "P_A_Pois", "P_Over25_Pois", "P_Under25_Pois"]
    ]
    df_merged = df_all.merge(
        df_mod_b_pois,
        on=["Season", "Div", "Date", "HomeTeam", "AwayTeam"],
        how="left",
    )

    blend_1x2_con = fit_1x2_market_blend(df_merged, constrained=True, use_residual=False)
    blend_ou_con = fit_ou25_market_blend(df_merged, constrained=True)

    df_val = df_merged[df_merged["Season"].isin(VAL_SEASONS)].copy()
    ftr_val = df_val["FTR"].to_numpy()

    # 1X2 probabilities on Validation
    p_pois_1x2 = df_val[["P_H_Pois", "P_D_Pois", "P_A_Pois"]].to_numpy(dtype=np.float64)
    p_dc_1x2 = df_val[["P_H_ModB", "P_D_ModB", "P_A_ModB"]].to_numpy(dtype=np.float64)
    p_v2a_1x2 = df_val[["P_H", "P_D", "P_A"]].to_numpy(dtype=np.float64)
    p_open_1x2 = devig_power(df_val[["PSH", "PSD", "PSA"]].to_numpy(dtype=np.float64))
    p_close_1x2 = devig_power(df_val[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=np.float64))
    p_blend_1x2 = predict_1x2_market_blend(blend_1x2_con, p_open_1x2, p_v2a_1x2)

    ll_1x2 = [
        compute_log_loss(p_pois_1x2, ftr_val),
        compute_log_loss(p_dc_1x2, ftr_val),
        compute_log_loss(p_v2a_1x2, ftr_val),
        compute_log_loss(p_blend_1x2, ftr_val),
        compute_log_loss(p_open_1x2, ftr_val),
        compute_log_loss(p_close_1x2, ftr_val),
    ]

    # O/U 2.5 probabilities on Validation (matched valid O/U closing subset N=3,466)
    valid_ou = df_val[["PC>2.5", "PC<2.5"]].notna().all(axis=1)
    sub_ou = df_val[valid_ou]
    ou_outcomes = np.where((sub_ou["FTHG"] + sub_ou["FTAG"]) > 2.5, 0, 1)

    p_pois_ou = sub_ou[["P_Over25_Pois", "P_Under25_Pois"]].to_numpy(dtype=np.float64)
    p_dc_ou = sub_ou[["P_Over25_ModB", "P_Under25_ModB"]].to_numpy(dtype=np.float64)
    p_v2a_ou = sub_ou[["P_Over25", "P_Under25"]].to_numpy(dtype=np.float64)
    p_open_ou = devig_power(sub_ou[["P>2.5", "P<2.5"]].to_numpy(dtype=np.float64))
    p_close_ou = devig_power(sub_ou[["PC>2.5", "PC<2.5"]].to_numpy(dtype=np.float64))
    p_blend_ou = predict_ou25_market_blend(blend_ou_con, p_open_ou, p_v2a_ou[:, 0])

    ll_ou = [
        compute_log_loss(p_pois_ou, ou_outcomes),
        compute_log_loss(p_dc_ou, ou_outcomes),
        compute_log_loss(p_v2a_ou, ou_outcomes),
        compute_log_loss(p_blend_ou, ou_outcomes),
        compute_log_loss(p_open_ou, ou_outcomes),
        compute_log_loss(p_close_ou, ou_outcomes),
    ]

    labels = [
        "Pure Poisson\n(Goals Only)",
        "Dixon-Coles\n(Modulo B)",
        "V2A-Full\n(Shots + Red Card)",
        "Market Blend\n(w_mod = 0.0000)",
        "Pinnacle Opening\n(Power Devig)",
        "Pinnacle Closing\n(Power Devig)",
    ]
    colors = ["#8c969e", "#5b7c99", "#2b5c8f", "#d9822b", "#2e8b57", "#1b5e20"]

    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.0), dpi=150)
    fig.patch.set_facecolor("white")

    x = np.arange(len(labels))
    bar_w = 0.58

    # Panel A: 1X2 Log-Loss
    bars1 = ax1.bar(x, ll_1x2, width=bar_w, color=colors, edgecolor="#222222", linewidth=0.7, zorder=3)
    ax1.set_title("A. 1X2 Multiclass Log-Loss — Validation (23/24–24/25, N = 3,504)", fontsize=10.8, fontweight="bold", pad=12)
    ax1.set_ylabel("Cross-Entropy Log-Loss (Lower is Better)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.8, rotation=18, ha="right")
    ax1.set_ylim(0.945, 0.990)
    ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax1.axhline(ll_1x2[-1], color="#1b5e20", linestyle=":", linewidth=1.2, zorder=2, label="Pinnacle Closing Benchmark")

    for b, val in zip(bars1, ll_1x2):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            val + 0.0007,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
        )

    # Panel B: O/U 2.5 Log-Loss
    bars2 = ax2.bar(x, ll_ou, width=bar_w, color=colors, edgecolor="#222222", linewidth=0.7, zorder=3)
    ax2.set_title("B. Over/Under 2.5 Binary Log-Loss — Validation (23/24–24/25, N = 3,466)", fontsize=10.8, fontweight="bold", pad=12)
    ax2.set_ylabel("Binary Log-Loss (Lower is Better)", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.8, rotation=18, ha="right")
    ax2.set_ylim(0.655, 0.683)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax2.axhline(ll_ou[-1], color="#1b5e20", linestyle=":", linewidth=1.2, zorder=2, label="Pinnacle Closing Benchmark")

    for b, val in zip(bars2, ll_ou):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            val + 0.0004,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
        )

    fig.suptitle(
        "Betta — Information Gap & Market Orthogonality Audit on Out-of-Sample Validation (Top 5 European Leagues)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def generate_fig2_adverse_selection_and_montecarlo(
    df_all: pd.DataFrame,
    out_path: Path,
) -> None:
    """Generate reports/figures/fig2_adverse_selection_and_montecarlo.png (150 dpi)."""
    df_val = df_all[df_all["Season"].isin(VAL_SEASONS)].copy()

    val_open_1x2 = df_val[["PSH", "PSD", "PSA"]].to_numpy(dtype=np.float64)
    val_close_1x2 = df_val[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=np.float64)
    val_p_v2a_1x2 = df_val[["P_H", "P_D", "P_A"]].to_numpy(dtype=np.float64)
    val_ftr = df_val["FTR"].to_numpy()

    val_open_ou = df_val[["P>2.5", "P<2.5"]].to_numpy(dtype=np.float64)
    val_close_ou = df_val[["PC>2.5", "PC<2.5"]].to_numpy(dtype=np.float64)
    val_p_v2a_ou = df_val[["P_Over25", "P_Under25"]].to_numpy(dtype=np.float64)
    val_ou_outcomes = np.where((df_val["FTHG"] + df_val["FTAG"]) > 2.5, 0, 1)

    res_1x2 = run_walk_forward_backtest(
        "V2A-Full 1X2 (EV>0%, c=5%)",
        "1X2",
        val_p_v2a_1x2,
        val_open_1x2,
        val_close_1x2,
        val_ftr,
        commission=0.05,
        ev_threshold=0.0,
    )
    res_ou = run_walk_forward_backtest(
        "V2A-Full O/U 2.5 (EV>0%, c=5%)",
        "O/U 2.5",
        val_p_v2a_ou,
        val_open_ou,
        val_close_ou,
        val_ou_outcomes,
        commission=0.05,
        ev_threshold=0.0,
    )

    _, _, eq_flat_1x2, _ = compute_equity_and_max_drawdown(
        won_mask=res_1x2.bets_df["won"].to_numpy(dtype=bool),
        exec_odds=res_1x2.bets_df["odds_taken"].to_numpy(dtype=np.float64),
        model_probs=res_1x2.bets_df["p_model"].to_numpy(dtype=np.float64),
        commission=0.05,
        flat_unit_pct=0.01,
    )
    _, _, eq_flat_ou, _ = compute_equity_and_max_drawdown(
        won_mask=res_ou.bets_df["won"].to_numpy(dtype=bool),
        exec_odds=res_ou.bets_df["odds_taken"].to_numpy(dtype=np.float64),
        model_probs=res_ou.bets_df["p_model"].to_numpy(dtype=np.float64),
        commission=0.05,
        flat_unit_pct=0.01,
    )

    mc_df = run_monte_carlo_risk_simulation()

    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.0), dpi=150)
    fig.patch.set_facecolor("white")

    # Panel A: Normalized Equity Curves showing Adverse Selection
    ax1.plot(
        np.arange(len(eq_flat_1x2)),
        eq_flat_1x2,
        color="#b22222",
        linewidth=1.8,
        label=f"V2A-Full 1X2 (N={res_1x2.n_bets}, ROI={res_1x2.roi_pct:+.1f}%, CLV={res_1x2.mean_clv_pct:+.2f}%)",
    )
    ax1.plot(
        np.arange(len(eq_flat_ou)),
        eq_flat_ou,
        color="#1f618d",
        linewidth=1.8,
        label=f"V2A-Full O/U 2.5 (N={res_ou.n_bets}, ROI={res_ou.roi_pct:+.1f}%, CLV={res_ou.mean_clv_pct:+.2f}%)",
    )
    ax1.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, label="Initial Bankroll (1.00)")
    ax1.axhline(0.0, color="#888888", linestyle=":", linewidth=0.9)
    ax1.set_title("A. Adverse Selection Equity Curve — Validation (c = 5%, Flat 1% Stake)", fontsize=11.5, fontweight="bold", pad=12)
    ax1.set_xlabel("Cumulative Placed Bets (Walk-Forward Chronological Order)", fontsize=10)
    ax1.set_ylabel("Normalized Bankroll (Initial = 1.00)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=8.8, frameon=True, facecolor="white", framealpha=0.9)

    # Panel B: Monte Carlo Max Drawdown Distribution (Median & 95th Percentile)
    x_mc = np.arange(len(mc_df))
    edge_labels = [f"Net Edge +{e:.0f}%\n(Loss Prob: {lp:.1f}%)" for e, lp in zip(mc_df["Net_Edge_pct"], mc_df["Flat_Loss_Prob_pct"])]
    w_b = 0.19

    b1 = ax2.bar(x_mc - 1.5 * w_b, mc_df["Flat_Median_MaxDD_pct"], width=w_b, color="#5b7c99", edgecolor="#222222", linewidth=0.6, label="Flat 1% — Median MaxDD")
    b2 = ax2.bar(x_mc - 0.5 * w_b, mc_df["Flat_P95_MaxDD_pct"], width=w_b, color="#2b5c8f", edgecolor="#222222", linewidth=0.6, label="Flat 1% — 95th Perc MaxDD")
    b3 = ax2.bar(x_mc + 0.5 * w_b, mc_df["Kelly_Median_MaxDD_pct"], width=w_b, color="#66bb6a", edgecolor="#222222", linewidth=0.6, label="Kelly 0.20x — Median MaxDD")
    b4 = ax2.bar(x_mc + 1.5 * w_b, mc_df["Kelly_P95_MaxDD_pct"], width=w_b, color="#2e7d32", edgecolor="#222222", linewidth=0.6, label="Kelly 0.20x — 95th Perc MaxDD")

    ax2.set_title("B. Monte Carlo Max Drawdown (10,000 Paths x 1,000 Bets, Odds = 2.10)", fontsize=11.5, fontweight="bold", pad=12)
    ax2.set_xticks(x_mc)
    ax2.set_xticklabels(edge_labels, fontsize=9.2)
    ax2.set_ylabel("Maximum Drawdown (%)", fontsize=10)
    ax2.set_ylim(0, 68)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="white", framealpha=0.9)

    for bar_group in [b1, b2, b3, b4]:
        for bar in bar_group:
            h_v = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                h_v + 0.9,
                f"{h_v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
                color="#222222",
            )

    fig.suptitle(
        "Betta — Empirical Adverse Selection vs Theoretical Monte Carlo Risk Benchmarks",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    level2a_preds_path = ROOT_DIR / "data" / "processed" / "level2a_predictions.csv"
    level1_csv_path = ROOT_DIR / "data" / "processed" / "level1_top5_1617_2526.csv"
    mod_b_csv_path = ROOT_DIR / "data" / "processed" / "modulo_b_predictions.csv"
    fig1_path = ROOT_DIR / "reports" / "figures" / "fig1_logloss_and_orthogonality.png"
    fig2_path = ROOT_DIR / "reports" / "figures" / "fig2_adverse_selection_and_montecarlo.png"

    print("[Charts] Loading evaluation dataset (excluding 2526)...", flush=True)
    df_all = load_evaluation_dataset(level2a_preds_path=level2a_preds_path, level1_csv_path=level1_csv_path)

    print(f"[Charts] Generating Figure 1 -> {fig1_path.relative_to(ROOT_DIR)}", flush=True)
    generate_fig1_logloss_and_orthogonality(df_all=df_all, mod_b_csv_path=mod_b_csv_path, out_path=fig1_path)

    print(f"[Charts] Generating Figure 2 -> {fig2_path.relative_to(ROOT_DIR)}", flush=True)
    generate_fig2_adverse_selection_and_montecarlo(df_all=df_all, out_path=fig2_path)
    print("[Charts] All portfolio figures generated successfully.", flush=True)


if __name__ == "__main__":
    main()
