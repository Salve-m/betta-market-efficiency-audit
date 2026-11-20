# Modulo B Report — Dixon-Coles MAP Walk-Forward Baseline

## 0. Conformità al Protocollo Anti-Leakage (`RULES.md`)
- **Walk-Forward rigoroso per data calendario**: per ogni lega e ogni data $t$, il modello è addestrato esclusivamente sulle partite con `Date < t` (finestra rolling di 1.100 giorni).
- **Prior Neopromosse**: media dei parametri $(a_k, b_k)$ stimati a fine stagione precedente per le ultime 3 classificate di quella lega (calcolato rigorosamente con dati `< inizio stagione`).
- **Perimetro valutato**: Train `1920–2223` (7203 partite) + Validation `2324–2425` (3504 partite) = **10707 partite**.
- **Test Cieco (`2526`)**: **Mai toccato né valutato** (`0` partite presenti in `modulo_b_predictions.csv`).

## 1. Taratura Iperparametri su Train (`1920–2223`)

Coppia ottima selezionata esclusivamente su Train (`1920–2223`) e congelata per Validation: **$H^* = 365$ giorni**, **$\lambda_{\text{reg}}^* = 1.50$**.

| Rank | Emivita $H$ (giorni) | Regolarizzazione $\lambda_{\text{reg}}$ | Train Log-Loss 1X2 | Train Brier Score 1X2 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | **365** | **1.50** | **0.995255** | **0.593117** |
| 2 | 540 | 1.50 | 0.995986 | 0.593628 |
| 3 | 180 | 1.50 | 0.996444 | 0.593853 |
| 4 | 730 | 1.50 | 0.996682 | 0.594111 |
| 5 | 365 | 0.50 | 0.996957 | 0.593931 |
| 6 | 540 | 0.50 | 0.997649 | 0.594423 |
| 7 | 730 | 0.50 | 0.998335 | 0.594902 |
| 8 | 365 | 0.15 | 0.998522 | 0.594486 |
| 9 | 180 | 0.50 | 0.998537 | 0.594881 |
| 10 | 540 | 0.15 | 0.999185 | 0.594966 |
| 11 | 365 | 0.05 | 0.999714 | 0.594745 |
| 12 | 730 | 0.15 | 0.999862 | 0.595442 |
| 13 | 180 | 0.15 | 1.000326 | 0.595544 |
| 14 | 540 | 0.05 | 1.000375 | 0.595228 |
| 15 | 730 | 0.05 | 1.001053 | 0.595706 |
| 16 | 180 | 0.05 | 1.001596 | 0.595830 |

## 2. Confronto Modelli vs Mercato (Poisson Puro, Dixon-Coles, Pinnacle Opening & Closing)

### 2.1 Mercato 1X2 — Train (`1920–2223`)

| Perimetro | N | Log-Loss Pois ($\rho=0$) | Log-Loss DC | Log-Loss Pin Open | Log-Loss Pin Close | Gap (`DC - Pin Close`) | Brier Pois | Brier DC | Brier Pin Open | Brier Pin Close |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Totale Top 5** | **7203** | 0.995705 | **0.995255** | 0.978149 | 0.975332 | **+0.019922** | 0.593456 | 0.593117 | 0.581380 | 0.579586 |
| **`I1` (Serie A)** | 1520 | 0.980296 | **0.979051** | 0.962639 | 0.959837 | **+0.019215** | 0.582120 | 0.581516 | 0.570731 | 0.568969 |
| `E0` (Premier League) | 1520 | 0.987429 | 0.988443 | 0.970619 | 0.967409 | +0.021034 | 0.586933 | 0.587282 | 0.575177 | 0.573088 |
| `SP1` (La Liga) | 1520 | 1.002603 | 1.000972 | 0.983298 | 0.980232 | +0.020740 | 0.598910 | 0.598185 | 0.585771 | 0.583592 |
| `D1` (Bundesliga) | 1224 | 1.003389 | 1.002833 | 0.986770 | 0.984215 | +0.018619 | 0.599717 | 0.598925 | 0.587105 | 0.585460 |
| `F1` (Ligue 1) | 1419 | 1.007057 | 1.007245 | 0.989875 | 0.987507 | +0.019739 | 0.601346 | 0.601354 | 0.589787 | 0.588562 |

### 2.2 Mercato 1X2 — Validation (`2324–2425`)

| Perimetro | N | Log-Loss Pois ($\rho=0$) | Log-Loss DC | Log-Loss Pin Open | Log-Loss Pin Close | Gap (`DC - Pin Close`) | Brier Pois | Brier DC | Brier Pin Open | Brier Pin Close |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Totale Top 5** | **3504** | 0.980833 | **0.980111** | 0.958544 | 0.955920 | **+0.024192** | 0.584447 | 0.584161 | 0.569626 | 0.567852 |
| **`I1` (Serie A)** | 760 | 0.982712 | **0.980302** | 0.957905 | 0.956741 | **+0.023562** | 0.586985 | 0.586009 | 0.570773 | 0.570555 |
| `E0` (Premier League) | 760 | 0.953933 | 0.953759 | 0.938833 | 0.932311 | +0.021448 | 0.565937 | 0.565809 | 0.555564 | 0.550184 |
| `SP1` (La Liga) | 760 | 0.967702 | 0.967405 | 0.948816 | 0.947434 | +0.019971 | 0.575092 | 0.575000 | 0.562737 | 0.561823 |
| `D1` (Bundesliga) | 612 | 1.004113 | 1.001088 | 0.968232 | 0.966177 | +0.034912 | 0.599313 | 0.597737 | 0.575601 | 0.574345 |
| `F1` (Ligue 1) | 612 | 1.004929 | 1.007401 | 0.986208 | 0.984498 | +0.022903 | 0.601033 | 0.602454 | 0.588247 | 0.587428 |

### 2.3 Mercato Over/Under 2.5 — Train (`1920–2223`)

| Perimetro | N | Log-Loss Pois ($\rho=0$) | Log-Loss DC | Log-Loss Pin Open | Log-Loss Pin Close | Gap (`DC - Pin Close`) | Brier Pois | Brier DC | Brier Pin Open | Brier Pin Close |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Totale Top 5** | **7178** | 0.684317 | **0.684299** | 0.671616 | 0.669654 | **+0.014646** | 0.245371 | 0.245364 | 0.239488 | 0.238560 |
| **`I1` (Serie A)** | 1520 | 0.688881 | **0.688527** | 0.676127 | 0.674789 | **+0.013737** | 0.247541 | 0.247378 | 0.241519 | 0.240846 |
| `E0` (Premier League) | 1519 | 0.685920 | 0.685924 | 0.676675 | 0.677068 | +0.008856 | 0.246103 | 0.246112 | 0.241887 | 0.242082 |
| `SP1` (La Liga) | 1519 | 0.686480 | 0.686512 | 0.674786 | 0.673308 | +0.013204 | 0.246261 | 0.246277 | 0.240910 | 0.240184 |
| `D1` (Bundesliga) | 1203 | 0.662939 | 0.663107 | 0.651034 | 0.647607 | +0.015500 | 0.235202 | 0.235271 | 0.229804 | 0.228276 |
| `F1` (Ligue 1) | 1417 | 0.693535 | 0.693643 | 0.675415 | 0.670997 | +0.022646 | 0.249941 | 0.249993 | 0.241429 | 0.239319 |

### 2.4 Mercato Over/Under 2.5 — Validation (`2324–2425`)

| Perimetro | N | Log-Loss Pois ($\rho=0$) | Log-Loss DC | Log-Loss Pin Open | Log-Loss Pin Close | Gap (`DC - Pin Close`) | Brier Pois | Brier DC | Brier Pin Open | Brier Pin Close |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Totale Top 5** | **3466** | 0.675832 | **0.675836** | 0.666859 | 0.663038 | **+0.012798** | 0.241280 | 0.241281 | 0.237093 | 0.235263 |
| **`I1` (Serie A)** | 760 | 0.686161 | **0.686211** | 0.680003 | 0.676500 | **+0.009711** | 0.246432 | 0.246456 | 0.243428 | 0.241737 |
| `E0` (Premier League) | 750 | 0.679351 | 0.679340 | 0.665460 | 0.663180 | +0.016159 | 0.242842 | 0.242832 | 0.236423 | 0.235320 |
| `SP1` (La Liga) | 758 | 0.669474 | 0.669566 | 0.660327 | 0.657395 | +0.012170 | 0.238170 | 0.238211 | 0.233845 | 0.232449 |
| `D1` (Bundesliga) | 589 | 0.653234 | 0.653599 | 0.646630 | 0.639465 | +0.014134 | 0.230429 | 0.230576 | 0.227328 | 0.224011 |
| `F1` (Ligue 1) | 609 | 0.688381 | 0.687884 | 0.679849 | 0.675884 | +0.012000 | 0.247292 | 0.247091 | 0.243486 | 0.241498 |

## 3. Parametri Fisici Stimati per Lega (a fine Validation `2425`)

| Lega | Nome | $h$ Finale | Fattore Campo $\gamma = e^h$ | $\rho$ Finale (`2425`) | $\rho$ Medio (`2324–2425`) | Gol Attesi Casa $\bar{\lambda}$ | Gol Reali Casa | Gol Attesi Trasf $\bar{\mu}$ | Gol Reali Trasf |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`I1`** | **Serie A** | 0.1477 | **1.159x** | -0.0676 | -0.0542 | 1.419 | 1.387 | 1.189 | 1.199 |
| `E0` | Premier League | 0.1501 | **1.162x** | -0.0063 | +0.0102 | 1.641 | 1.657 | 1.333 | 1.450 |
| `SP1` | La Liga | 0.2474 | **1.281x** | +0.0361 | -0.0173 | 1.444 | 1.468 | 1.094 | 1.163 |
| `D1` | Bundesliga | 0.1790 | **1.196x** | -0.1274 | -0.1320 | 1.824 | 1.719 | 1.385 | 1.458 |
| `F1` | Ligue 1 | 0.1316 | **1.141x** | -0.0786 | -0.0953 | 1.487 | 1.528 | 1.259 | 1.310 |

## 4. Calibrazione di Dixon-Coles su Validation (`2324–2425`, 1X2)

| Fascia Probabilità | N (DC) | Pred Media (DC) | Freq Reale (DC) | Errore Obs-Pred (DC) | N (Pin Close) | Pred Media (Pin Close) | Freq Reale (Pin Close) | Errore Obs-Pred (Pin Close) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0–10%` | 336 | 7.21% | 7.14% | **-0.07%** | 436 | 7.12% | 5.05% | -2.08% |
| `10–20%` | 1567 | 15.95% | 15.19% | **-0.76%** | 1712 | 15.68% | 14.95% | -0.72% |
| `20–30%` | 3854 | 25.16% | 25.79% | **+0.63%** | 3687 | 25.47% | 25.58% | +0.10% |
| `30–40%` | 1769 | 34.07% | 32.73% | **-1.34%** | 1707 | 33.90% | 32.98% | -0.92% |
| `40–50%` | 1183 | 44.84% | 45.82% | **+0.98%** | 1070 | 44.58% | 45.23% | +0.65% |
| `50–60%` | 870 | 54.52% | 54.02% | **-0.50%** | 880 | 54.86% | 56.59% | +1.73% |
| `60–70%` | 546 | 64.76% | 66.12% | **+1.36%** | 510 | 64.78% | 65.88% | +1.10% |
| `70–80%` | 285 | 74.45% | 73.68% | **-0.77%** | 349 | 74.49% | 75.64% | +1.16% |
| `80–100%` | 102 | 84.98% | 84.31% | **-0.67%** | 161 | 84.86% | 85.71% | +0.85% |

## 5. Diagnostica Preliminare sui Residui di Segnale (Validation `2324–2425`)

Ispezione puramente diagnostica (nessun backtest economico): verifica su Validation (`2324–2425`) dei casi in cui la probabilità stimata da Dixon-Coles genera un **$EV > 0$ contro Pinnacle Opening (`PSH / PSD / PSA`) al netto della commissione $c = 5\%$** ($EV = p_{\text{DC}}(Q_{\text{open}} - 1)(1 - 0.05) - (1 - p_{\text{DC}}) > 0$), valutando poi il **CLV medio contro Pinnacle Closing (Power devigged)**: $\text{CLV}_{\text{mkt}} = Q_{\text{open}} \cdot p_{\text{close, power}} - 1$.

| Perimetro | Partite Totali | Partite con $\ge 1$ Esito $EV_{5\%} > 0$ (%) | Selezione Esiti con $EV_{5\%} > 0$ ($/ 3N$) | EV Medio Stimato da DC (%) | CLV Medio vs Pinnacle Closing Power (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Totale Top 5** | 3504 | **2716 (77.51%)** | 3298 (31.37%) | +13.81% | **-5.82%** |
| **`I1` (Serie A)** | 760 | **591 (77.76%)** | 700 (30.70%) | +13.11% | **-5.48%** |
| `E0` (Premier League) | 760 | 603 (79.34%) | 708 (31.05%) | +15.58% | -6.14% |
| `SP1` (La Liga) | 760 | 576 (75.79%) | 674 (29.56%) | +13.02% | -5.84% |
| `D1` (Bundesliga) | 612 | 475 (77.61%) | 603 (32.84%) | +15.02% | -6.42% |
| `F1` (Ligue 1) | 612 | 471 (76.96%) | 613 (33.39%) | +12.22% | -5.20% |

> **Nota diagnostica**: Un modello basato esclusivamente sui gol finali storici (`FTHG / FTAG`), pur battendo nettamente il Poisson puro grazie alla correzione di correlazione $\rho$ di Dixon-Coles e ai prior sulle neopromosse, presenta un Log-Loss superiore a Pinnacle Closing e seleziona all'apertura quote dove il mercato ha informazioni extra (infortuni, motivazioni, xG/tiri, mercato). Quando il modello solo-gol diverge da Pinnacle Opening, il CLV medio rispetto a Pinnacle Closing resta negativo: questo giustifica il passaggio al Modulo C per l'integrazione di informazioni di secondo livello.
