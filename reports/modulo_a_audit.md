# Modulo A Audit Report — Market Benchmark, Devigging & Exchange Friction

## 0. Conformità allo Split Temporale Vincolante (`RULES.md`)
- **Warm-up (`1617–1819`)**: 3 stagioni (5.478 partite)
- **Train / Sviluppo (`1920–2223`)**: 4 stagioni (7203 partite)
- **Validation (`2324–2425`)**: 2 stagioni (3504 partite)
- **Test Cieco (`2526`)**: 1 stagione (1752 partite) — *Non utilizzato per scelte di tuning/modellazione; ispezionato esclusivamente per l'audit tecnico di copertura del buco Pinnacle.*
- **Holdout Live (`2627`)**: Esclusa categoricamente.

## 1. Confronto Metodi di Devigging (Pinnacle Closing 1X2 e O/U 2.5)

### 1.1 Mercato 1X2 (`PSCH / PSCD / PSCA`) — Train (`1920–2223`)

| Perimetro | Partite Valide | Log-Loss (Mult) | Log-Loss (Power) | Log-Loss (Shin) | Brier (Mult) | Brier (Power) | Brier (Shin) | Miglior Log-Loss |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Totale Top 5** | **7203** | 0.975140 | 0.975332 | 0.975258 | 0.579560 | 0.579586 | 0.579558 | **Multiplicative** |
| **`I1` (Serie A)** | 1520 | 0.959939 | 0.959837 | 0.959878 | 0.569085 | 0.568969 | 0.568985 | **Power** |
| `E0` (Premier League) | 1520 | 0.967022 | 0.967409 | 0.967318 | 0.573022 | 0.573088 | 0.573060 | Multiplicative |
| `SP1` (La Liga) | 1520 | 0.980189 | 0.980232 | 0.980139 | 0.583680 | 0.583592 | 0.583583 | Shin |
| `D1` (Bundesliga) | 1224 | 0.983357 | 0.984215 | 0.983913 | 0.585068 | 0.585460 | 0.585309 | Multiplicative |
| `F1` (Ligue 1) | 1419 | 0.987623 | 0.987507 | 0.987543 | 0.588621 | 0.588562 | 0.588572 | Power |

### 1.2 Mercato 1X2 (`PSCH / PSCD / PSCA`) — Validation (`2324–2425`)

| Perimetro | Partite Valide | Log-Loss (Mult) | Log-Loss (Power) | Log-Loss (Shin) | Brier (Mult) | Brier (Power) | Brier (Shin) | Miglior Log-Loss |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Totale Top 5** | **3504** | 0.956531 | 0.955920 | 0.955997 | 0.568149 | 0.567852 | 0.567899 | **Power** |
| **`I1` (Serie A)** | 760 | 0.957617 | 0.956741 | 0.956684 | 0.571005 | 0.570555 | 0.570594 | **Shin** |
| `E0` (Premier League) | 760 | 0.932981 | 0.932311 | 0.932484 | 0.550464 | 0.550184 | 0.550246 | Power |
| `SP1` (La Liga) | 760 | 0.948593 | 0.947434 | 0.947742 | 0.562549 | 0.561823 | 0.562010 | Power |
| `D1` (Bundesliga) | 612 | 0.966165 | 0.966177 | 0.966176 | 0.574194 | 0.574345 | 0.574290 | Multiplicative |
| `F1` (Ligue 1) | 612 | 0.984651 | 0.984498 | 0.984415 | 0.587474 | 0.587428 | 0.587398 | Shin |

### 1.3 Mercato Over/Under 2.5 (`PC>2.5 / PC<2.5`) — Train (`1920–2223`)

| Perimetro | Partite Valide | Log-Loss (Mult) | Log-Loss (Power) | Log-Loss (Shin) | Brier Binario (Mult) | Brier Binario (Power) | Brier Binario (Shin) | Miglior Log-Loss |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Totale Top 5** | **7178** | 0.669790 | 0.669654 | 0.669688 | 0.238612 | 0.238560 | 0.238572 | **Power** |
| **`I1` (Serie A)** | 1520 | 0.674703 | 0.674789 | 0.674742 | 0.240830 | 0.240846 | 0.240834 | **Multiplicative** |
| `E0` (Premier League) | 1519 | 0.677000 | 0.677068 | 0.677035 | 0.242053 | 0.242082 | 0.242068 | Multiplicative |
| `SP1` (La Liga) | 1519 | 0.673077 | 0.673308 | 0.673215 | 0.240087 | 0.240184 | 0.240145 | Multiplicative |
| `D1` (Bundesliga) | 1203 | 0.648512 | 0.647607 | 0.647900 | 0.228625 | 0.228276 | 0.228387 | Power |
| `F1` (Ligue 1) | 1417 | 0.671332 | 0.670997 | 0.671105 | 0.239443 | 0.239319 | 0.239358 | Power |

### 1.4 Mercato Over/Under 2.5 (`PC>2.5 / PC<2.5`) — Validation (`2324–2425`)

| Perimetro | Partite Valide | Log-Loss (Mult) | Log-Loss (Power) | Log-Loss (Shin) | Brier Binario (Mult) | Brier Binario (Power) | Brier Binario (Shin) | Miglior Log-Loss |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Totale Top 5** | **3466** | 0.663372 | 0.663038 | 0.663131 | 0.235417 | 0.235263 | 0.235306 | **Power** |
| **`I1` (Serie A)** | 760 | 0.676607 | 0.676500 | 0.676524 | 0.241796 | 0.241737 | 0.241752 | **Power** |
| `E0` (Premier League) | 750 | 0.663197 | 0.663180 | 0.663160 | 0.235330 | 0.235320 | 0.235312 | Shin |
| `SP1` (La Liga) | 758 | 0.657704 | 0.657395 | 0.657476 | 0.232603 | 0.232449 | 0.232490 | Power |
| `D1` (Bundesliga) | 589 | 0.640899 | 0.639465 | 0.639936 | 0.224630 | 0.224011 | 0.224212 | Power |
| `F1` (Ligue 1) | 609 | 0.675859 | 0.675884 | 0.675856 | 0.241499 | 0.241498 | 0.241491 | Shin |

## 2. Calibrazione del Benchmark per Decili (Power vs Multiplicative su 1X2)

### Calibrazione Train (`1920–2223`) (Esiti 1X2 aggregati $N \times 3$)

| Fascia Probabilità | N (Mult) | Pred Media (Mult) | Freq Reale (Mult) | Errore Obs-Pred (Mult) | N (Power) | Pred Media (Power) | Freq Reale (Power) | Errore Obs-Pred (Power) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0–10%` | 770 | 7.41% | 8.31% | +0.90% | 936 | 7.19% | 9.08% | +1.89% |
| `10–20%` | 3339 | 15.69% | 15.48% | -0.20% | 3366 | 15.69% | 15.60% | -0.09% |
| `20–30%` | 7781 | 25.74% | 25.28% | -0.46% | 7641 | 25.71% | 25.57% | -0.14% |
| `30–40%` | 3809 | 33.96% | 33.81% | -0.14% | 3695 | 33.92% | 33.83% | -0.09% |
| `40–50%` | 2248 | 44.72% | 45.37% | +0.65% | 2245 | 44.72% | 44.99% | +0.27% |
| `50–60%` | 1646 | 54.85% | 54.74% | -0.11% | 1612 | 54.92% | 54.65% | -0.27% |
| `60–70%` | 1098 | 64.69% | 68.21% | +3.52% | 1094 | 64.68% | 66.27% | +1.59% |
| `70–80%` | 668 | 74.36% | 73.80% | -0.56% | 693 | 74.40% | 73.88% | -0.51% |
| `80–100%` | 250 | 84.19% | 81.60% | -2.59% | 327 | 84.83% | 79.82% | -5.01% |

### Calibrazione Validation (`2324–2425`) (Esiti 1X2 aggregati $N \times 3$)

| Fascia Probabilità | N (Mult) | Pred Media (Mult) | Freq Reale (Mult) | Errore Obs-Pred (Mult) | N (Power) | Pred Media (Power) | Freq Reale (Power) | Errore Obs-Pred (Power) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0–10%` | 369 | 7.53% | 5.42% | -2.11% | 436 | 7.12% | 5.05% | -2.08% |
| `10–20%` | 1675 | 15.79% | 13.85% | -1.94% | 1712 | 15.68% | 14.95% | -0.72% |
| `20–30%` | 3751 | 25.51% | 25.51% | +0.00% | 3687 | 25.47% | 25.58% | +0.10% |
| `30–40%` | 1774 | 33.89% | 32.69% | -1.20% | 1707 | 33.90% | 32.98% | -0.92% |
| `40–50%` | 1091 | 44.63% | 46.47% | +1.84% | 1070 | 44.58% | 45.23% | +0.65% |
| `50–60%` | 883 | 54.78% | 57.08% | +2.30% | 880 | 54.86% | 56.59% | +1.73% |
| `60–70%` | 506 | 64.71% | 66.21% | +1.50% | 510 | 64.78% | 65.88% | +1.10% |
| `70–80%` | 337 | 74.14% | 77.15% | +3.01% | 349 | 74.49% | 75.64% | +1.16% |
| `80–100%` | 126 | 83.88% | 86.51% | +2.63% | 161 | 84.86% | 85.71% | +0.85% |

## 3. Audit Tecnico del Buco Pinnacle in `2526` e Confronto Pinnacle vs Betfair Exchange

### 3.1 Data Esatta di Scomparsa di Pinnacle in `2526` e Copertura Alternative
- **Ultima data con quote Pinnacle Closing (`PSCH/PSCD/PSCA`) presenti in `2526`**: **`2026-01-14`**
- **Data esatta in cui spariscono definitivamente le quote Pinnacle in `2526`**: **`2026-01-15`** (da questa data in poi la copertura Pinnacle è `0.00%` su `853` partite; prima singola assenza isolata il `2025-12-15`)
- **Copertura nel periodo post-scomparsa Pinnacle (`2026-01-15` → `2026-05-24`, 853 partite)**:
  - **`AvgCH / AvgCD / AvgCA` (Market Average Closing)**: **100.00%**
  - **`BFECH / BFECD / BFECA` (Betfair Exchange Closing)**: **86.99%**
- **Copertura nella seconda metà cronologica della stagione `2526` (876 partite, dal `2026-01-10` al `2026-05-24`)**:
  - **`PSCH / PSCD / PSCA`**: **2.63%**
  - **`AvgCH / AvgCD / AvgCA`**: **100.00%**
  - **`BFECH / BFECD / BFECA`**: **87.33%**

### 3.2 Confronto Diretto Pinnacle Closing vs Betfair Exchange Closing sulla Stagione `2425`
- **Partite in `2425` in cui convivono sia `PSCH` sia `BFECH`**: **1752 / 1752** (100.00%)
- **Overround Medio Pinnacle Closing (`2425`)**: **3.01%**
- **Overround Medio Betfair Exchange Closing (`2425`)**: **0.57%**

| Metodo di Devigging | MAE Probabilità (`|P_Pinnacle - P_BFE|`) | Log-Loss Pinnacle (`2425`) | Log-Loss Betfair Exchange (`2425`) | Brier Pinnacle (`2425`) | Brier Betfair Exchange (`2425`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Multiplicative** | 0.5361 pp | 0.960600 | 0.960280 | 0.570619 | 0.570624 |
| **Power** | 0.4986 pp | 0.960145 | 0.960202 | 0.570413 | 0.570596 |
| **Shin** | 0.4705 pp | 0.960242 | 0.960221 | 0.570452 | 0.570603 |

## 4. Curva di Attrito Exchange (Break-Even $Q_{\min}$ ed Edge Minimo Richiesto)

Dato un evento con probabilità reale $p$ e quota fair $Q_{\text{fair}} = 1/p$, in presenza di commissione Exchange $c$ sui profitti netti:
- **Quota minima di break-even**: $Q_{\min}(p, c) = 1 + \frac{1 - p}{p(1 - c)}$
- **Probabilità implicita grezza di break-even**: $q_{\text{mkt}} = 1 / Q_{\min}(p, c)$ (ovvero per andare a break-even pagando la quota $Q_{\min}$, la nostra probabilità reale $p$ deve superare la probabilità implicita di mercato $1/Q_{\min}$ di almeno $\Delta p = p - 1/Q_{\min}$ punti percentuali; equivalentemente, in termini di quota, serve un markup $\Delta Q = Q_{\min} - Q_{\text{fair}}$).

| Probabilità $p$ | Quota Fair ($1/p$) | $Q_{\min}$ ($c = 4.5\%$) | Markup Quota ($c = 4.5\%$) | Edge Probabilità $\Delta p$ ($c = 4.5\%$) | $Q_{\min}$ ($c = 5.0\%$) | Markup Quota ($c = 5.0\%$) | Edge Probabilità $\Delta p$ ($c = 5.0\%$) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **20%** | 5.0000 | 5.1885 | +0.1885 | **+0.73 pp** | 5.2105 | +0.2105 | **+0.81 pp** |
| **30%** | 3.3333 | 3.4433 | +0.1099 | **+0.96 pp** | 3.4561 | +0.1228 | **+1.07 pp** |
| **40%** | 2.5000 | 2.5707 | +0.0707 | **+1.10 pp** | 2.5789 | +0.0789 | **+1.22 pp** |
| **50%** | 2.0000 | 2.0471 | +0.0471 | **+1.15 pp** | 2.0526 | +0.0526 | **+1.28 pp** |
| **60%** | 1.6667 | 1.6981 | +0.0314 | **+1.11 pp** | 1.7018 | +0.0351 | **+1.24 pp** |
| **70%** | 1.4286 | 1.4488 | +0.0202 | **+0.98 pp** | 1.4511 | +0.0226 | **+1.09 pp** |
| **80%** | 1.2500 | 1.2618 | +0.0118 | **+0.75 pp** | 1.2632 | +0.0132 | **+0.83 pp** |

## 5. Sanity Check Diagnostico: Apertura vs Chiusura Pinnacle (`1920–2425`)

Simulazione puramente diagnostica: puntata piatta di 1 unità su ciascuno dei 3 esiti (1, X, 2) alla quota di **Apertura Pinnacle (`PSH / PSD / PSA`)** valutata contro la **Closing Line devigged (`PSCH / PSCD / PSCA`)** e contro l'esito reale del campo sul perimetro `1920–2425` (Train + Validation).

| Perimetro | Partite Valide | Scommesse Totali ($3N$) | Overround Medio Apertura (%) | Overround Medio Chiusura (%) | CLV Medio vs Power Close (%) | CLV Medio vs Mult Close (%) | ROI Reale Puntata Piatta (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Train (`1920–2223`)** | 7195 | 21585 | 2.72% | 2.59% | **-3.48%** | **-2.58%** | **-2.61%** |
| **Validation (`2324–2425`)** | 3504 | 10512 | 3.52% | 2.94% | **-4.55%** | **-3.54%** | **-6.16%** |
| **Totale `1920–2425`** | 10699 | 32097 | 2.98% | 2.70% | **-3.83%** | **-2.90%** | **-3.77%** |

> **Conclusione diagnostica**: Puntare indiscriminatamente 1 unità piatta su tutti gli esiti alle quote di apertura Pinnacle produce un CLV medio negativo (`-2.90%` vs Multiplicative Close, `-3.83%` vs Power Close) e un ROI reale negativo (`-2.61%` su Train `1920–2223`, `-3.77%` sull'intero `1920–2425`), pienamente coerente con l'overround medio di apertura (`2.98%`).
