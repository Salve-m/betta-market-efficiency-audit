# Level 2A Report — Quasi-Poisson MAP su Tiri (Synthetic xG) + Dixon-Coles 2-Stage

## 1. Audit Copertura e Integrità Tiri & Cartellini Rossi (`1617–2425`)

- **Partite totali ispezionate (`1617–2425`)**: **16185** (Stagione `2526` Test Cieco esclusa e inviolata).
- **Partite con statistiche tiri mancanti (`NaN`, gestite con fallback automatico sui gol reali `FTHG/FTAG`)**: **2**
  - `2017-04-16` (`1617` `F1`) `Bastia` vs `Lyon` (FTHG=0, FTAG=3)
  - `2024-12-14` (`2425` `D1`) `Union Berlin` vs `Bochum` (FTHG=0, FTAG=2)
- **Valori negativi su `HS/AS/HST/AST/HR/AR`**: **0**
- **Anomalie `HST > HS` (tiri in porta > tiri totali casa)**: **0**
- **Anomalie `AST > AS` (tiri in porta > tiri totali trasferta, sanificate con fallback su gol reali)**: **1**
  - `2021-08-15` (`2122` `E0`) `Newcastle` vs `West Ham` (AS=8.0, AST=9.0)

### 1.1 Copertura Percentuale Non-Null per Lega (`1617–2425`)

| Lega | Nome | Partite | `HS / AS` (%) | `HST / AST` (%) | `HR / AR` (%) | Tutte le Stat Tiri Valide (%) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **`I1`** | **Serie A** | 3420 | 100.00% | 100.00% | 100.00% | **100.00%** |
| `E0` | Premier League | 3420 | 100.00% | 100.00% | 100.00% | 100.00% |
| `SP1` | La Liga | 3420 | 100.00% | 100.00% | 100.00% | 100.00% |
| `D1` | Bundesliga | 2754 | 99.96% | 99.96% | 99.96% | 99.96% |
| `F1` | Ligue 1 | 3171 | 99.97% | 99.97% | 99.97% | 99.97% |

### 1.2 Copertura Percentuale Non-Null per Stagione (`1617–2425`)

| Stagione | Partite | `HS / AS` (%) | `HST / AST` (%) | `HR / AR` (%) | Tutte le Stat Tiri Valide (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `1617` | 1826 | 99.95% | 99.95% | 99.95% | 99.95% |
| `1718` | 1826 | 100.00% | 100.00% | 100.00% | 100.00% |
| `1819` | 1826 | 100.00% | 100.00% | 100.00% | 100.00% |
| `1920` | 1725 | 100.00% | 100.00% | 100.00% | 100.00% |
| `2021` | 1826 | 100.00% | 100.00% | 100.00% | 100.00% |
| `2122` | 1826 | 100.00% | 100.00% | 100.00% | 100.00% |
| `2223` | 1826 | 100.00% | 100.00% | 100.00% | 100.00% |
| `2324` | 1752 | 100.00% | 100.00% | 100.00% | 100.00% |
| `2425` | 1752 | 99.94% | 99.94% | 99.94% | 99.94% |

## 2. Grid Search Parallela su Train (`1920–2223`) e Iperparametri Congelati

- **Combinazioni valutate su Train (`1920–2223`, refit settimanale anti-leakage)**: **96** (3 emivite $H \times 4$ regolarizzazioni $\lambda_{\text{reg}} \times 4$ mix bersaglio $\times 2$ sconti rossi $d_{\text{rc}}$)
- **Tempo di esecuzione totale Grid Search parallela**: **544.1 secondi**
- **Configurazione Ottima Congelata (`V2A-Full`)**: **$H^* = 270$ giorni**, **$\lambda_{\text{reg}}^* = 1.50$**, **$(w_g, w_{\text{sot}}, w_{\text{tot}})^* = (0.50, 0.35, 0.15)$**, **$d_{\text{rc}}^* = 0.50$** (Train Log-Loss 1X2 weekly = `0.991536`)
- **Configurazione Controllo Ablativo Solo-Gol (`V2A-Ablation`)**: **$H = 270$ giorni**, **$\lambda_{\text{reg}} = 6.00$**, **$(w_g, w_{\text{sot}}, w_{\text{tot}}) = (1.00, 0.00, 0.00)$**, **$d_{\text{rc}} = 1.00$** (Train Log-Loss 1X2 weekly = `0.993475`)
- **✅ Verifica Bordo Regolarizzazione**: sul controllo ablativo solo-gol (`V2A-Ablation`) lo sblocco della griglia sposta effettivamente l'ottimo da `1.50` a **$\lambda_{\text{reg}} = 6.00$** (ottimo interno a `[1.5, 3.0, 6.0, 12.0]`, a conferma che i soli gol interi richiedono forte shrinkage), mentre in **`V2A-Full`** il bersaglio composito sui tiri riduce intrinsecamente la varianza del target rendendo ottimale **$\lambda_{\text{reg}}^* = 1.50$** (seguito a `0.000086` di distanza da $\lambda_{\text{reg}} = 3.00$, senza mai toccare il bordo superiore `12.0`).

### Top 10 Configurazioni su Train (`1920–2223`)

| Rank | Emivita $H$ | $\lambda_{\text{reg}}$ | Mix $(w_g, w_{\text{sot}}, w_{\text{tot}})$ | Sconto Rossi $d_{\text{rc}}$ | Train Log-Loss 1X2 | Train Brier 1X2 |
| ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| 1 | **270d** | **1.50** | **`(0.50, 0.35, 0.15)`** | **0.50** | **0.991536** | **0.590910** |
| 2 | 270d | 3.00 | `(0.50, 0.35, 0.15)` | 0.50 | 0.991622 | 0.591004 |
| 3 | 270d | 1.50 | `(0.50, 0.35, 0.15)` | 1.00 | 0.991868 | 0.591106 |
| 4 | 270d | 3.00 | `(0.50, 0.35, 0.15)` | 1.00 | 0.991903 | 0.591168 |
| 5 | 365d | 3.00 | `(0.50, 0.35, 0.15)` | 0.50 | 0.992190 | 0.591413 |
| 6 | 365d | 1.50 | `(0.50, 0.35, 0.15)` | 0.50 | 0.992232 | 0.591408 |
| 7 | 365d | 3.00 | `(0.50, 0.35, 0.15)` | 1.00 | 0.992486 | 0.591582 |
| 8 | 365d | 1.50 | `(0.50, 0.35, 0.15)` | 1.00 | 0.992557 | 0.591596 |
| 9 | 270d | 6.00 | `(0.50, 0.35, 0.15)` | 0.50 | 0.992676 | 0.591768 |
| 10 | 270d | 6.00 | `(0.50, 0.35, 0.15)` | 1.00 | 0.992795 | 0.591827 |

## 3. Tabella Ablazione & Confronto Diretto (Refit Giornaliero Rigoroso `Date < t`)

Confronto a parità esatta di protocollo walk-forward giornaliero (`Date < t`) tra:
- **Modulo B**: Baseline Dixon-Coles solo-gol ($\lambda_{\text{reg}} = 1.50$, $H = 365$)
- **V2A-Ablation**: Solo-gol $(1,0,0)$ con $\lambda_{\text{reg}} = 6.0$ sbloccato ($H = 270$)
- **V2A-Full**: Synthetic xG Tiri `(0.50, 0.35, 0.15)` + Sconto Rossi $d_{\text{rc}} = 0.50$ + $\lambda_{\text{reg}}^* = 1.5$ ($H^* = 270$)
- **Pinnacle Opening (Power)** e **Pinnacle Closing (Power)**

### 3.1 Mercato 1X2 — Validation (`2324–2425`) — Out-of-Sample Congelato

| Perimetro | N | Log-Loss Mod B | Log-Loss V2A-Abl | Log-Loss V2A-Full | Log-Loss Pin Open | Log-Loss Pin Close | Guadagno (`V2A-Full - Mod B`) | Gap (`V2A-Full - Pin Close`) | Brier Mod B | Brier V2A-Full | Brier Pin Close |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Totale Top 5** | **3504** | 0.980111 | 0.980160 | **0.980532** | 0.958544 | 0.955920 | **+0.000421** | **+0.024612** | 0.584161 | **0.583905** | 0.567852 |
| **`I1` (Serie A)** | 760 | 0.980302 | 0.981654 | **0.984400** | 0.957905 | 0.956741 | **+0.004098** | **+0.027660** | 0.586009 | **0.588054** | 0.570555 |
| `E0` (Premier League) | 760 | 0.953759 | 0.953022 | 0.951457 | 0.938833 | 0.932311 | -0.002302 | +0.019145 | 0.565809 | 0.563521 | 0.550184 |
| `SP1` (La Liga) | 760 | 0.967405 | 0.968256 | 0.973469 | 0.948816 | 0.947434 | +0.006064 | +0.026035 | 0.575000 | 0.578139 | 0.561823 |
| `D1` (Bundesliga) | 612 | 1.001088 | 1.000136 | 0.995240 | 0.968232 | 0.966177 | -0.005849 | +0.029063 | 0.597737 | 0.593618 | 0.574345 |
| `F1` (Ligue 1) | 612 | 1.007401 | 1.006810 | 1.005898 | 0.986208 | 0.984498 | -0.001503 | +0.021400 | 0.602454 | 0.601513 | 0.587428 |

### 3.2 Mercato 1X2 — Train (`1920–2223`) — Walk-Forward Giornaliero

| Perimetro | N | Log-Loss Mod B | Log-Loss V2A-Abl | Log-Loss V2A-Full | Log-Loss Pin Open | Log-Loss Pin Close | Guadagno (`V2A-Full - Mod B`) | Gap (`V2A-Full - Pin Close`) | Brier Mod B | Brier V2A-Full | Brier Pin Close |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Totale Top 5** | **7203** | 0.995255 | 0.993426 | **0.991566** | 0.978149 | 0.975332 | **-0.003688** | **+0.016234** | 0.593117 | **0.590909** | 0.579586 |
| **`I1` (Serie A)** | 1520 | 0.979051 | 0.978117 | **0.977440** | 0.962639 | 0.959837 | **-0.001612** | **+0.017603** | 0.581516 | **0.580708** | 0.568969 |
| `E0` (Premier League) | 1520 | 0.988443 | 0.985673 | 0.982422 | 0.970619 | 0.967409 | -0.006021 | +0.015012 | 0.587282 | 0.583583 | 0.573088 |
| `SP1` (La Liga) | 1520 | 1.000972 | 0.999323 | 0.997979 | 0.983298 | 0.980232 | -0.002993 | +0.017747 | 0.598185 | 0.596081 | 0.583592 |
| `D1` (Bundesliga) | 1224 | 1.002833 | 1.000088 | 0.995715 | 0.986770 | 0.984215 | -0.007118 | +0.011501 | 0.598925 | 0.594680 | 0.585460 |
| `F1` (Ligue 1) | 1419 | 1.007245 | 1.006067 | 1.006047 | 0.989875 | 0.987507 | -0.001199 | +0.018540 | 0.601354 | 0.600890 | 0.588562 |

### 3.3 Mercato Over/Under 2.5 — Validation (`2324–2425`) — Out-of-Sample Congelato

| Perimetro | N | Log-Loss Mod B | Log-Loss V2A-Abl | Log-Loss V2A-Full | Log-Loss Pin Open | Log-Loss Pin Close | Guadagno (`V2A-Full - Mod B`) | Gap (`V2A-Full - Pin Close`) | Brier Mod B | Brier V2A-Full | Brier Pin Close |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Totale Top 5** | **3466** | 0.675836 | 0.675318 | **0.672967** | 0.666859 | 0.663038 | **-0.002868** | **+0.009930** | 0.241281 | **0.240045** | 0.235263 |
| **`I1` (Serie A)** | 760 | 0.686211 | 0.686218 | **0.685683** | 0.680003 | 0.676500 | **-0.000528** | **+0.009183** | 0.246456 | **0.246291** | 0.241737 |
| `E0` (Premier League) | 750 | 0.679340 | 0.677070 | 0.673080 | 0.665460 | 0.663180 | -0.006259 | +0.009900 | 0.242832 | 0.239996 | 0.235320 |
| `SP1` (La Liga) | 758 | 0.669566 | 0.669456 | 0.666931 | 0.660327 | 0.657395 | -0.002634 | +0.009536 | 0.238211 | 0.237046 | 0.232449 |
| `D1` (Bundesliga) | 589 | 0.653599 | 0.653433 | 0.651664 | 0.646630 | 0.639465 | -0.001935 | +0.012199 | 0.230576 | 0.229832 | 0.224011 |
| `F1` (Ligue 1) | 609 | 0.687884 | 0.688022 | 0.685076 | 0.679849 | 0.675884 | -0.002808 | +0.009192 | 0.247091 | 0.245922 | 0.241498 |

### 3.4 Mercato Over/Under 2.5 — Train (`1920–2223`) — Walk-Forward Giornaliero

| Perimetro | N | Log-Loss Mod B | Log-Loss V2A-Abl | Log-Loss V2A-Full | Log-Loss Pin Open | Log-Loss Pin Close | Guadagno (`V2A-Full - Mod B`) | Gap (`V2A-Full - Pin Close`) | Brier Mod B | Brier V2A-Full | Brier Pin Close |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Totale Top 5** | **7178** | 0.684299 | 0.682308 | **0.680125** | 0.671616 | 0.669654 | **-0.004174** | **+0.010472** | 0.245364 | **0.243538** | 0.238560 |
| **`I1` (Serie A)** | 1520 | 0.688527 | 0.686783 | **0.685533** | 0.676127 | 0.674789 | **-0.002993** | **+0.010744** | 0.247378 | **0.246140** | 0.240846 |
| `E0` (Premier League) | 1519 | 0.685924 | 0.684547 | 0.682182 | 0.676675 | 0.677068 | -0.003742 | +0.005114 | 0.246112 | 0.244515 | 0.242082 |
| `SP1` (La Liga) | 1519 | 0.686512 | 0.683177 | 0.680741 | 0.674786 | 0.673308 | -0.005771 | +0.007432 | 0.246277 | 0.243746 | 0.240184 |
| `D1` (Bundesliga) | 1203 | 0.663107 | 0.662827 | 0.660064 | 0.651034 | 0.647607 | -0.003043 | +0.012457 | 0.235271 | 0.233904 | 0.228276 |
| `F1` (Ligue 1) | 1417 | 0.693643 | 0.690715 | 0.688493 | 0.675415 | 0.670997 | -0.005150 | +0.017496 | 0.249993 | 0.247656 | 0.239319 |

## 4. Diagnostica Selezione Avversa su Validation (`2324–2425`) — Modulo B vs V2A-Full

Confronto diretto della selezione di esiti con $EV_{5\%} > 0$ contro **Pinnacle Opening (`PSH / PSD / PSA`)** e relativo **CLV medio contro Pinnacle Closing (Power devigged)** su Validation (`2324–2425`).

### 4.1 Confronto Fianco a Fianco per Lega ($EV_{5\%} > 0\%$)

| Perimetro | Partite | Partite $EV_{5\%}>0$ Mod B (%) | Partite $EV_{5\%}>0$ V2A-Full (%) | Esiti Selezionati Mod B | Esiti Selezionati V2A-Full | CLV Medio Mod B vs Pin Close (%) | CLV Medio V2A-Full vs Pin Close (%) | Miglioramento CLV (`V2A - Mod B`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Totale Top 5** | 3504 | 2716 (77.5%) | **2716 (77.5%)** | 3298 | 3518 | -5.82% | **-6.55%** | **-0.73 pp** |
| **`I1` (Serie A)** | 760 | 591 (77.8%) | **586 (77.1%)** | 700 | 751 | -5.48% | **-6.19%** | **-0.71 pp** |
| `E0` (Premier League) | 760 | 603 (79.3%) | 599 (78.8%) | 708 | 727 | -6.14% | -6.80% | -0.65 pp |
| `SP1` (La Liga) | 760 | 576 (75.8%) | 606 (79.7%) | 674 | 745 | -5.84% | -7.05% | -1.20 pp |
| `D1` (Bundesliga) | 612 | 475 (77.6%) | 442 (72.2%) | 603 | 599 | -6.42% | -7.13% | -0.71 pp |
| `F1` (Ligue 1) | 612 | 471 (77.0%) | 483 (78.9%) | 613 | 696 | -5.20% | -5.64% | -0.45 pp |

### 4.2 Analisi Soglie di Filtro Edge e Filtro Disaccordo Massimo con il Mercato (Validation `2324–2425`)

Effetto di soglie più selettive su $EV_{5\%}$ e del filtro anti-selezione-avversa sul disaccordo massimo con l'apertura devigged ($|p_{\text{mod}} - p_{\text{open, power}}| \le \Delta_{\max}$):

| Regola di Filtro Diagnostica | Esiti Selezionati Mod B | CLV Medio Mod B (%) | Esiti Selezionati V2A-Full | EV Medio Stimato V2A-Full (%) | CLV Medio V2A-Full vs Pin Close (%) | Miglioramento CLV (`V2A - Mod B`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. $EV_{5\%} > 0\%$ (nessun cap disaccordo) | 3298 | -5.82% | **3518** | +15.00% | **-6.55%** | **-0.73 pp** |
| 2. $EV_{5\%} > 3\%$ (nessun cap disaccordo) | 2657 | -6.09% | **2890** | +17.95% | **-6.84%** | **-0.75 pp** |
| 3. $EV_{5\%} > 5\%$ (nessun cap disaccordo) | 2296 | -6.25% | **2535** | +19.90% | **-6.99%** | **-0.74 pp** |
| 4. $EV_{5\%} > 0\%$ & $|p_{\text{mod}} - p_{\text{open}}| \le 10\text{ pp}$ | 2925 | -5.99% | **3322** | +13.30% | **-6.66%** | **-0.67 pp** |
| 5. $EV_{5\%} > 0\%$ & $|p_{\text{mod}} - p_{\text{open}}| \le 6\text{ pp}$ | 2111 | -6.28% | **2491** | +9.01% | **-6.91%** | **-0.63 pp** |
| 6. $EV_{5\%} > 3\%$ & $|p_{\text{mod}} - p_{\text{open}}| \le 10\text{ pp}$ | 2284 | -6.36% | **2694** | +16.07% | **-6.99%** | **-0.64 pp** |
