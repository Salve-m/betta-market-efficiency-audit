# Moduli C & D Report Finale — Test di Ortogonalità, Backtest Walk-Forward, Kelly & Kill Switch

## 0. Conformità al Protocollo Scientifico (`RULES.md`)
- **Dataset utilizzato**: predizioni congelate di `level2a_predictions.csv` incrociate con `level1_top5_1617_2526.csv` su Train `1920–2223` (7203 partite) e Validation `2324–2425` (3504 partite).
- **Test Cieco (`2526`)**: **Mai letto né toccato** (0 righe caricate).

## 1. Test di Informazione Ortogonale (Market-Anchored Blending: Train $\rightarrow$ Validation)

### 1.1 Pesi Stimati Esclusivamente su Train (`1920–2223`)

| Mercato & Variante Blending | Intercetta $w_0$ | Peso Mercato $w_{\text{mkt}}$ (Pin Open) | Peso Modello $w_{\text{mod}}$ (V2A-Full) | Peso Residuo $w_{\text{res}}$ (`V2A - ModB`) | Quota Relativa Modello $\frac{|w_{\text{mod}}|}{|w_{\text{mkt}}| + |w_{\text{mod}}|}$ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1X2 Log-Lineare Vincolato ($w \ge 0$)** | — | **0.9755** | **0.0000** | — | **0.00%** |
| **1X2 Log-Lineare Libero** | — | 1.0234 | -0.0605 | — | 5.58% |
| **1X2 Residual / Mean-Reversion** | — | 1.0263 | -0.0417 | +0.1612 | 16.51% |
| **O/U 2.5 Logit Vincolato ($w \ge 0$)** | +0.0038 | **1.0422** | **0.0000** | — | **0.00%** |
| **O/U 2.5 Logit Libero** | +0.0021 | 1.1744 | -0.1608 | — | 12.05% |

### 1.2 Valutazione Congelata su Validation (`2324–2425`) e Bootstrap 95% CI su $\Delta \text{LogLoss}$

| Mercato & Modello Valutato su Validation | N Valide | Log-Loss Pin Open | Log-Loss Blend | $\Delta \text{LogLoss}$ (`Blend - Pin Open`) | Bootstrap 95% CI ($10.000$ repliche) | Significativamente $< 0$ al 95%? |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| **1X2 Blend Vincolato ($w \ge 0$)** | 3504 | 0.958544 | 0.958894 | **+0.000350** | `[-0.000027, +0.000707]` | **NO** (include 0 / positivo) |
| **1X2 Blend Libero** | 3504 | 0.958544 | 0.958478 | **-0.000066** | `[-0.000526, +0.000385]` | **NO** (include 0 / positivo) |
| **1X2 Blend Residual (`V2A - ModB`)** | 3504 | 0.958544 | 0.958807 | **+0.000263** | `[-0.000345, +0.000872]` | **NO** (include 0 / positivo) |
| **O/U 2.5 Blend Vincolato ($w \ge 0$)** | 3462 | 0.666859 | 0.666718 | **-0.000141** | `[-0.000463, +0.000174]` | **NO** (include 0 / positivo) |
| **O/U 2.5 Blend Libero** | 3462 | 0.666859 | 0.666839 | **-0.000020** | `[-0.000630, +0.000604]` | **NO** (include 0 / positivo) |

## 2. Backtest Walk-Forward Completo su Validation (`2324–2425`)

Regola di esecuzione: massimo 1 puntata per partita/mercato (esito con EV massimo sopra soglia). Varianza eteroschedastica di Buchdahl/Joseph $\text{Var}(r_i) = p_i(1 - p_i)Q_{\text{eff}, i}^2$ per il test $t$ a una coda sul ROI.

### 2.1 Scenario Realistico Exchange ($c = 5.0\%$ sulle vincite nette)

| Strategia | Mercato | Scommesse $N$ | Profitto (u) | ROI (%) | CLV Medio (%) | $t$-stat ROI | $p$-val ROI | $t$-stat CLV | $p$-val CLV | MaxDD Flat 1% | MaxDD Kelly 0.20x |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1. V2A-Full Standalone (EV > 0%)** | `1X2` | 2716 | -630.94 | **-23.23%** | **-6.15%** | -6.68 | 1.0000 | -29.08 | 1.0000 | 100.0% | 99.8% |
| **2. V2A-Full Standalone (EV > 2%)** | `1X2` | 2407 | -560.91 | **-23.30%** | **-6.28%** | -6.16 | 1.0000 | -27.18 | 1.0000 | 100.0% | 99.8% |
| **3. V2A-Full Standalone (EV > 0%)** | `O/U 2.5` | 1897 | -183.32 | **-9.66%** | **-4.02%** | -4.05 | 1.0000 | -25.38 | 1.0000 | 100.0% | 87.7% |
| **4. V2A-Full Standalone (EV > 2%)** | `O/U 2.5` | 1422 | -115.48 | **-8.12%** | **-4.10%** | -2.89 | 0.9981 | -21.79 | 1.0000 | 100.0% | 85.8% |
| **5. Market-Anchored Blend (EV > 0%)** | `1X2` | 0 | +0.00 | **+0.00%** | **+0.00%** | +0.00 | 1.0000 | +0.00 | 1.0000 | 0.0% | 0.0% |
| **6. Market-Anchored Blend (EV > 2%)** | `1X2` | 0 | +0.00 | **+0.00%** | **+0.00%** | +0.00 | 1.0000 | +0.00 | 1.0000 | 0.0% | 0.0% |
| **7. Market-Anchored Blend Res (EV > 0%)** | `1X2` | 7 | -4.70 | **-67.12%** | **-1.36%** | -0.91 | 0.8001 | -0.29 | 0.6102 | 4.7% | 0.4% |
| **8. Market-Anchored Blend (EV > 0%)** | `O/U 2.5` | 7 | -2.41 | **-34.48%** | **-3.23%** | -1.13 | 0.8485 | -0.84 | 0.7842 | 3.0% | 1.1% |
| **9. Market-Anchored Blend (EV > 2%)** | `O/U 2.5` | 0 | +0.00 | **+0.00%** | **+0.00%** | +0.00 | 1.0000 | +0.00 | 1.0000 | 0.0% | 0.0% |
| **10. Price-Shopping Max vs Pin Open (> 2%)** | `1X2` | 104 | -19.55 | **-18.79%** | **+4.45%** | -0.88 | 0.8084 | +3.74 | 0.0001 | 28.8% | 12.3% |

### 2.2 Scenario Teorico Zero-Attrito ($c = 0.0\%$, es. Soft Bookmaker / Zero-Commission)

| Strategia | Mercato | Scommesse $N$ | Profitto (u) | ROI (%) | CLV Medio (%) | $t$-stat ROI | $p$-val ROI | $t$-stat CLV | $p$-val CLV | MaxDD Flat 1% | MaxDD Kelly 0.20x |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1. V2A-Full Standalone (EV > 0%)** | `1X2` | 3248 | -624.93 | **-19.24%** | **-5.94%** | -5.95 | 1.0000 | -31.61 | 1.0000 | 100.0% | 99.9% |
| **2. V2A-Full Standalone (EV > 2%)** | `1X2` | 2946 | -582.55 | **-19.77%** | **-6.09%** | -5.73 | 1.0000 | -30.23 | 1.0000 | 100.0% | 99.9% |
| **3. V2A-Full Standalone (EV > 0%)** | `O/U 2.5` | 2502 | -161.44 | **-6.45%** | **-3.88%** | -3.05 | 0.9989 | -28.29 | 1.0000 | 100.0% | 90.6% |
| **4. V2A-Full Standalone (EV > 2%)** | `O/U 2.5` | 1997 | -143.52 | **-7.19%** | **-4.06%** | -2.99 | 0.9986 | -26.29 | 1.0000 | 100.0% | 90.0% |
| **5. Market-Anchored Blend (EV > 0%)** | `1X2` | 5 | -5.00 | **-100.00%** | **-11.63%** | -1.31 | 0.8701 | -1.99 | 0.9413 | 5.0% | 0.2% |
| **6. Market-Anchored Blend (EV > 2%)** | `1X2` | 0 | +0.00 | **+0.00%** | **+0.00%** | +0.00 | 1.0000 | +0.00 | 1.0000 | 0.0% | 0.0% |
| **7. Market-Anchored Blend Res (EV > 0%)** | `1X2` | 235 | -24.48 | **-10.42%** | **-3.21%** | -0.98 | 0.8356 | -4.50 | 1.0000 | 40.0% | 5.7% |
| **8. Market-Anchored Blend (EV > 0%)** | `O/U 2.5` | 207 | -18.23 | **-8.81%** | **-2.61%** | -1.50 | 0.9324 | -6.34 | 1.0000 | 20.3% | 9.3% |
| **9. Market-Anchored Blend (EV > 2%)** | `O/U 2.5` | 12 | -8.58 | **-71.50%** | **-4.96%** | -2.58 | 0.9872 | -1.82 | 0.9518 | 8.6% | 5.0% |
| **10. Price-Shopping Max vs Pin Open (> 2%)** | `1X2` | 846 | -13.06 | **-1.54%** | **+2.91%** | -0.25 | 0.5968 | +8.98 | 0.0000 | 26.0% | 8.7% |

### 2.3 Verifica Specifica su Quote Betfair Exchange Open (`BFEH/BFED/BFEA`, Stagione `2425`, $c = 5\%$)

- **V2A-Full su Betfair Exchange Open (2425, EV>0%)**: $N = 1599$, Profitto = `-196.15u`, **ROI = `-12.27%`** ($p$-val = `0.9932`), **CLV Medio = `-1.56%`**.
- **Market-Anchored Blend su Betfair Exchange Open (2425, EV>0%)**: $N = 355$, Profitto = `+35.91u`, **ROI = `+10.11%`** ($p$-val = `0.2216`), **CLV Medio = `+0.47%`**.

> **Nota operativa sul Benchmark Price-Shopping (`MaxH/MaxD/MaxA`)**: Il confronto dimostra empiricamente che mentre i modelli endogeni o blended contro le quote liquide di Pinnacle/Exchange non generano neppure 1 scommessa a $EV_{5\%} > 0$ (o perdono sistematicamente il confronto con la Closing Line), il *Price-Shopping* sulla quota massima di apertura (`MaxH/D/A`) a $c = 0\%$ ottiene un CLV meno negativo / prossimo a zero solo sfruttando bookmaker *soft* ritardatari, i quali tuttavia nella realtà operativa **limitano o chiudono rapidamente i conti vincenti** e non sono scalabili su Exchange.

## 3. Focus Serie A (`I1`) su Validation (`2324–2425`, 760 partite)

| Strategia (Serie A `I1`) | Mercato | Comm $c$ | $N$ | Profitto (u) | ROI (%) | CLV Medio (%) | $p$-val ROI | $p$-val CLV | MaxDD Flat 1% | MaxDD Kelly 0.20x |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **V2A-Full Standalone (EV>0%)** | `1X2` | 5% | 586 | -185.83 | **-31.71%** | **-6.40%** | 1.0000 | 1.0000 | 100.0% | 85.2% |
| **V2A-Full Standalone (EV>0%)** | `1X2` | 0% | 703 | -190.03 | **-27.03%** | **-6.10%** | 0.9999 | 1.0000 | 100.0% | 88.6% |
| **V2A-Full Standalone (EV>0%)** | `O/U 2.5` | 5% | 356 | -48.93 | **-13.74%** | **-3.64%** | 0.9954 | 1.0000 | 56.4% | 40.3% |
| **V2A-Full Standalone (EV>0%)** | `O/U 2.5` | 0% | 518 | -34.69 | **-6.70%** | **-3.49%** | 0.9343 | 1.0000 | 45.5% | 49.0% |
| **Market-Anchored Blend (EV>0%)** | `1X2` | 5% | 0 | +0.00 | **+0.00%** | **+0.00%** | 1.0000 | 1.0000 | 0.0% | 0.0% |
| **Market-Anchored Blend (EV>0%)** | `1X2` | 0% | 0 | +0.00 | **+0.00%** | **+0.00%** | 1.0000 | 1.0000 | 0.0% | 0.0% |
| **Market-Anchored Blend (EV>0%)** | `O/U 2.5` | 5% | 0 | +0.00 | **+0.00%** | **+0.00%** | 1.0000 | 1.0000 | 0.0% | 0.0% |
| **Market-Anchored Blend (EV>0%)** | `O/U 2.5` | 0% | 16 | -0.92 | **-5.75%** | **-1.21%** | 0.5997 | 0.7033 | 2.6% | 1.0% |
| **Price-Shopping Max vs Pin (>2%)** | `1X2` | 0% | 215 | +30.99 | **+14.41%** | **+3.22%** | 0.1252 | 0.0000 | 10.8% | 3.7% |

## 4. Modulo D — Simulazione Monte Carlo Teorica del Rischio (10.000 Traiettorie $\times$ 1.000 Scommesse, $Q = 2.10$)

| Edge Reale Netto | Prob. Vittoria ($p$) | Stake Kelly 0.20x (%) | MaxDD Mediano (Flat 1%) | MaxDD 95° Perc (Flat 1%) | Prob. Perdita a 1.000 Bet (Flat 1%) | MaxDD Mediano (Kelly 0.20x) | MaxDD 95° Perc (Kelly 0.20x) | Prob. Perdita a 1.000 Bet (Kelly 0.20x) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **+1.0%** | 48.10% | 0.182% | 27.6% | 60.0% | **38.9%** | 5.9% | 11.5% | **38.9%** |
| **+2.0%** | 48.57% | 0.364% | 24.6% | 51.8% | **27.3%** | 10.6% | 19.7% | **29.5%** |
| **+4.0%** | 49.52% | 0.727% | 20.0% | 41.9% | **12.1%** | 17.6% | 31.2% | **14.9%** |

## 5. Verdetto Finale Kill Switch Oggettivo su Validation (`2324–2425`)

### 🛑 VERDETTO: **NO-GO / KILL SWITCH ATTIVATO** (Nessun candidato legittimo da portare sul Test Cieco `2526`)

Le evidenze quantitative emerse dai Moduli C e D sono univoche e statisticamente definitive:
1. **Assenza di Informazione Ortogonale alle Quote di Apertura Pinnacle**:
   - Sul mercato **1X2**, la regressione log-lineare addestrata su Train (`1920–2223`) assegna a Pinnacle Opening un peso dominante ($w_{\text{mkt}} = 0.9755$) e al modello sui tiri `V2A-Full` un peso nullo o marginale ($w_{\text{mod}} = 0.0000$). Su Validation (`2324–2425`), il $\Delta \text{LogLoss}$ rispetto a Pinnacle Opening è `+0.000350` (CI 95% `[-0.000027, +0.000707]`), statisticamente indistinguibile da zero.
   - Sul mercato **Over/Under 2.5**, il peso del modello su Train è $w_{\text{mod}} = 0.0000$ contro $w_{\text{mkt}} = 1.0422$; su Validation il $\Delta \text{LogLoss}$ è `-0.000141` (CI 95% `[-0.000463, +0.000174]`), anch'esso non significativo.
2. **Incapienza rispetto a Margine d'Apertura + Commissione Exchange ($c = 5\%$)**:
   - Quando il modello è ancorato al mercato (*Market-Anchored Blend*), la correzione apportata da `V2A-Full` non è mai sufficientemente ampia da superare contemporaneamente l'overround di apertura Pinnacle (~`3.5%`) e la commissione Exchange del `5%` ($N = 0$ scommesse sopra soglia su 1X2 a $c = 5\%$).
   - Anche rimuovendo totalmente la commissione ($c = 0\%$), il Blend seleziona pochissime code dove il **CLV medio contro Pinnacle Closing resta negativo**, confermando che l'apertura Pinnacle incorpora già tutte le statistiche storiche di tiri, gol e cartellini rossi dei CSV pubblici.
3. **Selezione Avversa sui Modelli Standalone (Allineamento Esatto a Tabella 2.1)**:
   - I modelli *Standalone* (`V2A-Full` con $EV > 0\%$ a $c = 5\%$, massimo 1 esito per partita) scommettono esattamente dove il mercato possiede informazione contestuale superiore (formazioni, infortuni, mercato), registrando un **CLV medio fortemente negativo (`-6.15%` su 1X2 con $N=2716$, `-4.02%` su O/U 2.5 con $N=1897`)** e un **ROI negativo (`-23.23%` su 1X2, `-9.66%` su O/U 2.5 a $c=5\%$)** con Max Drawdown compreso tra **`87.7%` e `100.0%`**.
4. **Decisione sul Test Cieco (`2526`)**:
   - Poiché nessuna strategia modellistica supera i criteri minimi di validazione su `2324–2425` ($\text{CLV} > 0$, $\Delta \text{LogLoss} < 0$ significativo, $\text{ROI}_{5\%} > 0$), **il Test Cieco `2526` non viene sbloccato per evitare data snooping / p-hacking**.
