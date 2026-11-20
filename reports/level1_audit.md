# Level 1 Audit Report — Data Ingestion & Integrity (Top 5 Leagues, 2016/17 – 2025/26)

## 1. Riepilogo Generale & Download
- **File CSV totali previsti**: 50 (10 stagioni `1617`–`2526` × 5 leghe `I1`, `E0`, `SP1`, `D1`, `F1`)
- **Stagione Holdout `2627`**: **Esclusa categoricamente** (in conformità a `RULES.md`)
- **File scaricati ex-novo**: 0
- **File letti da cache locale (`data/raw`)**: 50
- **Errori di download**: 0
- **Righe grezze lette dai CSV**: 17937
- **Righe vuote di coda / padding rimosse**: 0
- **Numero totale di partite valide nel dataset consolidato**: **17937**
- **Range temporale**: `2016-08-12` → `2026-05-24`

## 2. Copertura Percentuale Non-Null per Stagione

| Stagione | Partite | Pinnacle Open 1X2 (%) | Pinnacle Close 1X2 (%) | Pinnacle Close O/U 2.5 (%) | Betfair Exchange Close 1X2 (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `1617` | 1826 | 99.73% | 100.00% | 0.00% | 0.00% |
| `1718` | 1826 | 99.78% | 99.95% | 0.00% | 0.00% |
| `1819` | 1826 | 99.89% | 99.95% | 0.00% | 0.00% |
| `1920` | 1725 | 99.88% | 100.00% | 100.00% | 0.00% |
| `2021` | 1826 | 99.78% | 100.00% | 99.67% | 0.00% |
| `2122` | 1826 | 99.89% | 100.00% | 99.78% | 0.00% |
| `2223` | 1826 | 100.00% | 100.00% | 99.18% | 0.00% |
| `2324` | 1752 | 100.00% | 100.00% | 98.86% | 0.00% |
| `2425` | 1752 | 100.00% | 100.00% | 98.97% | 100.00% |
| `2526` | 1752 | 51.48% | 51.26% | 50.80% | 93.66% |

## 3. Riepilogo per Lega

| Lega | Nome | Numero Partite | NaN su Pinnacle Closing 1X2 | Overround Medio Pinnacle Closing (%) | % Partite con almeno 1 Cartellino Rosso |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `I1` | Serie A | 3800 | 182 | 2.63% | 17.87% |
| `E0` | Premier League | 3800 | 170 | 2.53% | 10.68% |
| `SP1` | La Liga | 3800 | 193 | 2.61% | 20.50% |
| `D1` | Bundesliga | 3060 | 158 | 2.63% | 13.66% |
| `F1` | Ligue 1 | 3477 | 153 | 2.60% | 21.34% |

## 4. Riepilogo Anomalie Rilevate dal Reconciler (Fail-Fast & Data Integrity)

- **Date non parsate / righe corrotte scartate**: **0**
- **Partite con risultato finale mancante (FTHG/FTAG/FTR) scartate**: **0**
- **Partite duplicate rilevate (e deduplicate)**: **0**
- **Quote impossibili ($\le 1.01$) rilevate e bonificate in `NaN`**: **11**
  - `2025-04-19` (`2425` `F1`) `Paris SG` vs `Le Havre` — Colonna `P>2.5` = `0.0`
  - `2025-05-10` (`2425` `D1`) `Bayern Munich` vs `M'gladbach` — Colonna `P>2.5` = `0.0`
  - `2025-04-19` (`2425` `F1`) `Paris SG` vs `Le Havre` — Colonna `P<2.5` = `0.0`
  - `2025-05-10` (`2425` `D1`) `Bayern Munich` vs `M'gladbach` — Colonna `P<2.5` = `0.0`
  - `2025-04-19` (`2425` `F1`) `Paris SG` vs `Le Havre` — Colonna `PC>2.5` = `0.0`
  - `2025-05-10` (`2425` `D1`) `Bayern Munich` vs `M'gladbach` — Colonna `PC>2.5` = `0.0`
  - `2025-10-18` (`2526` `SP1`) `Barcelona` vs `Girona` — Colonna `PC>2.5` = `0.0`
  - `2025-04-19` (`2425` `F1`) `Paris SG` vs `Le Havre` — Colonna `PC<2.5` = `0.0`
  - `2025-05-10` (`2425` `D1`) `Bayern Munich` vs `M'gladbach` — Colonna `PC<2.5` = `0.0`
  - `2025-10-18` (`2526` `SP1`) `Barcelona` vs `Girona` — Colonna `PC<2.5` = `0.0`
  - `2020-06-01` (`1920` `D1`) `FC Koln` vs `RB Leipzig` — Colonna `AvgC>2.5` = `0.42`
- **Overround anomali Pinnacle Closing ($< -0.5\%$ o $> 12\%$)**: **0**
- **Overround anomali Betfair Exchange Closing ($< -0.5\%$ o $> 12\%$)**: **0**

## 5. Elenco Completo Nomi Unici Squadre in Serie A (`I1`)
Totale nomi squadra unici rilevati in Serie A (`1617`–`2526`): **34**

1. `Atalanta`
2. `Benevento`
3. `Bologna`
4. `Brescia`
5. `Cagliari`
6. `Chievo`
7. `Como`
8. `Cremonese`
9. `Crotone`
10. `Empoli`
11. `Fiorentina`
12. `Frosinone`
13. `Genoa`
14. `Inter`
15. `Juventus`
16. `Lazio`
17. `Lecce`
18. `Milan`
19. `Monza`
20. `Napoli`
21. `Palermo`
22. `Parma`
23. `Pescara`
24. `Pisa`
25. `Roma`
26. `Salernitana`
27. `Sampdoria`
28. `Sassuolo`
29. `Spal`
30. `Spezia`
31. `Torino`
32. `Udinese`
33. `Venezia`
34. `Verona`
