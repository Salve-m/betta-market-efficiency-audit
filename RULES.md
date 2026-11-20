# Regole Vincolanti del Progetto Betta

Queste regole devono essere sempre rilette e rispettate in ogni intervento futuro sul progetto **Betta**:

1. **Zero Data Leakage**: ogni calcolo o stima alla data $t$ può usare solo dati con data strettamente inferiore a $t$.
2. **Separazione Holdout**: la stagione 2026/27 in corso non va mai mescolata nel dataset storico di sviluppo (stagioni 2016/17 – 2025/26).
3. **Fail-Fast & Data Integrity**: non ignorare silenziosamente anomalie. Quote impossibili ($\le 1.01$), date non valide, overround anomali o nomi squadra duplicati vanno sempre tracciati e segnalati esplicitamente.
4. **Riproducibilità**: pipeline modulare, deterministica e testabile con `pytest`.
5. **Perimetro legale**: nessun scraping di bookmaker live, nessuna automazione di piazzamento scommesse, nessuna cattura schermo. Solo analisi offline su dataset storici puliti.

## Split Temporale Vincolante Ufficiale

| Partizione | Stagioni | Ruolo e Vincoli |
| :--- | :--- | :--- |
| **Warm-up** | `1617–1819` | Inizializzazione stati storici / rating / rolling features |
| **Train / Sviluppo** | `1920–2223` | Stima parametri e sviluppo modelli |
| **Validation** | `2324–2425` | Selezione iperparametri, confronto modelli e calibrazione |
| **Test Cieco** | `2526` | **Inviolabile per scelte di modellazione** (valutazione finale out-of-sample) |
| **Holdout Live** | `2627` | **Esclusa categoricamente** dal dataset storico |
