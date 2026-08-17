# Archivio — Setup infrastruttura (agosto 2026, concluso)

Decisioni di engineering e numeri del setup iniziale. Chiuso il 2026-08-17 con collaudo
E2E in cloud. Qui per consultazione; l'operatività corrente vive nella skill `ml-dev`.

## Architettura del sistema

Quattro sedi, ognuna col suo ruolo: **GitHub** (codice, pubblico — così Kaggle clona senza
token), **Kaggle** (GPU gratuita: 30 h/settimana, sessioni 12 h, secrets per le chiavi),
**HF Hub privato** (le cose pesanti: dataset tokenizzato e checkpoint — 100 GB gratis),
**W&B** (le cose leggere e interrogabili: metriche e config).

## Decisioni chiave e motivi

- **W&B come tracking** (vs ClearML/MLflow): unico con MCP server ufficiale *hosted*
  (`mcp.withwandb.com`) interrogabile da Claude Code senza infrastruttura. MLflow avrebbe
  richiesto un tracking server che un kernel Kaggle effimero non può esporre; ClearML ha
  solo MCP community single-maintainer.
- **Checkpoint su HF e non su W&B Artifacts**: un ckpt 33M con stato AdamW ≈ 400 MB;
  il free tier W&B (5 GB/mese) si saturerebbe in giorni.
- **Lightning "sottile"** (2.6.x): giustificato solo da resume completo su sessioni 12 h e
  WandbLogger. Niente DataModule, LightningCLI, torch.compile, StatefulDataLoader (l'ordine
  finestre diverso al resume è accettato — irrilevante per il confronto tra architetture).
- **Cloud invece di M2**: l'Air (M2, 16 GB, senza ventola) fa ~5,3k tok/s — solo debug e
  smoke. bf16 su MPS non dà guadagno (misurato). MLX scartato: il cloud è CUDA.
- **MCP Kaggle** aggiunto a scope user (token bearer in ~/.claude.json) per salvare/gestire
  il notebook da Claude Code.

## Collaudo E2E (2026-08-17)

Run di prova 10M token su T4: clone → secrets → dati da HF → training → W&B → checkpoint
su HF, tutto verificato. 610 step in 98 s, **~115k tok/s**, val_loss 5,62 (da 9,0).
Resume da checkpoint provato in locale (ripresa dallo step 1001 con loss in continuità).

## Note residue

- I token HF e Kaggle sono transitati in chiaro nei log della sessione di setup: se i log
  vengono condivisi, rigenerarli (HF settings/tokens; Kaggle settings/API).
- La cancellazione dei vecchi progetti W&B 2021 è stata fatta da UI (l'MCP è read-only e
  il bearer token MCP non autentica la Models API).
- Stima pre-collaudo M2 vs realtà T4: il fattore è ~22× — le stime di budget dello stadio 1
  sono state ricalibrate di conseguenza (griglia completa ≈ 2-2,5 h GPU).
