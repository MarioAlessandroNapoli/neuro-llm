# Research Log — neuro-llm

Log delle decisioni **scientifiche** del progetto: cosa abbiamo deciso, perché, cosa abbiamo
scartato e a quali condizioni una decisione va riconsiderata. Le scelte di puro engineering
(tracking, infrastruttura cloud, tooling) non vivono qui — stanno nel README e nella history git.

Regola di scrittura: ogni decisione ha un ID stabile (D1, D2, …), le questioni aperte hanno
ID Q1, Q2, …. Quando una Q si chiude diventa una D. Niente cronologia delle revisioni: il
documento riflette lo stato attuale del ragionamento.

---

## Domanda di ricerca

Le dinamiche oscillatorie — il meccanismo computazionale che il cervello implementa con le
bande delta/theta/alpha/beta/gamma — conferiscono un vantaggio misurabile a un language model,
a parità di parametri, dati e budget di training?

Formato dell'esperimento: confronto controllato tra un transformer baseline e varianti con
backbone oscillatorio (candidati: LinOSS, Wave-RNN), alla scala 8-33M parametri, from scratch.

**Gap di letteratura verificato (2026-08-17)**: nessun LM con backbone oscillatorio è stato
pubblicato su TinyStories o BabyLM. Ricerche web dedicate non hanno restituito alcun paper che
confronti LinOSS/D-LinOSS/Wave-RNN con un transformer su language modeling testuale. Il
confronto, anche con esito negativo, sarebbe un contributo originale.

---

## Mappa del campo (rilevata 2026-08-17)

Sintesi della rassegna; solo i punti che vincolano le nostre scelte.

**Reti oscillatorie (il filone che testiamo).** coRNN modella i neuroni come oscillatori
smorzati accoppiati, con bound sui gradienti che mitigano vanishing/exploding
(arXiv:2010.00951, ICLR 2021). LinOSS è lo state-space model costruito da oscillatori armonici
forzati: parallelizzabile con associative scan, ~2× meglio di Mamba su sequenze da 50k, ICLR
2025 Oral (arXiv:2410.03943). D-LinOSS aggiunge smorzamento apprendibile — il "forgetting" su
scale temporali apprese (arXiv:2505.12171). Wave-RNN codifica il passato recente come onde
viaggianti su un campo spaziale, apprende più in fretta con meno parametri (Keller, Muller,
Sejnowski, Welling — ICLR 2024, arXiv:2309.08045). Nessuno testato su LM testuale.

**Fase e sincronia.** AKOrN sostituisce i neuroni con oscillatori di Kuramoto: binding by
synchrony come meccanismo di rappresentazione (ICLR 2025, arXiv:2410.13821). RNN addestrate su
working memory sviluppano spontaneamente phase coding (Pals, Macke, Barak — PLOS Comp Biol
2024). Contro-evidenza da tenere presente: la fase theta-gamma potrebbe non codificare
l'ordine temporale né negli umani né nelle RNN (Nature Neuroscience 2025) — non dare per
scontato il phase coding come meccanismo, va misurato.

**Evidenze dentro i transformer.** Gli LLM pre-addestrati rappresentano i numeri con feature
di Fourier (NeurIPS 2024, arXiv:2406.03445). Il positional encoding sinusoidale è stato
proposto come analogo delle onde corticali viaggianti (Muller, Churchland, Sejnowski — Trends
in Neurosciences 2024, arXiv:2401.14267) — ponte concettuale, senza benchmark.

**Lezione metodologica da HRM/TRM.** HRM (27M, gerarchia a due timescale "corticali") ha fatto
rumore su ARC-AGI, ma le ablazioni indipendenti di ARC Prize mostrano che la gerarchia
brain-inspired vale ~5 punti e il grosso viene dal refinement loop e dall'augmentation; TRM la
elimina e fa meglio con 7M parametri (arXiv:2506.21734; arXiv:2510.04871). Morale che adottiamo
come principio: **testare meccanismi, non estetiche** — ogni claim "neuro" deve sopravvivere a
un'ablazione che lo isola.

**Banco di prova.** TinyStories: dati sintetici a lessico infantile, ~2,7M storie; modelli
1-33M producono inglese coerente (arXiv:2305.07759). BabyLM: pretraining from scratch con
budget 10M/100M parole e pipeline di valutazione standardizzata (BLiMP); lezione dalla
challenge: i curriculum "sviluppo-ispirati" ingenui non aiutano, contano architettura e
obiettivo di training (GPT-BERT, arXiv:2410.24159).

---

## Decisioni

### D1 — Regime sperimentale: from scratch, piccola scala, pipeline a due stadi
**Decisione.** Training from scratch di modelli 8-33M parametri. Stadio 1: TinyStories come
banco di iterazione rapida (val loss + probe). Stadio 2: la variante migliore ri-addestrata su
BabyLM Strict-Small (10M parole) e valutata con la pipeline BLiMP ufficiale.
**Perché.** Le varianti architetturali non hanno pesi pre-addestrati: il fine-tuning non è
un'opzione. TinyStories rende il from-scratch economico (run in ore); BabyLM fornisce metrica
standardizzata e decine di baseline pubblicate, e il suo framing (sample-efficiency a livello
umano) è nativo per una domanda neuro.
**Scartato.** Fine-tuning di modelli 1-4B (non testa l'architettura); valutazione solo su
TinyStories (nessun confronto esterno possibile).
**Riconsiderare se.** Lo stadio 1 non produce alcuna variante sana (allora il problema è a
monte, nel design del backbone).

### D2 — Direzione: backbone oscillatorio, candidati LinOSS/D-LinOSS e Wave-RNN
**Decisione.** Il "trattamento" sperimentale è il backbone che rimpiazza (o affianca)
l'attention: oscillatori armonici (LinOSS/D-LinOSS) e onde viaggianti (Wave-RNN) sono i
candidati; la scelta finale è aperta (→ Q1).
**Perché.** Sono i filoni oscillatori con la matematica più solida (gradienti stabili
dimostrati, risultati long-range forti), codice ufficiale replicabile a questa scala, e — per
LinOSS — parallelizzabilità compatibile col nostro budget. Sono anche i filoni dove il
principio HRM/TRM ci protegge: il meccanismo (ODE del secondo ordine, propagazione d'onda) è
isolabile in ablazione.
**Scartato.** Predictive coding come obiettivo di training (nessun LM competitivo esiste, il
demo resterebbe su MNIST); spiking LM (l'efficienza event-driven non è la nostra domanda);
architetture HRM-like (la lezione delle ablazioni è che la gerarchia in sé non regge).
**Riconsiderare se.** In fase di design emergesse che nessun candidato può fare LM causale
senza stravolgimenti (es. Wave-RNN puramente ricorrente troppo lenta per il budget).

### D3 — Tokenizer: BPE byte-level, vocabolario 8192, addestrato sul dominio, congelato
**Decisione.** Un solo tokenizer per tutte le architetture e tutte le run: BPE byte-level
(famiglia GPT-2) da 8192 token addestrato sul train split di TinyStories, `<|endoftext|>`=0.
È parte dell'apparato di misura, non del trattamento: non si cambia senza ripartire da zero
con tutte le baseline.
**Perché.** (a) Budget parametri: con d_model=256 e pesi legati, un vocab 50k costerebbe 12,9M
parametri di embedding contro i 2,1M dell'8k — i parametri devono stare nei layer, dove le
architetture differiscono. (b) Misurato sul corpus reale: 96% delle top-2000 parole è un token
singolo, compressione 4,09 byte/token (meglio del GPT-2 50k, 3,95, perché addestrato sul
dominio), roundtrip lossless. (c) La scaling law del vocabolario (NeurIPS 2024,
arXiv:2407.13623) predice vocabolari piccoli per compute piccolo: 8192 sta appena sopra il
lessico del dominio (~10k parole distinte).
**Scartato rispetto al SOTA 2025-26, con motivo.** Tokenizer GPT-2 riusato (embedding dominante
+ righe morte); SuperBPE (arXiv:2503.13423 — brilla a vocab ~200k, regime opposto al nostro);
direzione "Claude 5" di token più fini per qualità (trade sensato quando il compute per token
è enorme; a 8M parametri il compute per token è il collo di bottiglia); tokenizer-free con
chunking appreso (H-Net — potente ma introdurrebbe una seconda variabile architetturale che
confonderebbe il confronto).
**Riconsiderare se.** Mai dentro questo esperimento. Un'eventuale ablazione char-level è una
domanda separata (→ Q3), con baseline proprie.

### D4 — L'asse temporale dell'esperimento è il token
**Decisione.** Il passo temporale delle dinamiche oscillatorie è il token; le "frequenze" delle
architetture vivono nello spazio dei passi-token, non in secondi. Geometria misurata del
segnale: storia mediana 174 token (p90 259, max 1010), finestra di training 512 token (~2-3
storie), dipendenze narrative (coerenza dei personaggi) a scala 100-300 passi.
**Perché.** È l'unica definizione operativa disponibile a tokenizzazione fissa, e rende
confrontabili le architetture: tutte ricevono lo stesso segnale alla stessa granularità. La
scelta del tokenizer (D3) fissa quindi anche il contenuto in frequenza del compito — motivo in
più per congelarlo.
**Consapevolezza del limite.** Il token è un passo temporale *non uniforme* in termini di
testo (un token = 1-10 caratteri). Il SOTA tokenizer-free (H-Net, dynamic chunking) mostra che
la granularità temporale appresa batte quella fissa — concettualmente parente delle gerarchie
di timescale oscillatorie. Lo registriamo come asse futuro (→ Q3), non come variabile di
questo esperimento.

### D5 — Baseline riaddestrata, mai numeri di terzi
**Decisione.** Il transformer baseline viene addestrato da noi, con lo stesso tokenizer, dati,
token budget, conteggio parametri e scheduler delle varianti. I checkpoint pubblici di
TinyStories (GPT-Neo 1M/8M/33M) servono solo come àncora qualitativa.
**Perché.** La loss è confrontabile solo a parità totale di setup; i numeri pubblicati usano
tokenizer (50k) e budget diversi e la valutazione del paper è grading GPT-4, non loss. Un
confronto contro numeri altrui sarebbe invalido by design.

### D6 — Parità e statistica minima
**Decisione.** Confronti a parità di parametri (±5%) e di token budget; ogni configurazione
con ≥2-3 seed; si riportano media e range, e il segno del risultato si dichiara sempre (anche
se negativo per l'ipotesi oscillatoria).
**Perché.** A questa scala il rumore tra seed è materiale; un vantaggio architetturale
dichiarato su una singola run non è un risultato. Il costo è sostenibile (run da ~30-60 min su
GPU cloud).

---

## Questioni aperte (fase di design, in corso)

- **Q1 — Quale backbone oscillatorio, in quale forma.** LinOSS puro come mixer di sequenza?
  D-LinOSS (lo smorzamento apprendibile è la variante più "bande cerebrali": scale temporali
  multiple apprese)? Wave-RNN? Ibrido attention+oscillatori (dove l'ablazione isola il
  contributo oscillatorio)? Da decidere in brainstorming con criterio: il meccanismo deve
  essere isolabile in un'ablazione (principio HRM/TRM).
- **Q2 — Ipotesi falsificabile e probe.** La val loss non basta: servono probe che tocchino la
  *specificità* oscillatoria — es. dipendenze a lungo raggio (coerenza dei nomi dei personaggi
  a 100-300 token, dove vive la scala narrativa misurata in D4), degradazione con la distanza,
  eventuale analisi spettrale degli hidden state. Da definire prima di scrivere le architetture.
- **Q3 — Granularità temporale come ablazione futura.** Char-level (~4× più passi, dipendenze
  stirate: test più severo per la memoria oscillatoria) o chunking appreso stile H-Net. Fuori
  dallo scope dello stadio 1; richiede baseline dedicate.
- **Q4 — Token budget per lo stadio 1.** ~100M token per run (una via di mezzo tra il
  Chinchilla-ottimale ~170M per 8,5M parametri e il costo di 3 architetture × 3 seed)? Da
  fissare insieme a Q1 quando sapremo il costo per step delle varianti.

---

## Registro esperimenti

| ID | Data | Arch | Params | Token | Seed | val_loss | Note |
|----|------|------|--------|-------|------|----------|------|
| smoke-0 | 2026-08-17 | transformer | 8,5M | 8,4M (2 run, prova resume) | 1 | ~5,0 (non a convergenza) | Solo validazione pipeline su M2; nessun valore scientifico |

Ogni run vera aggiunge una riga; i dettagli vivono su W&B (progetto `neuro-llm`), qui solo
l'essenziale per leggere la storia dell'esperimento senza aprire dashboard.
