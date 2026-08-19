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
backbone oscillatorio (candidati: LinOSS, D-LinOSS, Wave-RNN), alla scala 8-33M parametri,
from scratch.

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

## Ancore quantitative e apparato di valutazione del campo (rilevate 2026-08-17)

Raccolta pre-design a supporto di D7 (che ha assorbito Q2) e Q4. Numeri di terzi = àncora,
mai braccio di confronto (D5).

**Ancore val loss su TinyStories.** I GPT-Neo ufficiali (roneneldan/TinyStories-1M/8M/33M) NON
hanno loss pubblicata — solo curve in Fig. 3 del paper; training ~20 epoch (≈9-10B token, stima
da HF Discussion #8), tokenizer GPT-Neo con vocab effettivo dibattuto (paper dice top-10k,
config dice 50257). Uniche àncore citabili: llama2.c (tokenizer Llama-2 32k) — 15M → **1.072**,
42M → **0.847**, 110M → **0.760** nats/token (README del repo); il 260K (vocab 512) → 1.297 non
è confrontabile. Nessuna soglia pubblicata loss↔coerenza. Dataset: 471,6M token train / 2,37M
val con BPE GPT-2 (arXiv:2601.23236).

**Confronto cross-tokenizer solo via bits-per-byte** (The Pile §3.1, arXiv:2101.00027:
BPB = loss_nats / ln2 / bytes_per_token). Nessun BPB pubblicato per TinyStories: l'àncora
esterna l'abbiamo misurata noi (2026-08-17, `src/eval/anchor_stories15m.py` — modello e
tokenizer ufficiali llama2.c a commit pinnato, mai reimplementati). **stories15M sul NOSTRO
val V2 a contesto 256**, stream packing identico al suo training (BOS per storia, finestre
non sovrapposte da 256): **loss 1,1511 nats/token → BPB 0,4407** (3,7687 byte/token Llama-2,
BOS esclusi). Coerenza verificata: il README llama2.c riporta 1,072 sul suo val V1 — il
+0,08 è l'effetto fuori-distribuzione V1→V2 atteso, quindi il numero è un limite superiore
e si dichiara come tale. Il nostro BPB si riporterà sia a contesto 256 (confronto con
l'àncora) sia a 512 (finestra di training). Equivalenza per leggere i nostri numeri: col
nostro tokenizer (4,0988 byte/token, EOT esclusi) BPB 0,4407 ≡ val loss 1,252 nats/token.
Convenzione BPB fissata: byte UTF-8 del solo testo, EOT/BOS esclusi dal conteggio token.
Cautela: la loss più bassa cross-tokenizer non predice la qualità soggettiva
(arXiv:2504.07989).

**Come valuta il campo.** TinyStories: GPT-Eval — LLM giudice su grammar/creativity/consistency/
plot, scala 1-10; i prompt ufficiali sono in `Evaluation_prompts.yaml` nel repo HF del dataset
(pipeline da ricostruire, rubric disponibile). BabyLM 2025: repo `babylm/evaluation-pipeline-2025`
— zero-shot BLiMP, BLiMP-supplement, EWoK, COMPS, WUG, reading/eye-tracking, AoA + fine-tuning
GLUE ridotto; modalità fast (checkpoint intermedi) e full. BabyLM classifica per score dei task,
non per perplexity.

**Assenza verificata (rafforza il gap).** LinOSS, D-LinOSS, Wave-RNN, coRNN: mai valutati su LM
testuale, nemmeno char-level (verificato sui paper). Termine di paragone più vicino al nostro
regime: BabyLM 10M parole — Mamba BLiMP 64.44 vs transformer baseline 62.64, HGRN2 67.05
(arXiv:2412.15978): le ricorrenti lineari battono già i transformer a questa scala.

**Probe long-range con precedenti citabili.** (a) Ablazione del contesto a distanza d — aumento
di loss quando il contesto oltre d viene troncato/mescolato (Khandelwal, arXiv:1805.04623; la
metodologia canonica per "quanto contesto usa davvero il modello"); (b) coerenza delle entità in
generazione — match entità generate vs gold, finestra di menzione del protagonista
(arXiv:2202.01709); (c) FFT 2D spazio-tempo delle attivazioni hidden — ricetta nel paper wRNN
stesso (arXiv:2309.08045). FFT su hidden state di un LM testuale: mai pubblicata — spazio di
contributo, insieme a oscillatori-su-LM e name-consistency su TinyStories.

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
**Decisione.** Confronti a parità di **backbone** (±5%): tutti i parametri esclusa la matrice
di embedding dei token (8192×256 = 2,10M, identica per costruzione in ogni braccio e legata
alla testa). Il position embedding conta nel backbone del transformer — "come si rappresenta
l'ordine" è parte del meccanismo confrontato — quindi una variante che non lo usa può
rispendere quel budget nei propri layer. Riferimento congelato: backbone baseline =
**6.449.664** parametri; enforcement nel codice (`train.py` rifiuta di partire fuori dal ±5%).
Scartato: parità sul totale (la tolleranza si scaricherebbe tutta sulla parte che differisce,
±6,8% effettivo, e i 131k di position embedding resterebbero non regolati). Parità anche di
token budget; **minimo 3 seed**
per configurazione (2 non bastano: la regola di verdetto scatterebbe per puro caso ~1 volta
su 6); prima della griglia, 5 seed della sola baseline per misurare σ della val loss, che
fissa la banda di equivalenza ε di D7. Si riportano media e range, e il segno del risultato
si dichiara sempre (anche se negativo per l'ipotesi oscillatoria).

**Ricetta di training (parità di ricetta, non solo di conteggio).** Regola simmetrica per
categoria di parametro, identica per tutte le braccia, cablata in `configure_optimizers`
(`src/lit_module.py`): AdamW betas (0.9, 0.95), weight decay 0,1 **solo sulle matrici dense**;
zero su bias, gain di normalizzazione, embedding e **parametri di stato** — frequenze,
smorzamento, passi di discretizzazione degli oscillatori, dichiarati da ogni architettura via
contratto `state_parameters()` del registry (il transformer ritorna vuoto). È la ricetta
canonica di entrambe le famiglie (GPT non decade norm/bias; S4/S5/LinOSS non decadono A e dt).
**Learning rate: mini-sweep pre-registrato per braccio.** Per ogni architettura (baseline
inclusa, stesso trattamento): 3 valori {1e-4, 3e-4, 1e-3}, 1 seed, budget ridotto a ~20M token,
selezione sulla val loss; il valore vincente si congela per la griglia. Le run di sweep non
sono risultati scientifici: vivono su W&B, non nel registro.
**Scartato.** Ricetta del transformer per tutti i bracci (decay sui parametri di stato =
prior contro il meccanismo in prova; il codice ufficiale LinOSS addestra senza decay); lr
unico non tunato (un esito negativo resterebbe attaccabile come "lr sbagliato per la
famiglia"); tuning completo per braccio (il budget esplode e la parità di compute salta).
**Perché.** A questa scala il rumore tra seed è materiale; un vantaggio architetturale
dichiarato su una singola run non è un risultato. Il costo è sostenibile (run da ~30-60 min su
GPU cloud).

### D7 — Apparato di valutazione stadio 1: loss + giudice LLM (scoring e Elo indipendenti)
**Decisione.** Tre strumenti, congelati prima di ogni run scientifica:
1. **Val loss** (nats/token) ad apparato congelato — metrica primaria del confronto interno.
   **BPB** (formula Pile) solo come aggancio esterno, con àncora calcolata da noi su stories15M.
2. **Giudice LLM**: `claude-opus-5` (ID fisso, senza suffisso di data), giudizio cieco (mai
   il nome dell'architettura, ordine randomizzato), su completamenti dei prompt ufficiali
   TinyStories (`Evaluation prompts.yaml` — con lo spazio — nel repo HF del dataset).
   **L'API attuale non espone sampling né seed** (temperature/top_p/top_k rimossi su Opus 5:
   400 se inviati; thinking adattivo di default): il determinismo del giudice non è
   richiedibile e si sostituisce con protocollo — (i) output strutturato via
   `output_config.format` con JSON schema validato (score e verdetti tipizzati, zero parsing
   di testo libero); (ii) effort dichiarato e fisso per tutta la campagna (`medium`);
   (iii) doppio ordine per coppia, inversione = pareggio; (iv) self-agreement del giudice
   misurata su un sottoinsieme ripetuto e riportata; (v) tutte le braccia giudicate nella
   stessa finestra via Batches API (−50% di costo): l'invarianza dello strumento *tra le
   braccia* conta più della riproducibilità assoluta nel tempo.
   Due strumenti **indipendenti**: (a) scoring assoluto multi-asse
   1-10 (grammar / consistency / plot / creativity) — misura la qualità in sé e aggancia la
   scala GPT-Eval del paper; (b) confronti pairwise ciechi A/B aggregati in rating Elo
   (fit Bradley-Terry, indipendente dall'ordine delle partite) — misura la discriminazione
   tra braccia. Il prompt set è **stratificato per lunghezza del prefisso** (corto ~50 token
   vs lungo 200-300, dove vive la scala narrativa di D4): score e Elo si riportano anche per
   strato, così la tenuta a lungo raggio è misurata dentro questo apparato.
3. **Pipeline BabyLM (BLiMP ecc.): solo stadio 2.** Allo stadio 1 il lessico è fuori dominio
   per modelli a lessico infantile; i valori sarebbero depressi e non decisionali.

**Protocollo di generazione (congelato, identico per tutte le braccia).** Quello del GPT-Eval
originale: **temperatura 1, 10 completamenti per prompt**, nessun troncamento — il paper non ne
dichiara alcuno (arXiv:2305.07759 §3, verificato: zero occorrenze di top-k/top-p), e i loro
stessi modelli 1-33M generavano inglese coerente così. `max_new_tokens` 200 con stop a
`<|endoftext|>`, totale entro la finestra 512 della baseline. Seed di generazione derivato da
(arch, seed della run, prompt, indice del completamento); poiché PyTorch non garantisce
riproducibilità cross-piattaforma (docs/notes/randomness), l'àncora vera è che **i testi
generati si salvano come artefatti versionati**: ogni giudizio è ri-eseguibile su testi
identici. Scartati con motivo: temp 0,8/top-k 40 (nessuna fonte lo usa — era un falso
ricordo); beam 5 deterministico (commento HF informale dell'autore, contraddice il testo del
paper e misura la moda, non la distribuzione); top-p 0,9 (convenzione llama2.c, non del paper).

**Prompt set: due strati, entrambi versionati in repo.** Strato corto = i 44 prompt ufficiali
(`Evaluation prompts.yaml`; mediana 59 token, max 118) — unico ponte col GPT-Eval del paper.
Strato lungo = **50 prefissi** da 250-300 token ritagliati dalle storie del val set V2 con
lunghezza ≥400 token (pool: 1.046 storie; resta ≥100 token di continuazione vera da giudicare),
selezione con seed dichiarato. Il file dei prompt è apparato di misura come il tokenizer (D3).

**Mappa generazione→giudizio (scelte nostre, senza precedente pubblicato: si dichiarano).**
Pairwise: verdetto **set-vs-set** — un giudizio per (coppia, prompt, ordine) con tutti i 10
completamenti per lato nello stesso prompt; i blocchi A/B si scambiano tra i due ordini,
l'ordine intra-blocco si mescola con seed; la rubric istruisce a giudicare la qualità
complessiva del set, non il singolo migliore/peggiore (mitiga la salienza degli outlier,
comunque simmetrica tra braccia). Razionale: a temp 1 × 10 il measurand è la *distribuzione*
del modello e il verdetto sul set la misura direttamente; accoppiare i completamenti uno-a-uno
(10 match per prompt) darebbe un guadagno di potenza illusorio — i match sullo stesso prompt
della stessa run sono correlati e il bootstrap clusterizzato li tratta come tale — mentre
20 storie per verdetto riducono il rumore per match a parità di confronti. Perdita accettata:
la win-rate per singolo completamento (la copre lo scoring). Scoring: **una chiamata per
(prompt, braccio, run)** che restituisce l'array dei 10 vettori di score per-completamento —
mai uno score aggregato sul set: gli score individuali servono al check score↔lunghezza e
all'aggancio GPT-Eval, che è per-completamento. Ordine dei 10 mescolato con seed;
l'ancoraggio intra-chiamata è rumore condiviso, non bias tra braccia (ogni chiamata è
mono-braccio, procedura identica ovunque).

**Dettagli del giudice, ciascuno con la sua fonte.** Swap d'ordine obbligatorio, inconsistenza
= tie (MT-Bench, arXiv:2306.05685: position bias fino al 75% su giudici deboli). Bradley-Terry
via MLE, mai Elo sequenziale, tie = ½ vittoria + ½ sconfitta (Chatbot Arena, arXiv:2403.04132).
Rating sempre con CI bootstrap; nel nostro caso il bootstrap è clusterizzato sulle run —
scelta nostra, più severa dei protocolli pubblicati che ricampionano i singoli match, perché
la nostra unità di replicazione è il seed. Scoring assoluto con **rubric ancorata per fascia**
sulle dimensioni del paper (grammar/consistency/plot/creativity): l'ancoraggio è l'intervento
con più evidenza di impatto (Prometheus, arXiv:2310.08491); l'aggancio assoluto alla scala
GPT-4 del paper è comunque rotto dal cambio di giudice, e si dichiara. Length bias: controllato
per costruzione (cap uniforme su tutte le braccia) + check diagnostico della correlazione
residua score↔lunghezza (razionale di AlpacaEval 2 LC, arXiv:2404.04475). La **self-agreement
su sottoinsieme ripetuto è un'aggiunta nostra**, non una pratica pubblicata: si dichiara come
scelta originale. Numerosità: nessuna regola pubblicata per coppia; 94 prompt
(44 corto + 50 lungo) × 2 ordini = **188 verdetti set-vs-set per coppia** (88 corto +
100 lungo), sul totale sopra l'euristica di potenza (~150-200 per risolvere ~100 punti Elo);
per strato decide comunque il CI, non il rating puntuale.

**Regola di verdetto (pre-registrata, esaustiva).** Ogni esito della griglia cade in una e
una sola cella della tabella; il verdetto è scritto qui, prima di qualunque numero.

*Stati dell'asse loss* (primario, per braccio vs baseline): **L+**/**L−** = test di
permutazione esatto sulla differenza delle medie, unilaterale, α=0,05 per braccio (con 3v3:
la differenza osservata è la più estrema delle 20 permutazioni); **L=** = non significativo
e |Δmedie| ≤ ε, con **ε = σ della baseline** misurata con 5 seed prima della griglia
— **misurata (2026-08-19, ricetta D11): ε = 0,007 nats** (5 seed a 170M, lr 3e-2, media
1,599, registro base2-1..5). Storica (2026-08-18, lr 1e-3): ε = 0,017 (media 1,8266,
range [1,8064, 1,8427], run base-1..5 nel registro);
*indeterminato* (non significativo, |Δ| > ε) → escalation pre-registrata: +2 seed per
braccio, una sola volta; se resta indeterminato si riporta come tale. Il familywise sui
bracci confrontati si dichiara nella scrittura dei risultati.

*Stati dell'asse Elo, strato lungo*: ogni coppia giudicata in entrambi gli ordini
(inversione = pareggio, neutralizza il position bias); CI 95% della differenza di rating
Bradley-Terry via bootstrap clusterizzato sulle run (unità di replicazione = seed).
**E+**/**E−** = CI esclude lo zero; **E=** = lo include ("non distinguibile con questo
apparato", non "uguali"). Lo strato corto non è un cancello: si riporta sempre e etichetta
la vittoria (corto pari → vantaggio *specifico* long-range; corto superiore → *generale*).
Eccezione: se lungo E+ ma corto E−, la vittoria decade a "mista" (trade-off, non vantaggio).

| | E+ (lungo superiore) | E= (indistinguibile) | E− (lungo inferiore) |
|---|---|---|---|
| **L+** | Supportata — esito forte → stadio 2 | Supportata via (a), specificità non dimostrata → stadio 2 | Supportata via (a), qualità in tensione → stadio 2 con riserva, prima diagnosi |
| **L=** | Supportata via (b), l'esito più interessante → stadio 2 | **Non supportata da questo apparato** (≠ "nessun effetto"); negativo pubblicato, niente stadio 2 | Falsificata sul lato qualità |
| **L−** | Mista, non supportata come formulata; segnale esplorativo per eventuale nuova ipotesi | Falsificata | Falsificata — esito netto |

**Probe di riserva = solo diagnostica.** Name-cloze a distanza, ablazione del contesto e
analisi spettrale possono essere eseguiti su qualunque esito come diagnostica esplorativa,
ma **non cambiano mai il verdetto dello stadio 1**: un loro eventuale segnale diventa
l'ipotesi di un nuovo esperimento pre-registrato, mai una vittoria retroattiva. Il controllo
strutturale contro i falsi positivi resta D1: la variante promossa deve replicare su
BabyLM/BLiMP.

**Scartato.** Benchmark name-cloze a distanza e ablazione del contesto (Khandelwal 1805.04623)
come strumenti di stadio 1 — scelta di semplicità dell'apparato; restano candidati per
ablazioni future (precedenti citabili in § Ancore). BLiMP a stadio 1 (mismatch di dominio).
Scoring-only o pairwise-only (le due misure rispondono a domande diverse: qualità assoluta vs
discriminazione). Pairwise a completamento singolo campionato (standard MT-Bench/Arena, ma
verdetti più rumorosi a parità di match e nessun guadagno statistico col bootstrap
clusterizzato); match uno-a-uno sui 10 completamenti (costo ×10 per potenza effettiva quasi
nulla, vedi Mappa generazione→giudizio).

**Riconsiderare se.** L'apparato produce sistematicamente esiti "indeterminato" anche dopo
l'escalation (σ tra seed più grande del previsto): allora il problema è la potenza, e si
riapre la questione del numero di seed o del budget — mai la tabella dei verdetti a
posteriori.

### D8 — Token budget stadio 1: 170M per run
**Decisione.** Ogni run di griglia dello stadio 1 usa 170.000.000 token di training
(`--tokens 170000000`), identico per tutte le braccia e i seed (parità D6). La run
pilota da epoch piena (536M) resta un'àncora interna, non un braccio.
**Perché.** (a) È il Chinchilla-ottimale per 8,5M parametri, quindi difendibile senza
argomenti ad hoc; (b) la curva del pilot (registro, pilot-1) mostra che a 170M il modello
ha completato ~96% della discesa dall'inizializzazione (1,80 vs 1,51 dell'epoch piena) ed
è pienamente nel regime "inglese coerente"; (c) il costo consente griglia completa + 5 seed
+ escalation D7 dentro la quota settimanale (run da ~16 min in DDP); (d) la domanda di
ricerca è a budget fisso moderato — il framing sample-efficiency è il terreno naturale
dell'ipotesi oscillatoria.
**Scartato.** 100M (lascia 0,48 nats sul tavolo, regime più rumoroso, risparmio marginale);
536M per tutte (×3 di costo per 0,15 nats di informazione in più, quota a rischio con le
varianti lente); budget differenziato per braccio (viola la parità D6 by design).
**Nota di misura (dal confronto sweep vs pilot).** A piccoli budget una run dedicata chiude
peggio del punto intermedio di una run lunga (20M dedicata: 3,67 vs ~3,4 della curva pilot
a 20M): l'annealing precoce costa più progresso di quanto rumore tolga. I punti intermedi
della curva pilot si leggono come stima centrale del risultato a quel budget, non come limite.
**Riconsiderare se.** La σ dei 5 seed rendesse la banda ε inutilizzabile (clausola già in D7,
il problema sarebbe la potenza); o il costo per step di una variante rendesse 170M
impraticabile nella quota — revisione esplicita, mai silenziosa.

### D9 — Assi di design dello stadio 1: cinque temi, ablazioni singole e combinate
**Decisione.** Il design dell'architettura dello stadio 1 si organizza su cinque assi,
distillati dalla mappa temi→transformer (lettura "Neurociencia del cuerpo", Castellanos,
cap. 1 — 2026-08-18): (1) **scale temporali multiple e parallele**; (2) **inibizione
selettiva e oblio strutturale**; (3) **scia percettiva**; (4) **metastabilità**;
(5) **gerarchia subcorticale→corticale**. Ogni asse va tradotto in "matematica da LLM"
— un meccanismo isolabile in ablazione — e adottato solo nella misura in cui l'ablazione
ne mostra il beneficio, singolarmente e in combinazione (feature seeking, non adozione
estetica: principio HRM/TRM).
**Traduzione operativa provvisoria** (da raffinare in D10): 1 → mixer oscillatorio con
frequenze apprese per unità (famiglia LinOSS); 2 → smorzamento apprendibile
(D-LinOSS-style: un parametro che è insieme alfa-inibizione e curva dell'oblio);
3 → proprietà emergente dello stato che decade — si *misura* (probe), non si implementa
come modulo; 4 → schema di inizializzazione/vincolo dei rapporti di frequenza (bande a
rapporti aurei/irrazionali vs spaziatura uniforme); 5 → disposizione per profondità
(gradiente di scale temporali tra layer; ibrido attention+oscillatori).
**Perché.** Sono gli assi dove il capitolo e la letteratura convergono su meccanismi
isolabili; gli overlap noti (3 emerge da 1+2; 4 è un vincolo su 1) riducono la matrice
reale a ~4 fattori indipendenti, compatibile col budget D8 (run da ~16 min).
**Scartato (con motivo).** Fase come canale computazionale (contro-evidenza Nature
Neuroscience 2025 già a registro; resta nei probe diagnostici); interocezione e valenza
emotiva (nessun analogo a questa scala, fuori scope); porting di temi senza ablazione
possibile.
**Riconsiderare se.** La matrice combinata sfora il budget → si procede a stadi
(1a: assi singoli; 1b: combinazioni dei soli assi con segnale); se nessun asse singolo
mostra segnale, le combinazioni non si esplorano a tappeto.

### D10 — Griglia 1a: sei bracci, un grado di libertà per confronto
**Decisione.** La griglia 1a confronta la baseline (fatta: base-1..3) con sei bracci;
ogni confronto adiacente cambia un solo meccanismo:

| Braccio | Asse D9 isolato | Confronto che lo decide |
|---|---|---|
| `linoss` | 1 scale temporali | vs baseline |
| `dlinoss` | 2 inibizione/oblio | vs `linoss` (solo lo smorzamento) |
| `dlinoss-phi` | 4 metastabilità | vs `dlinoss` (solo l'init) |
| `hyb-oa` (osc sotto, attn sopra) | 5 gerarchia | vs `dlinoss` e vs baseline |
| `hyb-ao` (inverso) | 5 direzione | vs `hyb-oa` |
| `wrnn` | onde vs oscillatori | vs `linoss` |

**Blocco (deviazione dichiarata dal blocco ufficiale LinOSS).** Scheletro transformer
identico per tutti i bracci (pre-norm → mixer → residuo → FFN → residuo): si scambia
*solo il mixer*. Il blocco ufficiale LinOSS (readout→GELU→GLU, niente FFN) è scartato
perché confonderebbe meccanismo di mixing e struttura del blocco (principio HRM/TRM).

**Mixer oscillatorio (equazioni paper-esatte, verificate 2026-08-18 sui full text).**
LinOSS-IMEX (arXiv:2410.03943): `z[k+1]=z[k]+Δt(−A·x[k]+B·u[k+1])`,
`x[k+1]=x[k]+Δt·z[k+1]`; A=ReLU(Â) init U[0,1], Δt=1 fisso. D-LinOSS
(arXiv:2505.12171): smorzamento *implicito* `z[k+1]=(z[k]+Δt(−A·x[k]+B·u[k+1]))/(1+Δt·G)`;
G=ReLU(Ḡ), Δt=σ(Δt̄), A clampata nella finestra di stabilità; init: autovalori
nell'anello complesso raggio [0,9, 1], angolo uniforme. Nota di merito: LinOSS puro ha
|λ|=1 (non può dimenticare — Funes), D-LinOSS apprende |λ|<1: l'ablazione 1-vs-2 misura
esattamente "l'oblio serve al linguaggio?". Parallelizzazione: prefix scan associativo
su matrici 2×2 diagonali (op binaria `(a₁,a₂)•(b₁,b₂)=(b₁∘a₁, b₁∘a₂+b₂)`), ~9 livelli
di raddoppio a 512 token, implementazione nostra in PyTorch. Parità: m=2d=512 oscillatori
per layer → B,C ≈ 262k ≈ i 4d² dell'attention; bracci puri senza position embedding
(l'ordine è nella ricorrenza; budget respendibile, come D6 prevede).

**Init aurea (`dlinoss-phi`).** Cambia solo la distribuzione degli *angoli* (=frequenze
in rad/token, D4): bande con periodi centrali in progressione aurea — periodi di
Fibonacci 377, 233, 144, 89, 55, 34, 21, 13, 8, 5, 3 token — jitter uniforme
intra-banda, ripartizione uniforme degli oscillatori tra bande; raggio come il default.
Dichiarato: è un prior, non un vincolo — il probe spettrale a fine run verifica se la
struttura a bande sopravvive al training.

**Ibridi.** 4+4 layer, mixer D-LinOSS (init default: le combinazioni con φ sono materia
1b) + attention della baseline; position embedding attivo (serve all'attention).

**wRNN (arXiv:2309.08045, verificato).** Mixer a campo d'onda: stato = campo circolare
N=512 (c=16 canali × 32), `h[t]=ReLU(u★h[t−1]+V·u[t]+b)`, kernel k=3 init shift-matrix
(ν=1), V init a iniezione puntuale — ricette del paper. Deviazioni dichiarate: impilata
nello scheletro a 8 layer (il paper è mono-layer); ricorrenza non lineare → niente scan,
loop sequenziale: uno smoke di velocità su T4 precede e condiziona il calendario dei
suoi seed. Il paper non ha alcun task linguistico: gap confermato.

**Percorso operativo (pre-registrato).** Smoke M2 per arch → smoke di velocità Kaggle
(gruppo bench) → sweep lr per *tutti e sei* i bracci (3 lr × 20M × 1 seed, D6) →
griglia 3 seed × 170M (D8), gruppo grid-stage1, checkpoint HF. Verdetti: asse loss
(permutazione 3v3, ε=0,007 dalla baseline D11) + asse Elo (campagna giudice in un'unica finestra Batches)
→ tabella D7 per ogni braccio vs baseline; i confronti interni (dlinoss vs linoss,
φ vs uniforme, oa vs ao) sono secondari, stesso test, dichiarati come tali. Costi:
5 bracci veloci ≈ 5h GPU inclusi sweep; wrnn secondo smoke.
**Scartato.** Blocco paper-faithful (confonde due variabili); φ sugli ibridi e gradiente
di frequenze per profondità (combinazioni → 1b); ibrido a direzione singola (la direzione
della gerarchia è domanda empirica); Wave-RNN esclusa per costo (si è scelto di includerla
accettando run lunghe).
**Riconsiderare se.** Lo smoke wrnn desse tok/s da rendere impraticabili 3×170M nella
quota → i suoi seed slittano o il braccio decade a esperimento separato (si dichiara);
instabilità numeriche dello scan in 16-mixed → fallback a precisione fp32 del solo scan,
dichiarato.

### D11 — Esiti smoke/sweep, revisione lr (clausola D6) e griglia 1a snellita
**Contesto.** Implementati i sei bracci (review avversariale: 3 difetti corretti pre-run,
tra cui init V della wrnn perso per copia e scan fp16 → fallback fp32 di D10 cablato).
Smoke e sweep su istanza vast dedicata (GPU singola, batch 32 = config storica D6).

**Revisione D6 — lr per categoria.** La clausola del bordo (vincitore sull'estremo della
griglia + trend monotono) è scattata per *tutte* le categorie, baseline inclusa: a 20M la
baseline passa da 3,67 (lr 1e-3 congelata ieri) a 2,41 (3e-2), i bracci guadagnano
0,5-1,2 nats oltre il vecchio bordo. Ricette congelate al punto di svolta misurato
(NaN/divergenza alla lr successiva): **dlinoss 1e-2 · dlinoss-phi 3e-3 · hyb-oa 1e-2 ·
hyb-ao 3e-3 · transformer: argmin tra 3e-2 (2,41) e sonde 1e-1/3e-1 in corso**.
Conseguenza: i 5 seed baseline e ε=0,017 (D8, registro base-1..5) si rifanno alla nuova
lr sulla stessa macchina della griglia; le righe vecchie restano come storia.

**Verdetti dai bracci malati (evidenza smoke/diagnosi, 5-20M):**
- **linoss puro — il fallimento È il risultato dell'asse 2.** In 16-mixed: NaN (overflow
  fp16 dello stato, |λ|=1 → crescita polinomiale su 512 token). In fp32: niente NaN ma
  impara pochissimo (6,35 vs 5,03 di dlinoss a 5M) a 3,6× il costo. L'ablazione
  "l'oblio serve al linguaggio?" ha risposta affermativa già qui. Nel registro entra una
  run fp32 dichiarata a budget sweep (20M); niente run 170M (−9h, informazione ~nulla).
- **wrnn — negativo di porting.** Non impara a nessuna lr (1e-3 diverge; 3e-4 ~random;
  1e-4 sopra il random): il campo ReLU con shift senza scarico accumula energia; il paper
  non aveva task linguistici (gap dichiarato in D10). Registrato com'è; una variante
  riparata (leak/norm sul campo) è materia esplicita della 1b, non un fix silenzioso.

**Griglia snellita (emendamento al percorso D10):** baseline 5 seed + hyb-ao, hyb-oa,
dlinoss × 3 seed × 170M; **dlinoss-phi è gated**: parte solo se dlinoss mostra segnale
(o parità) contro la baseline — altrimenti scala a domanda 1b. Tutta la griglia su una
sola istanza vast (GPU singola, hardware dichiarato nel registro): confronto interno,
stessa macchina per tutti i bracci.

**Primi segnali (1 seed, 20M — da confermare coi 3 seed):** dlinoss al passo della
baseline a parità di lr-ottima; **hyb-ao > hyb-oa a ogni lr** (2,80 vs 3,01 al meglio):
la gerarchia "anti-biologica" batte la direzione suggerita dal cap. 1. Ipotesi di lavoro
(testabile col probe name-cloze): gli oscillatori sotto sfocano l'identità dei token che
l'attention deve recuperare; sopra, integrano il contesto senza distruggere il retrieval.
**Scartato.** Riparare linoss (clamp/init-floor: cambierebbe il meccanismo che l'ablazione
vuole misurare); sweep wrnn esteso; 170M per linoss.
**Riconsiderare se.** dlinoss coi 3 seed contraddicesse il segnale a 20M → phi decade a
1b senza run; la sonda transformer 1e-1/3e-1 vincesse ancora sul bordo → si accetta il
bordo residuo dichiarandolo (il rendimento marginale per decade è già in calo).

**Emendamento (2026-08-19, mattina) — il bordo a 20M non regge a 170M.** Le sonde
transformer hanno chiuso il suo bordo (1e-1 e 3e-1 → NaN, lr baseline = 3e-2). Ma in
griglia hyb-oa@1e-2 (NaN 2/2 seed, terzo interrotto: informazione nulla) e
dlinoss@1e-2 (NaN 1/1) sono divergenti sull'intera run: 8,5× più step ad alta lr
fanno emergere instabilità invisibili a 20M. Colpiti esattamente i bracci congelati
sull'ultimo lr stabile dello sweep; illesi quelli già un gradino sotto il proprio
bordo (transformer, hyb-ao — che pure mostra fragilità: seed 3 a 2,07 vs 1,68 dei
fratelli). **Regola aggiunta alla ricetta:** se il vincitore dello sweep è all'ultimo
lr stabile della griglia, la run di griglia scende di un gradino: dlinoss → 3e-3
(rilanciato), hyb-oa → registrato instabile alla lr pre-registrata (le due run NaN
sono il suo dato; eventuale ripescaggio a 3e-3 è materia 1b, dichiarata).

**Findings di griglia (2026-08-19).** Il quadro con la baseline finalmente alla sua
lr onesta:
1. **Titolo provvisorio dello stadio 1**: a parità di parametri, budget e lr per
   categoria, su TinyStories a 512 token **gli oscillatori non battono l'attention** —
   baseline 1,599±0,007 · hyb-ao 1,68 (seed sani) · dlinoss 1,93. Segno dichiarato,
   come D6-parità impone.
2. **L'oblio è necessario ma non sufficiente (asse 2, ablazione più pulita della
   griglia)**: linoss (|λ|=1, non può dimenticare) è inaddestrabile in ogni regime
   provato; dlinoss, identico salvo lo smorzamento appreso, si addestra sempre. La
   funzione computazionale dell'inibizione (Castellanos cap. 1) è confermata nel senso
   forte — ma dimenticare bene non basta a vincere.
3. **Gerarchia (asse 5): il verso "biologico" perde** (hyb-oa instabile/peggiore di
   hyb-ao a ogni condizione provata). Ipotesi di lavoro registrata: sui token BPE — già
   simboli — il filtro oscillatorio all'ingresso può solo sfocare le identità che
   l'attention deve recuperare; il posto naturale del concetto è la granularità
   char-level (Q3), dove l'input è un flusso quasi-continuo. Testabile col probe
   name-cloze.
4. **Tema emerso non pre-registrato: l'omeostasi mancante.** Il filo rosso operativo
   (NaN al bordo, seed semi-esplosi, lr-bordo che non regge sulla distanza) è che i
   nostri sistemi dinamici non si autoregolano: la metastabilità richiede regolazione
   attiva, non un init fortunato. Candidato primario per la 1b insieme a φ (gate
   fallito ma non falsificato: mai corso a 170M) e alla wrnn riparata (il suo campo
   senza scarico è lo stesso difetto: manca l'oblio).

**Chiusura anticipata (decisione, 2026-08-19 sera).** dlinoss s2 interrotta a metà
corsa (val 2,39, traiettoria coerente con s1) e s3/linoss-registro cancellate: nessun
verdetto qui sopra cambierebbe con 3-4 ore in più di seed. Il **gate phi è FAIL per
aritmetica**: con s1 = 1,932, la media 3-seed non può scendere sotto la soglia 1,606
se non con seed che batterebbero la baseline stessa — dlinoss-phi resta materia 1b.
Al posto della coda gira il **controllo lr-matched**: baseline @3e-3 × 170M × 2 seed
(~15 min), per separare nel gap di hyb-ao (+11ε) il deficit di espressività (attention
= richiamo per contenuto, che un sistema LTI non fa) dal tetto di ottimizzazione
(hyb-ao corre a lr 10× più bassa della baseline). Se baseline@3e-3 ≈ 1,68 il gap è
ottimizzazione; se ≈ 1,60 è espressività.

**Esito del controllo (ctrl3-1..2): baseline@3e-3 = 1,692/1,709, media 1,700.** Il
verdetto si sdoppia, e raffina il finding 1:
- **hyb-ao (1,68 sui seed sani) ≥ baseline a lr pari (1,700)**: l'intero gap
  dell'ibrido dalla baseline D11 è **tetto di ottimizzazione**, non espressività — a
  parità di lr l'ibrido non perde nulla, anzi. Gli oscillatori in cima non costano
  capacità; costano la lr che il resto del modello potrebbe permettersi (10×).
- **dlinoss (1,932) resta a +0,23 anche dalla baseline lr-matched**: per l'oscillatore
  puro il deficit di espressività è reale — coerente con l'ipotesi LTI (nessun
  richiamo per contenuto), testabile col probe name-cloze.
Il titolo dello stadio 1 si riformula: non "gli oscillatori perdono", ma **"gli
oscillatori pagano due tasse distinte: una di addestrabilità (l'ibrido paga solo
questa) e una di espressività (solo l'oscillatore puro)"**. Per la 1b la tassa di
addestrabilità è il bersaglio dell'omeostasi; quella di espressività, della
selettività/gating.

---

### D12 — Design griglia 1b: due tasse, due bersagli (chiude Q5)

**Decisione.** La 1b attacca la tassa di addestrabilità (l'unica che l'ibrido paga) e
prepara il terreno con apparato migliore. Quattro passi in sequenza:

- **Pre-step — autopsia spettrale dei checkpoint 1a** (locale, CPU, costo ~0): dai
  pesi addestrati ricavare gli autovalori appresi (r, θ) per layer e confrontarli con
  l'init; misurare la saturazione del clamp (parametri che spingono contro la finestra
  di stabilità); confrontare il seed degenerato di hyb-ao (2,07) coi seed sani; leggere
  i checkpoint pre-NaN di hyb-oa/dlinoss@1e-2 (upload periodico) per la firma
  dell'instabilità; verificare con le norme di B/C se in hyb-ao il modello *usa* gli
  oscillatori o li ha silenziati (controllo dell'interpretazione "l'ibrido non perde
  nulla").
- **Fase 0 — apparato** (prima di ogni run 1b): (a) parametrizzazione log-polare LRU
  (r = exp(−exp(ν)): r < 1 per costruzione, update moltiplicativi ben condizionati al
  bordo) cablata in OscMixer per tutti i bracci; (b) fusione dello scan — spike
  `torch.associative_scan`, poi eventuale kernel Triton (bottleneck misurato: ~72
  lanci sequenziali di kernel, revert 3a04dfe). Vincolo: ogni modifica validata con
  A/B di training su GPU, non con equivalenza CPU.
- **Fase 1 — bracci**: (a) dlinoss log-polare + sweep lr (quanto della tassa di
  addestrabilità paga la sola parametrizzazione?); (b) omeostasi (regolazione attiva
  di r dalle statistiche di attivazione, analogo dello scaling sinaptico) **contro il
  controllo log-polare**, non contro il dlinoss 1a — altrimenti il merito della
  regolazione attiva non è isolato (meccanismi-non-estetiche); (c) ibrido intercalato
  A-O-A-O (terzo punto dell'asse gerarchia); (d) ripescaggio hyb-oa@3e-3 (già in D11).
- **Fase 2 — asintoto**: i vincitori di fase 1 a 536M token contro la curva pilot
  della baseline (1,509 a epoca piena).

**Perché.** Il controllo lr-matched (ctrl3) mostra che a lr pari hyb-ao ≥ baseline: la
strada più corta verso "gli oscillatori aiutano" passa dal rimuovere il tetto di lr,
non dall'aggiungere capacità. La fusione dello scan (ibridi oggi a 50k tok/s vs 439k
della baseline) riduce di ~4× il costo di ogni braccio futuro e rende economica la
fase 2.

**Scartato.** CUDA scritto a mano (il collo è il numero di lanci, non la matematica);
attaccare subito la tassa di espressività con selettività/gating stile Mamba —
dichiarata fuori scope, è il candidato naturale di una 1c.

**Riconsiderare se.** L'autopsia spettrale smentisce le premesse (es. hyb-ao ha
silenziato gli oscillatori: allora "ibrido ≥ baseline a lr pari" non dice nulla sugli
oscillatori e la 1b va ridisegnata); o se log-polare da sola chiude la tassa di
addestrabilità (allora l'omeostasi perde il bersaglio principale e resta solo come
claim biologico).

**Esito pre-step — autopsia spettrale (2026-08-19, 8 checkpoint HF).**
1. **Il training tira r verso il basso, con forza**: dlinoss sano converge a r mediana
   0,74 (layer 0) → 0,90 (layer 7) dall'init 0,95; orizzonti effettivi 4-10 token,
   frazione r>0,99 ≤ 4%. Il modello scarta la memoria lunga che l'init gli regala e
   tiene filtri locali — coerente con la tassa di espressività (un banco LTI non sa
   *usare* il passato remoto per contenuto, quindi lo butta).
2. **Emerge una gerarchia di scale temporali con la profondità** (r cresce
   monotonicamente col layer, entrambi i seed dlinoss): eco diretta del gradiente di
   timescale intrinseche lungo la gerarchia corticale (Murray et al. 2014). Non
   pre-registrato; da citare come osservazione, non come conferma.
3. **La firma del guasto è la perdita dello smorzamento**: dlinoss@1e-2 chiude con il
   40-60% degli oscillatori dei layer alti a r>0,99 (mediana 1,0000) e ||B||,||C||
   gonfiate ~3×; il seed degenerato di hyb-ao (2,07) mostra la stessa firma in forma
   graduata (29% a r>0,99 nel solo ultimo layer). Stesso meccanismo, dose diversa →
   l'omeostasi 1b ha un bersaglio misurabile: tenere r lontano da 1.
4. **In hyb-ao gli oscillatori NON sono silenziati**: ||C|| ≈ ||B|| ≈ 30, come nel
   dlinoss sano — la premessa di D12 regge (caveat: la norma non prova il contributo
   funzionale; la prova forte sarebbe l'ablazione del mixer a eval).
5. **hyb-oa è morto all'init**: al checkpoint step 3000 i suoi parametri spettrali
   erano ancora alla distribuzione iniziale (frac r>0,99 = 10% = valore esatto
   dell'init U[0,9;1]) e C ferma all'init (‖C‖=9,3 = valore atteso), mentre dlinoss
   allo stesso step aveva già portato r a 0,80-0,90. La stabilità è una **corsa**:
   imparare a smorzare prima di esplodere — con gli oscillatori in fondo allo stack a
   lr 1e-2 la corsa si perde.
Implicazioni di design accolte in fase 1: (e) braccio quasi gratis **init
post-autopsia** (r ~ U[0,7;0,9]: partire dove il training converge elimina il
transito pericoloso vicino al bordo); nota per il gate φ: le frequenze θ si muovono
poco in training (mediane dei periodi ~4 token ovunque, si muovono solo le code) —
se le bande φ sopravvivessero, potrebbe essere inerzia, non utilità: il probe
spettrale resta obbligatorio prima di ogni claim.

**Esito fase 0 — apparato (2026-08-19, sera, su RTX 3060).**
1. **Log-polare**: `dlinoss-lp` implementato (r = exp(−exp(ν)), θ = π·σ(θ̄), Δt fisso;
   autovalori esatti r·e^{±iθ}, parità −0,9%); smoke di training sano.
2. **Scan fuso: adottato, 9,7× sul modello intero** (10,5k → ~101k tok/s su 3060).
   Strada tortuosa e istruttiva: il backward che Inductor genera per
   `torch.associative_scan` (modo generic) è **rotto quando il combine contiene
   matmul/einsum** — gradienti ~100% errati, solo compile+CUDA, invisibile a ogni test
   eager-mode (bug di PyTorch, non nostro). Prima versione: +0,27 di val (7× il
   rumore) e un seed NaN. Fix: combine 2×2 in aritmetica elementwise esplicita → 
   gradienti corretti (2e-4 vs fp64), scan 1067→8 ms. A/B finale su 2 seed:
   hoo 3,540/3,463 vs eager 3,660/3,624 — nessuna regressione (semmai meglio, entro
   2-4× il rumore). Switch esplicito `NEURO_SCAN=hoo` nei lanci 1b, default eager.
3. **bf16-mixed: respinto dall'A/B** (previsione teorica falsificata): NaN a step
   ~300-319 a traiettoria già sana. Spiegazione: in 16-mixed il GradScaler controlla
   inf/NaN nei gradienti e **salta lo step** — un omeostata primitivo che scarta i
   colpi di coda della dinamica oscillatoria; bf16 non lo ha e il primo spike avvelena
   i pesi. Ricetta 1b: resta 16-mixed (scan sempre fp32). Rima col tema omeostasi.
4. **Regola di metodo acquisita: ogni A/B di training misura anche il pavimento di
   rumore seed-a-seed** (qui 0,037): senza, il backward rotto sarebbe passato — o un
   equivalente sano sarebbe stato respinto. Il criterio si fissa prima di vedere i
   dati.

**Primo esito fase 1 — sweep lr dlinoss-lp @20M (2026-08-19, RTX 4070TiS, 231k
tok/s):** 3e-3 → 2,642 · 1e-2 → 2,386 · **3e-2 → 2,282** · 1e-1 → NaN. La sola
parametrizzazione log-polare porta il tetto di lr dell'oscillatore puro **al livello
della baseline** (3e-2, un ordine sopra il 3e-3 forzato in 1a; il bordo è 1e-1, come
per il transformer): a 20M la tassa di addestrabilità è pagata dal formato, senza
regolazione attiva. Bracci 170M in gruppo grid-1b, batch 16 come lo sweep. Emendamenti
in corsa: **2 seed per braccio** (decisione a s1/s2 lp quasi identici: 1,9498/1,9672 —
la stabilità è dimostrata, il terzo seed non cambia verdetti) e **niente ripescaggio
hyb-oa col mixer vecchio** — mai confermare cose nuove con apparato vecchio: al suo
posto hyb-oa-lp (ibrido log-polare) con proprio mini-sweep lr.

**Verdetto dlinoss-lp @3e-2 × 170M (s1 1,9498 · s2 1,9672, media 1,958):** il tetto
regge sulla distanza — primo oscillatore puro addestrato alla lr della baseline, zero
NaN. Ma la loss non migliora sul classico@3e-3 (1,932): l'addestrabilità guadagnata
non compra perplexity → il collo residuo è tutta espressività.

**Autopsia dlinoss-lp (entrambi i seed, quadro identico):** a lr piena l'architettura
degenera spontaneamente in **"banco di filtri all'ingresso + pila feedforward"**:
layer 0 sano (r med ~0,79, periodi ~9 tok con coda di quasi-integratori a p90 ~75k,
‖B‖,‖C‖ raddoppiate a ~57: capacità temporale concentrata lì); layer 1-7 con r
collassato a ~0,002 (memoria di un token) — alcuni silenziati del tutto (‖B‖,‖C‖≈0,
scavalcati dal residual), altri riciclati come trasformazioni istantanee. Stessa loss
del classico che tiene r distribuiti (0,74-0,90): valle piatta — **oltre un layer di
filtraggio temporale, la memoria LTI extra non compra nulla** (tassa di espressività
nella sua forma più cruda). E il modo di guasto a 3e-2 non è più l'esplosione (frac
r>0,99 = 0% ovunque): la log-polare elimina la deriva verso il cerchio; il fenomeno
nuovo è la potatura. **Il bersaglio "stabilità" dell'omeostasi è evaporato** — resta
solo la domanda biologica (regolazione attiva vs formato), da ripesare in D13 col
quadro completo.

**Finding principale della 1b (2026-08-19, sera) — la degenerazione è l'ottimo, non
un incidente.** dlinoss-lp-init s1 (init r ~ U[0,7;0,9]) chiude a **1,9450** con la
**stessa struttura** di dlinoss-lp (init U[0,9;1]): layer 0 banco di filtri (r med
0,78, ‖B‖,‖C‖ ~54), layer superiori potati (r→0,002-0,03) con un sopravvissuto
parziale al layer 7. Init diverse, seed diversi, stessa destinazione, stessa loss
(1,9450/1,9498/1,9672 vs classico 1,932): la riorganizzazione "filtro periferico +
pila feedforward" è **l'ottimo dell'architettura LTI su questo compito**, robusto al
punto di partenza. Braccio lp-init chiuso a 1 seed (nessuna informazione nel secondo);
figura: `docs/figures/2026-08-autopsia-spettrale-1b.png`. Lettura: l'ipotesi
"oscillazione distribuita su tutta la gerarchia" è falsificata dal gradiente stesso —
il modello, libero di scegliere, converge verso un'architettura nota
(filtro-in-periferia → elaborazione feedforward, coclea→cortex); ciò che manca alla
pila per usare la memoria oltre il primo layer è l'indirizzamento per contenuto →
motiva la direzione **selettività** (1c) sopra l'omeostasi.

**Secondo finding dell'autopsia (2026-08-19, sera) — la memoria LTI sopravvive solo
dove ha un consumatore capace di indirizzarla.** hyb-oa-lp s1 @3e-2 chiude a **1,7225**
(l'ordine "biologico" riabilitato: in 1a moriva NaN; ora a +0,04 dal vincitore hyb-ao
e con la curva ancora in discesa a fine budget → candidato asintoto). Autopsia dei 4
layer oscillatori (identica a step 15k e a fine run, quindi struttura stabile): layer
0 banco di filtri come nel puro ma a memoria *più corta* (r 0,69 ≈ 3 token: l'attention
sopra copre il lungo raggio — divisione del lavoro); e la potatura si ammorbidisce
avvicinandosi all'attention: sopravvissuti r>0,5 = 8% → 17% → **34%** al layer 3
(quello che alimenta l'attention), con ‖C‖ crescente in parallelo (14→21→27). Nel
dlinoss-lp puro, senza consumatore con indirizzamento, gli stessi layer erano deserti.
È la tesi della selettività letta nei pesi. Caveat: un solo seed, alla lr di bordo
(la gemella s2 è NaN — hazard confermato per la categoria ibridi a 3e-2 → coppie
@1e-2 in corsa). Figura aggiornata (5° pannello): `docs/figures/2026-08-autopsia-spettrale-1b.png`.

**Verdetto finale fase 1 (2026-08-19, notte) — parità, non vittoria; e la baseline era
sotto-tarata.** La coppia hyb-oa-lp @1e-2 (1,5742/1,5729, seed al millesimo) era scesa
sotto la baseline D11 (1,599); il controllo di parità totale (transformer alla ricetta
esatta dell'ibrido, b16@1e-2) ha però dato **1,5708/1,5454 (media 1,558)**: il claim
"batte la baseline" muore per mano del controllo pre-registrato. Verità acquisite:
(a) **la baseline D11 non era al suo ottimo** — il batch è un iperparametro accoppiato
alla lr (b16@3e-2 = 1,737/NaN: bordo più basso; b16@1e-2 = 1,558: meglio di b32@3e-2)
e va sweepato insieme alla lr anche per la baseline; (b) il verdetto onesto è
**parità entro il rumore** (1,574 vs 1,558, n=2, spread transformer-b16 0,025):
enorme rispetto alla 1a (hyb-oa era morto), ma non un sorpasso; (c) la **gerarchia si
è invertita** rispetto alla 1a in modo simmetrico (là ao 1,68 stabile e oa NaN; qui
oa 1,573 stabile e ao 1,727+NaN): il "verdetto architetturale" della 1a era in realtà
un verdetto di *ottimizzazione* — l'ipotesi filtro-che-sfoca è da ritirare; (d)
l'ibrido a fine budget scende ancora mentre il transformer satura → la domanda vera
passa all'**asintoto** (536M). Nessun numero della 1a cambia; cambia il riferimento
per la 1b+: baseline onesta = min su (batch, lr), da ri-stabilire in D13.

*Riproducibilità (bracci lp/lp-init 170M):* codice a commit `cd60dbc`; run W&B gruppo
`grid-1b` (id = run name, es. `dlinoss-lp-d256-L8-t170M-s1-lr3e-2`), checkpoint su HF
`neuro-llm-ckpt/<run_name>/last.ckpt`; ricetta: BPE 8k congelato (D3-tokenizer),
170M token, batch 16×512, lr 3e-2 (da sweep 20M, bordo a 1e-1), 16-mixed con scan
sempre fp32, `NEURO_SCAN=hoo` (scan fuso, esiti fase 0), torch 2.11.0+cu128,
RTX 4070 Ti Super (~231k tok/s, ~15 min e ~0,03 $ per run). Riferimenti: baseline
D11 1,599±0,007 (b32×2 su 4080), dlinoss classico 1,932 @3e-3.

---

### D13 — Chiusura griglia 1b: baseline onesta, asintoto, selettività (2026-08-19)

**Decisione.** (a) **Baseline onesta**: il riferimento del progetto diventa il minimo
su (batch × lr) — oggi 1,558 (b16@1e-2, 2 seed, spread 0,025); sonda b8@1e-2 in coda
per sapere se il trend del batch continua; ogni confronto futuro usa la ricetta di
parità totale (stesso batch, lr, token, eval), mai più solo la parità di parametri.
(b) **Fase 2 asintoto in corsa**: hyb-oa-lp vs transformer a ricetta di parità
(b16@1e-2), 536M token × 2 seed — la domanda viva: l'ibrido scendeva ancora dove il
transformer saturava; le curve si incrociano? (c) **Selettività promossa a direzione
della 1c**: motivata dai pesi (degenerazione filtro+feedforward nel puro; gradiente di
sopravvivenza 8→17→34% verso l'attention nell'ibrido — la memoria LTI vale solo con
un consumatore che la indirizza). (d) **Omeostasi archiviata come domanda biologica
senza bersaglio ingegneristico**: la log-polare ha eliminato la deriva verso il
cerchio (frac r>0,99 = 0% ovunque nelle autopsie 1b); resta legittima solo come test
"regolazione attiva vs formato", non prioritaria.

**Perché.** Il controllo di parità ha ucciso il claim di sorpasso (1,574 vs 1,558) ma
ha scoperto la baseline sotto-tarata — lezione di metodo che vale da sola la fase.
La gerarchia invertita (oa da morto a pari; ao da vincitore a fragile) dimostra che i
verdetti architetturali della 1a erano in parte verdetti di ottimizzazione: prima di
confrontare architetture bisogna dare a ciascuna la sua migliore dinamica.

**Scartato.** Omeostasi come braccio 1c (bersaglio evaporato); sweep ao-lp dedicato
(il suo segno a 1e-2 è chiaro e la direzione oa è quella viva); wrnn e φ restano
fuori (invariati da D11/D12).

**Riconsiderare se.** L'asintoto mostra incrocio delle curve (→ il claim di sorpasso
rinasce a budget maggiore e la 1c si disegna attorno a quello); la sonda b8 scende
ancora significativamente (→ la baseline onesta va ri-stabilita prima di ogni claim).

---

## Questioni aperte (fase di design, in corso)

- **Q3 — Granularità temporale come ablazione futura.** Char-level (~4× più passi, dipendenze
  stirate: test più severo per la memoria oscillatoria) o chunking appreso stile H-Net. Fuori
  dallo scope dello stadio 1; richiede baseline dedicate.

---

## Registro esperimenti

| ID | Data | Arch | Params | Token | Seed | val_loss | Note |
|----|------|------|--------|-------|------|----------|------|
| smoke-0 | 2026-08-17 | transformer | 8,5M | 8,4M (2 run, prova resume) | 1 | ~5,0 (non a convergenza) | Solo validazione pipeline su M2; nessun valore scientifico |
| base-1 | 2026-08-18 | transformer | 8,5M | 170M (D8) | 1 | 1,8427 | Griglia 1a, braccio baseline; DDP b16×2+compile |
| base-2 | 2026-08-18 | transformer | 8,5M | 170M (D8) | 2 | 1,8064 | idem |
| base-3 | 2026-08-18 | transformer | 8,5M | 170M (D8) | 3 | 1,8114 | idem |
| base-4 | 2026-08-18 | transformer | 8,5M | 170M (D8) | 4 | 1,8336 | idem (seed extra per σ) |
| base-5 | 2026-08-18 | transformer | 8,5M | 170M (D8) | 5 | 1,8390 | idem (seed extra per σ). I 5 seed: media 1,8266, **σ = 0,017 = ε di D7** |
| base2-1..5 | 2026-08-19 | transformer | 8,5M | 170M ×5 seed | 1-5 | 1,5910 / 1,6027 / 1,6006 / 1,5927 / 1,6089 | Baseline D11 (lr 3e-2, vast RTX 4080, GPU singola b32). **Media 1,599, σ = ε = 0,007.** Sostituisce base-1..5 come riferimento |
| hybao-1..3 | 2026-08-19 | hyb-ao | 8,55M | 170M ×3 seed | 1-3 | 1,6815 / 1,6795 / 2,0721 | Griglia 1a, lr 3e-3 (D11). Seed 3: instabilità parziale a metà run, mai recuperata — fragilità del braccio alla sua lr. Sui seed sani: +0,08 (~11ε) dalla baseline |
| hyboa-x | 2026-08-19 | hyb-oa | 8,55M | 170M | 1-2 | NaN / NaN | Divergente alla lr pre-registrata 1e-2 (stabile a 20M): il bordo non regge a 170M (emendamento D11). Terzo seed non eseguito (informazione nulla) |
| dlin-x | 2026-08-19 | dlinoss | 8,43M | 170M | 1 | NaN | Stesso pattern a lr 1e-2 → griglia rilanciata a 3e-3 (emendamento D11) |
| dlin2-1 | 2026-08-19 | dlinoss | 8,43M | 170M | 1 | 1,9320 | Griglia 1a, lr 3e-3 (regola un-gradino-sotto, emendamento D11); +0,33 (+47ε) dalla baseline D11. Seed 2-3 interrotti (chiusura anticipata: nessun verdetto cambiava) |
| lp-1..2 | 2026-08-19 | dlinoss-lp | 8,42M | 170M ×2 seed | 1-2 | 1,9498 / 1,9672 | Griglia 1b @3e-2 (lr della baseline: primo oscillatore puro a reggerla). Autopsia: degenerazione in filtro+feedforward |
| lpi-1 | 2026-08-19 | dlinoss-lp-init | 8,42M | 170M | 1 | 1,9450 | Griglia 1b @3e-2, init r~U[0,7;0,9]. Stessa struttura e loss di lp → degenerazione robusta all'init; braccio chiuso a 1 seed |
| hoalp-1 | 2026-08-19 | hyb-oa-lp | 8,55M | 170M | 1 | 1,7225 | Griglia 1b @3e-2 (bordo: s2 gemella NaN a step 4k — hazard). Morta di rete a step 16k e ripresa da HF. Curva ancora in discesa a fine budget. Autopsia: gradiente di sopravvivenza 8→17→34% verso l'attention |
| hoalp2-1..2 | 2026-08-19 | hyb-oa-lp | 8,55M | 170M ×2 seed | 1-2 | 1,5742 / 1,5729 | Griglia 1b @1e-2 (un-gradino-sotto). Due seed al millesimo; curva in discesa a fine budget (candidato asintoto) |
| haolp-1..2 | 2026-08-19 | hyb-ao-lp | 8,55M | 170M ×2 seed | 1-2 | 1,7268 / NaN | Griglia 1b @1e-2 (lr di categoria dallo sweep di oa — caveat: mai sweepata per ao). Gerarchia invertita rispetto alla 1a: ora è ao il fragile |
| b16-3e2 | 2026-08-19 | transformer | 8,5M | 170M ×2 seed | 1-2 | 1,7371 / NaN | Controllo batch: a b16 la lr 3e-2 è oltre il tetto della baseline (batch piccolo = rumore alto = bordo più basso) |
| b16-1e2 | 2026-08-19 | transformer | 8,5M | 170M ×2 seed | 1-2 | 1,5708 / 1,5454 | **Controllo di parità totale (ricetta dell'ibrido). Media 1,558: la baseline D11 era sotto-tarata** — b16@1e-2 batte b32@3e-2 (1,599). Spread tra seed 0,025 (≫ ε=0,007 di b32) |
| ctrl3-1..2 | 2026-08-19 | transformer | 8,5M | 170M ×2 seed | 1-2 | 1,6918 / 1,7090 | **Controllo lr-matched** @3e-3 (la lr degli oscillatori): media 1,700. Non braccio di griglia; separa espressività da ottimizzazione nel gap oscillatori-vs-baseline |
| pilot-1 | 2026-08-18 | transformer | 8,5M | 536M (1 epoch) | 1 | 1,509 (val completo @512) | Pilot per Q4, non braccio di griglia. BPB 0,531 @512 · 0,551 @256 (àncora: 0,4407). Curva: 100M→1,99 · 170M→1,80 · 260M→1,66 · 390M→1,56. Nota di metodo: la run dedicata da 20M dello sweep (3,67) chiude PEGGIO del punto 20M di questa curva (~3,4) — a piccoli budget l'annealing precoce costa più del rumore che toglie; i punti intermedi si leggono come stima centrale, non come limite |

Ogni run vera aggiunge una riga; i dettagli vivono su W&B (progetto `neuro-llm`), qui solo
l'essenziale per leggere la storia dell'esperimento senza aprire dashboard.
