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

### D10+D11 — Griglia 1a: design ed esiti (archiviati)

Dettaglio integrale in `docs/archive/2026-08-griglia-1a.md`. L'essenziale: sei bracci
(linoss, dlinoss, dlinoss-phi, hyb-oa, hyb-ao, wrnn) a parita' D6; baseline D11
1,8266->**1,599 ± ε 0,007** dopo revisione lr (b32@3e-2); verdetti: linoss
inaddestrabile (oblio necessario), wrnn negativo di porting, hyb-ao 1,68, hyb-oa e
dlinoss@bordo NaN a 170M (emendamento **un-gradino-sotto**: il bordo lr a 20M non
regge a 170M); controllo lr-matched (baseline@3e-3 = 1,700) -> finding **due tasse**
(addestrabilita' | espressivita'); gate phi FAIL per aritmetica.

### D12 — Griglia 1b: apparato log-polare, findings di autopsia, parita' (archiviata)

Dettaglio integrale in `docs/archive/2026-08-griglia-1b-e-fase0.md`. L'essenziale:
**apparato 1b** = log-polare (r=exp(−exp(ν)), tetto lr -> 3e-2 come la baseline) +
scan fuso `NEURO_SCAN=hoo` (9,7×; bug backward Inductor aggirato in elementwise) +
16-mixed (bf16 respinto: GradScaler = omeostata) + regola pavimento-di-rumore.
**Findings di autopsia**: (1) il training tira r giu'; (2) il puro degenera in "banco
filtri layer 0 + feedforward" robusto a init/seed (~1,95); (3) sopravvivenza della
memoria vicino all'attention = ricetta-specifica (ridimensionato in D13); figura a 5
pannelli in `docs/figures/`. **Esiti**: hyb-oa-lp 1,574/1,573 (da morto in 1a a
parita'), hyb-ao-lp 1,727+NaN (gerarchia invertita: il verdetto 1a era
ottimizzazione), controllo di parita' -> baseline sotto-tarata (b16@1e-2 = 1,558).


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

**Esito (2026-08-19, notte fonda) — entrambe le clausole si chiudono in negativo, il
verdetto è definitivo.** (a) **Asintoto: niente incrocio** — transformer 1,4965/1,4967
(spread 0,0002!) vs hyb-oa-lp 1,5040/1,5150 a 536M: il gap delle medie passa da 0,016
(170M) a 0,013 (536M), stabile entro il rumore. La parità è *strutturale*, non un
artefatto del budget. (b) **Sonda b8 = 1,6813**: il trend del batch si inverte (a b8
il rumore batte il beneficio degli step extra) → b16@1e-2 è il punto dolce e la
baseline onesta **1,558** è confermata difendibile. (c) **Autopsia 536M** (obbligo di
skill): l'invariante di tutte le autopsie regge — banco di filtri al layer 0 (r 0,70,
100% vivi) + potatura pesante + orizzonti corti — ma il *gradiente di sopravvivenza
verso l'attention* del checkpoint @3e-2 **non replica** (qui 100/28/51/5%: superstiti
nei layer 0-2, layer 3 morto): il finding "consumatore" è ridimensionato a
osservazione ricetta-specifica; la posizione dei superstiti è idiosincratica (coerente
con la valle piatta), solo il front-end è legge. **Titolo dello stadio 1**: a parità
totale di ricetta, l'ibrido oscillatori-sotto-attention **eguaglia** il transformer a
170M e 536M (parità robusta a due budget); l'oscillatore puro resta staccato (~+0,4);
i meccanismi sono documentati dalle autopsie. La direzione 1c-selettività eredita la
domanda: cosa serve alla memoria ricorrente per *superare*, non solo eguagliare.

---

### D14 — Giudice D7 in-sessione: emendamento di canale, non di giudice (2026-08-20)

**Decisione.** Il giudizio pairwise D7-valutazione si esegue anche via **giudici-subagent
in-sessione**: `prepare-elo` scrive i 188 corpi ciechi su file (byte-identici a quelli del
canale Batches: stesse permutazioni seedate, doppio ordine A/B), un workflow lancia un
giudice **Opus 5, effort medium** (modello ed effort pre-registrati, invariati) *per
verdetto*, ognuno in contesto vergine col solo prompt di sistema D7 verbatim e il suo
corpo; `resolve` (codice deterministico, fail-loud su qualunque cid mancante o fuori
enum) rimappa display→run e produce lo stesso `.results.jsonl` del canale API. La mappa
sigillata non è mai letta né dai giudici né dall'orchestratore.

**Perché.** (a) Nessuna `ANTHROPIC_API_KEY` richiesta; costo API zero. (b) Cecità
*rafforzata*: i giudici-subagent non conoscono il progetto, le architetture né le loss —
il giudice Batches condivideva almeno il contesto del prompt di sistema, questi nemmeno
sanno che esiste un confronto in corso. (c) Ogni verdetto è un contesto indipendente:
nessuna contaminazione tra giudizi. **Modifica contestuale alla generazione**: i 10
completamenti di uno stesso prompt escono in un solo forward batchato (generatori CPU
per-riga: semantica di campionamento invariata, ~10× meno forward); introdotta prima
della prima generazione 536M, quindi nessuna àncora invalidata.

**Scartato.** Giudice = orchestratore in-sessione (io): cieco sui testi ma sa troppo
(loss, priori di parità); scartato a favore dei subagent naive. Batches API: resta il
canale di riferimento se servirà una campagna con chiave.

**Limite statistico scoperto.** Il bootstrap pre-registrato di `analysis elo` è
clusterizzato sui seed: con **una sola coppia di run** il ricampionamento è degenere
(CI di larghezza zero, verdetto E± privo di senso). Per giri a coppia singola la
statistica corretta è il **sign test** sui verdetti (per-match e, cluster-safe,
per-prompt sui vincitori netti nei due ordini). La regola E+/E=/E− pre-registrata resta
valida per campagne ≥2 seed per braccio.

**Esito preliminare (esplorativo: 1 coppia s1-s1, budget 536M fuori dal pre-registrato
D8).** 188 verdetti: transformer 82 · ibrido 77 · tie 29 (sign test p=0,75); vincitori
netti per prompt 28 vs 30 su 94 (p=0,90); nessun segnale per strato (short p=1,00,
long p=0,75). **Il giudice cieco conferma la parità strutturale vista dalla loss** —
due misure indipendenti convergono. Coerenza del giudice: 60/94 prompt concordi nei due
ordini, 9 flip netti. Analisi tematica delle 188 motivazioni: i modi di fallire sono
**gli stessi, distribuiti a caso** (lessico dei giudici perfettamente simmetrico: chi
perde "deriva e sputa word salad", chi vince "resta ancorato", indipendentemente
dall'architettura); i loop degenerativi vivono quasi solo sui prompt lunghi (29/100 vs
3/88); qualità assoluta bassa per entrambi (~78% delle motivazioni apre con "both sets
are degraded" — regime atteso a 8,5M parametri, temp 1,0).

**Riconsiderare se.** Una campagna ≥2 seed per braccio (generazioni su GPU) ribalta o
raffina il verdetto; oppure se un giudizio su modelli distanti (es. ibrido vs oscillatore
puro) mostra che il giudice non discrimina nemmeno differenze di loss grandi (sanity
check del potere del giudice, mai ancora misurato).

---

### D15 — Stadio char: byte come tempo, fase come indirizzo (2026-08-20)

**Decisione.** Si apre un nuovo stadio sperimentale ("stadio char") che revisiona tre
decisioni congelate *nel solo perimetro del nuovo stadio* — lo stadio 1 resta chiuso e
valido nel suo apparato:

- **D3-tokenizer → niente tokenizer**: input a **byte grezzi** (vocab 256). L'embedding
  scende da 2,1M (BPE 8k, tied) a ~65k: ~2M di parametri passano dal lessico
  memorizzato alla dinamica. Il BPB diventa esatto per costruzione (nats/byte / ln 2).
- **D4-asse-token → byte=tempo**: non è un tradimento ma il compimento — token=tempo
  era il compromesso pratico, il byte è più vicino al tempo fisico (era l'istinto di
  Q3-granularità: "test più severo per la memoria oscillatoria").
- **D6-parità → parità lasca di parametri totali** (±10%) + parità di *ricetta*
  (batch×lr sweeppati per entrambi i lati, lezione della baseline sotto-tarata di D13)
  + FLOPs/byte riportati onestamente. Dichiarato in chiaro: a parità di testo le
  sequenze byte sono ~4-5× più lunghe — l'attention paga quadratico, lo scan lineare
  no; il regime char è l'habitat strutturale della ricorrenza ed è una scelta di campo
  motivata, non un trucco da nascondere.

**Filone concettuale (preambolo di stadio).** Il cervello non usa le oscillazioni per
ricordare: le usa per **indirizzare** — la fase come puntatore ordinale (phase
precession: Reddy Nat. Comm. 2021, Qasim Cell 2021), il ritmo come scheduler
scrittura/lettura (LTP al picco theta, LTD al trough; Daume Nature 2024), il **reset di
fase su confini** come segmentatore (theta sillabico: Giraud & Poeppel; modello
Hovsepyan Nat. Comm. 2020, dove senza reset il riconoscimento di parlato a rate
variabile collassa). Lo stadio 1 ha falsificato l'oscillazione-come-memoria a parità
stretta (parità strutturale a due budget, degenerazione del puro in banco di filtri);
lo stadio char testa l'**oscillazione-come-indirizzamento**. Conferma indipendente dal
lato ML (rassegna `docs/2026-08-rassegna-continuazione-1c.md`): tutto ciò che eguaglia
o batte l'attention rende la dinamica input-dipendente; e le RNN sviluppano soluzioni
oscillatorie solo se il compito rende utile un orologio (Pals, PLOS CB 2024) — al
modello va dato un *mestiere* per la fase, non solo l'organo.

**Tre meccanismi, ognuno con l'interruttore che lo isola** (regola: meccanismi, non
estetiche):

1. **Fase-come-posizione**: attention **senza** positional embedding (si rimuove
   `self.pos`, 131k), l'ordine fornito solo dal banco oscillatorio sottostante — il
   "front-end di filtri al layer 0" (unica legge di tutte le autopsie) promosso a ruolo
   di progetto. Interruttore: togli il banco → l'attention senza posizione perde
   l'ordine → collasso; rimettilo → recupero.
2. **Reset-su-confini**: b_t ∈ [0,1] appreso dai byte, ricorrenza
   s_t = (1−b_t)·r·R(θ)·s_{t−1} + B·u_t (resta affine tempo-variante → lo scan fuso
   sopravvive). Dopo il reset la fase conta i byte dall'ultimo confine = coordinata
   ordinale *dentro l'unità* (sillaba/parola). Tre bracci: nessun reset (LTI, controllo)
   · hard reset con θ≡0 (puro chunking alla H-Net) · phase reset (la forma oscillatoria
   continua). Se phase ≈ hard, conta la segmentazione e non il ritmo: si dice.
3. **Lettura a fase** (ereditata dalla candidata 1c-int della rassegna): oggi il readout
   butta metà stato (`linoss.py`, legge solo la componente x); lettura
   y = C(cos φ ⊙ x + sin φ ⊙ z) con φ content-dependent, init a zero = bit-per-bit
   il modello attuale (attribuzione pulita).

**Apparato di confronto a tre livelli.** (1) Primario: char-oscillatorio vs
**char-transformer baseline nostra** (D5 intatta), stessa ricetta sweeppata — verdetti
in BPB sullo stesso testo di validazione. (2) Cross-stadio: entrambi vs i modelli BPE
dello stadio 1 **via BPB** (àncore già misurate: stories15M 0,4407; pilot 0,531 @512)
— "confronto ancorato, non verdetto" (ricette diverse). (3) Qualitativo: giudice cieco
D14 sugli **stessi prompt congelati** (sono testo: l'apparato D7 è tokenizer-agnostico
per costruzione) + probe meccanicistiche: decodifica lineare della posizione-nel-chunk
dalla fase; curva di capacità multi-item vs rapporto di frequenze (predizione
quantitativa di Ursino 2022); autopsia spettrale obbligatoria su ogni braccio.

**Loss/training come oggetto di ricerca**: ammesso (es. loss ausiliaria di confine,
multi-scala) con disciplina — ogni loss ausiliaria è un meccanismo e riceve la sua
ablazione; la valutazione resta likelihood pura (BPB) + giudice + probe.

**Criteri di successo pre-registrati.** Vietato ogni criterio sotto il pavimento di
rumore misurato della nuova baseline (da misurare al primo sweep, analogo dello 0,025
di stadio 1). Criteri ammessi: separazione BPB dal char-transformer oltre il rumore ·
separazione binaria su una probe (posizione-dalla-fase decodificabile in un braccio e
non nell'altro) · collasso/recupero nell'interruttore fase-come-posizione · autopsia
(la degenerazione da banco-di-filtri resta la variabile dipendente che solo noi
abbiamo). Protocollo a due budget con lettura della pendenza (warning
Phase-Associative-Memory 2026: i macchinari di fase possono essere piatti in loss a
~8,5M con pendenza diversa).

**Scartato.** (a) Proseguire la 1c su BPE (transizione selettiva r/θ, curva di
sostituzione attention): non morta, parcheggiata in Later — il regime char è più
coerente col filone e più favorevole alla ricorrenza. (b) Phase-locking sostenuto tra
layer come routing: ridimensionato dalla letteratura (Schneider Neuron 2021);
sopravvivono solo eventi di fase transitori guidati dall'input. (c) Tokenizer char
appreso (tipo BPE piccolo): i byte grezzi eliminano ogni decisione di apparato.

**Riconsiderare se.** Il char-transformer baseline risulta intrattabile al nostro
budget (seq 4-5× più lunghe); o se il primo sweep mostra che a 8,5M il regime byte è
troppo povero per qualunque segnale (BPB baseline fuori scala rispetto all'àncora
0,4407); o se le probe dicono che la fase non è decodificabile nemmeno nel braccio
migliore — allora il filone indirizzamento va riformulato prima di spendere oltre.

---

### D16 — Griglia char: design operativo (2026-08-20)

**Backbone char.** Byte grezzi UTF-8, vocab 256; il delimitatore letterale
`<|endoftext|>` del testo grezzo si sostituisce col byte 0x00 (mai presente nel testo)
come EOT. d256, L8, head tied; **seq_len 1024 byte** (≈ metà finestra-testo dello
stadio 1: 512 token ≈ 2130 byte — l'estensione a 2048 è un punto di arrivo dopo il
collaudo, non di partenza: l'attention paga ×4). Parametri: ~65k embedding + ~6,3M
corpo (+262k pos, solo dove previsto) ≈ **6,4-6,6M**; parità lasca ±10% tra bracci
(D15). Budget: **B1 = 700M byte** (≈ testo dei 170M token di stadio 1), **B2 = 2,2B
byte** (1 epoca) per i vincitori — protocollo pendenza a due punti obbligatorio.

**Fasi della griglia** (2 seed per braccio, autopsia obbligatoria, ordine = costo
crescente di implementazione):

- **Fase 0-char (collaudo, oggi)**: pipeline byte (`train_bytes.bin` uint8 dal txt
  grezzo, su HF), smoke M2, collaudo su GPU piccola (throughput byte/s, memoria a
  t=1024, prime curve). Gate: si prosegue solo con throughput ibrido ≥ ~30k byte/s.
- **Sweep ricetta char-baseline**: batch×lr a budget corto (~200M byte), poi 3 seed
  alla ricetta vincente → **pavimento di rumore σ_char** (l'analogo dello 0,025 di
  stadio 1; nessun criterio di successo sotto questo numero).
- **Fase A — fase-come-posizione** (l'interruttore più informativo per $ speso):
  `cb` char-transformer con pos emb (baseline) · `cb-nopos` senza pos emb (controllo
  negativo) · `osc0-nopos` layer 0 = banco oscillatorio log-polare + 7 layer attention
  senza pos emb (l'ordine lo fornisce la fase) · opz. `osc0-pos` (misura ridondanza).
  **Caveat dichiarato**: la letteratura (Haviv 2022; NoPE, Kazemnejad 2023) mostra che
  i decoder causali senza PE imparano comunque posizione dalla mask — quindi il
  controllo potrebbe non collassare; il verdetto è il confronto a tre, non il collasso:
  fase-come-posizione è supportata se osc0-nopos ≥ cb (entro σ_char) E > cb-nopos
  (oltre σ_char), con la probe a confermare che la posizione è *nella fase*.
- **Fase B — reset-su-confini** sull'ibrido char (hyb-oa-lp-char): 3 bracci — LTI
  (controllo) · hard reset θ≡0 (puro chunking) · phase reset — gate b_t ∈ [0,1] da
  conv causale sui byte (kernel ~7), per-gruppo. **Flag ingegneristico**: col gate la
  transizione diventa batch-dipendente → prima implementazione nello scan eager
  (corretto per costruzione), ottimizzazione del path `hoo` solo dopo il collaudo (il
  bug backward di Inductor è un precedente: prudenza motivata da evidenza).
- **Fase C — lettura a fase** sullo stack puro char (dlinoss-lp-char): LTI vs
  phase-read, init a zero = identità bit-per-bit col controllo.

**Probe e misure.** BPB = loss_nats/ln2 (esatto a livello byte); giudice cieco D14 sui
prompt congelati (generazione byte: max_new ~900 byte ≈ i 200 token attuali; serve il
codec byte in `src.eval.generate`); **probe posizione-nel-chunk**: regressione lineare
dai 2D-state della fase a "byte dall'ultimo confine (spazio/punteggiatura)" — la
capacità di decodifica deve separare i bracci; autopsia spettrale su tutti i bracci.

**Costi stimati** (ricaricabile; collaudo su GPU ~0,05 $/h, griglia su classe 4090
~0,35 $/h): collaudo <0,5 $ · sweep+pavimento ~1,5 $ · fase A ~2 $ · fase B ~2,5 $ ·
fase C ~1 $ · pendenza B2 vincitori ~3-4 $.

**Scartato.** Partire a seq 2048 (costo attention ×4 prima ancora del collaudo);
implementare subito il gate nello scan fuso (precedente Inductor); tokenizer char
appreso (D15: i byte eliminano ogni decisione di apparato).

**Riconsiderare se.** Throughput ibrido < ~30k byte/s anche su GPU seria (griglia
intrattabile → ridisegno budget); `cb-nopos` ≈ `cb` oltre ogni dubbio (il controllo
negativo di fase A evapora → il verdetto si sposta interamente sulla probe di
decodifica); BPB della char-baseline fuori scala rispetto all'àncora 0,4407 (regime
byte troppo povero a ~6,5M parametri).

---

## Questioni aperte (fase di design, in corso)

- (nessuna — Q3-granularità chiusa in D15-stadio-char)

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
| as-t1..2 | 2026-08-19 | transformer | 8,5M | **536M** ×2 seed | 1-2 | 1,4965 / 1,4967 | Fase 2 asintoto, ricetta di parità b16@1e-2. Spread tra seed 0,0002 |
| as-h1..2 | 2026-08-19 | hyb-oa-lp | 8,55M | **536M** ×2 seed | 1-2 | 1,5040 / 1,5150 | Fase 2 asintoto, stessa ricetta. **Niente incrocio**: gap medie 0,013 (era 0,016 a 170M) — parità stabile col budget |
| b8-1 | 2026-08-19 | transformer | 8,5M | 170M | 1 | 1,6813 | Sonda batch: a b8 il trend si inverte (rumore > beneficio step) → **b16@1e-2 è il punto dolce, baseline onesta 1,558 confermata** |
| ctrl3-1..2 | 2026-08-19 | transformer | 8,5M | 170M ×2 seed | 1-2 | 1,6918 / 1,7090 | **Controllo lr-matched** @3e-3 (la lr degli oscillatori): media 1,700. Non braccio di griglia; separa espressività da ottimizzazione nel gap oscillatori-vs-baseline |
| judge-s1 | 2026-08-20 | as-t1 vs as-h1 | — | — | 1 | — | **Giudizio cieco D14** (non training): 188 giudici Opus 5 in-sessione, doppio ordine. t 82 · h 77 · tie 29 (p=0,75); prompt netti 28 vs 30 (p=0,90) → **parità qualitativa, conferma la loss**. Preliminare (1 coppia, 536M). Artefatti: eval/judgments/elo-536M-s1.* |
| pilot-1 | 2026-08-18 | transformer | 8,5M | 536M (1 epoch) | 1 | 1,509 (val completo @512) | Pilot per Q4, non braccio di griglia. BPB 0,531 @512 · 0,551 @256 (àncora: 0,4407). Curva: 100M→1,99 · 170M→1,80 · 260M→1,66 · 390M→1,56. Nota di metodo: la run dedicata da 20M dello sweep (3,67) chiude PEGGIO del punto 20M di questa curva (~3,4) — a piccoli budget l'annealing precoce costa più del rumore che toglie; i punti intermedi si leggono come stima centrale, non come limite |

Ogni run vera aggiunge una riga; i dettagli vivono su W&B (progetto `neuro-llm`), qui solo
l'essenziale per leggere la storia dell'esperimento senza aprire dashboard.
