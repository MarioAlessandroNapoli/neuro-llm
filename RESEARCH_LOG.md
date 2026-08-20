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
come EOT. d256, L8, head tied; **seq_len 2048 byte**, deciso dall'analisi del dataset
(2026-08-20): storie TinyStories p50=720 byte, p90=1090, p99=2200 — 2048 contiene
intera il 98,6% delle storie (1024 solo l'88%, e soprattutto tronca il protocollo del
giudice: i prompt lunghi D7 arrivano a 1383 byte + ~900 di generazione = 2283
richiesti; a 2048 lo scorrimento perde solo ~235 byte di coda del prefisso a fine
generazione, dichiarato e uguale per tutti i bracci). 2048 ≈ la finestra-testo dello
stadio 1 (512 token ≈ 2130 byte): confronto cross-stadio a finestra quasi pari. Parametri: ~65k embedding + ~6,3M
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

**Esiti fase 0-char + sweep (2026-08-20 pomeriggio).** (a) **Collaudo 3090**: 450-467k
byte/s su tutti e tre i bracci di fase A (gate 30k superato 15×; il braccio osc0 non
paga nulla: 1 solo layer di scan su 8). Segnale precoce a ricetta di collaudo (lr 1e-3,
30M byte): osc0 1,84 vs baseline 2,30 — rimisurato in fase A a ricetta onesta.
(b) **Sweep lr** (b32, 200M byte): 3e-2→2,169 · **1e-2→0,639** · 3e-3→0,831 ·
1e-3→1,199; il regime char ha un **plateau a livello trigramma (~2,3 nats/byte ≈ 3,3
BPB)** che solo lr sufficiente attraversa entro il budget — la mappa è ripidissima.
(c) **Sonde batch @1e-2**: b64→2,041 · b32→0,639 · b16→0,506 · b8→0,490; le curve
b16-b64 si sovrappongono *in step* (siamo sotto il critical batch size: la moneta è il
numero di step, non i byte), b8 se ne stacca (transizione più tardiva, varianza
maggiore) — il critical batch del regime è tra 8 e 16. (d) **Pavimento a 200M**: b16 su
3 seed = 0,506/0,526/0,541 → **σ_char(200M) ≈ 0,018**, spread 0,035; il vantaggio di
b8 (0,014-0,016 sulle medie) è dentro il rumore e paga varianza doppia → **ricetta di
fase: b16@1e-2**. (e) **Gemelle cross-GPU** (stessa run b8-s2 su 3090 e 5090):
traiettorie sovrapposte per 12k step → il delta cross-scheda vive dentro il rumore di
seed, lo split della griglia su due GPU è legittimo. Caveat: la gemella 3090 è morta a
~metà run con Traceback non diagnosticato (esito finale perso; il confronto si regge
sulle traiettorie); la 5090 ha chiuso a 0,531, dentro il ventaglio dei seed b8.
(f) **Fase B implementata**: `prefix_scan_gated` con oracolo fwd/bwd a 7e-8 dal
sequenziale fp64, g=1 bit-identico allo scan LTI; gate conv causale k=7, 64 gruppi,
bias −4 (init quasi-LTI); bracci char-hyb / -hard(θ≡0) / -phase in parità (reset −0,9%,
controllo −7,6%: i parametri del gate sono il meccanismo, dichiarato). Codec byte in
`generate` (max 900 byte ≈ protocollo D7). Ops: la 3090 è morta due volte con crash
opachi (~7 min, traceback perso per colpa del filtro grep nel lancio — lezione: log
sempre integrale) → distrutta, griglia consolidata sulla 5090.

**Esito fase A (2026-08-20, 15:01-16:10, 7 run × 700M byte, b16@1e-2).**
L'interruttore fase-come-posizione ha funzionato **in entrambe le direzioni**:

| braccio | val loss (nats/byte) | media | probe posizione post-L0 (R²) |
|---|---|---|---|
| cb (pos emb) | 0,4148 · 0,4263 · 0,4146 | **0,419 (σ 0,007)** | 0,88 (residual) |
| osc0 (banco L0, no pos) | 0,4446 · 0,4115 | **0,428** | **0,90 dalla fase** (0,91 residual; 0,76 sola ampiezza) |
| nopos (niente) | 1,4485 · 2,0390 | — | **0,24** (residual) |

(1) **Senza posizione esplicita la transizione dal plateau a trigramma fallisce o
ritarda drammaticamente**: nopos resta a 2,04 (s2, mai transitato) o a 1,45 (s1, a
metà del guado) — la letteratura NoPE ("la mask causale basta") NON regge a 6,4M
parametri byte-level su questo budget; la sua posizione implicita è appena decodificabile
(R² 0,24). (2) **Il banco oscillatorio al layer 0 è un position encoder completo**:
parità di loss con la baseline (0,428 vs 0,419; divario 0,009 ≈ 1,3σ, sotto il criterio
pre-registrato) SENZA position embedding e con 260k parametri in meno; l'indirizzo
ordinale è leggibile linearmente **dalla fase** (R² 0,90 — phase precession artificiale,
il claim neuro di D15 operazionalizzato e verificato). (3) Il vantaggio osc0 del
collaudo (0,46 nats a 30M byte) era **velocità di transizione**, non asintoto: a 700M
la baseline raggiunge — coerente con lo stadio 1 (il vantaggio oscillatorio vive nella
dinamica di apprendimento). Caveat dichiarati: nopos-s1 stava ancora scendendo
(transizione ritardata ≠ impossibile); osc0 su 2 seed.

**Diagnosi NaN del braccio hard (fase B, 2026-08-20 sera).** fB-hard (θ≡0 + reset) in
16-mixed: loss patologica dall'init (44 vs ~4 in fp32) poi NaN in warmup. Matrice
diagnostica: hard@32-true sano · phase@16-mixed sano · compile scagionato (NaN anche
eager) → **interazione θ≡0 × fp16**. Causa: con θ≡0 il blocco 2×2 ha un doppio polo
reale in r (Jordan): la risonanza in continua guadagna ~1/(1−r)² ≈ 50-100× — senza
rotazione il DC dell'input si accumula coerentemente invece di mediarsi via; in fp32
il LayerNorm assorbe, in fp16 satura. **Il braccio hard gira a 32-true (dichiarato:
engine diverso dagli altri due)**. Lezione fisica in regalo: la rotazione non è solo
espressività — è anche il meccanismo che tiene le attivazioni piccole (media via il
DC); "un oscillatore è un filtro che non esplode in continua". Il banco NON è migrato verso
orologi lenti da posizione assoluta: r mediana 0,85-0,86 (p90 0,87-0,88; frac r>0,99 =
0), **orizzonte di memoria τ ≈ 7-8 byte**, periodi mediani 4 byte/giro (p90 21-25).
È una base temporale a **scala di parola**: τ coincide con la lunghezza mediana di una
parola TinyStories (~5-7 byte) e con il range della probe (posizione-nel-chunk, cap 16)
— il banco codifica la posizione *relativa alla parola corrente*, non quella globale
nella finestra; phase precession dentro finestre grandi quanto una parola. Coerente con
la legge di stadio 1 (front-end di filtri, r 0,74-0,88): stesso bacino, ma ora il
mestiere ha un nome. Implicazione per la fase B: il reset rende *esplicito* (confine
appreso) ciò che lo smorzamento fa *implicitamente* (oblio a τ≈7) — se phase-reset
non batte lti, una spiegazione candidata è che lo smorzamento breve è già un
"reset morbido" sufficiente.

**Esito fase B (2026-08-20, 16:48-18:30, 7 run × 700M byte, b16@1e-2): il reset duro
vince, la rotazione no.** Ibrido 4 osc + 4 attn, nessuna posizione esplicita:

| braccio | val loss (nats/byte) | media | probe posizione post-L0 (R²) |
|---|---|---|---|
| fB-lti (nessun reset) | 0,4548 · 0,4619 | 0,458 | 0,91 (fase) |
| fB-phase (reset + rotazione) | 0,4586 · 0,4573 | 0,458 | 0,94 (fase) |
| **fB-hard (reset, θ≡0)** | 0,4295 · 0,4251 | **0,427** | 0,95 ("fase" = rampa) |
| fB-lti32 (lti @32-true, controllo) | 0,4527 | — | — |
| cb (rif. fase A, pos emb) | — | 0,419 (σ 0,007) | 0,88 (residual) |

(1) **Il gate sui confini con θ≡0 — "dimentica quando vedi un confine" — vale 0,03
nats/byte (≈4σ)**: 0,427 contro lo 0,458 di lti e phase, a ~1σ dal transformer con
position embedding. (2) **Confound precisione escluso by design**: hard gira a 32-true
(diagnosi NaN); il controllo fB-lti32 (lti identica a 32-true) fa 0,4527 — dentro il
gruppo 16-mixed, non a 0,427: il residuo fp32 vale ≤0,004, un decimo del gap.
(3) **La rotazione non compra nulla, in nessuna combinazione**: phase = lti alla terza
cifra (col gate), e già la fase A aveva osc0 = cb (senza gate). Con reset, togliere la
rotazione *migliora*. (4) **Meccanismo (probe)**: la probe non discrimina i bracci
(tutti codificano la posizione-nella-parola a R²≈0,9) — il vantaggio di hard non è
*avere* l'indirizzo ma la sua *forma*: con doppio polo reale atan2(z,x) non è un angolo
che ruota ma il rapporto tra le componenti del blocco di Jordan, che cresce monotono dal
reset — una **rampa riavviata a ogni confine**, il contatore ordinale più letterale
possibile. Il gate duro rende esplicito e pulito ciò che lo smorzamento a τ≈7 byte
faceva in modo sfumato (la predizione dell'autopsia era giusta per phase — ridondante
col reset morbido — ma sottostimava il valore del reset *esatto*). Traduzione neuro
onesta: il vincitore non è l'oscillazione sostenuta ma il **segmentatore theta-sillabico
+ integratore a rampa** — più vicino a "reset di fase agli onset" (Giraud/Hovsepyan)
che a phase precession continua. Caveat dichiarati: 2 seed per braccio; hard su engine
32-true (quantificato dal controllo); budget 700M — l'asintoto B2 (2,2B, in corsa)
dice se il gap è di pendenza o di punto fisso.

**Test di estrapolazione in lunghezza (pre-registrato 2026-08-20, esecuzione post-B2
su 5090).** Domanda: il reset-su-confini è solo un modo alternativo di codificare la
posizione, o compra una proprietà che il position embedding appreso non può avere?
Protocollo: checkpoint B2 valutati a finestre 2048/4096/8192 senza riaddestrare
(`scripts/eval_estrapolazione.py`), nats/byte per bucket di posizione da 2048.
Predizioni: (a) cb è **strutturalmente incapace** oltre 2048 (la tabella non ha
indici — il test lo dichiara, quello è il risultato); (b) hard **degrada con grazia**
(coordinata rampa-dal-confine, invariante per traslazione: bucket >2048 ≈ bucket
<2048); (c) osc0 è il caso interessante intermedio (fasi LTI mai viste a
distanze >2048 — l'estrapolazione delle fasi è il test della qualità dell'orologio).
Verdetto: se (b) regge, il "fancy re-encoding" è falsificato — a parità di parametri
il meccanismo compra robustezza fuori distribuzione; se hard degrada quanto crolla
osc0/nopos, l'ipotesi scettica vince e si scrive col medesimo inchiostro.
**Anteprima sui checkpoint 700M (2026-08-20 sera, 1 seed/braccio, 64 finestre)**: le
tre predizioni confermate — cb incapace (tabella finita) · osc0 collassa (bucket
6144-8191: 1,175, +188% — il crollo è dell'attention su un candidato-set 4× mai visto,
lo scan LTI è invariante) · **hard regge: 4096 gratis (bucket lontano +0,008, totale
migliore che a 2048), 8192 a +13%** dove osc0 paga +188%. Il "fancy re-encoding" è
falsificato in anteprima: a parità di parametri il reset compra estrapolazione che il
pos emb appreso non può avere per costruzione. Caveat: manca il confronto esterno con
RoPE/ALiBi; ufficiale sui checkpoint B2.

---

## D17 — Tempo a eventi: convergenza sull'asse posizionale (2026-08-20, sera)

**Decisione.** Il progetto converge sulla tesi: *il reset di stato su confini è un
meccanismo posizionale — "il tempo scandito dagli eventi, non dall'indice" — e la sua
forma appresa e gerarchica è l'oggetto di studio*. Asse **posizionale**, non
compressivo (la compressione H-Net-style è satura; la nostra moneta è l'ablazione
controllata + estrapolazione, non i FLOP). Griglia in due mosse, ogni braccio misurato
su loss (700M), estrapolazione 2048→8192, probe, autopsia e metrica di allineamento
dei confini ℬ (SOMBRERO):

- **C1 — appreso vs euristico vs timescale** (le baseline che i reviewer esigeranno):
  (a) `hard-euristico` — reset cablato sui confini letterali (spazi/punteggiatura/EOT,
  il set BOUNDARIES della probe), gate senza parametri (delta di parità dichiarato);
  (b) `timescale` — gerarchia di sole scale temporali senza reset (Harmonic-style:
  init dei ν per layer su range scalati, nessun gate); contro `hard-appreso` (già in
  mano). Domande: il gate appreso batte lo spazio cablato? il reset aggiunge qualcosa
  sopra le scale imposte?
- **C2 — secondo livello di reset** (condizionata: solo se C1 promuove l'appreso E il
  corpus ha segnale alla scala frase — misurare prima BPIC del livello 1 e
  distribuzione lunghezze frasi TinyStories): gate al layer superiore sopra gli stati
  del primo, con le tre predizioni neuro pre-registrate come pagella (sotto).

**L'àncora neuro cambia mestiere: da giustificazione a generatore di predizioni con
pagella.** Ogni predizione neuro va registrata PRIMA dell'esperimento con esito
annotato. Pagella a oggi: byte=tempo ✓ · phase precession ✗ (rotazione inerte 2×) ·
reset theta-sillabico ✓✓ (fase B + estrapolazione) · smorzamento=reset-morbido ✓.
Pre-registrate per C2: (1) rapporto tra scale adiacenti ~5-15× (Hasson/Murray/Ding);
(2) reset di alto livello endogeno/top-down — ablando gli stati alti i gate devono
cambiare, se dipendono solo da punteggiatura è scorciatoia (Ding & Poeppel 2016);
(3) il canale top-down persistente tra i reset paga su ambiguità locale
(Kiebel-Friston) — contro-precedente dichiarato: in Harmonic l'accoppiamento
predittivo è rumore. Argomento di fondo: convergenza evolutiva come evidenza — quando
gradiente e corteccia scelgono la stessa soluzione (τ≈7 byte spontaneo; S4→Mamba =
lti→hard), il dato è sul calcolo, non sul substrato.

**Perché.** (1) Fase B: il reset appreso vale 4σ e la rotazione è inerte; anteprima
estrapolazione: hard regge (4096 gratis, 8192 +13%) dove cb è incapace per costruzione
e osc0 collassa (+188%). (2) Rassegna 4 fronti (docs/2026-08-rassegna-gerarchia-di-reset.md):
l'intersezione «SSM + confini appresi + reset di stato + estrapolazione/ablazione» è
vuota — H-Net comprime senza resettare (verificato nel codice), Harmonic ha gerarchia+
controlli ma timescale fisse, HM-RNN aveva il FLUSH ma è pre-SSM e senza controlli; le
cause della sua morte (stimatore instabile, cardinalità non controllata, euristica
vincente) sono state rimosse dal 2023-26. (3) La domanda residua è netta e nostra:
**il confine appreso batte la timescale fissa e l'euristica come meccanismo
posizionale?**

**Scartato.** Asse compressione (saturo: 72 citazioni H-Net in 13 mesi, industria
dentro, e il gruppo H-Net è migrato sul DNA); sintassi esplicita come livello
gerarchico (URNNG/ON-LSTM/StructFormer: ritorni marginali — si gerarchizza la
sorpresa, non la grammatica); fase C lettura-a-fase (declassata: la rotazione è
risultata inerte due volte, leggere a fase un orologio che vince da fermo non è più
la domanda); scala frase incondizionata (rischio no-signal su TinyStories, e SOMBRERO
mostra che forzare confini linguistici alti degrada).

**Autopsia del gate (fase B, stessa sera — domanda dell'utente: "sta convergendo a
una tokenizzazione classica?").** No — e la storia "reset a eventi" va corretta. Su
fB-hard-s2 (`scripts/autopsia_gate.py`, 32 finestre): il gate spara OVUNQUE (p media
~0,5 anche dentro le parole; recall 1,0 ma precision 0,24 sui confini letterali), con
modulazione ai confini (p 0,55→0,83 a L0) e 2-11 gruppi su 64 "portatori" (p<0,1,
memoria che scavalca le parole). Non è un tokenizer implicito (sarebbe: silenzio
intra-parola, colpo secco agli spazi — e comunque heu dimostra che quello vale zero):
è **oblio selettivo continuo per-canale, modulato dai confini** — il gradiente ha
trasformato il nostro rilevatore di confini in un gate selettivo alla Mamba, con in
più la struttura linguistica. Terza convergenza evolutiva (dopo τ≈7 spontaneo e
S4→Mamba=lti→hard). Il claim si raffina: "oblio selettivo appreso, allineato ai
confini"; la domanda per C2 si sposta: è la modulazione-ai-confini a essere portante,
o basta il decay selettivo? (controllo naturale: stesso gate cieco al contenuto).

**Emendamento post-bonifica (stessa sera).** La verifica sui vocabolari adiacenti
(rassegna § fronte 5) restituisce tre pezzi del claim alla letteratura: CoPE 2024
(posizione = conteggio di eventi, gate che si apre sui separatori), FoX ICLR 2025 (il
gate rende RoPE rimovibile a parità + estrapola 4× — lo schema argomentativo è già
pubblicato), GRAPE/Selective-RoPE (decay = posizione, formalizzato), Segatron 2021
(PE boundary-relative costruito a mano). **Vuoto verificato: "reset di stato =
posizione"** — il claim si riformula: *il reset ai confini produce spontaneamente una
coordinata posizionale locale, emergente (non progettata), su stato ricorrente (non
bias sui logit); l'estrapolazione va attribuita causalmente alla località della
coordinata*, non esibita come fattore (FoX ha già il 4×). C1 si arricchisce dei due
controlli obbligati: (a) probe della posizione ASSOLUTA dagli stati (attesa: bassa in
hard, alta in cb — il complemento della probe locale già fatta); (b) braccio
`cb-rel` alla Segatron: transformer con PE boundary-relative esplicito — se eguaglia
hard, il reset È la coordinata; se hard vince, il claim si restringe (igiene di stato
oltre la coordinata). Citazioni obbligate in rassegna § fronte 5. Il nostro claim di
partenza è un raffinamento dichiarato di Haviv 2022 (NoPE conta i predecessori): col
reset il conteggio riparte dal confine.

**Esito C1 (2026-08-20/21, notte — griglia completa in 6 run): lo spettro di
timescale è il meccanismo; confini, coordinata e gate sono comparse.** Decomposizione
a parità (700M, engine 32-true dove conta):

| braccio | cosa fornisce | val loss | estrapolazione (bucket 8192) |
|---|---|---|---|
| lti32 | nulla (ring 0,9-1) | 0,4527 | — |
| fC1-heu | confini cablati, reset totale | 0,4523 | — |
| fC1-rel (transformer) | SOLO coordinata dal confine | 0,7571 | — |
| fC1-ts | SOLO spettro τ multi-banda | 0,4357 / 0,4277 | **+15%** |
| fB-hard | gate appreso (spettro emergente + confini) | 0,4295 / 0,4251 | +13% |
| osc0 (rif.) | banda unica | — | +188% (collassa) |

(1) I confini da soli valgono zero (heu=lti32); la coordinata esplicita da sola resta
a metà strada (rel 0,757: sblocca la transizione da nopos 1,45-2,04 ma −0,33 da hard);
(2) **lo spettro di timescale da solo vale quasi tutto**: ts pareggia hard entro lo
spread dei seed (0,432 vs 0,427, seed intrecciati) E estrapola come lui (+15% vs +13%
a 4×; il contrasto interno osc0-vs-ts — banda unica collassa, quattro bande reggono —
inchioda lo spettro come meccanismo); (3) il gate appreso non aggiunge nulla di
misurabile sopra lo spettro imposto — e l'autopsia spiega perché: il gradiente aveva
USATO il gate per costruirsi uno spettro efficace (gruppi portatori + dimenticatori).
Harmonic sostanzialmente replicato su byte-LM ibrido (con estrapolazione verificata a
bucket); PoST confermato (lo spettro va imposto o costruito: l'init collassato di lti
non lo trova da solo). La predizione unexplored-states per ts è fallita: i canali
lenti fissi non sbandano a 4×. Pagella neuro aggiornata: gerarchia-di-timescale
(Murray/Hasson) ✓✓ · reset-theta-come-meccanismo ✗ al netto dello spettro (il
guadagno di fase B era lo spettro travestito). Conseguenza per C2: la gerarchia di
reset ANNIDATI perde il suo razionale a questa scala — la domanda utile diventa se
lo spettro basti anche all'asintoto (fB2-ts, in coda) e dove il gate torni a contare
(recall/MQAR, budget maggiori, o mai). Caveat: 2 seed, TinyStories, 6,8M.

**Addendum notturno (2026-08-21, 01-02): probe assoluta e autopsia spettrale
correggono la lettura di C1.** (1) *Controllo alla Haviv eseguito*: posizione ASSOLUTA
decodificabile da cb (residual R²=0,997, è la tabella) e da NESSUN braccio
oscillatorio (hard/ts/osc0: R²≈0, anche negativi) — combinato con la probe locale
(R² 0,90-0,95), l'attribuzione "estrapola perché la coordinata è locale" è ora una
misura. (2) *Autopsia spettrale*: lo spettro IMPOSTO di ts è morto in training (tutte
le bande collassate a τ max 7-59 byte, spettro finale ≈ lti); l'unico braccio con
spettro sopravvissuto è HARD — 65-200/512 canali per layer con τ nominale >500 byte
(max ~2·10⁵): **divisione del lavoro** — senza gate l'oblio vive nei pesi e il
gradiente collassa r alla scala-parola; col gate l'oblio migra nel data-dependent
(g~0,5/byte, autopsia gate) e r resta libero di tenere capacità lunga, modulata
selettivamente. Lettura rivista: la parità ts=hard a 700M è probabilmente un effetto
di TRAIETTORIA (le bande lente guidano la transizione, poi muoiono), mentre hard ha
capacità lunga reale a fine training. Discriminanti in coda: asintoto fB2-ts (se
resta parallelo a cb mentre hard converge → transizione vs capacità) ed
estrapolazione estrema 16k/32k (se la capacità nominale conta, emerge lì).
*Esito 16k/32k*: nessun dirupo per nessuno dei due (degradazione liscia), ma **hard
sotto ts su OGNI bucket oltre 4096** (~0,02-0,03 costante; a 16k: 0,584 vs 0,609; a
32k: ~0,65 vs ~0,68) — la capacità nominale tenuta viva dal gate paga in robustezza,
con margine modesto ma sistematico. Nota di metodo: TinyStories non ha dipendenze
oltre la storia (~900 byte) → la degradazione per-posizione misura la pura
fuori-distribuzione dello stato, non la difficoltà del task.

**Riconsiderare se.** (a) un terzo pubblica confini appresi + reset + estrapolazione
(la finestra si chiude — Harmonic va replicato, non ignorato); (b) C1 mostra
gate-appreso = euristica-spazi (il livello 1 non è contributo: lo si dichiara e ci si
sposta su C2 o si chiude); (c) la pagella neuro chiude in rosso lo stadio (àncora
declassata a ispirazione, dichiarato nel log); (d) hierarchical collapse in C2 non
curabile col decoupled training di SUNTA.

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
| fA-cb-1..3 | 2026-08-20 | char-transformer | 6,91M | **700M byte** ×3 | 1-3 | 0,4148 / 0,4263 / 0,4146 | **Griglia char fase A, baseline onesta: media 0,419, σ 0,007** (≈0,60 BPB). Ricetta b16@1e-2 sweeppata. RTX 5090 |
| fA-osc0-1..2 | 2026-08-20 | char-osc0 | 6,38M | 700M byte ×2 | 1-2 | 0,4446 / 0,4115 | Banco osc layer 0 + 7 attn SENZA pos emb: **parità (0,428)** con 260k param in meno; probe: posizione dalla fase R²=0,90 |
| fA-nopos-1..2 | 2026-08-20 | char-transformer-nopos | 6,38M | 700M byte ×2 | 1-2 | 1,4485 / 2,0390 | Controllo senza posizione: **transizione fallita (s2) o a metà (s1)** — la mask causale non basta a questa scala; probe R²=0,24 |
| fB-lti-1..2 | 2026-08-20 | char-hyb | 6,39M | 700M byte ×2 | 1-2 | 0,4548 / 0,4619 | Griglia char fase B, controllo senza reset (ibrido 4osc+4attn, no pos). Parità lasca −7,6%: i param del gate sono il meccanismo (dichiarato) |
| fB-phase-1..2 | 2026-08-20 | char-hyb-phase | 6,85M | 700M byte ×2 | 1-2 | 0,4586 / 0,4573 | Reset con rotazione: **identico a lti** — il gate con oscillazione non compra nulla |
| fB-hard-1..2 | 2026-08-20 | char-hyb-hard | 6,85M | 700M byte ×2 | 1-2 | 0,4295 / 0,4251 | **Vincitore fase B: reset θ≡0 = 0,427, −0,03 (≈4σ) da lti/phase**, a ~1σ da cb. 32-true (diagnosi NaN); s1 rilanciata (guard anti-resume su ckpt HF della run NaN, rimosso) |
| fB-lti32-1 | 2026-08-20 | char-hyb | 6,39M | 700M byte | 1 | 0,4527 | Controllo confound precisione: lti a 32-true resta nel gruppo 16-mixed → il vantaggio di hard è del reset, non di fp32 (residuo ≤0,004) |
| fB2-cb-1 | 2026-08-20 | char-transformer | 6,91M | **2,2B byte (1 epoca)** | 1 | 0,3879 | Asintoto B2, baseline (≈0,56 BPB). 66.990 step |
| fB2-cb-2 | 2026-08-20 | char-transformer | 6,91M | 2,2B byte | 2 | 0,3889 | Asintoto B2: coppia cb 0,3879/0,3889, spread 0,001 |
| fB2-osc0-1..2 | 2026-08-20/21 | char-osc0 | 6,38M | 2,2B byte ×2 | 1-2 | 0,3960 / 0,3870 | Asintoto B2: s2 SOTTO entrambi i cb; spread 0,009. Con hard (0,3938±0,0017): **all'epoca piena tutti i bracci convergono in ~0,005 attorno a cb** — regime data-limited, la loss perde potere discriminante (come parità strutturale stadio 1); il verdetto tra bracci vive su estrapolazione/probe/spettri |
| fB2-hard-1 | 2026-08-20 | char-hyb-hard | 6,85M | 2,2B byte | 1 | 0,3921 | Asintoto B2: gap vs cb **dimezzato** (0,008→0,004) — prima curva del progetto che CONVERGE verso la baseline. 32-true |
| fC1-heu-1 | 2026-08-20 | char-hyb-heu | 6,39M | 700M byte | 1 | 0,4523 | **C1: il reset totale cablato sugli spazi = lti32 (0,4527) — i confini da soli non comprano nulla**; l'ipotesi "gate=rilevatore di spazi" muore. 32-true. s2 tagliata (verdetto a 3,5σ) |
| fC1-ts-1..2 | 2026-08-20/21 | char-hyb-ts | 6,39M | 700M byte ×2 | 1-2 | 0,4357 / 0,4277 | **C1: lo spettro di timescale PAREGGIA il gate appreso** (media 0,432 vs 0,427, seed intrecciati) e **estrapola come lui** (+15% vs +13% al bucket 8192; osc0 banda-unica +188%) — il meccanismo è lo spettro. 32-true dopo NaN a 16-mixed (bande lente quasi-DC, stessa fisica di hard) |
| fC1-rel-1 | 2026-08-21 | char-transformer-rel | 6,40M | 700M byte | 1 | 0,7571 | **C1, controllo alla Segatron: la coordinata distanza-dal-confine esplicita da sola fa 0,757** — sblocca la transizione (vs nopos 1,45/2,04) ma resta a 0,33 da hard: il reset è coordinata + dinamica di oblio selettivo, non solo PE implicito |
| judge-s1 | 2026-08-20 | as-t1 vs as-h1 | — | — | 1 | — | **Giudizio cieco D14** (non training): 188 giudici Opus 5 in-sessione, doppio ordine. t 82 · h 77 · tie 29 (p=0,75); prompt netti 28 vs 30 (p=0,90) → **parità qualitativa, conferma la loss**. Preliminare (1 coppia, 536M). Artefatti: eval/judgments/elo-536M-s1.* |
| pilot-1 | 2026-08-18 | transformer | 8,5M | 536M (1 epoch) | 1 | 1,509 (val completo @512) | Pilot per Q4, non braccio di griglia. BPB 0,531 @512 · 0,551 @256 (àncora: 0,4407). Curva: 100M→1,99 · 170M→1,80 · 260M→1,66 · 390M→1,56. Nota di metodo: la run dedicata da 20M dello sweep (3,67) chiude PEGGIO del punto 20M di questa curva (~3,4) — a piccoli budget l'annealing precoce costa più del rumore che toglie; i punti intermedi si leggono come stima centrale, non come limite |

Ogni run vera aggiunge una riga; i dettagli vivono su W&B (progetto `neuro-llm`), qui solo
l'essenziale per leggere la storia dell'esperimento senza aprire dashboard.
