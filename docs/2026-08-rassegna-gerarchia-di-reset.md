# Rassegna: gerarchia di segmentazioni apprese (2026-08-20, sera)

Quattro fronti in parallelo (2 Opus: chunking gerarchico storico, era SSM; 2 Sonnet:
neuroscienze delle timescale, analogia visione). Materiale istruttorio per la D17.
Sintesi operativa in RESEARCH_LOG § D17; qui i report integrali (lievemente compressi
sui passaggi ridondanti, fatti e fonti intatti).

---

## Fronte 1 — Chunking gerarchico e appreso: censimento 2014-2026

**Sintesi in una riga**: la segmentazione appresa è satura sull'asse **compressione**;
l'asse **posizionale** (confine come sostituto dell'encoding di posizione, verificato
per ablazione e in estrapolazione) era deserto — a maggio 2026 è comparso un solo
occupante parziale (Harmonic).

### HM-RNN (Chung, Ahn, Bengio, ICLR 2017 — arXiv:1609.01704): l'antenato diretto

Confini appresi non supervisionati (hard-sigmoid binarizzata, straight-through +
slope annealing). Tre operazioni: UPDATE, COPY, **FLUSH = il nostro reset**: al
confine lo stato c_{t-1} viene scartato e il riassunto sale al livello sopra.
3 livelli: il layer 1 spara sugli spazi, il 2 su morfemi/2-3-grammi, il 3 su confini
sintattico-semantici. SOTA char-LM dell'epoca (Text8 1.29 BPC). **Perché la linea è
morta**: (1) ottimizzazione fragile (annealing non scala, nessun controllo sulla
cardinalità dei confini); (2) mai provato su NLP oltre char-LM; (3) l'euristica dello
spazio batteva i confini appresi (misurato da Nawrot 2023). Tutte e tre le cause sono
state rimosse nel 2024-26 (ratio loss, SSM, smoothing) da gente che poi ha usato la
soluzione per comprimere, non per la posizione.

### Downsampling nel transformer (fattore fisso)

- Funnel-Transformer (NeurIPS 2020): mean-pooling stride 2, pool-query-only attention.
- Hourglass (Nawrot 2021/22): fattore k fisso; attention upsampling con shift k−1
  (anti-leakage); enwik8 0.98 bpc con 146M vs T-XL 0.99 con 277M.
- CANINE (TACL 2022): stride r=4. Charformer/GBST (ICLR 2022): soft, nessuna riduzione
  dinamica reale. MANTa (EMNLP-F 2022): predittore-di-confine + pooling differenziabile.
- ToMe (ICLR 2023): merging bidirezionale, non trasferibile al causale.

### Dynamic Token Pooling (Nawrot et al., ACL 2023 — arXiv:2211.09761): il confronto pulito

A parità di architettura (Hourglass), quattro modi di scoprire un confine: Gumbel-sigmoid
end-to-end (con prior binomiale anti-collasso), Unigram supervisionato, entropy spikes,
whitespace. **Verdetto testuale: Gumbel e entropia «generally inferior», whitespace e
Unigram vincono** (text8: whitespace 1.133 vs Gumbel 1.136 vs vanilla 1.143). Avg-pooling
batte sempre il sub-sampling. Nessuna estrapolazione, nessuna ablazione posizionale,
nessuno stato ricorrente. Citano HM-RNN e scelgono di NON replicare il reset.

### Famiglia byte→patch (2023-2026)

- **MEGABYTE** (NeurIPS 2023): patch fissa P=8; reset di fatto (il local model riparte a
  ogni patch); 8k→16k non migliora.
- **SpaceByte** (NeurIPS 2024): confini sui byte "spacelike", blocchi globali solo ai
  confini; a parità di FLOP pareggia il subword transformer.
- **BLT** (Meta, ACL 2025 Outstanding): patch a entropia da byte-LM 100M congelato
  (nessun gradiente al segmentatore); 8B/4T byte; CUTE 54.1 vs 27.5 (robustezza char);
  −50% FLOP inferenza.
- **MambaByte** (COLM 2024): SSM piatto sui byte, controfattuale senza chunking.
- **HAT** (Aleph Alpha, ICLR 2025 + arXiv:2603.15953): confini whitespace; il decoder
  char si azzera a ogni parola (reset euristico, non studiato come meccanismo); Llama
  3.1 8B/70B convertiti; +68% LAMBADA.
- **AU-Net** (Meta FAIR, NeurIPS 2025): **4 scale** (byte→parola→2 parole→4 parole) ma
  confini **regex**; HellaSwag 70.2→73.7, MMLU 27.0→31.7 — le gerarchie profonde pagano,
  con confini a regola.
- **MBLM** (IBM 2025): patch fisse multi-stage; **unico con estrapolazione (8K→991K)** e
  ablazione posizionale: «SSM performs best without any positional information».
- Ortogonali: MrT5 (delete gate), Scratchpad Patching (DeepMind: "patch lag"),
  Fast BLT, Autocompleting Tokenizers (elimina i byte predicibili).

### H-Net e discendenza (72 citazioni in 13 mesi): l'asse compressione è saturo

**H-Net** (Hwang, Wang, Gu — arXiv:2507.07955, ICLR 2026): routing per similarità
coseno tra rappresentazioni adiacenti, smoothing EMA (differenziabilità), STE, ratio
loss (anti-collasso — il sostituto moderno dello slope annealing di HM-RNN). Encoder/
decoder Mamba-2 + main Transformer, **RoPE a ogni stage**. 2-stage: BPB 0.715 (XL) vs
0.756 BPE; chunk appresi: 4.8 byte (1-stage), 7.0 (2-stage). **Verificato nel codice
(dc.py): nessun reset dello stato SSM ai confini** — lo stato scorre continuo, lo
smoothing EMA è l'opposto di un reset. Nessuna estrapolazione, nessuna ablazione pos.
- **SOMBRERO** (Aleph Alpha, arXiv:2601.22805): misura l'allineamento dei confini
  (metrica boundary-enrichment ℬ); i confini di H-Net collassano sul whitespace
  (ℬ=1.19); supervisione da surprisal come regolarizzatore → ℬ=3.04 e BPB migliore;
  **forzare confini linguistici alti spesso degrada**; rendimenti decrescenti a 2B.
- **ATDC** (Fujitsu): curriculum sul compression ratio; 680M byte-level batte
  Llama-3.2 1.5B su zero-shot AVG.
- **FlexiTokens** (Findings ACL 2026): hinge a banda al posto del prior binomiale.
- **ByteFlow** (Amazon, ICLR 2026): Top-K su coding rate, grafo statico.
- **ReinPatch**: confini via RL. **H-Net++**: router BiGRU, persiano.
- **dnaHNet** (gruppo H-Net, feb 2026): la gerarchia ricorsiva profonda portata sul
  **DNA**, non sul testo — segnale che testo a 3+ scale apprese è considerato difficile.
- **SUNTA** (Tokyo, 2026, video): SSM gerarchico con confini da errore di predizione;
  documenta l'**hierarchical collapse** in training e la cura (decoupled training).

### Il vicino di maggio 2026: Harmonic (arXiv:2606.24650)

Preprint a **singolo autore, nessuna sede** — da trattare come ipotesi da replicare.
Tre livelli SSM a timescale **fisse** progressivamente lente (Clockwork moderno),
ogni livello riceve l'errore di predizione del sotto. **Fa i nostri due controlli**:
estrapolazione (enwik8 28M: +1.4%→+11.4% sul transformer da 1K a 32K) e ablazione
posizionale (TinyLlama 1.1B con HarmonicBlock: loss stabile 1K-8K senza RoPE).
**Ablazioni chiave**: flat-timescales costa −0.50 bpt (la gerarchia di scale è tutto);
l'accoppiamento predittivo tra livelli è rumore (≤0.022). Nessun reset, nessun confine.
→ Ci lascia la domanda residua, netta: **il confine appreso batte la timescale fissa
come meccanismo posizionale?**

### Reset di stato moderno, contorno

- **SurgicalMamba** (2026): *ruota* (non azzera) lo stato ai confini di chunk, video.
- **Never Reset Again** (dic 2024): elimina i reset in streaming — contro-argomento.
- **Understanding Length Generalization in Recurrent Models** (Buitrago & Gu,
  arXiv:2507.02782): *unexplored states hypothesis* — i ricorrenti non estrapolano
  perché a lunghezze nuove visitano stati mai visti; curato con State Passing/TBTT in
  ~500 step. **Non testano il reset intra-sequenza**: il nostro meccanismo è la
  risposta architetturale allo stesso problema (il reset riporta periodicamente lo
  stato in distribuzione). Complementari.
- **Stuffed Mamba** (arXiv:2410.07145): state explosion oltre la lunghezza di training
  = overfitting dello stato; il reset periodico impedisce per costruzione di superare
  la soglia.
- **HGRN** (NeurIPS 2023): forget gate con lower bound crescente con la profondità —
  gerarchia di timescale *continua*, senza confini. Competitor concettuale.
- **Mamba-3** (ICLR 2026): transizioni complesse (oscillatorie) per pattern periodici e
  posizione via fase — il backbone oscillatorio è diventato mainstream.
- **PoST** (arXiv:2604.07658): lo spettro di decay ottimale va **imposto** (log-decay
  geometrici); l'inizializzazione random collassa lo spettro — argomento pro-struttura.
- Meta-lavori scomodi: arXiv:2608.17325 (i tokenizer-free ottimizzano efficienza, non
  morfologia); arXiv:2608.03599 (LM e confini quasi indipendenti — position paper).

### Conclusioni fronte 1+2 (giudizio incrociato dei due agenti architetturali)

**Gap reale e stretto**: backbone SSM + confini appresi end-to-end + **reset dello
stato al confine** + verifica di estrapolazione con ablazione posizionale. Nessuno dei
~35 lavori censiti occupa l'intersezione (H-Net comprime senza resettare; Harmonic ha
gerarchia+ablazione+estrapolazione ma timescale fisse; HM-RNN aveva il reset appreso
ma è pre-SSM e senza i controlli). **Giudizio: direzione aperta, finestra che si
chiude, da inquadrare come posizionale (non compressiva).**

**Baseline obbligate dai reviewer futuri**: (1) reset su confine **euristico** (spazi
letterali) — se il gate appreso non lo batte, il livello 1 è un rilevatore di spazi
(SOMBRERO docet); (2) gerarchia di **sole timescale senza reset** (Harmonic-style, che
da sola vale −0.50 bpt nel loro setup); (3) HM-RNN citato generosamente come antenato.

**Rischi misurati**: livelli oltre la parola mai ottenuti con confini appresi (H-Net
si ferma a 7 byte/chunk; AU-Net arriva a 4 parole ma con regex); hierarchical collapse
documentato (SUNTA); su TinyStories il livello frase potrebbe non avere segnale
(frasi corte, vocabolario ristretto) — misurare prima la BPIC del livello 1.

**Da leggere per intero**: HM-RNN (1609.01704) · H-Net paper+codice (2507.07955) ·
Harmonic (2606.24650) · SOMBRERO (2601.22805) · Dynamic Token Pooling (2211.09761).

---

## Fronte 3 — Neuroscienze: la gerarchia di timescale come legge corticale

Sei filoni, ciascuno con traduzione architetturale:

1. **Temporal receptive windows** (Hasson 2008, J Neurosci 28:2539): gerarchia
   topografica di finestre di integrazione (ms sensoriali → minuti associative),
   dimostrata con scrambling a blocchi di durata crescente. *Traduzione*: i layer bassi
   devono essere ciechi al contesto lungo, gli alti sensibili allo scrambling di
   frase/discorso — test portabile al modello (scrambling controllato per scala,
   sensibilità per layer).
2. **Gradiente di timescale intrinseco** (Murray 2014, Nat Neurosci; Manea eLife
   2022): l'autocorrelazione dell'attività cresce lungo la gerarchia. *Traduzione*: la
   costante di decadimento tra un reset e l'altro deve crescere col livello — non solo
   la frequenza dei reset. Misurabile dall'autocorrelazione dello stato per layer.
3. **Theta-gamma nesting** (Lisman & Idiart 1995; Lisman & Jensen 2013): annidamento
   come **indirizzamento** — il reset del ciclo lento apre un nuovo "contenitore" di
   slot per il veloce, non un azzeramento indiscriminato. *Traduzione*: reset = nuovo
   frame di riferimento posizionale per il livello sotto.
4. **Tracciamento gerarchico del parlato** (Ding, Melloni, Zhang, Tian & Poeppel 2016,
   Nat Neurosci 19:158): oscillazioni che tracciano SIMULTANEAMENTE sillabe (4 Hz),
   frasi (2 Hz), periodi (1 Hz), senza marcatori acustici di confine; il tracciamento
   alto è **endogeno e top-down** (richiede attenzione/comprensione). *Traduzione*:
   scale simultanee e sovrapposte (non a cascata); i reset alti condizionati da stato
   appreso/contestuale, non da trigger locali cablati.
5. **Event segmentation** (Zacks 2007; Kumar CogSci 2023; Baldassano 2017, Neuron):
   confini = picchi di prediction error; eventi annidati a scale crescenti; picco
   ippocampale ai confini ALTI = scrittura in memoria. *Traduzione*: reset guidato da
   sorpresa; il reset di alto livello è il momento naturale del consolidamento.
6. **Predictive coding gerarchico** (Kiebel, Daunizeau & Friston 2008, PLoS Comput
   Biol): livelli lenti si aggiornano su sorpresa e mandano prior top-down ai veloci.
   *Traduzione*: senza canale top-down persistente tra i reset, la gerarchia è
   segmentazione piatta multi-scala, non predictive coding.

**Predizioni falsificabili per una gerarchia di reset artificiale**: (1) rapporti tra
scale adiacenti ~5-15× (se il modello ne apprende di molto diversi, è un finding);
(2) reset di alto livello endogeno: ablando gli stati alti, i gate bassi devono
cambiare — se dipendono solo da feature locali (punteggiatura) è una scorciatoia, non
gerarchia predittiva; (3) il canale top-down deve pagare su ambiguità locale, e i
reset alti devono coincidere con eventi di consolidamento distinguibili nello stato.

---

## Fronte 4 — L'analogia coi visual encoder: cosa importare, cosa no

- **Come la visione fa gerarchia**: CNN/Swin = raggruppamento **geometrico uniforme**
  (griglia, patch merging 2×2 ovunque); i tentativi content-driven (superpixel, ToMe)
  sono rimasti di nicchia. In visione la statistica spaziale è omogenea: la griglia è
  un buon prior.
- **Nel linguaggio la griglia perde**: MEGABYTE (patch fisse, vision-style) è il
  peggiore della serie byte; i vincitori (BLT, H-Net, Dynamic Pooling) usano confini
  content-driven a lunghezza variabile. **Il meccanismo dominante in visione è il
  pezzo sbagliato da copiare.**
- **La sintassi esplicita come livello ha una storia di ritorni marginali**: URNNG
  (parità col supervisionato, non oltre), ON-LSTM, StructFormer; "Schrödinger's Tree"
  (Kulmizev & Nivre): guadagni "scant, if any" — i LM potenti ricavano la sintassi
  dai dati, iniettarla è ridondante. I sistemi che funzionano inducono **punti di
  rottura predittiva** (sorpresa), non alberi.
- **Dove l'analogia si rompe**: ricorsione linguistica illimitata vs profondità fissa
  (ma umani e GPT-2 collassano entrambi a 2-3 livelli di center-embedding — la
  stratificazione di scala temporale è ciò che si modella davvero); dipendenze lunghe
  non locali (il "capitolo" referenzia liberamente, serve comunque attention globale
  sopra le unità compresse — come fanno MEGABYTE/BLT/H-Net).
- **Formulazione corretta**: non gerarchia sintattica né geometrica, ma **gerarchia di
  scale temporali di predicibilità/sorpresa** — coerente con TRW (Lerner J Neurosci
  2011) ed event segmentation. È ciò che BLT/H-Net implementano con l'entropia.

---

*Report generati da 4 agenti (2 Opus, 2 Sonnet) con ricerca web, 2026-08-20 sera.
Decisione derivata: D17 nel RESEARCH_LOG.*
