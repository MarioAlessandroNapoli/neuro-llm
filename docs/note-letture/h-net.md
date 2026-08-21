# H-Net (arXiv:2507.07955v2, Hwang/Wang/Gu) + codice goombalab/hnet @3673fe1

Lettura integrale paper (HTML v2) + audit del codice. Il vicino da citare di più;
la differenza chiave («loro comprimono senza resettare») è provata a livello di riga.

## Numeri da citare (con provenienza)

- Setup: FineWeb-Edu 100B token; 8192 byte/seq; batch 256; AdamW WSD; SWA 1024 nei
  layer T esterni. Ablazioni su 36B token, scala Large.
- Compressione appresa: **4,5-5 byte/chunk** (intro); BPIC Tab. 1: GPT-2 4,6 ·
  1-stage 4,8 · 2-stage 7,0 · spacelike 6,0.
- Crossover col Transformer BPE (Fig. 3, XL): space 200B · 1-stage 100B · 2-stage
  **30B byte**.
- Tab. 2 (L/XL): 2-stage BPB **0,743/0,715** vs Transformer 0,756/0,730; media 7
  task 55,5/58,2 vs 53,3/55,5 («Large 2-stage = XL BPE Transformer»).
- Robustezza (Tab. 3, HellaSwag perturbato, no augmentation): XL Transformer 32,3 →
  H-Net 2-stage 40,9.
- Cinese (XL): 0,7032 vs 0,7404; XWinograd-zh 0,663 vs 0,599. DNA: **3,6× data
  efficiency** (§3.2/Fig. 6 — l'abstract dice "nearly 4×": citare 3,6×).
- Costi dichiarati (§4): training **fino a 2× più lento** di un isotropico; mai
  provato 3-stage; scala max FLOP-matched 1,3B.

## PROVA NEL CODICE: nessun reset ai confini appresi (8 prove)

In parole: l'encoder Mamba-2 scorre su tutti i byte come nastro unico; il router
marca posizioni; il downsampler **seleziona** quei vettori (indicizzazione, non
inizializzazione). L'unico azzeramento di stato è al confine di DOCUMENTO (packing).

1. L'unico canale di reset è `seq_idx`, derivato da `cu_seqlens`
   (isotropic.py:143-148; block.py:124-126).
2. `cu_seqlens = arange(B+1)*L` (mixer_seq.py:103-110) = packing var-len del batch,
   nessuna relazione coi confini appresi.
3. `get_seq_idx` (utils.py:6-11): id di sequenza per posizione.
4. Nella kernel Mamba (ssd_state_passing.py:77-79) `scale = where(seq_idx_new ==
   seq_idx, scale, 0.0)`: lo stato si azzera SOLO al cambio documento. (Il
   `chunk_size=256` è il blocco di parallelizzazione SSD, non i chunk semantici.)
5. ChunkLayer (dc.py:174-179): `hidden_states[boundary_mask]` + rimappa dei
   fine-documento — il chunk appreso non genera mai entry in `cu_seqlens`.
6. Anche l'EMA del dechunk riceve `seq_idx` (dc.py:258-260, 285-293): la ricorrenza
   dello smoothing è CONTINUA attraverso i chunk del documento.
7. In generazione (generate.py:81-117): cache allocata una volta; prefill senza
   `cu_seqlens`; step aggiorna in-place; nulla condizionato a boundary_mask.
8. `IsotropicInferenceParams.reset()` esiste (isotropic.py:32-42) ma è CODICE MORTO
   (zero chiamanti in tutto il repo).

Sintesi da paper: «il confine appreso agisce solo come maschera di selezione
(dc.py:175) e indice di ri-espansione (dc.py:298-308); lo stato ricorrente attraversa
i confini senza discontinuità; l'unica discontinuità è al confine di documento,
imposta dal packing».

Contro-argomento da anticipare: il router FORZA un confine a inizio sequenza
(dc.py:95-96, 100, 150-154; p₁=1.0 «by definition») — unico ancoraggio posizionale
hard-coded; coincide col confine di documento e non resetta stato.

## Dettagli meccanismo (che un reviewer si aspetta)

- Routing: W_q,W_k init identità (`_no_reinit`, dettaglio assente dal paper);
  p_t = (1−cos(q_t,k_{t−1}))/2; b = 1{p≥0,5} via argmax.
- Downsampler: selezione diretta ≈ mean/max/cross-attn (Fig. 13) → «l'encoder
  comprime implicitamente il contesto nei vettori di confine».
- Smoothing (eq. 5): EMA sulla sequenza compressa, implementata come Mamba-2 con
  A=−1, dt=log(1/(1−p)), x=h/dt, B=p, C=1 (dc.py:275-293; p clampata [1e-4,1−1e-4]).
  Verifica algebrica: decay = 1−p, input = p·h ✓. Il loro meccanismo di frontiera è
  un mixer che INTERPOLA tra chunk — ma NON descritto da loro come "scelta di non
  resettare": formulazione difendibile = «l'EMA propaga deliberatamente informazione
  attraverso i confini con peso 1−P; a P→1 il contributo del passato tende a zero
  senza che lo stato venga azzerato — e questo solo sul canale del dechunk, non
  sullo stato SSM di encoder/decoder».
- Upsampler: c_t = p^b(1−p)^{1−b}; STE(c)=c+sg(1−c); espansione causale.
- Ratio loss (eq. 10): L = N/(N−1)·((N−1)FG+(1−F)(1−G)), α=0,03; può scendere <1
  con F≠G («which we indeed observe»); statistiche per-device (NOTE degli autori).
- Signal propagation (§2.3): RMSNorm a fine sotto-rete; proiezione solo sul residuo
  init a zero in fp32; LR modulation λ = sqrt(N_GPT·(ΠN)/(ΠN_tot)·D^S/D^s), tipici
  2,0/1,5/1,0; cambio dimensione per concat/troncamento (da SpaceByte).
- Ablazioni: senza smoothing i ratio OSCILLANO violentemente e la qualità crolla
  (il componente essenziale); M4-M4 > combinazioni con T anche su input BPE
  (Fig. 10); MoE FLOP-matched molto peggio (Fig. 12).

## Posizione ed estrapolazione: IL CAMPO È LIBERO

- **Zero esperimenti di estrapolazione in lunghezza** in tutto il paper+appendici
  (verificato per assenza: extrapolat*/length generaliz*/context length).
- mixer_seq.py:99-101: `assert position_ids is None ... not supported for HNet due
  to the subsampling hierarchical structure` — la posizione assoluta NON è
  propagabile attraverso il chunking (ammissione più forte del repo).
- RoPE per stage (rotary_emb_dim 32/48/64) applicata con cu_seqlens: nella main
  network la coordinata conta CHUNK, non byte; nello stage byte (Mamba puro) la
  posizione è solo implicita nella ricorrenza.
- Long context: «we hypothesize... may provide general long context improvements» —
  mai testato (§4). "Dynamic state allocation" nominata come futuro.
- Fig. 4/16: 1-stage confini sugli spazi; 2-stage anche sui primi caratteri di
  parola («once the initial positions of a word are identified, the remaining
  characters become highly predictable») — la cosa più vicina a una coordinata
  locale nel paper, ma è qualitativa e sul DOVE, mai sul COSA rappresenta la rete.

## Da non duplicare / rischi di contraddizione

Loro (citare come loro): byte-level end-to-end batte BPE compute-matched; chunk
~4,5-5 byte spontanei; robustezza ortografica; SSM>T nelle reti esterne; DC > MoE;
il lessico "dynamic chunking"/(N⁰,N¹)-DC. Rischi: (1) «H-Net sceglie di non
resettare» è attribuzione nostra — usare la formulazione implementativa; (2) mai
dire «in H-Net non c'è reset» senza «ai confini appresi» (il doc-reset esiste);
(3) non negare che il vettore di confine riassuma il chunk (misurato, Fig. 13) — il
nostro reset serve ad ALTRO (coordinata+estrapolazione); (4) costi: 2× dichiarato —
se il nostro è più economico, misurarlo prima di dirlo.

**Spazio libero per noi**: cosa codifica la rappresentazione post-confine (loro:
solo dove cadono); estrapolazione in lunghezza; ablazioni sulla continuità di stato
ai confini; la connessione posizione↔confine (da loro solo un assert di
impossibilità).
