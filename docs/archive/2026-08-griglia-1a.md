# Griglia 1a — design ed esiti (archivio, 2026-08-19)

Testo integrale di D10-griglia-1a e D11-esiti-1a, archiviato alla chiusura dello
stadio 1 (D13). Sintesi operativa nel RESEARCH_LOG; qui il dettaglio completo.

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

