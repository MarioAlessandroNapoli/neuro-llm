# Griglia 1b, fase 0 e asintoto — design, findings ed esiti (archivio, 2026-08-19)

Testo integrale di D12-design-1b con tutti gli esiti (autopsia 1a, fase 0 apparato,
sweep, findings, verdetto due-tasse -> parita'). Sintesi nel RESEARCH_LOG (D13).

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

