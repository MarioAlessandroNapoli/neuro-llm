# HM-RNN (Chung, Ahn, Bengio, ICLR 2017 — arXiv:1609.01704v7) + replica COLING 2018

Lettura integrale (13 pp. + Kádár et al., COLING 2018 pp. 3215-3227). Appunti per la
scrittura dello stadio char; ogni claim con provenienza.

## Equazioni UPDATE/COPY/FLUSH e boundary detector

Layer ℓ, tempo t; z ∈ {0,1}; z_t^0 = 1 sempre; ultimo layer senza detector né top-down.

Cella (Eq. 2), determinata da z_{t-1}^ℓ (proprio boundary al passo prima) e z_t^{ℓ-1}
(boundary del layer sotto, passo corrente):
- UPDATE (z_{t-1}^ℓ=0, z_t^{ℓ-1}=1): c_t = f⊙c_{t-1} + i⊙g
- COPY (0,0): c_t = c_{t-1} — conserva TUTTO senza leak (≠ leaky integration LSTM)
- FLUSH (z_{t-1}^ℓ=1): c_t = i⊙g — «EJECT (passa lo stato sopra) then RESET
  (reinitialize)... a hard reset... completely erases all the previous states»
  (p. 5). Il nostro reset, nove anni prima.

Stato (Eq. 3): h_t = h_{t-1} se COPY, altrimenti o⊙tanh(c_t).

Gate (Eq. 4-7): [f;i;o;g;z̃] = [sigm;sigm;sigm;tanh;hard_sigm](s_rec + s_topdown +
s_bottomup + b), con s_topdown = z_{t-1}^ℓ·U h_{t-1}^{ℓ+1} (attivo solo post-boundary)
e s_bottomup = z_t^{ℓ-1}·W h_t^{ℓ-1} (attivo solo se il sotto ha boundary).

Binarizzazione (Eq. 8-9): step function a 0,5 (o Bernoulli). hard_sigm(x) =
max(0, min(1, (a·x+1)/2)). **Straight-through** + slope annealing: PTB a =
min(5, 1+0,04·N_epoch); IAM a = min(3, 1+0,004·N_epoch); **Text8/enwik8: annealing
DISATTIVATO** — «difficulty of finding a good annealing schedule on large-scale
datasets» (p. 8): ammissione che il meccanismo non scala.

## Cosa scoprono i livelli

Fig. 3 (enwik8): z¹ spara su/dopo gli spazi («somewhat surprising because the model
self-organizes this structure without any explicit boundary information»); z² su fine
parola e 2-3-grammi. Fig. 4 (PTB): flush a metà parola «tele-FLUSH-phone» — «the
model uses to some extent the concept of surprise to learn the boundary». Conteggi
(p. 9): su 270 char, layer1 270 update, layer2 56, layer3 9 → 335 totali = −60% vs
810 di una RNN piatta. La sparsità dei boundary alti NON è imposta: emerge dal
trade-off implicito del FLUSH (reward = informazione fresca sopra; penalty =
cancellare l'accumulato).

## Numeri

PTB test BPC (Tab. 1): LN HM-LSTM step+annealing **1,24** — NON SOTA (LN
HyperNetworks 1,23); il paper dice onestamente "comparable". Text8 (Tab. 2): LN
HM-LSTM **1,29 = SOTA dell'epoca** (unico claim SOTA). enwik8: 1,32, pari a Recurrent
Highway Networks («a tie»). IAM-OnDB: 1167 (miglior log-likelihood). Setup: PTB 512×3
layer, Adam 2e-3, clip 1, b64, seq 100, LN; Text8/enwik8 1024 unità, b128, 1e-3.

## Fragilità (dalla replica COLING 2018)

- Riproduzione: PTB 1,29 vs 1,25 pubblicato; **Text8 1,36 vs 1,29 — gap 0,07 mai
  chiuso, pur col codice originale** («We did not pursue further experiments»).
- Cause documentate: COPY sull'ultimo layer implementato solo su c (non h),
  diversamente dal paper; LN nel paper ma non nel codice; lr schedule ricostruito «an
  informed guess»; init/weight-decay mai specificati.
- **α=1,0 → divergenza dopo 4k iterazioni** (z¹=z²=1: collasso tutto-boundary):
  soglia di instabilità non lineare dello slope.
- **Variante Elman (senza gating LSTM): collasso degenere** (z¹=0 sempre, z²=1
  sempre; BPC 1,40 vs 1,27) — il boundary detection NON è separabile dal gating LSTM.
- **Il risultato più critico**: «Our runs could not reproduce the segmentation
  results... we do not find a relationship between the performance of the model and
  segmentation behavior... Changing α to 0.25 from 0.5... same performance, but huge
  difference in segmentation behavior» → **la stessa loss è compatibile con
  segmentazioni molto diverse: il boundary non è identificato dall'obiettivo LM**
  (pattern ricorrente: Williams 2017 su SPINN/Gumbel-Tree). Da affrontare
  esplicitamente se usiamo i confini come segnale.

## Posizione/estrapolazione: ASSENTI

Verificato per assenza su tutto il testo: mai "position/positional/extrapolat*" nel
nostro senso. I framing di FLUSH sono solo: scoperta di struttura, efficienza (−60%
update), vanishing gradient, allocazione risorse, interpretabilità. **Mai come
meccanismo posizionale o di generalizzazione in lunghezza.**

## Formulazione onesta per il paper

HM-RNN ha dimostrato che il reset appreso (hard, non graduale) è possibile e può
allinearsi a unità linguistiche senza supervisione. Mancavano: (X)
**identificabilità** — stessa loss, segmentazioni diverse (COLING §4.3); (Y)
**portabilità** — il meccanismo dipende dal gating LSTM e lo slope annealing non
scala. Il nostro lavoro entra esattamente in quei due varchi (e nel terzo mai
nominato: il reset come coordinata posizionale con estrapolazione).
