# Harmonic (arXiv:2606.24650, Nyoma, singolo autore, mag 2026) — paper + AUDIT del codice

Lettura integrale + audit di github.com/Omibranch/Harmonic e /harmonic-logs (l'agente
ha anche ESEGUITO il modello per la verifica di causalità). Verdetto in tre righe:
le timescale non sono "vincolate all'init" — sono INGABBIATE per parametrizzazione
(il collasso che noi osserviamo è per loro impossibile per costruzione, e non lo
sanno); il modello 1B della §5 NON è causale (legge il token da predire); il
vantaggio a piccola scala regge ma si erode di 2/3 col budget (controllo a 20K step
presente nei log e MAI pubblicato).

## Architettura reale (train_fast.py)

Per livello: A(t) = a_min + (a_max−a_min)·σ(MLP_A(inp)); b(t) = (1−A)·tanh(MLP_B);
h(t) = A⊙h(t−1) + b → LayerNorm. UN layer con 3 livelli in cascata, stessa
risoluzione (nessun downsampling); il livello ℓ+1 riceve l'errore di predizione del
sotto. **A_RANGES = ((0.55,0.88),(0.90,0.975),(0.992,0.999)) sono float Python, non
parametri**: τ ammessi 2,2-8,3 / 10-40 / 125-1000 TOKEN — il livello 3 non può
scendere sotto τ=125 nemmeno volendo. Il paper dice «constrained by initialization»:
descrizione ERRATA del proprio codice (i valori delle bande non compaiono mai nel
paper). Nessuna diagnostica sugli spettri finali in tutto il repo.

Il blocco 1B (hallamonic) è un'architettura DIVERSA: bande assenti, gerarchia =
mean-pool stride 4, predictive-coding = codice morto. E **non è causale**: il
mean-pool+repeat_interleave fa dipendere l'uscita in t da t+1..t+15; verificato
eseguendo (perturbazione in p → Δ≠0 a posizioni precedenti; ai confini di blocco
Δ=0). Pistola fumante nel paper stesso: su token uniformi 3,83 bpt contro un limite
informativo di 15 (§5 "Verification", letto dall'autore al contrario). §5/Tab. 6 da
scartare integralmente. Il modello piccolo è invece strettamente causale (verificato).

## Numeri utilizzabili e caveat

- Tab. 1 (enwik8, equal-budget, ~28M... reali 13,4M a h=128, conteggi del paper
  sbagliati; le LOSS sono verificate sui log): gap H-TF +1,4%→+11,4% (1K→32K) — ma
  il baseline Transformer è **NoPE e post-LN** (nessuna PE di alcun tipo,
  train_fast.py:493-497): per noi è oro (gerarchia di timescale batte NoPE = la
  struttura temporale sostituisce il segnale posizionale), come misura
  dell'efficienza dell'attention è viziato.
- "byte-level" dichiarato ma è BPE GPT-2 (bpt = bit per token BPE, non confrontabile
  con bpb; le loro τ in token: banda veloce ≈ 9-33 byte).
- Seed: Tab. 1 a seed singolo 42 = il più fortunato per TUTTI i modelli (log a 5
  seed: 16K H 6,54±0,20 vs 6,196 pubblicato → il claim «la loss migliora col
  contesto» è falsificato dai loro stessi log). Il gap però sopravvive; H vs Mamba
  appaiato: t(4)=7,08, 5/5 seed, p≈0,002 (l'autore lo scarta per errore statistico
  — confronta con la sd marginale invece che appaiata).
- Riga 64K: NESSUN log (grep su 22 file: zero); l'OOM dei baseline è
  implementativo, non architetturale (FlashAttention è O(L) in memoria).
- Ablazioni (Tab. 4, 1 seed): NoPred +0,002 (**il predictive coding è inerte** —
  dichiarato onestamente); Flat +0,501 — MA Flat = tutti i livelli nella banda media
  [10,40]: rimuove gerarchia E banda lenta insieme → misura l'assenza di memoria
  lunga, non il valore della diversità. **L'ablazione "bande libere" non esiste nel
  loro spazio: non hanno mai tolto la scatola. Noi sì — e collassano.**
- **Controllo di convergenza SOPPRESSO** (results/SUMMARY.md, 20K step = 5× budget):
  gap a 8K da +6,7% a **+2,8%** — il vantaggio si erode di 2/3 col budget. Parallelo
  diretto e indipendente del nostro finding (ts pari a 700M, ultimo all'epoca):
  **non contraddiciamo Harmonic, completiamo una curva che l'autore ha misurato e
  non ha pubblicato.**
- Baseline Mamba = reimplementazione dell'autore, mai validata contro l'ufficiale.

## Uso nel paper

Citabile con fiducia: NoPred; Flat come evidenza qualitativa (dichiarando il
confondente); protocollo equal-budget; il vantaggio su Mamba (rafforzabile col
t-test appaiato che loro non fanno). Con riserva esplicita: i gap crescenti H-TF
(NoPE, 1 seed, erosione col budget). Mai: §5/Tab. 6, riga 64K.

**Posizionamento**: alleato, non bersaglio. Le nostre tre fessure: (i) le loro
timescale non possono muoversi → «cosa succede se le liberi?» è nostra (collassano);
(ii) nessuno ha mai guardato lo spettro finale di questa classe di modelli; (iii) la
struttura temporale può essere APPRESA (gate su confini) invece che ingabbiata — ed è
la sola versione che tiene capacità a budget pieno. La scatola di Harmonic è la prova
per contrasto che il problema esiste: se la gerarchia si reggesse da sola, non
avrebbero avuto bisogno di inchiodarla.
