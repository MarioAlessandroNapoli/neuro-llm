# SOMBRERO (arXiv:2601.22805, Aleph Alpha, gen 2026)

Lettura integrale (HTML arXiv, 5 passate; Tabella 1 e formule riestratte due volte).

## Metrica ℬ (boundary enrichment, §3.5) — da implementare identica

s_{t+1} = −log p(x_{t+1}|x_{1:t}) (surprisal del byte successivo), b_t ∈ {0,1}
indicatore inizio-chunk, h_t = s_{t+1}:

    ℬ = [ (1/Σb_t)·Σ(b_t·h_t) ] / [ (1/T)·Σh_t ]

ℬ>1 = i confini si concentrano su byte difficili. Significatività: null
"rate-matched circular shift" (rotazioni cicliche degli indicatori, Z_ℬ =
(ℬ−μ_null)/σ_null; numero di rotazioni non specificato). Metriche complementari
(App. C): gap entropy normalizzata, CUSUM range, runs test.

**CAVEAT PER NOI (decisione metodologica)**: ℬ richiede b_t BINARIO. Il nostro gate è
continuo (p~0,5 ovunque, 0,83 ai confini): o si soglia (perdendo l'informazione che
ci distingue da un segmentatore) o si generalizza con pesi continui w_t — estensione
naturale ma NON del paper: se riportiamo un ℬ va dichiarato come. I loro numeri sono
confrontabili solo con la versione binaria.

## Numeri (Tab. 1; 0,98B param, 839B byte, seq 16384, corpora EN/DE/codice/math)

| config | ℬ | BPB |
|---|---|---|
| Equal-Size (1 confine ogni 5 byte, fisso) | 1,001 | 0,8893 |
| H-Net (senza CAB) | 1,19 | 0,6701 |
| + byte-level smoothing | 2,612 | 0,6571 |
| + CAB (peso 0,01) = SOMBRERO | **3,035** | **0,6568** |
| H-Net + CAB senza smoothing | 2,774 | 0,664 (peggiora) |
| Scale-up 2,19B: H-Net / SOMBRERO | 1,998 / 3,133 | 0,6126 / 0,6128 |

Letture: H-Net "nudo" ha ℬ=1,19 (arricchimento debole, non nullo); CAB e smoothing
sono sinergici, non sostituti; a scale-up il BPB converge ma ℬ resta separato (ℬ
vede ciò che il BPB non vede più); downstream: vantaggio a 1B, NON a 2,19B (0,463 vs
0,473 — limite non discusso dal paper).

## CAB loss (§3.4)

L_CAB = (1 − sg[P_{t+1}] − p_t)², P = prob del byte reale successivo, clamp
[1e-6, 1−1e-6]. Target 1−P (non −log P) per match di range con p_t ∈ [0,1].
On-policy (niente modello ausiliario ≠ BLT); circolarità riconosciuta come trade-off.
Smoothing byte-level (§3.3): x^b_i = c_i·x^k_i + (1−c_i)·x^b_{i-1} sugli INPUT (≠
H-Net che smussa a valle). La ladder: solo smoothing porta ℬ 1,19→2,61; CAB completa.

## Correzione di provenienza (IMPORTANTE)

**Il claim "forzare confini linguistici degrada" NON è in SOMBRERO** (cercato
sistematicamente: assente). Ciò che c'è: (a) Equal-Size (fisso POSIZIONALE, non
linguistico) è pessimo; (b) App. D, qualitativo: SOMBRERO allinea i confini agli
spazi, H-Net al primo carattere della parola («compute aggiuntivo per predire il
secondo byte, controintuitivo») — osservazione emergente, non esperimento di
imposizione. Se serve il claim sul degrado, verificarlo su H-Net/SpaceByte.

## Rapporto col nostro risultato

Conferma: ℬ nasce proprio perché i chunker appresi NON sono commutatori 0/1 (anche
il loro migliore è un gate continuo con bias statistico verso l'alta sorpresa) — il
nostro pattern p 0,5→0,83 è il tipo di segnale che ℬ misura. Complica: la loro
pipeline discretizza comunque p_t per comprimere; il nostro gate non discretizza mai
— è la differenza reset-continuo vs segmentazione da tenere esplicita.
