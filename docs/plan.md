# Piano di esperimento: STT minuscolo con loop alla Inkling 2 Mini, solo su Mac

Scritto il 2026-11-23. Solo ricerca, nessun codice scritto e nessun run eseguito.
Segue `ideas/inkling-2-mini-study.md` (esperimento E4).
Marcatori: **[P]** letto su una fonte primaria (link accanto); **[P-s]** fonte primaria letta solo tramite riassunto, da ricontrollare sul PDF; **[S]** fonte secondaria; **[E]** stima mia, non misurata; **[C]** congettura.

## Summary

- **Domanda:** un encoder CTC da circa 3–8M parametri con blocchi condivisi ripetuti (loop) batte un encoder normale con gli stessi parametri? Regge il confronto con uno da pari FLOP? E il numero di loop scelto all'inferenza dà una curva velocità/accuratezza utile sul Mac?
- **Precedente diretto:** LARM (giugno 2026). Allenato su 100h con decoding greedy, un encoder da 4 blocchi ripetuto 12 volte (7.7M) ottiene WER 11.34/31.84. Il 4 blocchi normale fa 26.78/52.82, il 16 blocchi (28.9M) 14.43/37.23. Persino il 48 blocchi (85.7M, stessi FLOP) fa peggio: 12.24/34.06. Su 960h invece il 48 blocchi vince [P].
- **Dati:** LibriSpeech train-clean-100 (6.3 GB, CC BY 4.0), valutazione su dev/test clean e other [P]. Sul Mac il preprocessing costa circa 1 h [E].
- **Pilota (circa 20 h di Mac):** 3 bracci (normale a pari parametri, loop, normale a pari FLOP), stessi step, stesso LR e stesso seed. Poi la curva WER e RTF al variare dei loop da 1 a K.
- **Go** se il loop batte il controllo a pari parametri di almeno il 20% relativo, sta entro il 5% del controllo a pari FLOP e, con meno loop, è almeno 1.5× più veloce a pari WER. Altrimenti si conferma "più piccolo, non più veloce" e non si finanzia oltre.

---

## 1. Cosa si testa, in una frase per ipotesi

Predizioni da scrivere (o correggere) **prima** di ogni run, nel file `predictions.md` del repo dell'esperimento.

| # | ipotesi | predizione [C] | perché |
|---|---|---|---|
| H1 | Il loop batte il modello normale **a pari parametri** | WER dev-clean −30% relativo o più | LARM: −58% a 100h (26.78 → 11.34) [P] |
| H2 | Il loop regge il modello normale **a pari FLOP** | entro ±5% relativo | LARM a 100h: 11.34 contro 12.24 (48 blocchi) [P]. A 960h perde: 4.59 contro 3.87 [P]. Su 100h quindi in parte è regolarizzazione |
| H3 | Meno loop all'inferenza è una manopola di compute utile | a K/2 loop, WER ≤ 1.15× quella a K loop | LARM K=12, exit 4/8/12: 15.20 / 11.57 / 11.34 [P-s]; "the second checkpoint is within 0.70 points" (K=16) [P] |
| H4 | Il gate chiuso all'inizio di Inkling aiuta rispetto alla somma fissa di LARM | differenza ≤ 0.5 WER assoluti, cioè probabilmente nulla | il gate serve soprattutto nel *retrofit* di un modello già allenato [P, studio precedente §I2]. Da zero il vantaggio atteso è piccolo |
| H5 | Il loop è **più veloce** sul Mac a pari WER | solo con meno loop, non a K pieno | a FLOP uguali la velocità resta uguale (F5-TTS: RTF 0.143 contro 0.144 [P, studio precedente]). Il solo guadagno possibile: pesi in cache su CPU, oppure meno loop |

## 2. Dataset

| split | uso | download | fonte |
|---|---|---|---|
| train-clean-100 (100h, 28.5k utterance) | training | 6.3 GB | [P] https://www.openslr.org/12 ; righe [P] https://huggingface.co/datasets/openslr/librispeech_asr |
| dev-clean / dev-other | early stopping, metriche del pilota | 337 MB / 314 MB | [P] openslr |
| test-clean / test-other | solo nel run finale, **una volta** | 346 MB / 328 MB | [P] openslr |
| train-clean-360 (opzionale, braccio di distillazione) | pseudo-label | 23 GB | [P] openslr |

- Licenza: CC BY 4.0 [P].
- Perché 100h e non meno: LARM ha le sue ablation proprio su 100h [P], quindi abbiamo numeri esterni con cui confrontarci. Un sottoinsieme più piccolo (per esempio 50h) farebbe perdere il confronto e spingerebbe ancora di più l'effetto di regolarizzazione.
- **Preprocessing [E]:** si decodificano i FLAC e si calcolano fbank a 80 dimensioni, 100 frame/s, salvati in float16 su disco. Sono circa 100h × 3600 × 100 × 80 × 2 B ≈ 5.8 GB. Su un M5 con multiprocessing a 4–8 processi la stima è 30–60 min, più il download (dipende dalla linea). Spazio libero sul disco: 169 GB (misurato con `df`).
- Tokenizzazione: caratteri (come LARM, 30 simboli [P-s]) **oppure** BPE da 256 token con sentencepiece. Consiglio BPE 256: accorcia le sequenze target ed evita il bug MPS di §8.

## 3. Modelli

Encoder Transformer pre-norm stile LARM [P]: subsampling conv 2D 4× (25 frame/s), d=256, 4 teste, FFN 1024, RoPE, testa CTC. Niente Conformer nel pilota, per restare confrontabili con LARM, che usa Transformer.

Parametri per blocco con d=256: circa 12·d² ≈ 0.79M [E, formula standard]. Con d=384 sono circa 1.77M, coerente con i 7.6M del 4 blocchi di LARM [P].

| braccio | struttura | parametri [E] | blocchi eseguiti per frame | ruolo |
|---|---|---|---|---|
| **A** | 4 blocchi normali | ≈3.5M | 4 | controllo **a pari parametri** |
| **B** | Inkling-style: preludio 1 blocco + core 2 blocchi condivisi × K loop + coda 1 blocco. Messaggio dal passaggio precedente tramite gate di rango basso **inizializzato a 0**; K campionato in training | ≈3.5M | 2 + 2K (K=8 → 18) | braccio di test |
| **C** | 18 blocchi normali | ≈14.5M | 18 | controllo **a pari FLOP** di B a K=8 |
| B-LARM (dopo il go) | 4 blocchi condivisi × K, somma α·r + β·h⁰ (init 0.5), FiLM di profondità, CTC ogni 4 loop [P] | ≈3.5M | 4K | isola il contributo del gate Inkling (H4) |
| A-lungo (dopo il go) | come A, ma allenato con gli stessi **FLOP di training** di B (≈4.5× gli step) | ≈3.5M | 4 | controllo "basta allenare di più o abbassare il LR" |

Dettagli di B, presi da Inkling [P, via studio precedente]:
- Loop sui layer **centrali**, non sugli ultimi. Ogni passaggio riparte dall'uscita del preludio. Il messaggio dal passaggio precedente entra attraverso un gate appreso che parte chiuso.
- Rapporto di campionamento dei loop in training ispirato a 1:2:6:2:1: per K_max=8, campiona K ∈ {2,4,6,8,10} con pesi 1:2:6:2:1.
- Loss CTC sull'uscita della coda. Opzionale: CTC anche ai loop intermedi, come LARM, che media la loss su {c, 2c, …, K} [P].

Riferimenti esterni (non da allenare, solo da misurare sullo stesso Mac):

| modello | parametri | WER test-clean / test-other | fonte |
|---|---|---|---|
| Whisper tiny.en | 39M (37.8M per Moonshine) | 5.66 / 15.45 (greedy, setup OpenASR) | [P] Moonshine Tab. 1, 3 https://arxiv.org/abs/2410.15608 |
| Moonshine Tiny | 27.1M | 4.52 / 11.71 | [P] stessa fonte. Allenato su circa 200k ore |
| Conformer(S), Transducer, 960h | 10.3M | 2.7 / 6.3 senza LM | [P] Tab. 2 https://arxiv.org/abs/2005.08100 |
| Squeezeformer-XS, 960h | 9.0M | 3.74 / 9.09 | [P-s] Tab. 3 https://arxiv.org/abs/2206.00888 (500 epoche su TPU v3) |
| LARM 7.7M, **100h**, greedy CTC | 7.7M | 11.34 / 31.84 | [P] Tab. 1 https://arxiv.org/abs/2606.04678 |
| Transformer normale 4 blocchi, **100h** | 7.6M | 26.78 / 52.82 | [P] stessa tabella |
| ESPnet Conformer, 100h, beam + CTC | non indicati (>10M) | 6.5 / 17.3, 15.9 h su una V100 | [P] https://github.com/espnet/espnet/tree/master/egs2/librispeech_100/asr1 |

Cosa aspettarsi in assoluto: con 100h e meno di 10M parametri, la letteratura sta tra 11 e 27 di WER greedy su test-clean [P, LARM]. **Non** si arriva vicino a Whisper o Moonshine, che vedono da 2 a 3 ordini di grandezza di dati in più. Il confronto vero è interno (A contro B contro C). I riferimenti esterni servono solo a dire quanto si è lontani da un prodotto.

Non ho trovato modelli pubblicati sotto i 10M parametri allenati solo su 100h, a parte LARM.

## 4. Controlli (regola del progetto precedente)

- **Tutti i bracci hanno gli stessi step, lo stesso LR peak e la stessa schedule** (AdamW, warmup lineare 1000 step e poi coseno, come LARM [P]; peak 7e-4 [P-s]). Anche lo stesso seed, lo stesso ordine dei batch, la stessa SpecAugment leggera e la stessa batch size.
- **Sweep minimo del LR per A** (3 valori: 3e-4, 7e-4, 1.5e-3), su metà degli step. Serve a evitare che il baseline perda solo perché il LR è sbagliato. Nel progetto precedente un semplice calo del LR batteva tutto. Se B batte A solo al LR di default e non al migliore di A, il risultato non conta.
- **A-lungo** (pari FLOP di training) nel run completo. È la versione ASR del controllo "basta allenare di più".
- 2 seed per braccio nel run completo. Le differenze sotto 2× la distanza tra i seed si riportano come "nessuna differenza".
- Il test set si apre una volta sola, a fine run completo. Il pilota usa solo dev.

## 5. Metriche

| metrica | come | note |
|---|---|---|
| WER dev/test clean/other | decoding CTC greedy, normalizzazione testo uguale per tutti (maiuscole, punteggiatura), WER con `jiwer` | metrica primaria. LM 4-gram solo come extra, come in LARM (beam 100, peso LM 0.5 [P]) |
| WER al variare dei loop | B valutato a k = 1, 2, 4, 6, 8, 10 | curva di H3 |
| parametri | `sum(p.numel())` | |
| dimensione su disco | `state_dict` fp32 e fp16, int8 con quantizzazione dinamica dei Linear | va verificato che la quantizzazione dinamica int8 giri su arm64 (backend qnnpack) [non verificato]. In alternativa, export ONNX o Core ML int8 |
| RTF | tempo di calcolo / durata audio, su 200 utterance di dev-clean, batch 1, CPU (1 thread e tutti i thread) e MPS, dopo un warmup | per B a ogni k. Stesso harness per Whisper tiny.en e Moonshine Tiny |
| latenza | tempo totale su una clip da 10 s, p50 e p95 | il caso "dettato" |
| FLOP per secondo di audio | conteggio analitico o `torch.utils.flop_counter` | serve per tenere onesto il confronto a pari FLOP |

## 6. Tempo di training sul Mac [tutto E]

Hardware misurato: Apple M5, 24 GB. PyTorch **non è installato** nel Python di sistema.

Stima del compute:
- FLOP di training per epoca ≈ 6 × (parametri eseguiti per frame) × frame. Con 100h a 25 frame/s ci sono circa 9M frame per epoca.
- B e C (18 blocchi × 0.79M ≈ 14M parametri eseguiti): ≈ 6 × 14M × 9M ≈ 7.7e14 FLOP per epoca, più l'attenzione e il frontend.
- Con un throughput effettivo MPS fp32 di 1–2 TFLOP/s per un modello così piccolo [E, nessuna fonte, **da misurare nel passo 0**], viene 6–13 min per epoca per B e C, e 1.5–3 min per A, senza overhead.
- Con dataloader e padding, la mia stima prudente è **10–20 min per epoca per B e C e 3–5 min per A**.

| run | epoche | stima |
|---|---|---|
| Pilota: A + B + C, 1 seed | 15 | A circa 1 h, B circa 2.5–5 h, C circa 2.5–5 h. **Totale 6–11 h**, più lo sweep LR di A (circa 1.5 h) |
| Completo: A, A-lungo, B, B-LARM, C × 2 seed | 50 (come LARM [P]) | circa 4–6 giorni di Mac acceso [E] |
| Distillazione (opzionale) | pseudo-label di 360h con Moonshine Tiny, poi training su 460h | da 1 a 2 giorni in più [E] |

## 7. Pilota: al massimo circa 1 giorno di Mac

**Passo 0, circa 2 h:**
- Installare PyTorch (versione recente, vedi §8).
- Scaricare train-clean-100 e le parti dev, poi calcolare le fbank.
- Test `ctc_loss` su MPS con lo script di repro di pytorch#203356.
- **Calibrazione:** 200 step per A, B e C, misurando secondi/step. Se la proiezione del pilota supera 20 h, si riducono le epoche (minimo 10) o si passa a d=192. La scelta va scritta **prima** di partire.

**Passo 1, circa 8–12 h (di notte):** A, B e C per 15 epoche, stessi step, stesso LR e stesso seed. Lo sweep LR di A gira a metà degli step.

**Passo 2, circa 1 h:**
- WER su dev-clean e dev-other.
- Curva di B a k = 1…10.
- RTF su CPU e MPS di A, B a ogni k, e C.
- Stesso RTF per Whisper tiny.en e Moonshine Tiny come riferimento.

**Criterio go/no-go**, fissato prima del pilota, tutto su dev-clean greedy:

1. **Sanità:** B < 30% di WER e A < 45%. LARM a 50 epoche con d=384 fa 11.34 e 26.78 [P]; noi abbiamo meno epoche e d più piccolo. Se fallisce, è un bug: si corregge e si ripete, non è un no-go.
2. **H1:** B ≤ 0.8 × WER(A migliore dello sweep). Altrimenti **no-go**: l'effetto principale di LARM non si riproduce.
3. **H2:** B ≤ 1.05 × WER(C).
4. **Velocità, quella che interessa a te:** esiste un k < K tale che B(k) ha WER ≤ WER(C) + 5% relativo **ed** è almeno 1.5× più veloce di C su CPU, misurato in wall clock.
5. Decisione:
   - **Go** se valgono 2, 3 e 4.
   - Se valgono 2 e 3 ma non 4: il risultato è "più piccolo, non più veloce", come nello studio precedente. Si finanzia **solo** se il target è la dimensione su disco o in RAM (per esempio embedded), non la velocità.
   - In tutti gli altri casi, no-go.

## 8. Codice da cui partire

| cosa | licenza | uso | note |
|---|---|---|---|
| PyTorch, `nn.functional.ctc_loss` su MPS | BSD-3 [non riletto] | training | forward nativo dal 26-06-2026 (PR #187716), backward dal 29-06-2026 (PR #188187) [P] https://github.com/pytorch/pytorch/pull/187716 https://github.com/pytorch/pytorch/pull/188187. **Bug** con target ≥ 512 label (gradienti inf), che LibriSpeech a caratteri raggiunge (516 caratteri). Chiuso il 07-11-2026 [P] https://github.com/pytorch/pytorch/issues/203356. Versione di rilascio non verificata. Rimedi: BPE 256, oppure filtrare le utterance oltre 500 caratteri, oppure CPU fallback |
| torchaudio | BSD-2 [non verificato] | fbank, `models.Conformer` | in **manutenzione**: alcune API rimosse in 2.9, migrazione completata in 2.10, I/O passato a TorchCodec; `models` resta [P] https://github.com/pytorch/audio/issues/3902. Leggere i FLAC con `soundfile` e calcolare le fbank con le funzioni torchaudio ancora presenti, verificandole |
| HF `openslr/librispeech_asr` | CC BY 4.0 | dati in parquet | [P] https://huggingface.co/datasets/openslr/librispeech_asr |
| ESPnet `egs2/librispeech_100` | Apache-2.0 | ricetta di riferimento (iperparametri, SpecAugment, normalizzazione) | [P] https://github.com/espnet/espnet. CI su macOS, MPS non documentato |
| SpeechBrain | Apache-2.0 | recipe LibriSpeech CTC/Conformer | [P] https://github.com/speechbrain/speechbrain. Lo YAML del Conformer piccolo non è verificato |
| icefall (k2) | Apache-2.0 | baseline Zipformer-CTC (il più piccolo visto: 22.1M, 2.57/5.95 su 960h) | [P] https://github.com/k2-fsa/icefall. Dipende da k2 e lhotse; Mac e MPS non verificati, quindi **sconsigliato come base** |
| LARM | paper CC BY 4.0 | design di riferimento | codice "will be released soon", per ora nessun repo [P] |
| Moonshine | MIT (modelli inglesi MIT) | riferimento e maestro per la distillazione | [P] https://github.com/usefulsensors/moonshine (`pip install moonshine-voice`, macOS supportato) |
| Whisper | MIT | riferimento e maestro | [P] https://github.com/openai/whisper (richiede ffmpeg) |
| jiwer | Apache-2.0 [a memoria, non verificato] | WER | |

**Consiglio [C]:** un training loop in PyTorch puro, di circa 400 righe: dataset da fbank precalcolate, encoder Transformer, CTC, greedy decode, jiwer. Da ESPnet si copiano solo iperparametri e normalizzazione. I framework (ESPnet, icefall, NeMo) costano giorni di setup su Mac, e NeMo richiede GPU NVIDIA per il training [P].

## 9. Distillazione (opzionale, solo dopo il go)

Si trascrive train-clean-360 con Moonshine Tiny (MIT) e si allena su 100h di etichette vere più 360h di pseudo-label. Per CTC è distillazione a livello di sequenza (testo del maestro), non KL per frame, perché il maestro è seq2seq con un tokenizer diverso [C].

Il controllo è lo stesso braccio con gli stessi step solo su 100h. Questo braccio misura quanto del divario con Moonshine si chiude con i dati e non con l'architettura. Il loop si giudica su A, B e C.

## 10. Rischi aperti

- **Throughput MPS sconosciuto.** Tutte le stime di tempo sono [E]. Il passo 0 serve proprio a misurarlo.
- **Bug di `ctc_loss` su MPS** con target lunghi (fix del 07-11-2026, release non verificata). In più, MPS non ha float64.
- **torchaudio in manutenzione.** Alcune funzioni per le fbank potrebbero essere state rimosse in 2.9 o 2.10 (non verificato quali).
- **Su 100h il loop potrebbe vincere per regolarizzazione, non per efficienza.** LARM perde contro il 48 blocchi su 960h [P]. Un go sul pilota non garantisce il risultato su più dati.
- **H5 (velocità) è improbabile per costruzione:** a pari FLOP il loop non accelera [P, F5-TTS]. Il go dipende quasi tutto dalla manopola "meno loop".
- **LARM è letto in parte da un riassunto.** Tabella 4, ablation e iperparametri esatti sono [P-s]. Nessun codice, quindi la riproduzione non è garantita.
- **int8 su arm64:** la quantizzazione dinamica PyTorch su Mac non è verificata. Potrebbe servire Core ML o ONNX, e il confronto di velocità diventerebbe un confronto tra runtime.
- **Whisper e Moonshine** sono allenati su molti più dati: il confronto esterno dice solo quanto si è lontani da un prodotto, non se il loop funziona.

## Fonti

- LARM: https://arxiv.org/abs/2606.04678 (Tab. 1–2 verbatim; Tab. 3–5 da riassunto)
- Studio precedente: `ideas/inkling-2-mini-study.md` (Inkling, F5-TTS ricorsivo arXiv 2609.29768)
- LibriSpeech: https://www.openslr.org/12 ; https://huggingface.co/datasets/openslr/librispeech_asr
- Moonshine: https://arxiv.org/abs/2410.15608 ; https://github.com/usefulsensors/moonshine
- Whisper: https://arxiv.org/abs/2212.04356 ; https://github.com/openai/whisper
- Conformer: https://arxiv.org/abs/2005.08100 ; Squeezeformer: https://arxiv.org/abs/2206.00888
- ESPnet librispeech_100: https://github.com/espnet/espnet/tree/master/egs2/librispeech_100/asr1
- icefall: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md
- PyTorch MPS CTC: PR #187716, PR #188187, issue #203356 ; torchaudio: https://github.com/pytorch/audio/issues/3902
