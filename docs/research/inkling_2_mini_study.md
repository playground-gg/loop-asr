# Inkling 2 Mini: studio e cosa trasferire su dLLM mascherati e modelli vocali

Scritto il 2026-11-23. Solo lettura: nessun codice modificato, nessun esperimento eseguito.
Marcatori: **[P]** = letto su una fonte primaria (pagina indicata), **[S]** = fonte secondaria, **[M]** = misurato nel repo `wsp-adaptive-decision-models`, **[C]** = congettura mia, non verificata.

## Summary

- Il modello esiste e non è ambiguo: **Inkling 2 Mini**, di Thinking Machines Lab, uscito il **20 novembre 2026**, pesi aperti con licenza **Apache 2.0**. È un MoE da 26.4B parametri totali e 3.4B attivi. Su 24 layer, i 12 centrali vengono eseguiti 3 volte con pesi condivisi (*looping*).
- Da copiare c'è soprattutto il **looping con un messaggio passato tra un passaggio e l'altro**, controllato da un gate che parte chiuso e da un numero di passaggi variabile in training. Per un modello di diffusione mascherato si trasferisce quasi uno a uno, ed esiste già un paper (LoopMDM) che lo prova anche su Sudoku.
- La seconda cosa da copiare è la **distillazione on-policy**, usata da Thinking Machines nel post-training. Applicata al tuo caso diventa "il modello a 81 passi insegna a quello a 4 passi, sugli stati che visita lo studente". Attacca direttamente il divario 4→81 passi che hai misurato.
- Training in 4 bit (NVFP4), pruning di 2/3 degli esperti, contesto da 1M, attenzione ibrida e monitorabilità della CoT **non servono** al tuo setup: richiedono GPU Blackwell, oppure il problema che risolvono a 1.86M parametri non c'è.
- Lato voce l'idea che trasferisce è sempre il looping: due paper del 2026 riportano −47% di parametri su F5-TTS e un ASR da 7.7M che batte uno da 28.9M. Allenare da zero un modello vocale sul Mac però costa molto più di un esperimento Sudoku.

---

## 1. Quale modello è (verifica)

| campo | valore | fonte |
|---|---|---|
| Nome | Inkling 2 Mini (repo HF `thinkingmachines/Inkling-2-Mini`) | [P] https://thinkingmachines.ai/news/inkling-2-mini/ |
| Laboratorio | Thinking Machines Lab | [P] stesso post |
| Data | 20 novembre 2026 | [P] post e model card https://thinkingmachines.ai/model-card/inkling-2-mini/ |
| Licenza | Apache 2.0, più una Acceptable Use Policy (citata da LM Studio) | [P] model card; [P] tag HF https://huggingface.co/thinkingmachines/Inkling-2-Mini ; [S] https://lmstudio.ai/models/inkling-2-mini |
| Famiglia | Il più piccolo della seconda generazione Inkling (Inkling 2 Small: 594B totali / 17B attivi, uscito il 5 novembre 2026) | [P] https://thinkingmachines.ai/news/inkling-2-small/ |
| Modello gemello | `TML-d24-varloop-research`: stessa architettura, numero di passaggi scelto all'inferenza (allenato con 1–5), solo per ricerca | [P] https://thinkingmachines.ai/model-card/tml-d24-varloop-research/ |

**Non c'è un technical report o un paper dedicato.** Le fonti primarie sono il post di lancio, le due model card, la documentazione Transformers (https://huggingface.co/docs/transformers/en/model_doc/inkling2_loop) e il blog HF (https://huggingface.co/blog/inkling2mini). Il paper "How to Loop Your Transformer" (arXiv 2610.40325), che compare nelle ricerche accanto a Inkling, è di **FAIR/Meta** e non di Thinking Machines. Va letto come lavoro correlato, non come fonte su Inkling.

## 2. Scheda tecnica

### Architettura [P salvo dove indicato]
- **24 layer** decoder-only in 4 blocchi da 6. I layer 0–5 girano una volta, i layer **6–17 tre volte con pesi condivisi**, i layer 18–23 una volta. In totale 48 applicazioni di layer per token (post di lancio, docs Transformers).
- **Parametri:** 26.4B totali e 3.4B attivi a un passaggio. Ogni passaggio in più aggiunge circa 1.0B di parametri attivi effettivi, quindi 5.4B a 3 passaggi (model card). Artificial Analysis scrive "3.1B unique weights per text token", in conflitto con il 3.4B ufficiale e senza spiegazione [S] https://artificialanalysis.ai/articles/inkling-2-mini .
- **Messaggio tra passaggi:** ogni passaggio riparte dall'output del preludio (layer 0–5). Nei passaggi 2 e 3 ogni layer del core legge un "messaggio" dal passaggio precedente, con una propria query appresa. Un **gate appreso** fonde messaggio e preludio e **all'inizio blocca l'intero messaggio**; la sua apertura cresce durante il training (post di lancio). Il gate ha rango 64 (`loop_gate_rank: 64`) e quattro gate mescolano il messaggio negli input di Q, K, V e MoE (docs Transformers; blog HF).
- **Attention Residuals** (Kimi Team, arXiv 2603.15031): al posto della somma residua a peso fisso, ogni layer fa attenzione softmax sulle uscite dei blocchi precedenti con pesi che dipendono dall'input (`n_layers_per_attn_res_block: 6`). Dall'abstract l'overhead è "minimal", ma i numeri non li ho letti [P, solo abstract] https://arxiv.org/abs/2603.15031 .
- **Dove mettere il loop:** nello studio interno, il loop sui layer centrali dà 1.51–1.91× di guadagno compute-equivalente contro 1.24–1.36× sugli ultimi layer, e vince in 27 valutazioni su 28 (post di lancio).
- **Loop contro nessun loop, a parità di loss:** il modello non-loop costa il 7.5% in meno da allenare ma richiede il **140% in più di compute in inferenza** per raggiungere la stessa loss su tracce agentiche da 512k token (post di lancio).
- **MoE:** pre-training con 384 esperti e top-8. A metà del midtraining sono stati **potati due terzi degli esperti**, tenendo i 128 più usati per layer. Il modello rilasciato ha 128 esperti e top-8, routing con sigmoide, `moe_intermediate_size` 1152 (post; docs). Goodfire scrive "147,456 neurons" per MoE, cioè 384×384: non quadra con 128 esperti, ed è una discrepanza non risolta [S] https://www.goodfire.com/research/interpreting-inkling-2-mini/ .
- **Attenzione:** ibrida locale/globale. Ogni blocco da 6 ha 5 layer a finestra scorrevole (1024 token) e 1 layer globale. 24 head di query e 8 KV (GQA), head_dim 128, bias di posizione relativa appreso, *sink* appreso, output gate, convoluzioni corte (kernel 4) su K/V nei layer globali, RMSNorm, softcap su logit e MLP. Hidden size 2304 (docs Transformers; blog HF).
- **KV cache:** ogni passaggio tiene le proprie K/V, quindi la cache ha 48 "layer" e non 24 (docs Transformers). Il loop **non fa risparmiare memoria di cache**: la memoria risparmiata è solo quella dei pesi.

### Tokenizer e modalità [P]
- Vocabolario 200,058 (201,024 con il padding), identico a Inkling 2 Small. Il tipo di tokenizer non è nominato (docs Transformers).
- **Input testo e immagini** (encoder ViT; quello di Inkling 2 Small ha 292M parametri). **Output solo testo.** Contesto fino a **1M token**. **Niente audio:** le firme del forward contengono campi `audio_outputs`/`video_outputs`, ma nessuna pagina descrive un encoder audio [P, docs Transformers; non verificato oltre].

### Dati e ricetta [P]
- Dati: "text, images, video" da fonti pubbliche, terze, interne e sintetiche, con deduplica e filtri (model card). **Il numero di token di pre-training di Mini non è pubblicato.** Per confronto, Small ha usato 8.4T + 3.8T (a 512K) + ~320B (a 1M) + ~34B token [P, pagina Small].
- Pre-training dalla "mainline predictive scaling ladder", poi midtraining. Durante il midtraining: introduzione del loop, pruning 384→128 esperti, **"healing" con 1T token distillati da un modello Inkling interno più grande** (non nominato) (post di lancio).
- **Precisione:** NVFP4 nel forward dei layer esperti dal pre-training all'RL, MXFP8 negli ultimi 4 layer. Il checkpoint rilasciato ha lo stesso formato (20 layer NVFP4 e 4 MXFP8), quindi non c'è perdita da conversione. Dettagli della ricetta [P] https://thinkingmachines.ai/blog/end-to-end-4-bit-training/ : attivazioni NVFP4 per token con scaling adattivo "4/6"; backward in MXFP8 con stochastic rounding; Hadamard a 32 dimensioni solo nel *combine* MoE; l'ultimo sesto dei layer in MXFP8. Costa circa il 2% di efficienza di compute rispetto a MXFP8 pieno, con speedup dell'8% nel pre-training e del 20% nell'RL. **Usa istruzioni specifiche di GB300/Blackwell.**

### Post-training [P]
- SFT con un mix simile a Small, poi **RL per 2.2T token in 2,000 step**, poi **distillazione on-policy** per l'allineamento (post di lancio; model card).
- Reasoning effort regolabile da 0.0 a 0.99 (none 0.0, minimal 0.1, low 0.2, medium 0.7, high 0.9 di default, xhigh/max 0.99) (blog HF). Il meccanismo non è descritto. "Minimal" non spegne del tutto il ragionamento (model card).
- Variante varloop: midtraining con 1–5 passaggi campionati in rapporto **1:2:6:2:1**, post-training a 3 passaggi. SWE-bench Verified da 1 a 5 passaggi: 56.3 / 76.5 / 78.7 / 79.3 / 79.6. DeepSWE: 0.7 / 28.3 / 47.4 / 46.9 / 48.2. La sequenza 1→5 l'ha ricostruita il fetcher dal testo del grafico [P, model card varloop; ordinamento da confermare a occhio].
- Nel varloop il tempo utile senza CoT sale da 16 s con 1 passaggio a 37 s con 3, poi resta circa piatto. N-Hop Lookup migliora fino a 10 passaggi. I benchmark di conoscenza (AA-Omniscience, SimpleQA…) **non migliorano con i loop** (post di lancio).

### Inferenza [P]
- Checkpoint misto NVFP4/MXFP8 di circa 20 GB. Consigliati ≥32 GB, Ollama/Exo/MLX su Apple Silicon. In MLX il checkpoint originale gira "without reconversion" (blog HF).
- GGUF IQ4_NL e Q8_0 (`ggml-org/Inkling-2-Mini-GGUF`). LM Studio indica un download da 17.5 GB e 18 GB di memoria minima [S]. **Su un Mac da 24 GB l'IQ4_NL ci sta, al limite, con poco contesto [C].**
- **Speculative decoding DSpark:** un drafter semi-autoregressivo propone un blocco intero in un solo passaggio, con una testa di Markov per le dipendenze interne al blocco e una testa di confidenza che stima l'accettazione. Legge gli stati delle visite 5, 23, 41 e 48 (blog HF; https://docs.nvidia.com/nemo/automodel/latest/recipes-e2e-examples/dspark-speculative-decoding ). RadixArk dichiara 1.89× su 8×B200, numero suo e non verificato [S].
- **Niente early exit né loop adattivo** nel modello di produzione: "only supports a fixed number of looping passes" (post).

### Benchmark (dichiarati dal vendor; "evaluation setups vary") [P]
SWE-bench Verified 79.2, DeepSWE v1.1 48.2 (Qwen3.8-27B: 42.2), Terminal Bench v2.1 71.6, BrowseComp 79.0 (con compaction), GPQA Diamond 86.6, ARC-AGI-1/2 84.5/49.3, MMMU Pro 74.9, LongBench v2 66.9, AA Index 20.1. Artificial Analysis, indipendente, gli dà 20 sull'Intelligence Index, alla pari con Qwen3.8 27B, con circa 2.8× meno compute (parametri attivi × token) ma circa il 38% di token di output in più [S]. Limiti dichiarati: ragionamento ripetitivo, conoscenza fattuale debole, istruzioni con molti vincoli, lingue diverse dall'inglese.

### Cosa fanno i loop dentro (Goodfire, "preliminary") [S]
Nei passaggi 2 e 3 i lettori cross-loop leggono quasi sempre il blocco 3 del passaggio precedente. Una coordinata outlier (d2027) si alza al layer 18 nei passaggi 1 e 2 (a loro avviso segnala "leggi qui") e scende di colpo nel passaggio 3, dove non c'è un passaggio successivo. Goodfire **non misura** convergenza o raffinamento iterativo: non attribuire loro quella lettura.

---

## 3. Le innovazioni, una per una, con verdetto

Legenda della matrice: **S** = più intelligente, **F** = più veloce, **P** = più piccolo. **Sì** = trasferisce ed è da provare. **Forse** = plausibile ma incerto o costoso. **No** = non trasferisce o non serve.

| # | innovazione | fonte | Diff S | Diff F | Diff P | Voce S | Voce F | Voce P |
|---|---|---|---|---|---|---|---|---|
| I1 | Loop del blocco centrale (3 passaggi, pesi condivisi) | post di lancio | **Sì** | Forse | **Sì** | Forse | No | **Sì** |
| I2 | Messaggio tra passaggi con gate a inizio chiuso | post; docs | **Sì** | No | No | Forse | No | No |
| I3 | Numero di passaggi variabile in training (1:2:6:2:1) → manopola di compute in inferenza | model card varloop | **Sì** | **Sì** | No | Forse | **Sì** | No |
| I4 | Distillazione on-policy | post; model card | **Sì** | **Sì** | Forse | Forse | Forse | Forse |
| I5 | Pruning degli esperti a metà training + healing per distillazione | post | No | No | Forse | No | No | Forse |
| I6 | Training nativo NVFP4/MXFP8 | blog 4-bit | No | No | No | No | No | Forse |
| I7 | Attention Residuals | post; arXiv 2603.15031 | Forse | No | No | Forse | No | No |
| I8 | Attenzione ibrida locale/globale, sink, posizione relativa, conv corte | docs; blog HF | No | No | No | Forse | **Sì** | No |
| I9 | Drafter DSpark (bozza parallela a blocchi + testa di confidenza) | blog HF; docs NVIDIA | Forse | Forse | No | No | **Sì** | No |
| I10 | RL lungo (2.2T token) + reasoning effort | post; blog HF | No | No | No | No | No | No |
| I11 | Contesto 1M, monitorabilità della CoT | model card | No | No | No | No | No | No |

### I1. Loop del blocco centrale
**Cosa dice Inkling.** Loop sui layer centrali e non sugli ultimi (1.51–1.91× contro 1.24–1.36×). A parità di loss, il modello senza loop richiede il 140% in più di compute in inferenza [P].
**Lavoro vicino al tuo caso.** *Looped Diffusion Language Models* (LoopMDM, arXiv 2605.26106; NeurIPS 2026 secondo la pagina poster https://neurips.cc/virtual/2026/poster/152196). DiT da 170M con un loop sui primi layer centrali (layer 1–2), fino a 12 iterazioni in training, numero variabile in inferenza. Letto verbatim: stessa NLL di test con 3.34× / 2.95× / 2.34× meno FLOP (LM1B / OWT / FineWeb-Edu). Il **Sudoku con ordine di generazione ristretto risolto solo con il loop** e l'aumento di GSM8K al crescere dei loop vengono invece da un riassunto, non dal testo [P parziale]. Teoria: Svete e Sabharwal (ICLR 2026, arXiv 2510.13117) confrontano l'espressività dei MDM con i looped transformer [P, solo snippet].
**Diffusione.**
- **S, Sì.** Il tuo modello ha 6 layer. In fase 4 un grid fallito è "uno o due errori presto, poi celle coerenti con quegli errori" [M], e a 4 passi il modello deve fissare circa 14 celle per passo con un solo forward. Più profondità per passo è esattamente ciò che manca quando i passi sono pochi [C].
- **F, Forse.** Il loop aggiunge compute per passo. Il guadagno c'è solo se 4 passi × 3 loop battono circa 7 passi senza loop a pari FLOP, ed è la domanda da misurare.
- **P, Sì.** A parità di qualità, meno pesi salvati. A 1.86M parametri però "più piccolo" non ti serve: conta come leva di efficienza, non di memoria.

**Voce.**
- *Depth through recurrence* (arXiv 2609.29768): F5-TTS Small modificato, da 158.0M a 83.6M parametri (−47.1%) con lo schema "SEQUENCE 9×2". WER 2.03 contro 2.23 (Seed-TTS) e 2.26 contro 2.29 (LibriSpeech-PC) a 32 passi. Memoria 877 → 574 MB, **RTF invariato** (0.143–0.144). Allenato su 8 GPU per circa 500k update [P, HTML letto per intero].
- LARM (arXiv 2606.04678): encoder ASR condiviso da 4 blocchi ripetuto 12 volte, 7.7M parametri contro 28.9M del modello standard a 16 blocchi. WER 11.34/31.84 contro 14.43/37.23 (100h, clean/other) e 4.59/11.75 contro 4.79/13.26 (960h) [P, HTML].

Il loop rende i modelli vocali **più piccoli, non più veloci**: il compute per frame resta lo stesso o cresce.

### I2. Messaggio tra passaggi con gate inizializzato chiuso
**Cosa dice Inkling.** Il gate all'inizio blocca tutto il messaggio e si apre col training [P]. Il vantaggio pratico è che **inserire un loop parte dalla funzione originale**: niente shock iniziale.
**Diffusione, S: Sì, con due usi.**
1. Dentro il loop, come in Inkling, per fare *retrofitting* di un checkpoint esistente senza regressione iniziale [C].
2. **Tra un passo di denoising e il successivo** [C]. Il tuo sampler oggi passa al passo dopo solo i token committati e perde lo stato interno, comprese le confidenze. Un messaggio gated dallo stato nascosto del passo t al passo t+1 è un parente stretto del *self-conditioning* (Chen et al., *Analog Bits*, arXiv 2208.04202, non riaperto in questa sessione). Attacca il collo di bottiglia che hai misurato in 4.5b: la riparazione con oracolo arriva a 0.907, mentre nessun detector disponibile supera un AUROC di circa 0.6 [M]. Che lo stato portato avanti aiuti a *rilevare* gli errori è una congettura mia.

**Voce, Forse.** In TTS flow-matching, a ogni passo ODE, lo stesso trucco si chiama di nuovo self-conditioning [C].

### I3. Passaggi variabili in training → manopola in inferenza
**Cosa dice Inkling.** Rapporto 1:2:6:2:1 su 1–5 passaggi. A 1 passaggio crolla (SWE-V 56.3), da 3 a 5 guadagna poco [P].
**Diffusione, S e F: Sì.** È la cosa più vicina alla tua domanda aperta ("confidence-based adaptive step budgets look like the obvious next thing", REPORT.md). Con i passaggi variabili hai **due manopole, passi di denoising × loop per passo**, e puoi spenderle in modo diverso lungo la traiettoria: più loop nei primi passi (dove nascono gli errori [M]) e meno negli ultimi, quando quasi tutto è fissato [C]. LoopMDM fa già halting adattivo (soglia 0.1), ma il dato viene dal riassunto e non dal testo.
**Voce, F: Sì.** LARM tratta la profondità come manopola test-time per l'ASR, con checkpoint di riconoscimento intermedi [P]: uscita anticipata nei frame facili, cioè latenza per il riconoscimento in streaming. Il loop-exit di LARM era nella parte letta solo come riassunto, e il fetcher l'ha segnalato come sospetto.

### I4. Distillazione on-policy
**Cosa dice Inkling.** Usata nel post-training per l'allineamento. L'"healing" invece è distillazione off-policy da 1T token [P]. Il metodo è descritto da Thinking Machines nel post "On-Policy Distillation" (https://thinkingmachines.ai/blog/on-policy-distillation/, ottobre 2025): lo studente campiona, il maestro dà la distribuzione per token e la loss è un reverse KL per token. Non l'ho riaperto in questa sessione.
**Diffusione, S e F: Sì.** Il tuo modello a 4 passi risolve 0.634, lo stesso modello a 81 passi 0.897 [M]. Il maestro gratis è **lo stesso checkpoint con più passi** (o con più loop, se fai I1+I3). Lo studente fa rollout a 4 passi; in ogni stato intermedio che visita, il maestro dà la distribuzione per cella e la loss è il KL per cella. Rispetto alle tue RL:
- il segnale è **denso** e non binario (in 4.4 il 72% dei gruppi aveva reward tutti uguali [M]);
- **non c'è il termine che spinge giù senza limite**, quello che ha rovinato wd1 [M];
- è on-policy, quindi corregge proprio gli stati in cui lo studente sbaglia.

La distillazione a pochi passi per diffusione discreta esiste già (ad esempio SDTT, arXiv 2410.21035, non riaperto). La versione on-policy per stato, sul tuo Sudoku, non l'ho cercata in modo specifico.
**Voce.** È la strada standard per TTS a pochi passi e per studenti STT più piccoli. Distil-Whisper (arXiv 2311.00430, non riaperto) è off-policy. Per la voce è Forse: serve un maestro vocale che giri in locale.

### I5. Pruning degli esperti e healing
Pruning di 2/3 degli esperti in base all'uso, poi 1T token di distillazione [P]. Il tuo modello è denso e da 1.86M: **non trasferisce**. Per la voce l'analogo è potare layer o head di un modello pre-allenato e "guarire" per distillazione (è la ricetta di Distil-Whisper) [C]. Forse, solo per P.

### I6. Training in 4 bit
Richiede GB300/Blackwell (istruzioni PTX dedicate, stochastic rounding in hardware) [P]. Sul Mac non accelera nulla; al massimo si emula, ed è più lento. A 1.86M parametri la memoria non è un problema. Per la voce in deployment conta la *quantizzazione post-training* (MLX/GGUF), non questo training. **No** su tutto, salvo un Forse per la voce P nel senso di "quantizza il modello finale".

### I7. Attention Residuals
Utile su modelli profondi contro la "PreNorm dilution" [P, abstract]. Con 6 layer l'effetto atteso è piccolo [C]. Diventa interessante solo se fai I1, perché nei modelli loopati è il canale da cui passa il messaggio. Forse.

### I8. Attenzione ibrida, sink, convoluzioni corte
Servono per contesti lunghi. Il Sudoku ha 81 token: **No**. Nella voce le sequenze sono lunghe (50–100 frame al secondo), quindi finestre locali più qualche layer globale riducono il costo dello streaming. È però pratica già standard nei modelli vocali, non un'idea nuova di Inkling [C].

### I9. Drafter DSpark
Lo speculative decoding vale per modelli **autoregressivi**. I tuoi dLLM generano già in parallelo, quindi per la diffusione il punto è l'inverso: DSpark *è* un piccolo generatore parallelo a blocchi con **testa di confidenza che predice l'accettazione** [P, docs NVIDIA]. Quella testa è una forma di detector degli errori allenato con target espliciti, la tua direzione 5 [C]. Forse. Per i TTS autoregressivi su codec (token audio) lo speculative decoding è una leva di velocità reale: Sì, F.

### I10–I11. RL lungo, reasoning effort, contesto 1M, monitorabilità
Sono risultati di scala (2.2T token di RL) o proprietà di prodotto. Non trasferiscono a un modello da 1.86M né a un modello vocale piccolo. **No.**

---

## 4. I primi esperimenti più economici (Mac da 24 GB, MPS)

Una regola valida per tutti, imparata in fase 4: ogni braccio con fine-tuning ha un **controllo con lo stesso numero di step allo stesso learning rate senza la modifica**. In 4.6–4.9 il drop del learning rate valeva più di ogni RL [M].

### E1. Retrofit di un loop con gate chiuso sul modello 9×9 (I1+I2+I3), circa 2–3 h
- **Partenza:** il checkpoint supervisionato dopo il drop del LR (0.634 / 0.874 / 0.897 [M]). Config attuale: `n_layers: 6, d_model: 160` (`configs/phase4.yaml:23-25`).
- **Modifica:** preludio = layer 0–1, core = layer 2–3 (come in LoopMDM, nei primi layer centrali), coda = layer 4–5. Il messaggio dal passaggio precedente entra con un gate di rango basso inizializzato a 0, così a step 0 il modello è identico all'originale. Passaggi campionati in {1,2,3,4,5} in rapporto 1:2:6:2:1. Loss SFT invariata, lr 1e-5, circa 4,000 step come il controllo di 4.6.
- **Misura:** solve rate hard test a (passi ∈ {2, 4, 8, 16}) × (loop ∈ {1, 3, 5}), più FLOP per puzzle. Confronto **a pari FLOP** contro il modello senza loop a 6–7 passi (da misurare, circa 2 min).
- **Predizioni [C]:** a 4 passi, 3 loop guadagnano almeno +0.05 sul controllo; a 81 passi il guadagno è vicino a zero, perché lì l'ordine compensa già. Se 4×3 a pari FLOP non batte circa 7 passi senza loop, per la velocità il loop non serve e resta solo la leva "più piccolo".
- **Fallback se il retrofit non si muove:** training da zero, 5 h, come il supervisionato.

### E2. Distillazione on-policy da 81 a 4 passi (I4), circa 1.5 h
- **Studente:** lo stesso checkpoint a 4 passi. **Maestro:** una copia congelata, che in ogni stato visitato dallo studente dà le distribuzioni per cella ottenute "come se" continuasse a molti passi. Variante più semplice: la distribuzione del maestro con un forward sullo stesso stato, a temperatura bassa. Loss: reverse KL per cella sulle celle ancora mascherate, con pesi uguali per tutti i passi.
- **Riuso:** i rollout di `src/phase3/rl.py`. Nessun reward e nessun checker.
- **Controllo:** stesso numero di step con la loss SFT allo stesso lr.
- **Predizione [C]:** colma almeno metà del divario 4-passi tra 0.634 e 0.874 (il 16-passi). Se non lo fa, il collo di bottiglia non sono le distribuzioni per stato ma l'ordine di commit. In quel caso il maestro deve dare anche *quali* celle committare: una distillazione dell'ordine, che si collega al tuo oracolo d'ordine al 98% di fase 3 [M].

### E3. Messaggio tra passi di denoising (I2 trasposto), circa 1–1.5 h
- **Modifica:** al passo t+1 il modello riceve, oltre ai token, lo stato nascosto finale del passo t, proiettato e moltiplicato per un gate inizializzato a 0. In training si simula con due passi: si fa un primo forward su una maschera, si rimaschera, si fa il secondo forward con il messaggio (lo schema del self-conditioning).
- **Misura:** solve rate a 4/16 passi, e AUROC di un probe lineare sui wrong cell *dentro i grid falliti* rispetto a 4.5b (0.41–0.61 [M]).
- **Predizione [C]:** a 4 passi guadagna più a 4 passi che a 16; l'AUROC del probe sale sopra 0.7.

### E4 (voce, il più economico che abbia senso): un ASR loopato in stile LARM, mezza giornata o più, stima non misurata
- Encoder CTC da 4 blocchi condivisi ripetuti K volte, d=256–384, su un sottoinsieme di LibriSpeech train-clean-100. Confronto con uno standard da 4 e uno da 16 blocchi. LARM ha usato 100h e 960h, su hardware non indicato nel testo letto [P].
- **Onestà:** è molto più lento dei tuoi esperimenti Sudoku. Con lo stesso budget il ritorno di E1–E3 è maggiore. L'unico motivo per farlo è se l'obiettivo diventa la voce on-device.
- **Alternativa senza training:** usare Inkling 2 Mini (IQ4_NL, circa 17.5 GB) come "cervello" testuale in una pipeline STT → LLM → TTS. È uso del modello, non trasferimento di innovazioni, e i 24 GB sono al limite.

---

## 5. Classifica per valore atteso / costo

1. **E1: retrofit del loop con gate chiuso e passaggi variabili.** È l'innovazione centrale di Inkling, ha un precedente diretto su MDM e Sudoku (LoopMDM), parte senza regressione e dà la manopola loop × passi che il tuo REPORT indica come prossimo passo naturale. Costo circa 2–3 h.
2. **E2: distillazione on-policy da 81 a 4 passi.** Segnale denso, niente termine negativo illimitato, maestro gratis. Attacca il divario 4→16 passi (0.634 → 0.874), che finora nessuna RL ha toccato. Costo circa 1.5 h.
3. **E3: messaggio gated tra passi di denoising.** Economico, mira al collo di bottiglia della detection (4.5b) e si combina con E1. Valore più incerto.

Per la voce, nessun esperimento entra nel top 3 per rapporto costo/valore. La lezione utile da Inkling e dai paper TTS/ASR del 2026 è "loop = meno parametri a parità di qualità, non più velocità".

## 6. Cosa non trasferisce, detto chiaro
- **Training in 4 bit:** serve hardware Blackwell, non ha senso a 1.86M e non accelera sul Mac.
- **MoE e pruning degli esperti:** il tuo modello è denso e minuscolo.
- **Contesto 1M e attenzione ibrida:** 81 token. Per la voce è già pratica corrente.
- **RL da 2.2T token, reasoning effort, monitorabilità della CoT:** risultati di scala o di prodotto.
- **Benchmark di Inkling:** riguardano coding, agenti e visione. Non dicono nulla su diffusione o voce.

## 7. Non verificato
- Niente technical report: numero di token di pre-training di Mini, identità del maestro dell'healing, top-k prima del pruning, meccanismo del reasoning effort.
- Conflitti aperti: 3.4B attivi (TML) contro 3.1B (Artificial Analysis); "147,456 neurons" per MoE (Goodfire) contro 128 esperti.
- Attention Residuals: letto solo l'abstract, il PDF non era leggibile qui.
- LoopMDM: Sudoku, GSM8K per numero di loop, halting adattivo e hardware vengono da un riassunto, non dal testo. Il codice è solo annunciato.
- LARM: tabella di loop-exit non verificata e segnalata come sospetta. Codice "will be released soon".
- Paper "How to Loop Your Transformer" (FAIR/Meta, 2610.40325): letto per meno di un quarto, e nessuna legge di scaling esplicita trovata.
- DSpark 1.89×: numero di RadixArk, non indipendente.
- Citati a memoria e non riaperti in questa sessione: il post On-Policy Distillation di Thinking Machines, Analog Bits (2208.04202), SDTT (2410.21035), Distil-Whisper (2311.00430).
- Marktechpost (403) non letto. Benchmark non riprodotti.
- Che Inkling 2 Mini IQ4_NL giri bene su 24 GB: non provato.
- Tutte le predizioni e i tempi di E1–E4 sono **[C]**.

## 8. Fonti
Primarie (Thinking Machines e HF):
- https://thinkingmachines.ai/news/inkling-2-mini/
- https://thinkingmachines.ai/model-card/inkling-2-mini/
- https://thinkingmachines.ai/model-card/tml-d24-varloop-research/
- https://huggingface.co/thinkingmachines/Inkling-2-Mini
- https://huggingface.co/docs/transformers/en/model_doc/inkling2_loop
- https://huggingface.co/blog/inkling2mini
- https://thinkingmachines.ai/blog/end-to-end-4-bit-training/
- https://thinkingmachines.ai/news/inkling-2-small/

Paper e documentazione:
- Attention Residuals: https://arxiv.org/abs/2603.15031
- LoopMDM: https://arxiv.org/abs/2605.26106
- Depth through recurrence (TTS): https://arxiv.org/abs/2609.29768
- LARM (ASR): https://arxiv.org/abs/2606.04678
- How to Loop Your Transformer (FAIR/Meta): https://arxiv.org/abs/2610.40325
- Svete e Sabharwal: https://arxiv.org/abs/2510.13117
- DSpark (NVIDIA NeMo): https://docs.nvidia.com/nemo/automodel/latest/recipes-e2e-examples/dspark-speculative-decoding

Secondarie:
- https://artificialanalysis.ai/articles/inkling-2-mini
- https://www.goodfire.com/research/interpreting-inkling-2-mini/
- https://lmstudio.ai/models/inkling-2-mini
- https://huggingface.co/RadixArk/Inkling-DSpark-Preview

Repo dell'esperimento: `README.md`, `results/REPORT.md`, `docs/RL_DLLM_DIRECTIONS.md`, `configs/phase4.yaml`.
