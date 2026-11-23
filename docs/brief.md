status: project playground-gg/loop-asr

# Tiny STT: un voice-to-text minuscolo con il loop di Inkling

**Problema.** Si può avere un modello di trascrizione da pochi milioni di parametri che si avvicina a uno molto più grande? Inkling 2 Mini (Thinking Machines, 2026-11-20) ripete i blocchi centrali con pesi condivisi. LARM (arXiv 2606.04678) mostra che su ASR un encoder da 7.7M con loop arriva vicino a un 85.7M a pari FLOP.

**Utente.** Il proprietario: un esperimento onesto e riproducibile sul suo Mac, nello stile di wsp-adaptive-decision-models (previsioni scritte prima di ogni run, controlli appaiati).

**Prima versione.**
- Inglese, frasi libere, LibriSpeech train-clean-100, solo Mac (24 GB, MPS).
- Bracci: A, 4 blocchi normali (~3.5M); B, blocchi condivisi in loop con gate che parte chiuso e numero di loop variabile; C, 18 blocchi normali a pari FLOP di B.
- Misure: WER test-clean/test-other per ogni numero di loop, RTF su CPU e MPS, dimensione fp32/int8. Whisper tiny.en e Moonshine Tiny come riferimento.
- Pilota di circa un giorno di calcolo con un criterio go/no-go (vedi plan.md).

**Fuori.** Italiano, cloud GPU, TTS, streaming, app.

**Fatto quando.** Il pilota ha girato e un report dice go o no-go contro il criterio, con i numeri dei tre bracci.

**Stack.** Python, PyTorch (MPS), torchaudio o soundfile, sentencepiece (BPE da 256 token), jiwer.

**Nomi.** loop-asr (consigliato), tiny-stt, wee-asr: tutti liberi su giovannigiofre.

File: plan.md (piano dell'esperimento), research.md (studio su Inkling 2 Mini).
