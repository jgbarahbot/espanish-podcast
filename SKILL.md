---
name: espanish-podcast
version: 2.2.0
description: "Use when synthesizing a Spanish podcast MP3 from a script."
author: Jesus Gonzalez-Barahona
license: MIT
requires:
  - "ffmpeg; CosyVoice3 (Fun-CosyVoice3-0.5B-2512) under ~/.hermes/cache/cosyvoice/ (default engine) OR piper + piper voices in ~/.hermes/cache/piper-voices/ (fallback --engine piper)"
---

# Spanish two-voice podcast

## When to Use

- A ready-made Spanish script exists and you need it as a spoken MP3.
- Any Spanish show/newsletter pipeline can call this as its audio step
  (e.g. a daily newspaper that wants a spoken version of each edition).
- NOT for: writing the script itself (the caller owns content), for English-only
  podcasts (voices are Spanish; English terms are spoken in English).

Turn a Spanish script written by the agent into ONE MP3 with two local voices
alternating (CosyVoice3 cloned voices by default, with per-block emotion),
English terms pronounced in English, all numbers spoken in Spanish. Fully
offline.

**Division of labor:** THIS skill synthesizes audio from a given script file.
Writing the script (content, tone, length) is the calling skill's job — it
passes `--script <path>`. Keep that boundary: do not invent news content here.

## Requirements (one-time)

See "Requirements (one-time)" in the Synthesis section below: CosyVoice3
(default) or Piper (fallback). Both need `ffmpeg`.

## Script format

Plain UTF-8 text. Blocks separated by a blank line; the speaker letter goes on
the SAME line as the text:

```text
A: Bienvenidos al programa. Hoy repasamos las novedades de la semana.

C: Empecemos por la primera noticia, que llega de la comunidad.

A: Seguimos con lo segundo...
```

Rules:

- `A` and `C` are the only valid markers; a block without one defaults to A.
- Optional emotion tag between marker and colon: `A [alegre]: ...`
  (see Emotion section).
- Alternate voices block by block (A, C, A, C...) for a natural conversation.
- Natural spoken Spanish, present tense — this is speech, not an article.
- Typical length: 15-25 blocks (about 4-6 minutes at this pace).
- **Never name the narrator(s) or mention how many voices speak** — just use
  them. (Standing user rule.)

## English terms + numbers

Wrap any English word/phrase in asterisks: `*notebook*`, `*Hugging Face*`,
`*MCP*`. The pipeline applies, in order:

1. **Dictionary first.** `scripts/pron_dict.json` maps terms that do NOT get
   English pronunciation (acronyms and names Spanish speakers say
   letter-by-letter or in Spanish) to their Spanish spelling:
   `"llama.cpp" → "llama ce pe pe"`, `"GGUF" → "ge guf"`,
   `"r/LocalLLaMA" → "erre LocalLlama"`. Lookup is case-insensitive. New
   terms the user pronounces in Spanish go here.
2. **Otherwise** the asterisks are stripped and the term stays PLAIN —
   CosyVoice3 (multilingual) reads ordinary English words in English on its
   own; that path is verified and clean.

   - **CosyVoice3: do NOT use CMU/ARPAbet phoneme bracket notation**
     (`[D_E_B_A_G]`). The README's "Pronunciation Inpainting" feature is NOT
     supported by the Fun-CosyVoice3-0.5B-2512 checkpoint: the v3
     `CosyVoice-BlankEN` vocab (151,643 entries) contains ZERO bracket
     phoneme tokens, and the test proved the model reads bracket letters
     aloud or garbles the block (verified 2026-09-29 with control
     phonemes). Worse, the wetext English normalizer corrupts bracket
     strings (spaces: `[N OW1 T B UH1 K]` → `[N OW one T B UH one K]`;
     underscores: hard `AssertionError` crash). The worker runs with
     `text_frontend=False` to keep wetext out of the way.
   - **Piper fallback:** no espeak on this host, so `*terms*` also stay
     plain; Piper reads them in Spanish — acceptable degradation.
3. **Numbers are ALWAYS spoken in Spanish**, by `spanish_digits()` in
   `make_podcast.py`: `3.0` → "tres puntos cero", `45%` → "cuarenta y
   cinco por ciento", `99,99%` → "noventa y nueve punto noventa y nueve por
   ciento", `1,200` → "mil doscientos", `2.5 M` → "dos punto cinco millones",
   `1e6` → "un millón", `2026` → "dos mil veintiséis", `v2.1.4` → "ve dos
   punto uno punto cuatro" (letter v before the number), `8K` → "ocho kilas",
   `16 bits` → "dieciséis bits". Ranges `a-b` are split and converted
   separately.

## Emotion / intonation (CosyVoice3, default engine)

The default engine is **CosyVoice3 (Fun-CosyVoice3-0.5B-2512)** (runs on
CPU, 24 kHz). It **clones both voices** from reference clips and supports
per-block emotion via `inference_instruct2` with a **SHORT ENGLISH**
instruction:

```text
A [alegre]: Bienvenidos a IA Abierta Hoy...
C: ...
A [solemne]: Esto ha sido IA Abierta Hoy.
```

- Emotion tag on the same line: `A [tono]: texto`. Valid tones: `alegre`,
  `solemne`, `serio`, `calmo`, `sorpresa`, `curiosidad`, `urgencia`, `cierre`
  (see `INSTRUCT` in `cosyvoice_worker.py`). Unknown/absent tone → neutral
  (`inference_zero_shot`). Use sparingly: openings/closings and the one
  genuinely serious news item; a news digest is mostly neutral.
- **Instruction format (critical):** `"<1-2 English words><|endofprompt|>"`
  (e.g. `"cheerful<|endofprompt|>"`). NO prefix sentences, NO Spanish, NO
  long descriptions — anything longer leaks into the audio (the Qwen LLM
  reads the instruction out loud). "shouting"/"screaming" words cause
  repetition; use "extremely angry, raising your voice" if needed.
- **`instruct2` clones only the TIMBRE** of the reference (not its full
  voice), so emotional blocks sound a bit less like the neutral voice —
  known, accepted.
- Voices A/C are **cloned from reference clips**
  `~/.hermes/cache/cosyvoice/refA.wav` / `refC.wav` (24 kHz). Current refs:
  A = the user's own voice (12 s), C = a female reference (6.8 s). `--voice-a`
  / `--voice-c` (Piper paths) are still accepted as the source for
  regenerating the reference clips.
- **CPU cost:** `OMP_NUM_THREADS=4` MANDATORY (the worker sets it itself,
  before importing torch — libgomp reads it at import time; a later
  `torch.set_num_threads` is too late). ~70-107 s per block on i7-12700H
  → a 4-5 min podcast (17-20 blocks) takes ~25-35 min.
- One worker process (lazy-loaded, ~1 min warmup on CPU) — never fork per
  block. Worker env: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4`.

## Synthesis

**NEVER run the CosyVoice engine from inside a Hermes worker** (any
`terminal` tool call, cron or not): worker cgroups are hard-capped at 4 GiB
(`process_registry._WORKER_MEMORY_MAX_CAP_BYTES`) and CosyVoice3 needs
~4.2 GiB RSS (up to 7.9 GiB peak with OMP=4) → OOM-kill, exit 137, OMP does
not help (it's the model weights). Two working routes:

1. **`podcast_systemd.sh` (canonical, what the iaahoy cron uses):**
   `bash ~/.hermes/projects/iaahoy/scripts/podcast_systemd.sh --script <p> --out <p.mp3>`
   → one-shot systemd user unit `iaahoy-podcast.service` with `MemoryMax=8G`;
   run it as a background terminal and poll `systemctl --user show
   iaahoy-podcast.service` (watch `MemoryCurrent`/`MemoryPeak`) and the WAV
   count in `<out>.build/`. Per-run logs: `~/.hermes/projects/iaahoy/runs/podcast-*.log`.
2. **Direct, only from a gateway-scope foreground process:**
   ```bash
   python3 <skill-dir>/scripts/make_podcast.py \
       --script <path-to-script.txt> --out <path-to-output.mp3>
   # default: --engine cosy
   # --engine piper            : old Piper-only path (no emotion, seconds, fallback)
   # --voice-a/--voice-c       : Piper .onnx paths (cosy: clip source; piper: voices)
   # --ref-dir/--cosy-home     : override clip dir / CosyVoice install
   # --only START END --assemble --clean : chunked synthesis helpers
   ```
   `--engine piper` fits in the 4 GiB cap, so it is the in-worker fallback.
   (Verified 2026-09-29: same 17-block script → 137 in a worker, clean run
   in a MemoryMax=8G unit.)

Success output: `OK podcast <out> (<bytes> bytes, <N> blocks, <nA> A / <nC> C, <nEmo> emotion, <engine>, <secs>s)`.

Pipeline (cosy): each block → `cosyvoice_worker.py` (stdin JSON protocol) →
`inference_instruct2` (emotion) or `inference_zero_shot` (neutral), both with
named params (`tts_text=, prompt_text=/instruct_text=, prompt_wav=,
stream=False, text_frontend=False`) → wav → concatenate with 380 ms silence →
`libmp3lame` MP3 (VBR q3, 24 kHz mono). Piper pipeline: piper per block →
resample 22050 Hz → same concat/MP3.

### Requirements (one-time)

- CosyVoice3 (default): code + model + venv under `~/.hermes/cache/cosyvoice/`
  (`~/.hermes/venvs/cosyvoice/` with torch CPU, `code/` incl.
  `third_party/Matcha-TTS`, `model_v3/` = Fun-CosyVoice3-0.5B-2512, 19 files),
  reference clips in `~/.hermes/cache/cosyvoice/`. `ffmpeg` on PATH.
  NOTE: the repo moved from `FunAudioLLM/CosyVoice` to `QwenAudio/CosyVoice`.
- Piper (fallback `--engine piper`): `piper` in the Hermes venv, voice models
  in `~/.hermes/cache/piper-voices/` (A `es_ES-davefx-medium.onnx`, C
  `es_MX-claude-high.onnx`, 22050 Hz).

## Verification (do this)

```bash
ffprobe -v error -show_entries format=duration,bit_rate -of default=nw=1 <out.mp3>
```

- Duration in the expected range (roughly 10-15 s per block).
- Optional: transcribe with faster-whisper (`medium`, cpu int8) and confirm
  no block is missing and no English instruction text leaks. **Whisper is
  unreliable on Spanish** (it often hears English/Chinese) — use it only to
  detect leaks/repetition, never as a quality verdict.

## Pitfalls

- **CosyVoice reference audio is a FILE PATH**, not a tensor: passing a
  resampled tensor makes the model reject it ("Invalid file: tensor(...)").
  If the clip's rate differs from 24 kHz, resample to a temp wav and pass
  that path (the worker does this).
- **Exit 137 from a worker = the 4 GiB cgroup cap, not OMP:** set
  `OMP_NUM_THREADS=4` anyway (the worker does) — it only trims the peak
  (~7.9 GiB → lower); the fix for the cap is `podcast_systemd.sh`
  (MemoryMax=8G) or `--engine piper`. Never synthesize blocks in parallel
  anyway (CPU-bound).
- **Emotion instructions: short English + `<|endofprompt|>` only.** Long or
  Spanish instructions bleed into the audio; "shouting/screaming" cause
  repetition.
- **`inference_instruct2` yields single items, not tuples:** iterate
  `for chunk in gen`, never `for _, chunk in`.
- **`CosyVoice3()` takes `load_trt`/`load_vllm`/`fp16`** — there is no
  `load_jit` kwarg (that was CosyVoice2).
- **No CMU phoneme brackets in CosyVoice3** (see English terms section): the
  v3 checkpoint's vocab has no phoneme tokens and wetext corrupts the
  strings; plain English terms are the supported path.
- CosyVoice prints model-download progress to stderr on first load —
  harmless; the worker speaks JSON on stdout only.
- 16 kHz voice models (e.g. `es_ES-mls_9972-low`) sound fast/high-pitched in
  a mixed MP3 — the Piper path resamples every block to 22050 Hz to prevent
  this; keep both voices at the same rate regardless.
- Missing voice C falls back to A with a WARNING (the script never crashes on
  a missing optional voice); missing A is a hard ERROR.
- The script text is passed via stdin — never shell-interpolated (Piper: into
  piper; CosyVoice: as JSON lines to the worker).
- `pkill -f "<worker script>"` can match the calling shell and kill it; use
  `pgrep -v` to verify before killing.

## Keeping this skill in sync with GitHub
This directory is a git repo tracking `jgbarahbot/espanish-podcast`. After any local edit, run `./sync.sh` (commits, fast-forwards, and pushes using `GITHUB_CONTENT_RW_TOKEN` from `~/.hermes/.env` — read it with `grep '^GITHUB_CONTENT_RW_TOKEN=' ~/.hermes/.env | cut -d= -f2- | tr -d '\r'`, never `source .env`, and push as `jgbarahbot`, never `x-access-token`).
