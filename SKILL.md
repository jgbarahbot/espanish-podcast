---
name: espanish-podcast
version: 1.0.0
description: "Use when synthesizing a Spanish podcast MP3 from a script."
author: Jesus Gonzalez-Barahona
license: MIT
requires:
  - "piper (in the Hermes venv), ffmpeg, piper voice models in ~/.hermes/cache/piper-voices/"
---

# Spanish two-voice podcast

## When to Use

- A ready-made Spanish script exists and you need it as a spoken MP3.
- Any Spanish show/newsletter pipeline can call this as its audio step
  (e.g. a daily newspaper that wants a spoken version of each edition).
- NOT for: writing the script itself (the caller owns content), for English-only
  podcasts (voices are Spanish; only inline English terms get English IPA).

Turn a Spanish script written by the agent into ONE MP3 with two local Piper
voices alternating, English terms pronounced in English. Fully offline.

**Division of labor:** THIS skill synthesizes audio from a given script file.
Writing the script (content, tone, length) is the calling skill's job — it
passes `--script <path>`. Keep that boundary: do not invent news content here.

## Requirements (one-time)

- `piper` CLI in the Hermes venv (`~/.hermes/hermes-agent/venv/bin/piper`).
- `ffmpeg` on PATH.
- Voice models in `~/.hermes/cache/piper-voices/`:
  - A → `es_ES-davefx-medium.onnx` (22050 Hz)
  - C → `es_MX-claude-high.onnx` (22050 Hz)
  - Any other Spanish model works via `--voice-a/--voice-c`, but BOTH must be
    the same sample rate (the script resamples every block to 22050 Hz anyway,
    see Pitfalls).

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
- Alternate voices block by block (A, C, A, C...) for a natural conversation.
- Natural spoken Spanish, present tense — this is speech, not an article.
- Typical length: 15-25 blocks (about 4-6 minutes at this pace).
- **Never name the narrator(s) or mention how many voices speak** — just use
  them. (Standing user rule.)

## English terms (pronounced in English)

Wrap any English word/phrase in asterisks: `*notebook*`, `*Hugging Face*`,
`*MCP*`. Before synthesis each span is replaced by a `[[...]]` raw-phoneme
block holding the term's English IPA (espeak-ng `en-us` via piper's bundled
espeakbridge). Piper feeds `[[...]]` characters straight into the phoneme
id-map; every English IPA codepoint espeak emits exists in the Spanish models'
maps, and unknown phonemes are skipped silently — worst case a term degrades
to plain Spanish, never a crash.

If espeak is unavailable the script warns once and `*terms*` stay plain.

### Pronunciation rules for English terms (do not "simplify" away)

1. **Dictionary first.** `scripts/pron_dict.json` maps terms that do NOT get
   English IPA (acronyms and names Spanish speakers say letter-by-letter or in
   Spanish) to their Spanish pronunciation: `"llama.cpp" → "llama ce pe pe"`,
   `"GGUF" → "ge guf"`, `"r/LocalLLaMA" → "erre LocalLlama"`. Lookup is
   case-insensitive. New terms the user pronounces in Spanish go here.
2. **Numbers inside a term are read in Spanish, digit by digit**: `Llama 3.45`
   → "Llama **tres cuarenta y cinco**"; `Qwen-Image-2.1` → "Qwen Image **dos
   punto uno**" (each decimal point is said "punto").
3. **Otherwise** the term is phonemized with espeak en-US IPA as described
   above.
4. **Best-effort + report.** If a term sounds off, synthesize it anyway
   (never block on pronunciation) but the script prints a
   `UNSURE: <term>` list at the end (every term that went through the
   espeak fallback); the caller should surface that list to the user so
   uncertain terms can be added to `pron_dict.json`.

## Synthesis

```bash
python3 <skill-dir>/scripts/make_podcast.py \
    --script <path-to-script.txt> --out <path-to-output.mp3>
# optional: --voice-a <model.onnx> --voice-c <model.onnx>
```

Success output: `OK podcast <out> (<bytes> bytes, <N> blocks, <nA> A / <nC> C)`.

Pipeline: each block → piper (its voice) → wav → resample 22050 Hz mono →
concatenate with a 380 ms silence between blocks → `libmp3lame` MP3 (VBR q3,
22050 Hz mono). Temp dir `<out>.build` is removed on success.

## Verification (do this)

```bash
ffprobe -v error -show_entries format=duration,bit_rate -of default=nw=1 <out.mp3>
```

- Duration in the expected range (roughly 10-15 s per block).
- Optional: transcribe with faster-whisper (`medium`, cpu int8) and confirm
  English terms sound English and no block is missing.

## Pitfalls

- 16 kHz voice models (e.g. `es_ES-mls_9972-low`) sound fast/high-pitched in a
  mixed MP3 — the script resamples every block to 22050 Hz to prevent this;
  keep both voices at the same rate regardless.
- `EspeakPhonemizer` must be instantiated ONCE per process (the script does
  this): a second init segfaults the espeakbridge thread.
- Missing voice C falls back to A with a WARNING (the script never crashes on
  a missing optional voice); missing A is a hard ERROR.
- The script text is passed to piper via stdin — never shell-interpolated.

## Keeping this skill in sync with GitHub
This directory is a git repo tracking `jgbarahbot/espanish-podcast`. After any local edit, run `./sync.sh` (commits, fast-forwards, and pushes using `GITHUB_CONTENT_RW_TOKEN` from `~/.hermes/.env`).

