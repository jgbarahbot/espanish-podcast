#!/usr/bin/env python3
"""Qwen3-TTS worker for make_podcast.py / qwen3_podcast.py (GPU container or CPU).

Protocol (stdin, one JSON object per line) — SAME shape as cosyvoice_worker:
    {"spk": "A"|"C", "text": "...", "emotion": null|<Spanish tone>, "out": "/abs/path.wav"}
Progress (stdout, one JSON per line):
    {"ready": true, "sr": 24000}
    {"ok": <path>}
    {"error": <msg>}

TWO MODES (env QWEN3_MODE):

  clone   (default) — Qwen3-TTS-12Hz-1.7B-Base, voice-clone, NO per-block
          emotion (the Base model has no instruction channel; emotion tags
          are ignored, the two voices are simply cloned from refA.wav /
          refC.wav). ICL mode: each voice builds ONE reusable
          voice_clone_prompt from (ref clip + transcript) and reuses it for
          every block of that speaker ("clone once, reuse").

  design  — Qwen3-TTS-12Hz-1.7B-VoiceDesign, 100% synthetic (NO cloning, no
          ref clips): each voice is described in English (Madrid narrator,
          male A / female C, Castilian accent) and per-block emotion is
          appended to the description in English. This is the "voice by
          description" method.

Config via env (all have defaults):
    QWEN3_MODEL    model dir (default /model)   [the checkpoint to load]
    QWEN3_MODE     "clone" (default) or "design"
    QWEN3_REFS     refs dir (default /refs)     [clone only: refA/refC .wav + .txt]
    QWEN3_DEVICE   "cuda:0" (default) or "cpu" / "" (cpu smoke test)
    QWEN3_DTYPE    "bf16" (default) or "fp32" (cpu smoke test)
    QWEN3_XVEC     "0" (default, ICL) or "1" (speaker-embedding-only clone)
    QWEN3_LANG     language name (default "Spanish")
    QWEN3_DO_SAMPLE "0" (default, deterministic) or "1"

Output is 24 kHz mono in both modes — identical to CosyVoice, so
make_podcast.py's assemble path (380 ms gap, libmp3lame, 24 kHz MP3) is
unchanged.
"""
import contextlib
import json
import os
import sys

# CPU smoke test: hide the GPU + cap threads BEFORE torch import (libgomp
# reads OMP_NUM_THREADS at import time; setting it later is too late).
os.environ.setdefault("OMP_NUM_THREADS", "4")
_device = os.environ.get("QWEN3_DEVICE", "cuda:0")
if _device in ("", "cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _device = "cpu"

import torch  # noqa: E402

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

MODEL_DIR = os.environ.get("QWEN3_MODEL", "/model")
MODE = os.environ.get("QWEN3_MODE", "clone").lower()
REF_DIR = os.environ.get("QWEN3_REFS", "/refs")
LANG = os.environ.get("QWEN3_LANG", "Spanish")
XVEC_ONLY = os.environ.get("QWEN3_XVEC", "0") == "1"
_DO_SAMPLE = os.environ.get("QWEN3_DO_SAMPLE", "0") == "1"
_DTYPE = (torch.bfloat16 if os.environ.get("QWEN3_DTYPE", "bf16") == "bf16"
          else torch.float32)
# SDPA is GPU-accelerated and needs no flash-attn build; on CPU use eager
# (SDPA-on-CPU is supported but eager is the safest for the smoke test).
_ATTN = "eager" if _device == "cpu" else "sdpa"

# --- design mode: voice-by-description (100% synthetic, Madrid accent) ----
# The validated iter-4 prompts: same two narrators (male A / female C),
# Castilian accent, natural balanced pace. Emotion is appended in English
# (the instruct channel is English) so the same base voice varies tone.
PROMPT_A = (
    "A professional middle-aged male narrator from Madrid, Spain, "
    "speaking with a distinct Castilian Spanish accent. His voice has "
    "a clear, crisp northern-central Spain pronunciation, standard "
    "European Spanish ceceo/distinción, and a natural, balanced "
    "speaking pace suited for an audiobook."
)
PROMPT_C = (
    "A professional middle-aged female narrator from Madrid, Spain, "
    "speaking with a distinct Castilian Spanish accent. Her voice has "
    "a clear, crisp northern-central Spain pronunciation, standard "
    "European Spanish ceceo/distinción, and a natural, balanced "
    "speaking pace suited for an audiobook."
)
BASES = {"A": PROMPT_A, "C": PROMPT_C}

# Spanish emotion tag (script) -> short English phrase appended to the
# description. Unknown/absent tag -> neutral (base description only).
EMO = {
    "alegre": "sounding very happy and energetic",
    "animado": "sounding very happy and energetic",
    "serio": "in a calm, serious and informative tone",
    "calmo": "in a calm, serious and informative tone",
    "solemne": "in a calm, serious and informative tone",
    "sorpresa": "with surprise",
    "curiosidad": "with a curious, intrigued tone",
    "urgencia": "with urgency and a fast, tense delivery",
    "cierre": "in a calm, warm closing tone",
}


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def read_ref_text(spk):
    p = os.path.join(REF_DIR, "ref%s.txt" % spk)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return f.read().strip()


def build_prompt(model, spk):
    """Build a reusable voice-clone prompt (a list of VoiceClonePromptItem)
    for one speaker (clone mode only)."""
    ref_wav = os.path.join(REF_DIR, "ref%s.wav" % spk)
    if not os.path.exists(ref_wav):
        raise SystemExit("missing reference clip: %s" % ref_wav)
    ref_text = read_ref_text(spk)
    if not XVEC_ONLY and not ref_text:
        # ICL mode requires a transcript; if absent, degrade to speaker-
        # embedding-only rather than crash.
        emit({"warn": "ref%s: no transcript, using x_vector_only" % spk})
        return model.create_voice_clone_prompt(ref_audio=ref_wav,
                                               x_vector_only_mode=True)
    return model.create_voice_clone_prompt(
        ref_audio=ref_wav,
        ref_text=ref_text,
        x_vector_only_mode=XVEC_ONLY,
    )


def main():
    # qwen_tts/__init__ imports the 25 Hz tokenizer module, which prints a
    # "flash-attn is not installed" banner to STDOUT when flash-attn is
    # absent (CPU venv, and harmless for the 12 Hz path anyway). Redirect
    # stdout for the whole import so the JSON protocol stays clean.
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        from qwen_tts import Qwen3TTSModel

    emit({"loading": True, "model": MODEL_DIR, "mode": MODE,
          "device": _device, "dtype": str(_DTYPE), "attn": _ATTN,
          "xvec": XVEC_ONLY})
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR,
        device_map=_device,
        dtype=_DTYPE,
        attn_implementation=_ATTN,
    )
    sr = int(model.model.speech_tokenizer.get_output_sample_rate())
    emit({"ready": True, "sr": sr, "mode": MODE, "supported_languages":
          sorted(l.lower() for l in (model.model.get_supported_languages()
                                     or []))})

    if MODE == "clone":
        # Build the two reusable voice-clone prompts (encodes each ref clip
        # through the speech tokenizer + speaker encoder — done ONCE).
        prompt_A = build_prompt(model, "A")
        prompt_C = build_prompt(model, "C")
        prompts = {"A": prompt_A, "C": prompt_C}
        emit({"prompts": True, "A": len(prompt_A), "C": len(prompt_C)})

        # Warmup: one short block per voice so first-real-block is fast.
        try:
            for spk in ("A", "C"):
                model.generate_voice_clone(text="Carga inicial.",
                                           language=LANG,
                                           voice_clone_prompt=prompts[spk])
            emit({"warmup": True})
        except Exception as e:
            emit({"warmup_error": str(e)})

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                job = json.loads(line)
                spk = job["spk"]
                text = job["text"]
                out = job["out"]
                # emotion tag is intentionally IGNORED in clone mode: the
                # Base model has no instruction channel.
                wavs, fs = model.generate_voice_clone(
                    text=text, language=LANG,
                    voice_clone_prompt=prompts[spk])
                import soundfile as sf
                sf.write(out, wavs[0], fs)
                emit({"ok": out, "sr": fs, "secs":
                      round(float(len(wavs[0]) / fs), 3)})
            except SystemExit:
                raise
            except Exception as e:
                emit({"error": str(e)})

    elif MODE == "design":
        # No prompts to pre-build (the voice is the English description).
        # Warmup: one short block per voice to prime the code path.
        try:
            for spk in ("A", "C"):
                model.generate_voice_design(text="Carga inicial.",
                                            language="Spanish",
                                            instruct=BASES[spk],
                                            non_streaming_mode=True,
                                            do_sample=_DO_SAMPLE)
            emit({"warmup": True})
        except Exception as e:
            emit({"warmup_error": str(e)})

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                job = json.loads(line)
                spk = job["spk"]
                text = job["text"]
                out = job["out"]
                emo = (job.get("emotion") or "").strip().lower()
                base = BASES.get(spk, BASES["A"])
                instruct = base if not emo else base + " " + EMO.get(emo, "")
                wavs, fs = model.generate_voice_design(
                    text=text, language="Spanish", instruct=instruct,
                    non_streaming_mode=True, do_sample=_DO_SAMPLE)
                import numpy as np
                import soundfile as sf
                wav = np.asarray(wavs[0], dtype=np.float32)
                # Safety cut: a repetition loop would otherwise run for
                # minutes (observed on long blocks) — cap at 60 s.
                max_samples = fs * 60
                if len(wav) > max_samples:
                    emit({"warn": "block %s cut at 60s (repetition loop)"
                          % spk})
                    wav = wav[:max_samples]
                sf.write(out, wav, fs)
                emit({"ok": out, "sr": fs, "secs":
                      round(float(len(wav) / fs), 3)})
            except SystemExit:
                raise
            except Exception as e:
                emit({"error": str(e)})

    else:
        raise SystemExit("ERROR: unknown QWEN3_MODE '%s' (clone|design)"
                         % MODE)


if __name__ == "__main__":
    main()
