#!/usr/bin/env python3
"""Qwen3-TTS voice-clone worker for make_podcast.py (GPU container).

Protocol (stdin, one JSON object per line) — SAME shape as cosyvoice_worker:
    {"spk": "A"|"C", "text": "...", "emotion": null, "out": "/abs/path.wav"}
Progress (stdout, one JSON per line):
    {"ready": true, "sr": 24000}
    {"ok": <path>}
    {"error": <msg>}

Engine: Qwen3-TTS-12Hz-1.7B-Base (voice-clone, NO per-block emotion — the
Base model has no instruction channel; emotion tags in the script are ignored,
the two voices are simply cloned from refA.wav / refC.wav).

Cloning uses ICL mode (x_vector_only_mode=False): each voice builds ONE
reusable voice_clone_prompt from (ref clip + transcript) and reuses it for
every block of that speaker — the same "clone once, reuse" pattern as the
README's `create_voice_clone_prompt`.

Config via env (all have defaults):
    QWEN3_MODEL    model dir (default /model)   [1.7B-Base, self-contained]
    QWEN3_REFS     refs dir (default /refs)     [refA.wav refC.wav refA.txt refC.txt]
    QWEN3_DEVICE   "cuda:0" (default) or "cpu" / "" (cpu smoke test)
    QWEN3_DTYPE    "bf16" (default) or "fp32" (cpu smoke test)
    QWEN3_XVEC     "0" (default, ICL) or "1" (speaker-embedding-only clone)
    QWEN3_LANG     language name (default "Spanish")

Output is 24 kHz mono — identical to CosyVoice, so make_podcast.py's
assemble path (380 ms gap, libmp3lame, 24 kHz MP3) is unchanged.
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
REF_DIR = os.environ.get("QWEN3_REFS", "/refs")
LANG = os.environ.get("QWEN3_LANG", "Spanish")
XVEC_ONLY = os.environ.get("QWEN3_XVEC", "0") == "1"
_DTYPE = (torch.bfloat16 if os.environ.get("QWEN3_DTYPE", "bf16") == "bf16"
          else torch.float32)
# SDPA is GPU-accelerated and needs no flash-attn build; on CPU use eager
# (SDPA-on-CPU is supported but eager is the safest for the smoke test).
_ATTN = "eager" if _device == "cpu" else "sdpa"


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
    for one speaker."""
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

    emit({"loading": True, "model": MODEL_DIR, "device": _device,
          "dtype": str(_DTYPE), "attn": _ATTN, "xvec": XVEC_ONLY})
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR,
        device_map=_device,
        dtype=_DTYPE,
        attn_implementation=_ATTN,
    )
    sr = int(model.model.speech_tokenizer.get_output_sample_rate())
    emit({"ready": True, "sr": sr, "supported_languages":
          sorted(l.lower() for l in (model.model.get_supported_languages()
                                     or []))})

    # Build the two reusable voice-clone prompts (encodes each ref clip
    # through the speech tokenizer + speaker encoder — done ONCE).
    prompt_A = build_prompt(model, "A")
    prompt_C = build_prompt(model, "C")
    prompts = {"A": prompt_A, "C": prompt_C}
    emit({"prompts": True, "A": len(prompt_A), "C": len(prompt_C)})

    # Warmup: synthesize one short block per voice so first-real-block is fast.
    try:
        for spk in ("A", "C"):
            w, _ = model.generate_voice_clone(text="Carga inicial.",
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
            # emotion tag is intentionally IGNORED: the Base model has no
            # instruction channel (per-block emotion is a CosyVoice feature).
            wavs, fs = model.generate_voice_clone(text=text,
                                                  language=LANG,
                                                  voice_clone_prompt=prompts[spk])
            import soundfile as sf
            sf.write(out, wavs[0], fs)
            emit({"ok": out, "sr": fs, "secs": round(float(len(wavs[0]) / fs), 3)})
        except SystemExit:
            raise
        except Exception as e:
            emit({"error": str(e)})


if __name__ == "__main__":
    main()
