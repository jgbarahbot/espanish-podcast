#!/usr/bin/env python3
"""CosyVoice3 worker: synthesizes blocks on CPU for make_podcast.py.

Protocol (stdin, one JSON object per line):
    {"spk": "A"|"C", "text": "...", "emotion": "alegre"|"solemne"|null, "out": "/abs/path.wav"}
Progress (stdout, one JSON per line):
    {"ready": true, "sr": 24000}
    {"ok": <path>}
    {"error": <msg>}

Emotion is expressed via inference_instruct2 (voice cloning from a 5-10 s
reference clip + a SHORT ENGLISH instruction); no emotion -> inference_zero_shot.
Short (1-2 word) English instructions are required: long or Spanish
instructions bleed into the audio (the Qwen LLM reads them out loud).

The reference clips live next to the CosyVoice code dir (refA.wav, refC.wav)
and are resampled to the model rate (24 kHz) before each synthesis call.

Run with the cosyvoice venv:  ~/.hermes/venvs/cosyvoice/bin/python
"""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Cgroup memory cap: OMP>4 OOM-kills CosyVoice3 on this host. Must be set
# BEFORE torch is imported (libgomp reads it at import time; a later
# torch.set_num_threads does not limit already-loaded OpenMP).
os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.environ.get("COSYVOICE_CODE",
                          os.path.expanduser("~/.hermes/cache/cosyvoice/code"))
MODEL_DIR = os.environ.get("COSYVOICE_MODEL",
                           os.path.expanduser("~/.hermes/cache/cosyvoice/model_v3"))
REF_DIR = os.environ.get("COSYVOICE_REFS",
                         os.path.expanduser("~/.hermes/cache/cosyvoice"))

sys.path.append(CODE_DIR)
sys.path.append(os.path.join(CODE_DIR, "third_party", "Matcha-TTS"))

import torchaudio  # noqa: E402
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

# Short ENGLISH emotion instructions (long/Spanish ones bleed into the audio).
INSTRUCT = {
    "alegre": "cheerful",
    "animado": "cheerful",
    "solemne": "solemn",
    "serio": "serious",
    "calmo": "calm",
    "sorpresa": "surprised",
    "curiosidad": "curious",
    "urgencia": "urgent",
    "cierre": "warm",
}
# CosyVoice3 (Qwen LLM) hard-requires <|endofprompt|> in text or prompt_text.
PROMPT_TEXT = ("<|endofprompt|>")

# text_frontend=False: skip wetext English text normalization. It corrupts
# bracketed phoneme strings and we handle Spanish numbers/units ourselves.
TEXT_FRONTEND = False


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


_REF_TMP = os.path.join(REF_DIR, "ref_tmp.wav")


def ref_wav(spk):
    """Reference clip resampled to the model rate (CosyVoice reads the file
    itself; a resampled tensor without sample-rate metadata is rejected)."""
    global _SAMPLE_RATE
    p = os.path.join(REF_DIR, "ref%s.wav" % spk)
    if not os.path.exists(p):
        raise SystemExit("missing reference clip: %s" % p)
    w, sr = torchaudio.load(p)
    if sr != _SAMPLE_RATE:
        w = torchaudio.functional.resample(w, sr, _SAMPLE_RATE)
        torchaudio.save(_REF_TMP, w, _SAMPLE_RATE)
        return _REF_TMP
    return p


def main():
    global _SAMPLE_RATE
    cv = AutoModel(model_dir=MODEL_DIR)
    _SAMPLE_RATE = cv.sample_rate
    emit({"ready": True, "sr": cv.sample_rate})

    # Warmup once: first synthesis after load is slower (kernel/caches).
    try:
        for _ in cv.inference_zero_shot(tts_text="Carga inicial.",
                                        prompt_text=PROMPT_TEXT,
                                        prompt_wav=ref_wav("A"),
                                        text_frontend=TEXT_FRONTEND):
            pass
        emit({"warmup": True})
    except Exception as e:
        emit({"warmup_error": str(e)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            spk, text, emo, out = job["spk"], job["text"], job.get("emotion"), job["out"]
            ref = ref_wav(spk)
            if emo and emo in INSTRUCT:
                gen = cv.inference_instruct2(tts_text=text,
                                             instruct_text="%s<|endofprompt|>" % INSTRUCT[emo],
                                             prompt_wav=ref,
                                             stream=False,
                                             text_frontend=TEXT_FRONTEND)
            else:
                gen = cv.inference_zero_shot(tts_text=text,
                                             prompt_text=PROMPT_TEXT,
                                             prompt_wav=ref,
                                             stream=False,
                                             text_frontend=TEXT_FRONTEND)
            full = None
            for chunk in gen:
                seg = chunk["tts_speech"]
                full = seg if full is None else torch.cat([full, seg], dim=1)
            if full is None:
                raise RuntimeError("no speech produced")
            torchaudio.save(out, full, cv.sample_rate)
            emit({"ok": out})
        except SystemExit:
            raise
        except Exception as e:
            emit({"error": str(e)})


if __name__ == "__main__":
    main()
