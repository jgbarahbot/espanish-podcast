#!/usr/bin/env python3
"""Two-voice Spanish podcast — Spanish script -> MP3 via local Piper TTS.

Part of the `espanish-podcast` skill. See that SKILL.md for the script
format, English-term marking (*term* -> English IPA) and pitfalls.

Script format: paragraphs prefixed with the speaker letter on the same line:

    A: Bienvenidos...
    C: Empecemos por la portada...

A = davefx (es_ES-davefx-medium), C = Claude (es_MX-claude-high). The two
voices alternate block by block; blocks without a marker default to A.

ENGLISH TERMS: mark any English word/phrase with asterisks — `*notebook*`,
`*Hugging Face*` — and it is pronounced in English. Resolution order before
synthesis: (1) scripts/pron_dict.json (case-insensitive) — terms Spanish
speakers say in Spanish, e.g. "llama.cpp" -> "llama ce pe pe", "GGUF" ->
"ge guf", "r/LocalLLaMA" -> "erre LocalLlama"; (2) numbers inside a term are
read in Spanish digit by digit and each dot is said "punto" ("Llama 3.45" ->
"Llama tres cuarenta y cinco", "Qwen-Image-2.1" -> "Qwen Image dos punto
uno"), then espeak en-US IPA; (3) otherwise the whole term goes to a `[[...]]`
raw-phoneme block holding its English IPA (espeak-ng en-US via piper's bundled
espeakbridge). Piper feeds `[[...]]` characters straight into the model's
phoneme id-map (voice.py: `phonemes[-1].extend(text_part[2:-2].strip())`);
every English IPA codepoint used by espeak en-US exists in both Spanish
models' maps, and unknown phonemes are skipped silently (phoneme_ids.py), so
worst case a term degrades to plain Spanish — it never crashes.

EVERY term resolved via the espeak fallback is reported at the end as
`UNSURE: <term>` so the caller can offer the user a list to grow
pron_dict.json. Best effort: a bad-sounding term is still synthesized.

Pipeline: each block --(piper, the block's voice)--> wav, blocks concatenated
(380ms silence between) --(ffmpeg, libmp3lame)--> mp3.

Usage:
    python make_podcast.py --script script.txt --out podcast.mp3
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

VENV = os.path.expanduser("~/.hermes/hermes-agent/venv")
PIPER = os.path.join(VENV, "bin", "piper")
VOICE_DIR = os.path.expanduser("~/.hermes/cache/piper-voices")
VOICE_A = os.path.join(VOICE_DIR, "es_ES-davefx-medium.onnx")   # davefx
VOICE_C = os.path.join(VOICE_DIR, "es_MX-claude-high.onnx")     # Claude
FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
GAP_S = 0.38  # pause between blocks
SPEAKERS = ("A", "C")
PRON_DICT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "pron_dict.json")

# Spanish reading of digits inside English terms: "3.45" -> "tres cuarenta y
# cinco" (digit by digit), every dot -> "punto".
ES_DIGIT = {str(d): w for d, w in enumerate(
    ["cero", "uno", "dos", "tres", "cuatro", "cinco",
     "seis", "siete", "ocho", "nueve"])}


def load_pron_dict(path):
    """Load the Spanish-pronunciation dictionary (case-insensitive keys)."""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return {str(k).lower(): str(v) for k, v in d.items()}
    except Exception as e:  # missing/invalid dict must never break a podcast
        print("WARNING: pronunciation dictionary unavailable (%s); using espeak only" % e)
        return {}


def _init_espeak_en():
    """Init piper's bundled espeak-ng with the en-US voice, for English IPA.

    piper's EspeakPhonemizer is used only to initialize the espeakbridge
    library (it strips (lang) flags, so it cannot itself produce English
    phonemes); we then call espeakbridge directly with the en-US voice.
    """
    sp = os.path.join(VENV, "lib",
                      "python%d.%d" % sys.version_info[:2], "site-packages")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    from piper.phonemize_espeak import EspeakPhonemizer
    from piper import espeakbridge
    EspeakPhonemizer()               # initializes the espeak-ng bridge
    espeakbridge.set_voice("en-us")  # English dictionary + voice rules
    return espeakbridge


_ESPEAK_EN = None
_IPA_CACHE = {}


def eng_ipa(term):
    """English IPA for `term` (espeak en-US), cached. None on any failure."""
    global _ESPEAK_EN
    if _ESPEAK_EN is None:
        try:
            _ESPEAK_EN = _init_espeak_en()
        except Exception as e:  # pragma: no cover
            print("WARNING: espeak EN unavailable (%s); *terms* stay plain" % e)
            _ESPEAK_EN = False
    if not _ESPEAK_EN:
        return None
    if term not in _IPA_CACHE:
        try:
            ipa = "".join(p for p, _, _ in _ESPEAK_EN.get_phonemes(term))
            _IPA_CACHE[term] = ipa if ipa.strip() else None
        except Exception:
            _IPA_CACHE[term] = None
    return _IPA_CACHE[term]


def _spanish_digits(num):
    """Read a number in Spanish, digit by digit; each dot is said "punto".

    "3.45" -> "tres punto cuarenta y cinco"; "2.1" -> "dos punto uno".
    """
    parts = []
    for ch in num:
        if ch == ".":
            parts.append("punto")
        elif ch in ES_DIGIT:
            parts.append(ES_DIGIT[ch])
    return " ".join(parts)


def mark_english(text, pron_dict=None, unsure=None):
    """Replace `*English term*` spans per the pronunciation rules:
    dict (Spanish) -> numbers in Spanish + espeak en-US IPA per word chunk.
    Only the alphabetic chunks go through espeak; digit chunks and the word
    "punto" stay as plain Spanish text so Piper says them in Spanish.
    Any term that used the espeak fallback is added to `unsure` (order kept)."""
    if pron_dict is None:
        pron_dict = {}

    def sub(m):
        term = m.group(1).strip()
        low = term.lower()
        if low in pron_dict:
            return pron_dict[low]
        out, used_fallback = [], False
        for part in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d[\d.,/]*", term):
            if part[0].isdigit():
                out.append(_spanish_digits(part))
            else:
                ipa = eng_ipa(part)
                if ipa:
                    used_fallback = True
                    out.append("[[%s]]" % ipa)
                else:
                    out.append(part)  # no espeak: best-effort plain text
        if used_fallback and unsure is not None and term not in unsure:
            unsure.append(term)
        return " ".join(out) if out else term

    return re.sub(r"\*([^\*\n]+)\*", sub, text)


def parse_blocks(path):
    """Split the script into (voice, text) blocks. Voice is 'A' or 'C';
    blocks without a marker default to 'A'."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    blocks = []
    for chunk in raw.split("\n\n"):
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            continue
        voice = "A"
        m = re.match(r"^([AC])\s*:\s*(.*)", lines[0])
        if m:
            voice = m.group(1)
            rest = m.group(2).strip()
            lines = [rest] + lines[1:]
        text = " ".join(lines).strip()
        if text:
            blocks.append((voice, text))
    return blocks


def synth(out_dir, voice, idx, text):
    out = os.path.join(out_dir, "blk_%02d.wav" % idx)
    r = subprocess.run([PIPER, "-m", voice, "-f", out],
                       input=text.encode("utf-8"), capture_output=True)
    if r.returncode != 0 or not os.path.exists(out):
        print("PIPER FAILED (%s, block %d):\n%s" % (voice, idx, r.stderr.decode("utf-8", "replace")))
        sys.exit(1)
    # Some voices emit 16 kHz (e.g. es_ES-mls_9972-low); the concat target is
    # 22050 Hz, so resample every block to the same rate to avoid pitch/speed
    # shifts on the final MP3.
    norm = os.path.join(out_dir, "blk_%02d_norm.wav" % idx)
    r = subprocess.run([FFMPEG, "-y", "-i", out, "-ar", "22050", "-ac", "1",
                        "-acodec", "pcm_s16le", norm], capture_output=True)
    if r.returncode != 0:
        print("FFMPEG RESAMPLE FAILED (block %d):\n%s" % (idx, r.stderr.decode("utf-8", "replace")))
        sys.exit(1)
    os.remove(out)
    return norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="path to two-voice Spanish script")
    ap.add_argument("--out", required=True, help="output MP3 path")
    ap.add_argument("--voice-a", default=VOICE_A, help="davefx voice model")
    ap.add_argument("--voice-c", default=VOICE_C, help="Claude voice model")
    args = ap.parse_args()

    if not os.path.exists(PIPER):
        print("ERROR: piper not found at", PIPER)
        sys.exit(1)
    voices = {}
    for sp in SPEAKERS:
        path = getattr(args, "voice_%s" % sp.lower())
        if not os.path.exists(path):
            print("WARNING: voice %s missing (%s), falling back to A" % (sp, path))
            path = args.voice_a
        voices[sp] = path
    if not os.path.exists(args.voice_a):
        print("ERROR: voice A not found:", args.voice_a)
        sys.exit(1)

    blocks = parse_blocks(args.script)
    if not blocks:
        print("ERROR: empty script")
        sys.exit(1)
    counts = {sp: sum(1 for v, _ in blocks if v == sp) for sp in SPEAKERS}
    print("script: %d blocks (%d A, %d C)" % (len(blocks), counts["A"], counts["C"]))

    tmp = os.path.abspath(args.out + ".build")
    os.makedirs(tmp, exist_ok=True)
    pron_dict = load_pron_dict(PRON_DICT)
    unsure = []
    wavs = []
    for i, (voice, text) in enumerate(blocks, 1):
        wavs.append(synth(tmp, voices[voice], i, mark_english(text, pron_dict, unsure)))

    listfile = os.path.join(tmp, "concat.txt")
    with open(listfile, "w", encoding="utf-8") as f:
        f.write("file '%s'\n" % os.path.join(tmp, "gap.wav"))
        for w in wavs:
            f.write("file '%s'\n" % w)
    subprocess.run([FFMPEG, "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=22050:cl=mono", "-t", str(GAP_S),
                    "-acodec", "pcm_s16le", os.path.join(tmp, "gap.wav")],
                   capture_output=True, check=True)

    joined = os.path.join(tmp, "joined.wav")
    r = subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                        "-acodec", "pcm_s16le", joined],
                       capture_output=True)
    if r.returncode != 0:
        print("FFMPEG CONCAT FAILED:\n" + r.stderr.decode("utf-8", "replace"))
        sys.exit(1)

    r = subprocess.run([FFMPEG, "-y", "-i", joined,
                        "-codec:a", "libmp3lame", "-qscale:a", "3",
                        "-ar", "22050", "-ac", "1", args.out],
                       capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode != 0 or not os.path.exists(args.out):
        print("FFMPEG FAILED:\n" + r.stderr.decode("utf-8", "replace"))
        sys.exit(1)

    size = os.path.getsize(args.out)
    print("OK podcast %s (%d bytes, %d blocks, %d A / %d C)"
          % (args.out, size, len(blocks), counts["A"], counts["C"]))
    if unsure:
        print("Terms pronounced via espeak fallback — review, and add the ones "
              "you say in Spanish to scripts/pron_dict.json:")
        for term in unsure:
            print("UNSURE: %s" % term)
    else:
        print("All English terms resolved via the pronunciation dictionary.")


if __name__ == "__main__":
    main()
