#!/usr/bin/env python3
"""Two-voice Spanish podcast — Spanish script -> MP3.

Default engine: CosyVoice2-0.5B (Apache-2.0) on CPU — emotion/intonation
control via Spanish instructions + voice cloning from short reference clips.
Fallback engine: local Piper (fast, no emotion) via `--engine piper`.

Part of the `espanish-podcast` skill. See that SKILL.md for the full
script format and pitfalls.

Script format: paragraphs prefixed with the speaker letter on the same line:

    A: Bienvenidos...
    C: Empecemos por la portada...

A and C are cloned from reference clips (refA.wav / refC.wav, 5-10 s each,
generated from the Piper voices when absent). Blocks without a marker
default to A.

EMOTION: tag a block with one of the supported tones right after the
speaker marker (or on its own line in the block):

    A [alegre]: Hoy repasamos...
    A [solemne]: Y para cerrar...

Supported: alegre, animado, solemne, serio, calmo, sorpresa, curiosidad,
urgencia, cierre. No tag -> neutral reading. Tones are expressed in Spanish
via CosyVoice2 `instruct2`; the same reference clip keeps the voice
identity, only the delivery changes.

ENGLISH TERMS: mark any English word/phrase with asterisks — `*notebook*`,
`*Hugging Face*` — and it is read in English. Resolution order: (1)
scripts/pron_dict.json (case-insensitive) — terms Spanish speakers say in
Spanish; (2) otherwise the asterisks are stripped and the term stays as
plain text for the engine (CosyVoice2 is multilingual and reads English
terms in English; Piper gets the raw term as best effort — grow
pron_dict.json for exact Spanish readings).

Pipeline: each block --(engine, the block's voice)--> wav, blocks
concatenated (380 ms silence between) --(ffmpeg, libmp3lame)--> mp3.

Usage:
    python make_podcast.py --script script.txt --out podcast.mp3
    python make_podcast.py --script s.txt --out p.mp3 --engine piper
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
PIPER = os.path.join(HOME, ".hermes", "hermes-agent", "venv", "bin", "piper")
VOICE_DIR = os.path.join(HOME, ".hermes", "cache", "piper-voices")
VOICE_A = os.path.join(VOICE_DIR, "es_ES-davefx-medium.onnx")   # davefx
VOICE_C = os.path.join(VOICE_DIR, "es_MX-claude-high.onnx")     # Claude
FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
GAP_S = 0.38  # pause between blocks
SPEAKERS = ("A", "C")
PRON_DICT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "pron_dict.json")

# CosyVoice (CPU, emotion) — worker + persistent env/model
CV_PY = os.environ.get("COSYVOICE_PYTHON",
                       os.path.join(HOME, ".hermes", "venvs",
                                    "cosyvoice", "bin", "python"))
CV_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cosyvoice_worker.py")
CV_REF_DIR = os.path.join(HOME, ".hermes", "cache", "cosyvoice")
CV_SR = 24000

# Piper reference clip (seconds) used to seed the CosyVoice clone refs.
REF_SEED_TEXTS = {
    "A": "Bienvenidos a iaahoy, el diario de la inteligencia artificial. Hoy repasamos las noticias del mundo de los modelos de lenguaje.",
    "C": "Empecemos por la portada. Repasemos las noticias de hoy sobre los modelos de lenguaje abiertos y su avance.",
}
REF_TARGET_S = 6.5


def load_pron_dict(path):
    """Load the Spanish-pronunciation dictionary (case-insensitive keys)."""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return {str(k).lower(): str(v) for k, v in d.items()}
    except Exception as e:  # missing/invalid dict must never break a podcast
        print("WARNING: pronunciation dictionary unavailable (%s); "
              "terms stay as plain text" % e)
        return {}


def mark_english(text, pron_dict=None):
    """Resolve `*English term*` spans: pron_dict (Spanish) first, else strip
    the asterisks and let the engine read the term in English."""
    if pron_dict is None:
        pron_dict = {}

    def sub(m):
        term = m.group(1).strip()
        return pron_dict.get(term.lower(), term)

    return re.sub(r"\*([^\*\n]+)\*", sub, text)


# ---------------------------------------------------------------------------
# Numbers in Spanish (standing rule: every figure is spoken in Spanish, not
# spelled out in the target language).
# ---------------------------------------------------------------------------
_ES_DIGIT = {0: "cero", 1: "uno", 2: "dos", 3: "tres", 4: "cuatro",
             5: "cinco", 6: "seis", 7: "siete", 8: "ocho", 9: "nueve",
             10: "diez", 11: "once", 12: "doce", 13: "trece", 14: "catorce",
             15: "quince", 16: "dieciséis", 17: "diecisiete", 18: "dieciocho",
             19: "diecinueve", 20: "veinte", 21: "veintiuno", 22: "veintidós",
             23: "veintitrés", 24: "veinticuatro", 25: "veinticinco",
             26: "veintiséis", 27: "veintisiete", 28: "veintiocho",
             29: "veintinueve", 30: "treinta", 40: "cuarenta", 50: "cincuenta",
             60: "sesenta", 70: "setenta", 80: "ochenta", 90: "noventa",
             100: "cien"}
_ES_TEN_Y = {3: "treinta y ", 4: "cuarenta y ", 5: "cincuenta y ",
             6: "sesenta y ", 7: "setenta y ", 8: "ochenta y ", 9: "noventa y "}
_ES_HUND = {1: "", 2: "doscientos", 3: "trescientos", 4: "cuatrocientos",
            5: "quinientos", 6: "seiscientos", 7: "setecientos",
            8: "ochocientos", 9: "novecientos"}


def _es_below_100(n):
    if n <= 30:
        return _ES_DIGIT[n]
    t, u = divmod(n, 10)
    if u == 0:
        return _ES_DIGIT[t * 10]
    return _ES_TEN_Y[t] + _ES_DIGIT[u]


def _es_below_1000(n):
    h, r = divmod(n, 100)
    if h == 0:
        return _es_below_100(n)
    base = _ES_HUND[h]
    if r == 0:
        return (base + " cien") if h == 1 else base
    return base + " " + _es_below_100(r)


def _es_below_10000(n):
    th, r = divmod(n, 1000)
    if th == 0:
        return _es_below_1000(n)
    base = "mil" if th == 1 else _es_below_1000(th) + " mil"
    if r == 0:
        return base
    return base + " " + _es_below_1000(r)


def _es_number(n):
    if n < 0:
        return "menos " + _es_number(-n)
    if n < 10000:
        return _es_below_10000(n)
    if n < 1000000:
        t, r = divmod(n, 1000)
        base = "mil" if t == 1 else _es_below_1000(t) + " mil"
        return base if r == 0 else base + " " + _es_below_1000(r)
    if n < 1000000000:
        m, r = divmod(n, 1000000)
        base = ("un millón" if m == 1 else _es_below_1000(m) + " millones")
        return base if r == 0 else base + " " + _es_below_100000(r)
    return str(n)


_NUM_DEC_RE = re.compile(r"(?<![\w.\-])(\d{1,4})[.,](\d{1,4})(?![\w.])")
_NUM_PCT_RE = re.compile(r"(?<![\w.\-])(\d{1,7}(?:[.,]\d{1,4})?)\s*%")
_NUM_INT_RE = re.compile(r"(?<![\w.\-])(\d{1,9})(?![\w.])")


def _int_spanish(digits):
    if not digits.isdigit():
        return digits
    n = int(digits)
    return _es_number(n) if n < 1000000000 else digits


def spanish_digits(text):
    """Speak every figure in Spanish: 2026 -> dos mil veintiséis,
    3.0 -> tres punto cero, 45% -> cuarenta y cinco por ciento.
    Digits glued to letters (v2, llama-3.1) are left alone."""
    def pct(m):
        s = m.group(1)
        if "," in s:
            whole, frac = s.split(",", 1)
            f = " ".join(_ES_DIGIT[int(c)] for c in frac)
            return _int_spanish(whole) + " punto " + f + " por ciento"
        return _int_spanish(s) + " por ciento"

    def dec(m):
        w = int(m.group(1))
        f = " ".join(_ES_DIGIT[int(c)] for c in m.group(2))
        return _int_spanish(m.group(1)) + " punto " + f
    text = _NUM_PCT_RE.sub(pct, text)
    text = _NUM_DEC_RE.sub(dec, text)
    text = _NUM_INT_RE.sub(lambda m: _int_spanish(m.group(1)), text)
    return text


_EMOTION_RE = re.compile(r"\[([A-Za-záéíóúñ]+)\]")
EMOTIONS = ("alegre", "animado", "solemne", "serio", "calmo",
            "sorpresa", "curiosidad", "urgencia", "cierre")


def parse_blocks(path):
    """Split the script into (voice, emotion, text) blocks.

    Voice is 'A' or 'C' (default 'A'); emotion is one of EMOTIONS given as
    a [tag] right after the speaker marker or anywhere in the block (first
    tag wins and is removed from the text)."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    blocks = []
    for chunk in raw.split("\n\n"):
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            continue
        voice, text = "A", ""
        m = re.match(r"^([AC])\s*:(.*)", lines[0])
        if m:
            voice = m.group(1)
            lines = [m.group(2).strip()] + lines[1:]
        text = " ".join(ln for ln in lines if ln).strip()
        if not text:
            continue
        emotion = None
        for ln in lines:
            mt = _EMOTION_RE.search(ln)
            if mt and mt.group(1).lower() in EMOTIONS:
                emotion = mt.group(1).lower()
                break
        text = _EMOTION_RE.sub("", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        if text:
            blocks.append((voice, emotion, text))
    return blocks


def _ffmpeg(args):
    r = subprocess.run([FFMPEG] + args, capture_output=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode("utf-8", "replace")[-1500:])
    return r.returncode == 0


def synth_piper(out_dir, voice_onnx, idx, text):
    out = os.path.join(out_dir, "blk_%02d.wav" % idx)
    r = subprocess.run([PIPER, "-m", voice_onnx, "-f", out],
                       input=text.encode("utf-8"), capture_output=True)
    if r.returncode != 0 or not os.path.exists(out):
        print("PIPER FAILED (block %d):\n%s" % (idx, r.stderr.decode("utf-8", "replace")))
        sys.exit(1)
    norm = os.path.join(out_dir, "blk_%02d_norm.wav" % idx)
    if not _ffmpeg(["-y", "-i", out, "-ar", "22050", "-ac", "1",
                    "-acodec", "pcm_s16le", norm]):
        print("FFMPEG RESAMPLE FAILED (block %d)" % idx)
        sys.exit(1)
    os.remove(out)
    return norm


def ensure_ref_clips(ref_dir, voice_a, voice_c):
    """Seed refA.wav / refC.wav (5-10 s) from the Piper voices when absent."""
    os.makedirs(ref_dir, exist_ok=True)
    for spk, onnx in (("A", voice_a), ("C", voice_c)):
        ref = os.path.join(ref_dir, "ref%s.wav" % spk)
        if os.path.exists(ref):
            return ref
        raw = os.path.join(ref_dir, "seed_%s.wav" % spk)
        r = subprocess.run([PIPER, "-m", onnx, "-f", raw],
                           input=REF_SEED_TEXTS[spk].encode("utf-8"),
                           capture_output=True)
        if r.returncode != 0 or not os.path.exists(raw):
            print("ERROR: cannot seed CosyVoice reference clip %s (piper failed)" % ref)
            sys.exit(1)
        _ffmpeg(["-y", "-i", raw, "-t", str(REF_TARGET_S),
                 "-ar", str(CV_SR), "-ac", "1", "-acodec", "pcm_s16le", ref])
        os.remove(raw)
        if not os.path.exists(ref):
            print("ERROR: failed to create reference clip %s" % ref)
            sys.exit(1)
        print("reference clip %s created (%.1f s)" % (ref, REF_TARGET_S))


class CosyWorker:
    """Long-lived CosyVoice2 worker subprocess (JSON over stdin/stdout).

    Loads the model once, then synthesizes one block per job line. The
    worker resamples its own references and writes 24 kHz mono WAVs.
    """

    def __init__(self, worker_py, worker, ref_dir):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
                   OMP_NUM_THREADS="4", COSYVOICE_REFS=ref_dir)
        self.proc = subprocess.Popen([worker_py, worker],
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL,
                                     env=env, text=True, bufsize=1)
        deadline = time.time() + 420
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            msg = json.loads(line)
            if msg.get("ready"):
                self.sr = msg.get("sr", CV_SR)
                return
            if "warmup_error" in msg:
                print("WARNING: CosyVoice warmup: %s" % msg["warmup_error"])
        raise SystemExit("ERROR: CosyVoice worker did not become ready (load timeout)")

    def synth(self, out, spk, text, emotion):
        job = {"spk": spk, "text": text, "emotion": emotion, "out": out}
        self.proc.stdin.write(json.dumps(job, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise SystemExit("ERROR: CosyVoice worker died (pid=%s)" % self.proc.pid)
            msg = json.loads(line)
            if msg.get("ok"):
                return msg["ok"]
            if "error" in msg:
                raise SystemExit("ERROR: CosyVoice synthesis failed: %s" % msg["error"])

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def synth_cosyworker(worker, out_dir, idx, spk, text, emotion):
    out = os.path.join(out_dir, "blk_%02d.wav" % idx)
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        print("  block %02d already done, skipping" % idx)
        return out
    return worker.synth(out, spk, text, emotion)


def assemble(wavs, gap_s, sr, out_mp3):
    tmp = os.path.abspath(out_mp3 + ".build")
    os.makedirs(tmp, exist_ok=True)
    gap = os.path.join(tmp, "gap.wav")
    if not _ffmpeg(["-y", "-f", "lavfi", "-i", "anullsrc=r=%d:cl=mono" % sr,
                    "-t", str(gap_s), "-acodec", "pcm_s16le", gap]):
        print("FFMPEG GAP FAILED")
        sys.exit(1)
    listfile = os.path.join(tmp, "concat.txt")
    with open(listfile, "w", encoding="utf-8") as f:
        f.write("file '%s'\n" % gap)
        for w in wavs:
            f.write("file '%s'\n" % w)
    joined = os.path.join(tmp, "joined.wav")
    if not _ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", listfile,
                    "-acodec", "pcm_s16le", joined]):
        print("FFMPEG CONCAT FAILED")
        sys.exit(1)
    if not _ffmpeg(["-y", "-i", joined, "-codec:a", "libmp3lame",
                    "-qscale:a", "3", "-ar", str(sr), "-ac", "1", out_mp3]):
        print("FFMPEG ENCODE FAILED")
        sys.exit(1)
    shutil.rmtree(tmp, ignore_errors=True)
    if not os.path.exists(out_mp3):
        print("FFMPEG ENCODE FAILED (no output)")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="path to two-voice Spanish script")
    ap.add_argument("--out", required=True, help="output MP3 path")
    ap.add_argument("--engine", choices=("cosy", "piper"), default="cosy",
                    help="TTS engine: cosy=cosyvoice (emotion, CPU, slower), "
                         "piper=local piper (fast, no emotion)")
    ap.add_argument("--voice-a", default=VOICE_A, help="davefx voice model (piper/seed)")
    ap.add_argument("--voice-c", default=VOICE_C, help="Claude voice model (piper/seed)")
    ap.add_argument("--only", nargs=2, type=int, metavar=("START", "END"),
                    help="synth only blocks START..END (1-based) into the build dir, "
                         "no assembly. Lets long podcasts run in chunks that fit a "
                         "single supervised process window; run --assemble once every "
                         "block's WAV exists.")
    ap.add_argument("--assemble", action="store_true",
                    help="only assemble the MP3 from the WAVs already in the build "
                         "dir (no synthesis, no worker)")
    ap.add_argument("--clean", action="store_true",
                    help="delete the build dir before starting (fresh run)")
    args = ap.parse_args()

    blocks = parse_blocks(args.script)
    if not blocks:
        print("ERROR: empty script")
        sys.exit(1)
    counts = {sp: sum(1 for v, _, _ in blocks if v == sp) for sp in SPEAKERS}
    n_emo = sum(1 for _, e, _ in blocks if e)
    n = len(blocks)
    print("script: %d blocks (%d A, %d C, %d with emotion)"
          % (n, counts["A"], counts["C"], n_emo))

    if args.only and not (1 <= args.only[0] <= args.only[1] <= n):
        print("ERROR: --only %d %d out of range (1..%d)" % (args.only[0], args.only[1], n))
        sys.exit(1)

    sr = CV_SR if args.engine == "cosy" else 22050
    tmp = os.path.abspath(args.out + ".build")
    if args.clean:
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    # --assemble: build the MP3 from the WAVs already on disk (no worker)
    if args.assemble:
        wavs = [os.path.join(tmp, "blk_%02d.wav" % i) for i in range(1, n + 1)]
        missing = [w for w in wavs if not os.path.exists(w)]
        if missing:
            print("ERROR: cannot assemble, missing %d of %d WAV(s); first missing: %s"
                  % (len(missing), n, os.path.basename(missing[0])))
            sys.exit(1)
        t0 = time.time()
        assemble(wavs, GAP_S, sr, args.out)
        size = os.path.getsize(args.out)
        print("OK podcast %s (%d bytes, %d blocks, %s, %.0fs assemble)"
              % (args.out, size, n, args.engine, time.time() - t0))
        return

    pron_dict = load_pron_dict(PRON_DICT)
    t0 = time.time()
    do_assemble = not args.only

    if args.engine == "cosy":
        if not (os.path.exists(CV_PY) and os.path.exists(CV_WORKER)):
            print("ERROR: cosyvoice unavailable (python: %s, worker: %s); "
                  "retry with --engine piper" % (CV_PY, CV_WORKER))
            sys.exit(1)
        ensure_ref_clips(CV_REF_DIR, args.voice_a, args.voice_c)
        worker = CosyWorker(CV_PY, CV_WORKER, CV_REF_DIR)
        sr = worker.sr
        print("cosyvoice ready (sr=%d)" % sr)
        wavs = []
        try:
            for i, (voice, emotion, text) in enumerate(blocks, 1):
                if args.only and not (args.only[0] <= i <= args.only[1]):
                    continue
                t = time.time()
                wavs.append(synth_cosyworker(worker, tmp, i, voice,
                                             spanish_digits(mark_english(text, pron_dict)), emotion))
                print("block %02d %s%s %6.1fs" %
                      (i, voice, " [%s]" % emotion if emotion else "", time.time() - t))
        finally:
            worker.close()
        if do_assemble:
            assemble(wavs, GAP_S, sr, args.out)
            shutil.rmtree(tmp, ignore_errors=True)
            size = os.path.getsize(args.out)
            print("OK podcast %s (%d bytes, %d blocks, %d A / %d C, %d emotion, %s, %.0fs)"
                  % (args.out, size, n, counts["A"], counts["C"],
                     n_emo, args.engine, time.time() - t0))
        else:
            print("chunk done: blocks %d-%d synthesized into %s (%.0fs)"
                  % (args.only[0], args.only[1], tmp, time.time() - t0))
            print("NEXT: run --only <next range> for the remaining blocks, then "
                  "--assemble once blocks 1-%d all exist" % n)
    else:
        # piper: fast enough that chunking is rarely needed, but it still works
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
        wavs = []
        for i, (voice, emotion, text) in enumerate(blocks, 1):
            if args.only and not (args.only[0] <= i <= args.only[1]):
                continue
            wavs.append(synth_piper(tmp, voices[voice], i,
                                    spanish_digits(mark_english(text, pron_dict))))
        if do_assemble:
            assemble(wavs, GAP_S, sr, args.out)
            shutil.rmtree(tmp, ignore_errors=True)
            size = os.path.getsize(args.out)
            print("OK podcast %s (%d bytes, %d blocks, %s, %.0fs)"
                  % (args.out, size, n, args.engine, time.time() - t0))
        else:
            print("chunk done (piper): blocks %d-%d into %s"
                  % (args.only[0], args.only[1], tmp))


if __name__ == "__main__":
    main()
