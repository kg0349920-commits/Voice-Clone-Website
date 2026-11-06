import os
import re
import uuid
import time
import glob
import hashlib
import threading
import traceback

import numpy as np
import librosa
import soundfile as sf
import torch
from scipy.signal import butter, sosfilt
from flask import Flask, request, jsonify, send_from_directory
from TTS.api import TTS

# Optional packages (na hon to bhi server chalega)
try:
    import noisereduce as nr
except Exception:
    nr = None
try:
    from num2words import num2words
except Exception:
    num2words = None

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_FOLDER = os.path.join(BASE_DIR, "voices")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(VOICE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SR = 24000
MAX_REF_SECONDS = 30
MIN_REF_SECONDS = 6
MAX_CHUNK_CHARS = 200
KEEP_OUTPUT_HOURS = 6

# Quality settings (XTTS ke balanced defaults)
GEN_SETTINGS = dict(
    temperature=0.7,
    repetition_penalty=10.0,
    top_k=50,
    top_p=0.85,
    length_penalty=1.0,
    speed=1.0,
)

SEED = 42  # same seed = har baar consistent awaaz

print("\n========================================")
print("      VoxClone AI - Starting")
print("========================================\n")
print("Loading XTTS-v2 model...")

try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("NVIDIA GPU detected." if DEVICE == "cuda" else "Running on CPU.")
    tts = TTS(MODEL_NAME).to(DEVICE)
    xtts = tts.synthesizer.tts_model
    print("XTTS-v2 loaded successfully.")
except Exception as error:
    print("\nMODEL ERROR:")
    print(error, "\n")
    tts = None
    xtts = None

print("Noise reduction:", "ON" if nr else "OFF (pip install noisereduce)")
print("Number to words:", "ON" if num2words else "OFF (pip install num2words)")

generation_lock = threading.Lock()
latent_cache = {}  # file hash -> (gpt_cond_latent, speaker_embedding)


# --------------------------------------------------
# REFERENCE AUDIO PREPARATION
# --------------------------------------------------

def prepare_reference(src, dst):
    """Reference ko saaf karta hai: mono 24kHz, rumble filter, noise reduce,
    beech ki lambi khamoshi hata kar, volume normalize."""
    y, _ = librosa.load(src, sr=SR, mono=True)

    # Low-frequency rumble hatao
    sos = butter(4, 80, btype="highpass", fs=SR, output="sos")
    y = sosfilt(sos, y).astype(np.float32)

    # Background noise kam karo
    if nr is not None:
        try:
            y = nr.reduce_noise(y=y, sr=SR, stationary=True, prop_decrease=0.75).astype(np.float32)
        except Exception as e:
            print("Noise reduction skipped:", e)

    # Sirf bolne wale hisse rakho, khamoshi hatao
    intervals = librosa.effects.split(y, top_db=32, frame_length=2048, hop_length=512)
    gap = np.zeros(int(SR * 0.12), dtype=np.float32)
    parts, total = [], 0
    for start, end in intervals:
        seg = y[start:end]
        if end - start < SR * 0.15:
            continue
        parts.append(seg)
        parts.append(gap)
        total += len(seg) + len(gap)
        if total >= SR * MAX_REF_SECONDS:
            break
    if not parts:
        raise ValueError("Voice sample me awaaz nahi mili. Saaf recording upload karein.")

    y = np.concatenate(parts)[: SR * MAX_REF_SECONDS]
    if len(y) < SR * MIN_REF_SECONDS:
        raise ValueError(
            "Voice sample bohot chhota hai. Kam az kam 10-15 second ki saaf bolti hui clip use karein."
        )

    peak = float(np.max(np.abs(y))) or 1.0
    y = (y / peak * 0.95).astype(np.float32)
    sf.write(dst, y, SR)


def get_latents(reference_path, clean_path):
    h = hashlib.sha1()
    with open(reference_path, "rb") as f:
        h.update(f.read())
    key = h.hexdigest()

    if key in latent_cache:
        print("Using cached voice profile.")
        return latent_cache[key]

    prepare_reference(reference_path, clean_path)
    with torch.inference_mode():
        latents = xtts.get_conditioning_latents(
            audio_path=[clean_path],
            gpt_cond_len=30,
            gpt_cond_chunk_len=6,
            max_ref_length=MAX_REF_SECONDS,
        )
    if len(latent_cache) >= 5:
        latent_cache.pop(next(iter(latent_cache)))
    latent_cache[key] = latents
    return latents


# --------------------------------------------------
# TEXT CLEANING AND SPLITTING
# --------------------------------------------------

def clean_text(text, language):
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("…", "...").replace("—", ", ").replace("–", ", ")
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)   # emoji
    text = re.sub(r"[\[\]\(\)\{\}<>*_#~^|\\/@]", " ", text)

    if language == "en":
        for a, b in {"Dr.": "Doctor", "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miss",
                     "vs.": "versus", "etc.": "et cetera", "&": " and ", "%": " percent"}.items():
            text = text.replace(a, b)

    if num2words is not None:
        def repl(m):
            try:
                return num2words(int(m.group(0)), lang=language if language != "zh-cn" else "zh")
            except Exception:
                return m.group(0)
        try:
            text = re.sub(r"\d+", repl, text)
        except Exception:
            pass

    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?。！？":
        text += "."
    return text


def split_text(text, max_len=MAX_CHUNK_CHARS):
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    chunks, cur = [], ""

    def flush():
        nonlocal cur
        if cur.strip():
            chunks.append(cur.strip())
        cur = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        while len(s) > max_len:            # bohot lambi sentence ko comma/space par todo
            cut = max(s.rfind(",", 0, max_len), s.rfind(" ", 0, max_len))
            cut = cut if cut > 40 else max_len
            piece, s = s[:cut].strip(" ,"), s[cut:].strip(" ,")
            flush()
            chunks.append(piece + ("," if not piece.endswith((",", ".")) else ""))
        if len(cur) + len(s) + 1 <= max_len:
            cur = (cur + " " + s).strip()
        else:
            flush()
            cur = s
    flush()
    return chunks


# --------------------------------------------------
# AUDIO POST PROCESSING
# --------------------------------------------------

def tidy_chunk(wav):
    wav = np.asarray(wav, dtype=np.float32)
    trimmed, _ = librosa.effects.trim(wav, top_db=40)
    if len(trimmed) > SR * 0.1:
        wav = trimmed
    n = int(SR * 0.012)   # click hataane ke liye chhota fade
    if len(wav) > 2 * n:
        wav[:n] *= np.linspace(0, 1, n, dtype=np.float32)
        wav[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
    return wav


def cleanup_old_outputs():
    cutoff = time.time() - KEEP_OUTPUT_HOURS * 3600
    for path in glob.glob(os.path.join(OUTPUT_FOLDER, "voice_*.wav")):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except Exception:
            pass


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/outputs/<filename>")
def output_audio(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": xtts is not None,
        "device": DEVICE
    })


@app.route("/api/clone", methods=["POST"])
def clone_voice():

    if xtts is None:
        return jsonify({
            "error": "The AI model could not load. Check the terminal."
        }), 500

    text = request.form.get("text", "").strip()
    language = request.form.get("language", "en").strip()
    voice_file = request.files.get("voice")

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    if len(text) > 3000:
        return jsonify({
            "error": "Please keep the text under 3000 characters."
        }), 400

    if language != "en":
        return jsonify({
            "error": "Only English language is supported."
        }), 400

    if voice_file is None or not voice_file.filename:
        return jsonify({
            "error": "Please upload a voice sample."
        }), 400

    extension = os.path.splitext(
        voice_file.filename.lower()
    )[1]

    if extension not in (".wav", ".mp3", ".m4a"):
        return jsonify({
            "error": "Please upload WAV, MP3, or M4A audio."
        }), 400

    if not generation_lock.acquire(blocking=False):
        return jsonify({
            "error": "Another generation is running. Please wait."
        }), 429

    uid = uuid.uuid4().hex

    reference_path = os.path.join(
        VOICE_FOLDER, "reference_" + uid + extension
    )

    clean_path = os.path.join(
        VOICE_FOLDER, "clean_" + uid + ".wav"
    )

    output_name = "voice_" + uid + ".wav"

    output_path = os.path.join(
        OUTPUT_FOLDER, output_name
    )

    try:
        cleanup_old_outputs()

        voice_file.save(reference_path)

        text = clean_text(text, language)
        chunks = split_text(text)

        if not chunks:
            return jsonify({
                "error": "Text me bolne layak kuch nahi mila."
            }), 400

        print("\n----------------------------------------")
        print("Generating voice...")
        print("Language:", language)
        print("Chunks:", len(chunks))
        print("----------------------------------------")

        gpt_cond_latent, speaker_embedding = get_latents(
            reference_path,
            clean_path
        )

        pieces = []

        pause = np.zeros(
            int(SR * 0.22),
            dtype=np.float32
        )

        for i, chunk in enumerate(chunks, 1):

            print(f"[{i}/{len(chunks)}] {chunk}")

            torch.manual_seed(SEED + i)

            with torch.inference_mode():

                out = xtts.inference(
                    text=chunk,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    enable_text_splitting=False,
                    **GEN_SETTINGS
                )

            pieces.append(tidy_chunk(out["wav"]))

            if i < len(chunks):
                pieces.append(pause)

        wav = np.concatenate(pieces)

        peak = float(np.max(np.abs(wav))) or 1.0

        wav = (wav / peak * 0.92).astype(np.float32)

        sf.write(output_path, wav, SR)

        print("Generation complete.")

        return jsonify({
            "success": True,
            "audio_url": "/outputs/" + output_name
        })

    except ValueError as error:

        print("Input error:", error)

        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:

        print("\nGENERATION ERROR:")
        traceback.print_exc()

        return jsonify({
            "error": "Voice generation failed: " + str(error)
        }), 500

    finally:

        for p in (reference_path, clean_path):

            try:
                if os.path.exists(p):
                    os.remove(p)

            except Exception:
                pass

        generation_lock.release()

    if not generation_lock.acquire(blocking=False):
        return jsonify({"error": "Another generation is running. Please wait."}), 429

    uid = uuid.uuid4().hex
    reference_path = os.path.join(VOICE_FOLDER, "reference_" + uid + extension)
    clean_path = os.path.join(VOICE_FOLDER, "clean_" + uid + ".wav")
    output_name = "voice_" + uid + ".wav"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    try:
        cleanup_old_outputs()
        voice_file.save(reference_path)

        text = clean_text(text, language)
        chunks = split_text(text)
        if not chunks:
            return jsonify({"error": "Text me bolne layak kuch nahi mila."}), 400

        print("\n----------------------------------------")
        print(f"Generating voice... Language: {language} | Chunks: {len(chunks)}")
        print("----------------------------------------")

        gpt_cond_latent, speaker_embedding = get_latents(reference_path, clean_path)

        pieces = []
        pause = np.zeros(int(SR * 0.22), dtype=np.float32)

        for i, chunk in enumerate(chunks, 1):
            print(f"[{i}/{len(chunks)}] {chunk}")
            torch.manual_seed(SEED + i)
            with torch.inference_mode():
                out = xtts.inference(
                    text=chunk,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    enable_text_splitting=False,
                    **GEN_SETTINGS,
                )
            pieces.append(tidy_chunk(out["wav"]))
            if i < len(chunks):
                pieces.append(pause)

        wav = np.concatenate(pieces)
        peak = float(np.max(np.abs(wav))) or 1.0
        wav = (wav / peak * 0.92).astype(np.float32)
        sf.write(output_path, wav, SR)

        print("Generation complete.")
        return jsonify({"success": True, "audio_url": "/outputs/" + output_name})

    except ValueError as error:
        print("Input error:", error)
        return jsonify({"error": str(error)}), 400

    except Exception as error:
        print("\nGENERATION ERROR:")
        traceback.print_exc()
        print()
        return jsonify({"error": "Voice generation failed: " + str(error)}), 500

    finally:
        for p in (reference_path, clean_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        generation_lock.release()


if __name__ == "__main__":
    print("\n========================================")
    print("Website:")
    print("http://127.0.0.1:5000")
    print("========================================\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)