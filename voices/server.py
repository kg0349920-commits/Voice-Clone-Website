import os
import re
import time
import uuid
import threading
import traceback

import numpy as np
import librosa
import soundfile as sf
import torch

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from TTS.api import TTS


# ==========================================
# PROJECT FOLDERS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))

VOICE_FOLDER = os.path.join(BASE_DIR, "voices")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(VOICE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# SETTINGS
# ==========================================

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 24000

MAX_TEXT_CHARS = 5000          # index.html mein bhi yahi limit rakho
OUTPUT_MAX_AGE_SECONDS = 3600  # purani audio 1 ghante baad delete

# Languages jo XTTS-v2 support karta hai (Urdu support nahi hai)
SUPPORTED_LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
}

GEN_SETTINGS = {
    "temperature": 0.65,
    "repetition_penalty": 5.0,
    "top_k": 40,
    "top_p": 0.8,
    "length_penalty": 1.0,
    "speed": 1.0,
}


# ==========================================
# CPU / GPU SETTINGS
# ==========================================

torch.set_num_threads(4)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/outputs/*": {"origins": "*"},
})


@app.route("/", methods=["GET"])
def home():
    return send_from_directory(SERVER_DIR, "index.html")


# ==========================================
# LOAD XTTS MODEL
# ==========================================

print("Loading XTTS-v2 on:", DEVICE, flush=True)

tts = TTS(MODEL_NAME).to(DEVICE)
xtts_model = tts.synthesizer.tts_model

print("XTTS-v2 loaded successfully!", flush=True)


# ==========================================
# GENERATION LOCK
# ==========================================

generation_lock = threading.Lock()


# ==========================================
# HELPERS
# ==========================================

def cleanup_old_outputs():
    """Purani generated files delete karo taake disk na bhare."""
    now = time.time()
    try:
        for name in os.listdir(OUTPUT_FOLDER):
            if not (name.startswith("voice_") and name.endswith(".wav")):
                continue
            path = os.path.join(OUTPUT_FOLDER, name)
            try:
                if now - os.path.getmtime(path) > OUTPUT_MAX_AGE_SECONDS:
                    os.remove(path)
            except OSError:
                pass
    except OSError:
        pass


def prepare_reference(src, dst, max_seconds=30):
    """Voice sample ko saaf karo: mono, trim, normalize."""

    y, _ = librosa.load(src, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = y[:SAMPLE_RATE * max_seconds]

    if len(y) < SAMPLE_RATE * 3:
        raise ValueError("Please upload at least 3 seconds of clear speech.")

    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak * 0.95

    sf.write(dst, y, SAMPLE_RATE)


def split_text(text, max_chars=200):
    """Lamba text sentences ke hisaab se chhote chunks mein todta hai."""

    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?;:])\s+", text)

    chunks = []
    current = ""

    for s in sentences:

        # bohat lamba sentence ho to comma ya space par todo
        while len(s) > max_chars:
            cut = s.rfind(", ", 0, max_chars)
            if cut > 0:
                cut += 1
            else:
                cut = s.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars

            part = s[:cut].strip()
            s = s[cut:].strip()

            if current:
                chunks.append(current)
                current = ""
            if part:
                chunks.append(part)

        if not s:
            continue

        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s

    if current:
        chunks.append(current)

    return chunks


def generate_long_wav(text, language, gpt_cond_latent, speaker_embedding):
    """Text ko chunks mein generate karke ek audio banata hai."""

    max_chars = 200 if language == "en" else 140
    chunks = split_text(text, max_chars)
    total = len(chunks)

    gap = np.zeros(int(0.18 * SAMPLE_RATE), dtype=np.float32)
    parts = []

    for i, chunk in enumerate(chunks, 1):

        # Bina punctuation ke XTTS aakhir mein ajeeb awaaz nikalta hai
        if chunk[-1] not in ".!?":
            chunk = chunk + "."

        print(f"[{i}/{total}] {len(chunk)} chars: {chunk[:50]}...", flush=True)

        with torch.inference_mode():
            result = xtts_model.inference(
                text=chunk,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                enable_text_splitting=False,
                **GEN_SETTINGS
            )

        wav = np.asarray(result["wav"], dtype=np.float32)

        # chunk ke start/end ki extra khamoshi hatao
        wav, _ = librosa.effects.trim(wav, top_db=40)

        if len(wav) == 0:
            continue

        # click/pop rokne ke liye chhota fade in/out
        fade = min(int(0.02 * SAMPLE_RATE), len(wav) // 2)
        if fade > 0:
            wav[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            wav[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)

        parts.append(wav)

        if i < total:
            parts.append(gap)

    if not parts:
        raise ValueError("No audio was generated.")

    final = np.concatenate(parts)

    # volume barabar karo
    peak = float(np.max(np.abs(final)))
    if peak > 0:
        final = final / peak * 0.95

    return final.astype(np.float32)


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": xtts_model is not None,
        "device": DEVICE
    })


# ==========================================
# VOICE GENERATION API
# ==========================================

@app.route("/api/clone", methods=["POST"])
def clone_voice():

    text = request.form.get("text", "").strip()
    language = request.form.get("language", "en").strip().lower()
    voice_file = request.files.get("voice")

    # Validate text
    if not text:
        return jsonify({"error": "Please enter text."}), 400

    if len(text) > MAX_TEXT_CHARS:
        return jsonify({
            "error": f"Keep text under {MAX_TEXT_CHARS} characters."
        }), 400

    # Validate language
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({
            "error": f"Language '{language}' is not supported."
        }), 400

    # Validate uploaded voice
    if voice_file is None or not voice_file.filename:
        return jsonify({"error": "Please upload a voice sample."}), 400

    extension = os.path.splitext(voice_file.filename.lower())[1]

    if extension not in (".wav", ".mp3", ".m4a"):
        return jsonify({"error": "Use WAV, MP3, or M4A."}), 400

    # Prevent simultaneous generations
    if not generation_lock.acquire(blocking=False):
        return jsonify({
            "error": "Generation busy. Try again shortly."
        }), 429

    uid = uuid.uuid4().hex

    reference_path = os.path.join(VOICE_FOLDER, uid + extension)
    clean_path = os.path.join(VOICE_FOLDER, uid + "_clean.wav")

    output_name = "voice_" + uid + ".wav"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    try:

        print("Voice generation started...", flush=True)

        cleanup_old_outputs()

        # Save uploaded reference audio
        voice_file.save(reference_path)

        # Clean and prepare reference audio
        prepare_reference(reference_path, clean_path)

        print("Preparing voice reference...", flush=True)

        # Extract voice conditioning
        gpt_cond_latent, speaker_embedding = (
            xtts_model.get_conditioning_latents(
                audio_path=[clean_path],
                gpt_cond_len=30,
                gpt_cond_chunk_len=4,
                max_ref_length=30
            )
        )

        print("Generating speech...", flush=True)

        wav = generate_long_wav(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding
        )

        sf.write(output_path, wav, SAMPLE_RATE)

        print("Generation completed successfully!", flush=True)

        return jsonify({
            "success": True,
            "audio_url": "/outputs/" + output_name
        })

    except Exception as e:

        print("\n========== VOICE GENERATION ERROR ==========", flush=True)
        traceback.print_exc()
        print("ERROR MESSAGE:", repr(e), flush=True)

        return jsonify({"error": str(e)}), 500

    finally:

        # Remove temporary reference files
        for path in (reference_path, clean_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as cleanup_error:
                print("Cleanup warning:", repr(cleanup_error), flush=True)

        # Release generation lock
        generation_lock.release()


# ==========================================
# SERVE GENERATED AUDIO
# ==========================================

@app.route("/outputs/<filename>", methods=["GET"])
def output_audio(filename):

    if not filename.startswith("voice_") or not filename.endswith(".wav"):
        return jsonify({"error": "Invalid filename"}), 400

    response = send_from_directory(
        OUTPUT_FOLDER,
        filename,
        mimetype="audio/wav"
    )

    response.headers["Cache-Control"] = "no-store"

    return response


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    print("Flask server starting on port 5000", flush=True)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )