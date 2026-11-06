import modal
from pathlib import Path

HERE = Path(__file__).parent
APP_DIR = HERE / "voices"


def download_model():
    from TTS.api import TTS
    TTS("tts_models/multilingual/multi-dataset/xtts_v2")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch",
        "torchaudio",
        "torchcodec",
        "six",
        "coqui-tts",
        "flask",
        "flask-cors",
        "soundfile",
        "librosa",
        "scipy",
        "num2words",
    )
    .env({"COQUI_TOS_AGREED": "1"})
    .run_function(download_model)
    .add_local_file(APP_DIR / "server.py", "/root/app/server.py")
    .add_local_file(APP_DIR / "index.html", "/root/app/index.html")
)

app = modal.App("voxclone", image=image)

@app.function(
    gpu="T4",
    timeout=1800,
    scaledown_window=120,
    max_containers=1,
    secrets=[modal.Secret.from_name("voxclone-auth")],
)
@modal.concurrent(max_inputs=4)
@modal.wsgi_app()
def web():
    import os
    import sys
    import hmac

    sys.path.insert(0, "/root/app")

    import server
    from flask import request, Response

    user = os.environ["VOX_USER"].encode()
    password = os.environ["VOX_PASS"].encode()

    @server.app.before_request
    def require_login():
        auth = request.authorization
        if auth:
            u = (auth.username or "").encode()
            p = (auth.password or "").encode()
            if hmac.compare_digest(u, user) and hmac.compare_digest(p, password):
                return None
        return Response(
            "Login required",
            401,
            {"WWW-Authenticate": 'Basic realm="VoxClone"'},
        )

    return server.app

