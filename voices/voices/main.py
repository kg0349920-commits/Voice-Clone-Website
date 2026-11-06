import uuid
import threading
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
jobs = {}  # job_id -> {"progress": int, "status": str, "file": str | None, "error": str | None}


def run_job(job_id, text, ref_path):
    try:
        jobs[job_id]["status"] = "running"
        # TODO: yahan asli voice clone model ka code aayega (XTTS etc.)
        for i in range(1, 101, 10):
            time.sleep(0.3)
            jobs[job_id]["progress"] = i
        jobs[job_id]["progress"] = 100
        jobs[job_id]["status"] = "done"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.post("/api/generate")
async def generate(text: str = Form(...), reference: UploadFile = File(None)):
    job_id = uuid.uuid4().hex
    ref_path = None
    if reference:
        ref_path = OUT_DIR / f"{job_id}_ref_{reference.filename}"
        ref_path.write_bytes(await reference.read())
    jobs[job_id] = {"progress": 0, "status": "queued", "file": None, "error": None}
    threading.Thread(target=run_job, args=(job_id, text, ref_path), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/progress/{job_id}")
def progress(job_id: str):
    j = jobs.get(job_id)
    if not j:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"progress": j["progress"]}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    j = jobs.get(job_id)
    if not j:
        return JSONResponse({"error": "not found"}, status_code=404)
    return 