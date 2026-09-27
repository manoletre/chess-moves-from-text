"""Demo web app for the chess move & board-location tagger.

    pip install -r app/requirements.txt
    python -m uvicorn app.server:app --port 8000    # from the repo root
    open http://localhost:8000

The model folder defaults to ./chess-move-tagger; override with CHESS_TAGGER_MODEL.
"""

import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chess_tagger import ChessTagger  # noqa: E402

MODEL_DIR = Path(os.environ.get("CHESS_TAGGER_MODEL", ROOT / "chess-move-tagger"))
STATIC_DIR = Path(__file__).parent / "static"
MAX_CHARS = 5000

state: dict = {}
lock = threading.Lock()  # one inference at a time keeps memory flat on small machines


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not (MODEL_DIR / "model.safetensors").exists():
        raise RuntimeError(f"No model found in {MODEL_DIR}. Download it from the GitHub release (see README).")
    started = time.perf_counter()
    state["tagger"] = ChessTagger(MODEL_DIR)
    print(f"Loaded {MODEL_DIR} in {time.perf_counter() - started:.1f}s")
    yield
    state.clear()


app = FastAPI(title="Chess move & board-location tagger", lifespan=lifespan)


class TagRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


@app.post("/api/tag")
def tag(request: TagRequest) -> dict:
    tagger = state.get("tagger")
    if tagger is None:
        raise HTTPException(503, "Model is still loading")
    started = time.perf_counter()
    with lock:
        result = tagger.tag(request.text)
    return {**result, "elapsed_ms": round((time.perf_counter() - started) * 1000)}


@app.get("/api/info")
def info() -> dict:
    tagger = state["tagger"]
    return {"base_model": tagger.base_model, "labels": tagger.labels,
            "max_length": tagger.max_length, "device": str(tagger.device), "max_chars": MAX_CHARS}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
