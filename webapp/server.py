"""
FastAPI backend for the CatFlow web demo.

    python server.py            # serves on 0.0.0.0:8000

Endpoints
    GET  /                   -> mobile-first single-page UI
    GET  /api/adsorbates     -> adsorbate menu
    GET  /api/compositions   -> slab composition menu (structure prediction)
    POST /api/generate       -> run CatFlow sampling (10 samples, 50 steps)
    POST /api/eval           -> UMA relaxation + adsorption energy (+ volcano)
"""
from __future__ import annotations

import json
import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
STATIC_DIR = HERE / "static"
DEVICE = os.environ.get("CATFLOW_DEVICE", "cuda:0")

ENGINE = None     # CatFlowEngine
EVALUATOR = None  # EnergyEvaluator
ADSORBATES: list[dict] = []
COMPOSITIONS: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE, EVALUATOR, ADSORBATES, COMPOSITIONS
    ADSORBATES = json.loads((DATA_DIR / "adsorbates.json").read_text())
    COMPOSITIONS = json.loads((DATA_DIR / "compositions.json").read_text())

    from catflow_engine import CatFlowEngine
    from energy import EnergyEvaluator

    ENGINE = CatFlowEngine(device=DEVICE)
    ENGINE.load_models()
    EVALUATOR = EnergyEvaluator(device=DEVICE)
    EVALUATOR.load()
    print("[server] ready.", flush=True)
    yield


app = FastAPI(title="CatFlow Demo", lifespan=lifespan)

# CORS — the static UI is served from GitHub Pages while this backend runs on
# the GPU server behind a tunnel, so cross-origin requests must be allowed.
# Override the allowed origins with CATFLOW_CORS_ORIGINS (comma-separated).
_cors = os.environ.get(
    "CATFLOW_CORS_ORIGINS",
    "https://minkyu1022.github.io,http://localhost:8000,http://127.0.0.1:8000",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors.split(",") if o.strip()],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _ads_by_id(ads_id: int) -> dict:
    for a in ADSORBATES:
        if a["id"] == ads_id:
            return a
    raise HTTPException(404, f"adsorbate id {ads_id} not found")


def _comp_by_id(comp_id: str) -> dict:
    for c in COMPOSITIONS:
        if c["id"] == comp_id:
            return c
    raise HTTPException(404, f"composition id {comp_id} not found")


# --------------------------------------------------------------------------
@app.get("/api/adsorbates")
def get_adsorbates():
    return ADSORBATES


@app.get("/api/compositions")
def get_compositions():
    return COMPOSITIONS


def parse_composition(formula: str) -> dict:
    """Parse a free-text slab composition such as 'Pt3Ni', 'Pd Ag', 'Cu4'."""
    import re
    from ase.data import atomic_numbers, chemical_symbols

    tokens = re.findall(r"([A-Z][a-z]?)\s*(\d*)", formula.strip())
    numbers: list[int] = []
    for sym, count in tokens:
        if not sym:
            continue
        if sym not in atomic_numbers:
            raise HTTPException(400, f"unknown element '{sym}'")
        numbers.extend([atomic_numbers[sym]] * (int(count) if count else 1))
    if not (1 <= len(numbers) <= 48):
        raise HTTPException(
            400, "composition must have between 1 and 48 atoms")
    numbers.sort()
    from collections import Counter
    cnt = Counter(chemical_symbols[z] for z in numbers)
    raw = "".join(f"{el}{cnt[el]}" for el in sorted(cnt))
    sub = raw.translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
    return {
        "id": f"custom:{raw}", "formula": raw, "raw": raw,
        "display": sub, "n_atoms": len(numbers), "numbers": numbers,
        "elements": sorted(cnt.keys()), "custom": True,
    }


class GenerateRequest(BaseModel):
    mode: str                       # "denovo" | "structure"
    adsorbate_id: str
    composition_id: str | None = None
    custom_composition: str | None = None   # free-text formula (overrides id)


@app.post("/api/generate")
def generate(req: GenerateRequest):
    adsorbate = _ads_by_id(req.adsorbate_id)
    composition = None
    if req.mode == "structure":
        if req.custom_composition and req.custom_composition.strip():
            composition = parse_composition(req.custom_composition)
        elif req.composition_id:
            composition = _comp_by_id(req.composition_id)
        else:
            raise HTTPException(
                400, "composition required for structure mode")
    try:
        result = ENGINE.generate(req.mode, adsorbate, composition)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"generation failed: {e}")
    result["adsorbate"] = {
        "id": adsorbate["id"], "name": adsorbate["name"],
        "display": adsorbate["display"], "evaluable": adsorbate["evaluable"],
    }
    if composition:
        result["composition"] = {
            "id": composition["id"], "display": composition["display"],
            "custom": composition.get("custom", False)}
    return result


class EvalRequest(BaseModel):
    job_id: str
    sample_idx: int
    adsorbate_id: str


@app.post("/api/eval")
def evaluate(req: EvalRequest):
    adsorbate = _ads_by_id(req.adsorbate_id)
    try:
        atoms = ENGINE.get_structure(req.job_id, req.sample_idx)
    except KeyError as e:
        raise HTTPException(404, str(e))
    try:
        res = EVALUATOR.evaluate(atoms)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"energy evaluation failed: {e}")

    from energy import volcano_data
    res["volcano"] = volcano_data(adsorbate["name"], res["e_ads_relaxed"])
    res["adsorbate_name"] = adsorbate["name"]
    res["chemical_formula"] = (
        f"{res['slab_formula']} + {res['adsorbate_formula']}")
    return JSONResponse(res)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)),
                timeout_keep_alive=600)
