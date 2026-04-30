"""
VLM HTTP Server
Run on the remote server to expose VLMDecider via a simple REST API.

Start:
    python server/vlm_server.py

Then on local machine, open SSH tunnel:
    ssh -L 8000:localhost:8000 root@69.30.85.125 -p 22037 -N
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "vlmDecider", Path(__file__).parent.parent / "modules/vlmDecider.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
VLMDecider = _mod.VLMDecider
from PIL import Image
import base64
from io import BytesIO

# ── config ────────────────────────────────────────────────────────────────────

MODEL_PATH   = "/workspace/models/Qwen3.5-9B"
LOAD_IN_4BIT = True
PORT         = 8000

# ── app ───────────────────────────────────────────────────────────────────────

app     = FastAPI()
decider = None   # loaded on startup


@app.on_event("startup")
def load_model():
    global decider
    print("Loading VLMDecider...")
    decider = VLMDecider(model_path=MODEL_PATH, load_in_4bit=LOAD_IN_4BIT)
    print("VLMDecider ready.")


# ── request / response schema ─────────────────────────────────────────────────

class DecideRequest(BaseModel):
    image_b64:        str             # JPEG image encoded as base64
    keypoints_2d:     Dict[str, Tuple[float, float]]
    task_instruction: str
    mode:             str             # "pick" or "place"


class DecideResponse(BaseModel):
    w_grasp_axis: float
    w_safety:     float
    confidence:   float
    reasoning:    str
    is_fallback:  bool


# ── endpoint ──────────────────────────────────────────────────────────────────

@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest):
    img_bytes = base64.b64decode(req.image_b64)
    rgb_image = Image.open(BytesIO(img_bytes)).convert("RGB")

    decision = decider.decide(
        rgb_image        = rgb_image,
        keypoints_2d     = req.keypoints_2d,
        task_instruction = req.task_instruction,
        mode             = req.mode,
    )

    return DecideResponse(
        w_grasp_axis = decision.w_grasp_axis,
        w_safety     = decision.w_safety,
        confidence   = decision.confidence,
        reasoning    = decision.reasoning,
        is_fallback  = decision.is_fallback,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": decider is not None}


# ── batch endpoint ────────────────────────────────────────────────────────────

class BatchDecideRequest(BaseModel):
    items: List[DecideRequest]


class BatchDecideResponse(BaseModel):
    results: List[DecideResponse]


@app.post("/decide_batch", response_model=BatchDecideResponse)
def decide_batch(req: BatchDecideRequest):
    """Process a list of DecideRequests sequentially and return all decisions."""
    results = []
    for item in req.items:
        img_bytes = base64.b64decode(item.image_b64)
        rgb_image = Image.open(BytesIO(img_bytes)).convert("RGB")

        decision = decider.decide(
            rgb_image        = rgb_image,
            keypoints_2d     = item.keypoints_2d,
            task_instruction = item.task_instruction,
            mode             = item.mode,
        )

        results.append(DecideResponse(
            w_grasp_axis = decision.w_grasp_axis,
            w_safety     = decision.w_safety,
            confidence   = decision.confidence,
            reasoning    = decision.reasoning,
            is_fallback  = decision.is_fallback,
        ))

    return BatchDecideResponse(results=results)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)