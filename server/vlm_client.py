"""
VLM HTTP Client
Run on the local machine to send pipeline results to the VLM server.

Reads:   images/results/pipeline_results.json  (produced by visual pipeline)
Sends:   POST http://localhost:8000/decide_batch  (via SSH tunnel)
Writes:  images/results/vlm_results.json

Before running, open the SSH tunnel in a separate terminal:
    ssh -L 8000:localhost:8000 root@69.30.85.125 -p 22037 -N

Then run:
    python server/vlm_client.py
"""

import base64
import json
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

# ── config ────────────────────────────────────────────────────────────────────

SERVER_URL    = "http://localhost:8000"
RESULTS_DIR   = Path("images/results")
INPUT_JSON    = RESULTS_DIR / "pipeline_results.json"
OUTPUT_JSON   = RESULTS_DIR / "vlm_results.json"
REQUEST_TIMEOUT = 120   # seconds per batch


# ── helpers ───────────────────────────────────────────────────────────────────

def _image_to_base64(image_path: Path) -> str:
    img = Image.open(image_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _check_health():
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        data = resp.json()
        assert data.get("model_loaded"), "model not loaded yet"
        print(f"  server health: OK  (model_loaded={data['model_loaded']})")
    except Exception as e:
        raise RuntimeError(f"VLM server not reachable at {SERVER_URL}: {e}")


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("VLM Client  --  batch decide")
    print("=" * 60)

    # ── 1. health check ───────────────────────────────────────────
    _check_health()

    # ── 2. load pipeline results ──────────────────────────────────
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"{INPUT_JSON} not found. Run the visual pipeline first."
        )

    with open(INPUT_JSON) as f:
        records = json.load(f)
    print(f"  loaded {len(records)} record(s) from {INPUT_JSON}")

    # ── 3. build batch request ────────────────────────────────────
    items = []
    for rec in records:
        annotated_path = Path(rec["annotated_path"])
        if not annotated_path.exists():
            # try resolving relative to RESULTS_DIR (cross-machine path mismatch)
            annotated_path = RESULTS_DIR / annotated_path.name
        if not annotated_path.exists():
            raise FileNotFoundError(f"annotated image not found: {annotated_path}")

        items.append({
            "image_b64":        _image_to_base64(annotated_path),
            "keypoints_2d":     rec["keypoints_2d"],
            "task_instruction": rec["instruction"],
            "mode":             rec["mode"],
        })

    # ── 4. POST to server ─────────────────────────────────────────
    print(f"  sending {len(items)} item(s) to {SERVER_URL}/decide_batch ...")
    t0 = time.perf_counter()

    resp = requests.post(
        f"{SERVER_URL}/decide_batch",
        json={"items": items},
        timeout=REQUEST_TIMEOUT * len(items),
    )
    resp.raise_for_status()
    elapsed = time.perf_counter() - t0

    decisions = resp.json()["results"]
    print(f"  received {len(decisions)} decision(s)  ({elapsed:.1f}s total)")

    # ── 5. merge and save ─────────────────────────────────────────
    output = []
    for rec, dec in zip(records, decisions):
        output.append({
            "instruction": rec["instruction"],
            "mode":        rec["mode"],
            "keypoints_2d": rec["keypoints_2d"],
            "decision":    dec,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  saved → {OUTPUT_JSON}")

    # ── 6. summary ────────────────────────────────────────────────
    print()
    print("  instruction                          src       w_grasp  w_safety  conf")
    print("  " + "-" * 70)
    for entry in output:
        d   = entry["decision"]
        src = "fallback" if d["is_fallback"] else "VLM"
        print(f"  {entry['instruction'][:36]:<36}  {src:<8}  "
              f"{d['w_grasp_axis']:.2f}     {d['w_safety']:.2f}      {d['confidence']:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    run()