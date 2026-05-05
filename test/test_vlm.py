"""
test_vlm.py  —  Experiment 2: VLM Constraint Decision
Created by Yinghao Ho on 2026-04

Reads pipeline_results.json produced by Experiment 1, runs VLMDecider on
each case. Kept as a separate process to avoid VRAM conflicts with SAM3.

Run Experiment 1 first:
    python test/test_visual_pipeline.py

Then run on the server:
    python test/test_vlm.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

# Import vlmDecider directly to avoid modules/__init__.py pulling in mujoco/IKSolver
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "vlmDecider", PROJECT_ROOT / "modules/vlmDecider.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
VLMDecider = _mod.VLMDecider

# ── config ────────────────────────────────────────────────────────────────────

MODEL_PATH   = Path("/workspace/models/Qwen3.5-9B")
LOAD_IN_4BIT = True
RESULTS_JSON = Path("/workspace/PartKep/images/results/pipeline_results.json")
OUTPUT_JSON  = Path("/workspace/PartKep/results/vlm_results.json")

# Set to a list of instruction strings to run only a subset; None = run all
VLM_FILTER: Optional[List[str]] = [
    "pick up the cup",
]

def run_vlm_case(record: Dict, decider: VLMDecider) -> Dict:
    """Run VLM decision for one record from pipeline_results.json."""
    instruction  = record["instruction"]
    mode         = record["mode"]
    keypoints_2d = {k: tuple(v) for k, v in record["keypoints_2d"].items()}

    # remap annotated_path to server location (JSON was written on a different machine)
    annotated_path = (
        Path("/workspace/PartKep/images/results") /
        Path(record["annotated_path"]).name
    )
    annotated = Image.open(annotated_path).convert("RGB")

    kp_str = "  ".join(f"{k}=({v[0]:.0f},{v[1]:.0f})" for k, v in keypoints_2d.items())
    print(f"  {instruction!r}  mode={mode}  kp=[{kp_str}]")

    t0       = time.perf_counter()
    decision = decider.decide(
        rgb_image        = annotated,
        keypoints_2d     = keypoints_2d,
        task_instruction = instruction,
        mode             = mode,
    )
    elapsed = time.perf_counter() - t0

    meta = decider.last_inference_meta
    if meta:
        print(f"  t={elapsed:.2f}s  in={meta['input_tokens']}tok"
              f"  out={meta['output_tokens']}tok")
        print(f"  raw: {meta['raw_response']}")
    else:
        print(f"  t={elapsed:.2f}s  (fallback)")

    print(f"  w_grasp={decision.w_grasp_axis:.2f}  w_safety={decision.w_safety:.2f}"
          f"  conf={decision.confidence:.2f}  fallback={decision.is_fallback}")
    print(f"  reasoning: {decision.reasoning}")

    return {
        "instruction":  instruction,
        "mode":         mode,
        "keypoints_2d": keypoints_2d,
        "decision":     decision,
        "elapsed":      elapsed,
        "meta":         meta,
    }

def run_all_vlm_cases() -> List[Dict]:
    print("=" * 60)
    print("Experiment 2: VLM Constraint Decision")
    print("=" * 60)

    if not RESULTS_JSON.exists():
        print(f"ERROR: {RESULTS_JSON} not found.")
        print("Run test/test_visual_pipeline.py first.")
        return []

    with open(RESULTS_JSON) as f:
        records = json.load(f)

    if VLM_FILTER is not None:
        records = [r for r in records if r["instruction"] in VLM_FILTER]
    print(f"{len(records)} case(s) loaded\n")

    decider = VLMDecider(model_path=str(MODEL_PATH), load_in_4bit=LOAD_IN_4BIT)

    vlm_results = []
    for i, record in enumerate(records, 1):
        print(f"[{i}/{len(records)}]")
        result = run_vlm_case(record, decider)
        vlm_results.append(result)

    # summary
    print(f"\n{'=' * 60}")
    for r in vlm_results:
        d   = r["decision"]
        src = "fallback" if d.is_fallback else "VLM"
        print(f"  {r['instruction']!r}  [{src}]"
              f"  w_grasp={d.w_grasp_axis:.2f}  w_safety={d.w_safety:.2f}"
              f"  conf={d.confidence:.2f}  t={r['elapsed']:.2f}s")
    print("=" * 60)

    # save results
    output = []
    for r in vlm_results:
        d = r["decision"]
        output.append({
            "instruction":  r["instruction"],
            "mode":         r["mode"],
            "keypoints_2d": {k: list(v) for k, v in r["keypoints_2d"].items()},
            "decision": {
                "w_grasp_axis": d.w_grasp_axis,
                "w_safety":     d.w_safety,
                "confidence":   d.confidence,
                "reasoning":    d.reasoning,
                "is_fallback":  d.is_fallback,
            },
        })
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {OUTPUT_JSON}")

    return vlm_results


if __name__ == "__main__":
    run_all_vlm_cases()