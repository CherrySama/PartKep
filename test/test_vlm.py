"""
Experiment 2: VLM Constraint Decision Test
Created by Yinghao Ho on 2026-04

Reads pipeline_results.json produced by Experiment 1 (separate process),
then runs VLMDecider on each case. Keeps visual models and VLM in separate
processes to avoid VRAM conflicts.

Run Experiment 1 first:
    python test/test_visual_pipeline.py

Then run this:
    python test/test_vlm_decider.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from modules.vlmDecider import VLMDecider

# ── config ────────────────────────────────────────────────────────────────────

MODEL_PATH      = PROJECT_ROOT / "models/Qwen3.5-9B"
LOAD_IN_4BIT    = True
RESULTS_JSON    = PROJECT_ROOT / "images/results/pipeline_results.json"

# subset of instructions to run VLM on (None = run all from JSON)
VLM_FILTER: Optional[List[str]] = [
    "pick up the cup",
]


# ── single case ───────────────────────────────────────────────────────────────

def run_vlm_case(record: Dict, decider: VLMDecider) -> Dict:
    """Run VLM decision for one record loaded from pipeline_results.json."""
    instruction  = record["instruction"]
    mode         = record["mode"]
    keypoints_2d = {k: tuple(v) for k, v in record["keypoints_2d"].items()}
    annotated    = Image.open(record["annotated_path"]).convert("RGB")

    kp_str = "  ".join(
        f"{k}=({v[0]:.0f},{v[1]:.0f})" for k, v in keypoints_2d.items()
    )
    print(f"  instruction  : {instruction!r}")
    print(f"  mode         : {mode}")
    print(f"  keypoints    : {kp_str}")

    t0 = time.perf_counter()
    decision = decider.decide(
        rgb_image=annotated,
        keypoints_2d=keypoints_2d,
        task_instruction=instruction,
        mode=mode,
    )
    elapsed = time.perf_counter() - t0

    meta = decider.last_inference_meta

    if meta:
        print(f"  [vlm]        {elapsed:.3f}s  "
              f"in={meta['input_tokens']}tok  "
              f"out={meta['output_tokens']}tok")
        print(f"  raw          : {meta['raw_response']}")
    else:
        print(f"  [vlm]        {elapsed:.3f}s  (fallback)")

    print(f"  w_grasp_axis : {decision.w_grasp_axis:.2f}")
    print(f"  w_safety     : {decision.w_safety:.2f}")
    print(f"  confidence   : {decision.confidence:.2f}")
    print(f"  reasoning    : {decision.reasoning}")
    print(f"  is_fallback  : {decision.is_fallback}")

    return {
        "instruction": instruction,
        "keypoints_2d": keypoints_2d,
        "decision":    decision,
        "elapsed":     elapsed,
        "meta":        meta,
    }


# ── batch runner ──────────────────────────────────────────────────────────────

def run_all_vlm_cases() -> List[Dict]:
    print("=" * 70)
    print("Experiment 2: VLM Constraint Decision")
    print("=" * 70)

    # load pipeline results from disk
    if not RESULTS_JSON.exists():
        print(f"ERROR: {RESULTS_JSON} not found.")
        print("Run Experiment 1 first: python test/test_visual_pipeline.py")
        return []

    with open(RESULTS_JSON) as f:
        records = json.load(f)

    # filter by instruction if VLM_FILTER is set
    if VLM_FILTER is not None:
        records = [r for r in records if r["instruction"] in VLM_FILTER]

    print(f"  {len(records)} case(s) loaded from {RESULTS_JSON.name}\n")

    # load VLM
    print("-- Loading VLMDecider --")
    decider = VLMDecider(
        model_path=str(MODEL_PATH),
        load_in_4bit=LOAD_IN_4BIT,
    )

    # run decisions
    print("\n-- VLM decisions --")
    vlm_results = []
    for i, record in enumerate(records, 1):
        print(f"\n[{i}/{len(records)}]")
        result = run_vlm_case(record, decider)
        vlm_results.append(result)

    # summary
    print(f"\n{'=' * 70}")
    print("Experiment 2 Summary")
    print(f"{'=' * 70}")
    for r in vlm_results:
        d = r["decision"]
        src = "fallback" if d.is_fallback else "VLM"
        print(f"  {r['instruction']!r}")
        print(f"    src={src}  w_grasp={d.w_grasp_axis:.2f}  "
              f"w_safety={d.w_safety:.2f}  conf={d.confidence:.2f}  "
              f"time={r['elapsed']:.3f}s")
        print(f"    reasoning: {d.reasoning}")
    print("=" * 70)

    return vlm_results


if __name__ == "__main__":
    run_all_vlm_cases()