"""
Experiment 2: VLM Constraint Decision Test
Created by Yinghao Ho on 2026-04

Takes structured results from Experiment 1 and passes them to VLMDecider.
Only runs the specified VLM_TEST_CASES through the visual pipeline,
not the full Experiment 1 suite.

Usage:
    cd /workspace/PartKep
    python test/test_vlm_decider.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.vlmDecider import VLMDecider, VLMDecision
from modules import GroundingDINODetector, ImageProcessor, SAM3Segmenter, TaskParser

sys.path.insert(0, str(PROJECT_ROOT / "test"))
from test_visual_pipeline import run_single_case

# ── config ────────────────────────────────────────────────────────────────────

MODEL_PATH   = PROJECT_ROOT / "models/Qwen3.5-9B"
LOAD_IN_4BIT = True

# cases to run through the visual pipeline before VLM decision
VLM_TEST_CASES = [
    {
        "instruction": "pick up the cup",
        "image_path":  PROJECT_ROOT / "images/cup3.jpg",
    },
]


# ── single VLM case ───────────────────────────────────────────────────────────

def run_vlm_case(
    pipeline_result: Dict,
    decider:         VLMDecider,
) -> Optional[Dict]:
    """Run VLM constraint decision for one pipeline result."""
    instruction  = pipeline_result["instruction"]
    image        = pipeline_result["image"]
    keypoints_2d = pipeline_result["keypoints_2d"]
    mode         = pipeline_result["mode"]

    kp_str = "  ".join(
        f"{k}=({v[0]:.0f},{v[1]:.0f})" for k, v in keypoints_2d.items()
    )
    print(f"  instruction  : {instruction!r}")
    print(f"  mode         : {mode}")
    print(f"  keypoints    : {kp_str}")

    t0 = time.perf_counter()
    decision = decider.decide(
        rgb_image=image,
        keypoints_2d=keypoints_2d,
        task_instruction=instruction,
        mode=mode,
    )
    elapsed = time.perf_counter() - t0

    meta = decider.last_inference_meta  # populated by _infer_local()

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
        "instruction":  instruction,
        "keypoints_2d": keypoints_2d,
        "decision":     decision,
        "elapsed":      elapsed,
        "meta":         meta,
    }


# ── batch runner ──────────────────────────────────────────────────────────────

def run_all_vlm_cases() -> List[Optional[Dict]]:
    print("=" * 70)
    print("Experiment 2: VLM Constraint Decision")
    print("=" * 70)
    print(f"  {len(VLM_TEST_CASES)} case(s) to run\n")

    # step 1: run only the specified cases through the visual pipeline
    print("-- Visual pipeline (Experiment 1) --")
    parser    = TaskParser()
    detector  = GroundingDINODetector()
    processor = ImageProcessor(output_dir=PROJECT_ROOT / "images/objectlist")
    segmenter = SAM3Segmenter()

    pipeline_results = []
    for case in VLM_TEST_CASES:
        result = run_single_case(
            instruction=case["instruction"],
            image_path=case["image_path"],
            parser=parser,
            detector=detector,
            processor=processor,
            segmenter=segmenter,
            detection_prompt=case.get("detection_prompt"),
        )
        pipeline_results.append(result)

    passed = [r for r in pipeline_results if r is not None]
    print(f"\n  visual pipeline: {len(passed)}/{len(VLM_TEST_CASES)} passed")

    # step 2: load VLM once
    print("\n-- Loading VLMDecider --")
    decider = VLMDecider(
        model_path=str(MODEL_PATH),
        load_in_4bit=LOAD_IN_4BIT,
    )

    # step 3: VLM decision for each successful pipeline result
    print("\n-- VLM decisions --")
    vlm_results = []
    for i, result in enumerate(passed, 1):
        print(f"\n[{i}/{len(passed)}]")
        vlm_result = run_vlm_case(result, decider)
        vlm_results.append(vlm_result)

    # summary
    print(f"\n{'=' * 70}")
    print("Experiment 2 Summary")
    print(f"{'=' * 70}")
    for r in vlm_results:
        if r is None:
            continue
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