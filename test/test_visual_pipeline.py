"""
Experiment 1: Visual Pipeline Test
Tests the full visual pipeline from task instruction to annotated keypoint image.

Pipeline:
    instruction -> TaskParser -> GroundingDINO -> ImageProcessor
                -> SAM3Segmenter -> build_annotated_image -> save result

Each case returns a structured dict that feeds directly into Experiment 2
(VLM constraint decision) without re-running the visual pipeline.

Usage:
    cd /workspace/PartKep
    python test/test_visual_pipeline.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from modules import GroundingDINODetector, ImageProcessor, SAM3Segmenter
from modules import TaskParser, TaskSpec
from modules.vlmDecider import build_annotated_image


# ── test cases ──────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "instruction": "pick up the cup",
        "image_path":  PROJECT_ROOT / "images/cup3.jpg",
    },
    {
        "instruction": "pick up the leftmost cup",
        "image_path":  PROJECT_ROOT / "images/cups.png",
    },
    # kitchen scene -- detection_prompt overrides TaskParser output for GroundingDINO
    {
        "instruction":      "pick up the cup",
        "detection_prompt": "orange cup on table",
        "image_path":       PROJECT_ROOT / "images/mujoco.png",
    },
    {
        "instruction":      "pick up the bottle",
        "detection_prompt": "brown bottle with VOSS label",
        "image_path":       PROJECT_ROOT / "images/kitchen.png",
    },
]

OUTPUT_DIR = PROJECT_ROOT / "images/results"

# keypoints closer than this fraction of image short-side are flagged
_PROXIMITY_THRESHOLD_RATIO = 0.05


# ── helpers ──────────────────────────────────────────────────────────────────

def _action_to_mode(action: str) -> str:
    # visual pipeline always targets the pick object; place phase is handled separately
    return "pick"


def _check_keypoint_proximity(
    keypoints_2d: Dict[str, Tuple[float, float]],
    image_size:   Tuple[int, int],
) -> None:
    # warn if two keypoints are suspiciously close (likely SAM3 mask collapse)
    threshold = min(image_size) * _PROXIMITY_THRESHOLD_RATIO
    parts = list(keypoints_2d.keys())
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1, p2 = parts[i], parts[j]
            x1, y1 = keypoints_2d[p1]
            x2, y2 = keypoints_2d[p2]
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if dist < threshold:
                print(f"  WARNING: '{p1}' and '{p2}' are {dist:.1f}px apart "
                      f"(threshold {threshold:.1f}px) -- possible mask collapse")


# ── single case ──────────────────────────────────────────────────────────────

def run_single_case(
    instruction: str,
    image_path:  Path,
    parser:           TaskParser,
    detector:         GroundingDINODetector,
    processor:        ImageProcessor,
    segmenter:        SAM3Segmenter,
    detection_prompt: Optional[str] = None,
) -> Optional[Dict]:
    """Run the full visual pipeline for one instruction-image pair.

    detection_prompt overrides the TaskParser-derived prompt for GroundingDINO,
    allowing richer spatial/attribute descriptions without changing the instruction.
    Returns a dict ready for Experiment 2 (VLMDecider), or None on failure.
    """
    print(f"\n  instruction : {instruction!r}")
    print(f"  image       : {Path(image_path).name}")

    timings = {}

    # 1. parse instruction
    t0 = time.perf_counter()
    try:
        spec = parser.parse(instruction)
    except ValueError as e:
        print(f"  [parse] FAILED -- {e}")
        return None
    timings["parse"] = time.perf_counter() - t0

    # detection_prompt: explicit override > TaskParser-derived prompt
    det_prompt = detection_prompt if detection_prompt else spec.get_detection_prompt()
    print(f"  [parse]     {timings['parse']:.3f}s  "
          f"object='{spec.object_label}'  "
          f"det_prompt='{det_prompt}'")

    # 2. load image
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"  [image] FAILED -- file not found: {image_path}")
        return None

    # 3. object detection
    t0 = time.perf_counter()
    detection_results = detector.detect(
        image=image,
        text_prompt=det_prompt,
        box_threshold=0.35,
        text_threshold=0.25,
    )
    timings["detection"] = time.perf_counter() - t0
    if not detection_results:
        print(f"  [detect]    FAILED -- '{det_prompt}' not found")
        return None
    bbox = detection_results[0]["bbox"]
    print(f"  [detect]    {timings['detection']:.3f}s  "
          f"bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]  "
          f"score={detection_results[0]['score']:.3f}")

    # 4. crop detected region
    t0 = time.perf_counter()
    crop_results = processor.crop_objects_batch(
        image=image,
        detection_results=detection_results,
        padding=10,
    )
    timings["crop"] = time.perf_counter() - t0
    if not crop_results:
        print("  [crop]      FAILED")
        return None
    crop_result = crop_results[0]
    w, h = crop_result["crop_size"]
    print(f"  [crop]      {timings['crop']:.3f}s  size={w}x{h}")

    # 5. part segmentation and keypoint extraction
    t0 = time.perf_counter()
    seg_results = segmenter.segment_parts(
        cropped_image=crop_result["cropped_image"],
        label=spec.object_label,
        crop_bbox=crop_result["bbox_pixel"],
    )
    timings["segment"] = time.perf_counter() - t0
    if not seg_results:
        print("  [segment]   FAILED -- no parts detected")
        return None
    print(f"  [segment]   {timings['segment']:.3f}s  {len(seg_results)} parts")

    keypoints_2d: Dict[str, Tuple[float, float]] = {
        r["part_name"]: tuple(r["keypoint"]) for r in seg_results
    }

    # print keypoint coordinates and scores
    for r in seg_results:
        x, y = r["keypoint"]
        print(f"    {r['part_name']:8s}  ({x:6.1f}, {y:6.1f})  score={r['score']:.3f}")

    _check_keypoint_proximity(keypoints_2d, image.size)

    # 6. generate and save annotated image
    t0 = time.perf_counter()
    annotated = build_annotated_image(image, keypoints_2d)
    modifier_tag = f"{spec.spatial_modifier}_" if spec.spatial_modifier else ""
    out_name = f"{Path(image_path).stem}_{modifier_tag}{spec.object_label}_result.jpg"
    out_path = OUTPUT_DIR / out_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated.save(out_path, quality=95)
    timings["annotate"] = time.perf_counter() - t0

    total = sum(timings.values())
    print(f"  [annotate]  {timings['annotate']:.3f}s  saved: {out_path.name}")
    print(f"  total: {total:.3f}s")

    return {
        "success":        True,
        "instruction":    instruction,
        "spec":           spec,
        "image":          image,
        "keypoints_2d":   keypoints_2d,
        "mode":           _action_to_mode(spec.action),
        "annotated_path": out_path,
        "timings":        timings,
    }


# ── batch runner ─────────────────────────────────────────────────────────────

def run_all_cases() -> List[Optional[Dict]]:
    """Initialise all modules once and run all TEST_CASES.

    Returns the list of result dicts (None on failure) for use in Experiment 2.
    """
    print("=" * 70)
    print("Experiment 1: Visual Pipeline")
    print("=" * 70)

    parser    = TaskParser()
    detector  = GroundingDINODetector()
    processor = ImageProcessor(output_dir=PROJECT_ROOT / "images/objectlist")
    segmenter = SAM3Segmenter()

    pipeline_results: List[Optional[Dict]] = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}]")
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

    # summary
    print(f"\n{'=' * 70}")
    passed = sum(1 for r in pipeline_results if r is not None)
    print(f"Results: {passed}/{len(pipeline_results)} passed\n")
    for case, result in zip(TEST_CASES, pipeline_results):
        if result is not None:
            kps = result["keypoints_2d"]
            kp_str = "  ".join(
                f"{k}=({v[0]:.0f},{v[1]:.0f})" for k, v in kps.items()
            )
            total_t = sum(result["timings"].values())
            print(f"  OK  {case['instruction']!r}")
            print(f"      keypoints: {kp_str}")
            print(f"      time: {total_t:.3f}s")
        else:
            print(f"  FAIL  {case['instruction']!r}")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("=" * 70)

    return pipeline_results


if __name__ == "__main__":
    run_all_cases()