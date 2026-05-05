"""
Pipeline:
    instruction -> TaskParser -> GroundingDINO -> ImageProcessor
               -> SAM3Segmenter -> build_annotated_image -> save result
Created by Yinghao Ho on 2026-04-10
"""

import json
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

TEST_CASES = [
    {
        "instruction": "pick up the cup",
        "image_path":  PROJECT_ROOT / "images/cup3.jpg",
    },
    {
        "instruction": "pick up the leftmost cup",
        "image_path":  PROJECT_ROOT / "images/cups.png",
    },
    {
        "instruction":      "pick up the cup",
        "detection_prompt": "blue ceramic mug on the left side of the table",
        "image_path":       PROJECT_ROOT / "images/kitchen.png",
    },
    {
        "instruction":      "pick up the bottle",
        "detection_prompt": "brown bottle with VOSS label",
        "image_path":       PROJECT_ROOT / "images/kitchen.png",
    },
]

OUTPUT_DIR              = PROJECT_ROOT / "images/results"
PROXIMITY_THRESHOLD     = 0.05     # fraction of image short-side; flags collapsed keypoints


def _check_keypoint_proximity(
    keypoints_2d: Dict[str, Tuple[float, float]],
    image_size:   Tuple[int, int],
) -> None:
    """Warn if two keypoints are suspiciously close (likely SAM3 mask collapse)."""
    threshold = min(image_size) * PROXIMITY_THRESHOLD
    parts = list(keypoints_2d.keys())
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1, p2 = parts[i], parts[j]
            x1, y1 = keypoints_2d[p1]
            x2, y2 = keypoints_2d[p2]
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if dist < threshold:
                print(f"  [warn] '{p1}' and '{p2}' are {dist:.1f}px apart"
                      f" (threshold {threshold:.1f}px) -- possible mask collapse")


def run_single_case(
    instruction:      str,
    image_path:       Path,
    parser:           TaskParser,
    detector:         GroundingDINODetector,
    processor:        ImageProcessor,
    segmenter:        SAM3Segmenter,
    detection_prompt: Optional[str] = None,
) -> Optional[Dict]:
    """Run the full visual pipeline for one instruction-image pair.

    detection_prompt overrides the TaskParser-derived prompt for GroundingDINO.
    Returns a structured dict for Experiment 2, or None on failure.
    """
    print(f"\n  {instruction!r}  ({Path(image_path).name})")
    timings = {}

    # parse
    t0 = time.perf_counter()
    try:
        spec = parser.parse(instruction)
    except ValueError as e:
        print(f"  [parse]   FAILED -- {e}")
        return None
    timings["parse"] = time.perf_counter() - t0
    det_prompt = detection_prompt if detection_prompt else spec.get_detection_prompt()
    print(f"  [parse]   {timings['parse']*1000:.0f}ms  object='{spec.object_label}'")

    # load image
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"  [image]   FAILED -- not found: {image_path}")
        return None

    # detect
    t0 = time.perf_counter()
    detections = detector.detect(image, text_prompt=det_prompt,
                                 box_threshold=0.35, text_threshold=0.25)
    timings["detection"] = time.perf_counter() - t0
    if not detections:
        print(f"  [detect]  FAILED -- '{det_prompt}' not found")
        return None
    bbox = detections[0]["bbox"]
    print(f"  [detect]  {timings['detection']*1000:.0f}ms"
          f"  bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
          f"  score={detections[0]['score']:.3f}")

    # crop
    t0 = time.perf_counter()
    crops = processor.crop_objects_batch(image=image, detection_results=detections, padding=10)
    timings["crop"] = time.perf_counter() - t0
    if not crops:
        print("  [crop]    FAILED")
        return None
    w, h = crops[0]["crop_size"]
    print(f"  [crop]    {timings['crop']*1000:.0f}ms  size={w}x{h}")

    # segment
    t0 = time.perf_counter()
    seg_results = segmenter.segment_parts(
        cropped_image=crops[0]["cropped_image"],
        label=spec.object_label,
        crop_bbox=crops[0]["bbox_pixel"],
    )
    timings["segment"] = time.perf_counter() - t0
    if not seg_results:
        print("  [segment] FAILED -- no parts detected")
        return None
    print(f"  [segment] {timings['segment']*1000:.0f}ms  {len(seg_results)} parts")

    keypoints_2d: Dict[str, Tuple[float, float]] = {
        r["part_name"]: tuple(r["keypoint"]) for r in seg_results
    }
    for r in seg_results:
        x, y = r["keypoint"]
        print(f"            {r['part_name']:8s} ({x:.0f}, {y:.0f})  score={r['score']:.3f}")

    _check_keypoint_proximity(keypoints_2d, image.size)

    # annotate and save
    t0 = time.perf_counter()
    annotated    = build_annotated_image(image, keypoints_2d)
    modifier_tag = f"{spec.spatial_modifier}_" if spec.spatial_modifier else ""
    out_name     = f"{Path(image_path).stem}_{modifier_tag}{spec.object_label}_result.jpg"
    out_path     = OUTPUT_DIR / out_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated.save(out_path, quality=95)
    timings["annotate"] = time.perf_counter() - t0

    total = sum(timings.values())
    print(f"  [annotate] {timings['annotate']*1000:.0f}ms  saved: {out_path.name}"
          f"  total={total:.2f}s")

    return {
        "success":        True,
        "instruction":    instruction,
        "spec":           spec,
        "image":          image,
        "keypoints_2d":   keypoints_2d,
        "mode":           "pick",
        "annotated_path": out_path,
        "timings":        timings,
    }


def run_all_cases() -> List[Optional[Dict]]:
    print("=" * 60)
    print("Experiment 1: Visual Pipeline")
    print("=" * 60)

    parser    = TaskParser()
    detector  = GroundingDINODetector()
    processor = ImageProcessor(output_dir=PROJECT_ROOT / "images/objectlist")
    segmenter = SAM3Segmenter()

    pipeline_results: List[Optional[Dict]] = []
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}]")
        result = run_single_case(
            instruction      = case["instruction"],
            image_path       = case["image_path"],
            parser           = parser,
            detector         = detector,
            processor        = processor,
            segmenter        = segmenter,
            detection_prompt = case.get("detection_prompt"),
        )
        pipeline_results.append(result)

    # summary
    passed = sum(1 for r in pipeline_results if r is not None)
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(pipeline_results)} passed")
    for case, result in zip(TEST_CASES, pipeline_results):
        if result is not None:
            kp_str   = "  ".join(f"{k}=({v[0]:.0f},{v[1]:.0f})"
                                  for k, v in result["keypoints_2d"].items())
            total_t  = sum(result["timings"].values())
            print(f"  ok    {case['instruction']!r}  [{kp_str}]  {total_t:.2f}s")
        else:
            print(f"  FAIL  {case['instruction']!r}")
    print("=" * 60)

    # save for Experiment 2
    records = []
    for r in pipeline_results:
        if r is None:
            continue
        records.append({
            "instruction":    r["instruction"],
            "mode":           r["mode"],
            "keypoints_2d":   {k: list(v) for k, v in r["keypoints_2d"].items()},
            "annotated_path": str(r["annotated_path"]),
            "timings":        r["timings"],
        })
    out_json = OUTPUT_DIR / "pipeline_results.json"
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved: {out_json}")

    return pipeline_results


if __name__ == "__main__":
    run_all_cases()