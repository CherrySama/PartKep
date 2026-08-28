#!/usr/bin/env python3
"""Run the offline D435 bottle vision chain.

This command only consumes a saved RGB/depth fixture and calibration JSON. It
does not initialize a robot, call Panthera motion APIs, or execute a grasp.
The bundled SDK capture writes a RealSense ``bgr8`` frame with OpenCV, so the
loader converts it to RGB before calling GroundingDINO and SAM3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.camera_config import CameraConfig
from utils import CoordinateTransformer


SAM_MASK_THRESHOLD = 0.50


PANTHERA_TOOL_LINK_BRANCHES = (
    {
        "name": "tool_y_same_as_grasp_y",
        "axis_mapping": {
            # Panthera tool_link: +X points from link6 toward the fingertips,
            # +Y is the gripper opening axis, and +Z completes the right hand.
            "grasp_+x": "tool_link_-z",
            "grasp_+y": "tool_link_+y",
            "grasp_+z": "tool_link_+x",
        },
        # T_tool_link_grasp maps grasp-frame coordinates into tool_link.
        # Consequently T_base_tool_link = T_base_grasp @ inv(T_tool_link_grasp).
        "R_tool_link_grasp": np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]),
    },
    {
        "name": "tool_y_opposite_grasp_y",
        "axis_mapping": {
            "grasp_+x": "tool_link_+z",
            "grasp_+y": "tool_link_-y",
            "grasp_+z": "tool_link_+x",
        },
        "R_tool_link_grasp": np.array([
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
    },
)


def build_panthera_tool_link_targets(T_base_grasp: np.ndarray) -> list[Dict[str, Any]]:
    """Build the two symmetric-gripper tool_link orientations for one grasp."""
    targets = []
    for branch in PANTHERA_TOOL_LINK_BRANCHES:
        T_tool_link_grasp = np.eye(4, dtype=np.float64)
        T_tool_link_grasp[:3, :3] = branch["R_tool_link_grasp"]
        T_base_tool_link = T_base_grasp @ np.linalg.inv(T_tool_link_grasp)
        transform_closed = np.allclose(
            T_base_tool_link @ T_tool_link_grasp,
            T_base_grasp,
            atol=1e-10,
        )
        rotations_right_handed = (
            np.isclose(np.linalg.det(T_base_grasp[:3, :3]), 1.0, atol=1e-8)
            and np.isclose(np.linalg.det(T_base_tool_link[:3, :3]), 1.0, atol=1e-8)
        )
        origins_coincident = np.allclose(
            T_base_tool_link[:3, 3],
            T_base_grasp[:3, 3],
            atol=1e-12,
        )
        if not (transform_closed and rotations_right_handed and origins_coincident):
            raise RuntimeError("Panthera grasp/tool_link adapter validation failed")
        targets.append({
            "orientation_branch": branch["name"],
            "axis_mapping": branch["axis_mapping"],
            "T_tool_link_grasp": T_tool_link_grasp.tolist(),
            "T_base_tool_link_target": T_base_tool_link.tolist(),
            "validation": {
                "transform_closed": bool(transform_closed),
                "rotations_right_handed": bool(rotations_right_handed),
                "origins_coincident": bool(origins_coincident),
            },
        })
    return targets


def load_sdk_fixture(fixture_dir: Path) -> Tuple[Image.Image, np.ndarray, Dict[str, Any]]:
    """Load the fixture produced by the Panthera SDK RealSense script."""
    fixture_dir = Path(fixture_dir)
    color_path = fixture_dir / "color.png"
    depth_path = fixture_dir / "depth_raw.npy"
    info_path = fixture_dir / "camera_info.json"
    for path in (color_path, depth_path, info_path):
        if not path.is_file():
            raise FileNotFoundError(f"fixture file not found: {path}")

    bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to decode color image: {color_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb, mode="RGB")
    depth_raw = np.load(depth_path, allow_pickle=False)
    if depth_raw.ndim != 2 or depth_raw.dtype.kind != "u":
        raise ValueError(
            f"depth must be a 2-D unsigned integer array, got {depth_raw.shape} {depth_raw.dtype}"
        )
    with info_path.open("r", encoding="utf-8") as handle:
        camera_info = json.load(handle)
    return image, depth_raw, camera_info


def build_camera_config(camera_info: Dict[str, Any], calibration: Dict[str, Any]) -> CameraConfig:
    """Build ``CameraConfig`` directly from color intrinsics and T_base_camera."""
    convention = calibration.get("transform_convention")
    if convention != "p_base = T_base_camera @ p_camera":
        raise ValueError(f"unsupported calibration convention: {convention!r}")
    if calibration.get("calibration_type") != "eye_to_hand":
        raise ValueError("expected eye_to_hand calibration")
    intrinsics = camera_info["color_intrinsics"]
    return CameraConfig(
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
        extrinsic_matrix=np.asarray(calibration["T_base_camera"], dtype=np.float64),
    )


def convert_keypoints_to_base(
    keypoints_2d: Dict[str, Tuple[float, float]],
    depth_raw: np.ndarray,
    depth_scale_m_per_unit: float,
    camera_config: CameraConfig,
) -> Dict[str, Dict[str, Any]]:
    """Convert color pixels and aligned raw depth to camera/base coordinates."""
    depth_scale = float(depth_scale_m_per_unit)
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError(f"invalid depth scale: {depth_scale}")

    converted: Dict[str, Dict[str, Any]] = {}
    for part_name, (x_pixel, y_pixel) in keypoints_2d.items():
        depth_raw_value = CoordinateTransformer.get_depth_bilinear(
            depth_raw, float(x_pixel), float(y_pixel)
        )
        depth_m = depth_raw_value * depth_scale
        if depth_m <= 0:
            raise ValueError(
                f"invalid depth for {part_name} at ({x_pixel:.2f}, {y_pixel:.2f}): "
                f"{depth_raw_value:.3f} raw units"
            )
        point_camera = CoordinateTransformer.pixel_to_camera_3d(
            float(x_pixel), float(y_pixel), depth_m, camera_config
        )
        point_base = CoordinateTransformer.camera_to_world_3d(
            point_camera, camera_config
        )
        converted[part_name] = {
            "pixel": [float(x_pixel), float(y_pixel)],
            "depth_raw_bilinear": float(depth_raw_value),
            "depth_m": float(depth_m),
            "p_camera_m": point_camera.tolist(),
            "p_base_m": point_base.tolist(),
        }
    return converted


def solve_partkep_grasp(
    keypoints_3d: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the existing PartKep SAP/PoseSolver pick stage without robot IK."""
    from modules.constraintsInst import ConstraintInstantiator, FINGER_LENGTH
    from modules.poseSolver import PoseSolver
    from modules.vlmDecider import VLMDecision

    points_base = {
        part_name: np.asarray(values["p_base_m"], dtype=np.float64)
        for part_name, values in keypoints_3d.items()
    }
    decision = VLMDecision(
        w_grasp_axis=1.0,
        w_safety=2.0,
        confidence=0.0,
        reasoning="Rule-based fallback (pick mode): conservative SAP defaults.",
        is_fallback=True,
    )

    # The current pick branch accepts T_current for interface compatibility but
    # does not use it. Avoid loading either the Panda model or Panthera FK here.
    T_current_placeholder = np.eye(4, dtype=np.float64)
    instantiator = ConstraintInstantiator(object_class="bottle", verbose=True)
    cost_fn, x0, meta = instantiator.instantiate(
        points_base,
        decision,
        T_current_placeholder,
    )
    solved = PoseSolver(max_iter=200, tol=1e-6, verbose=True).solve(
        cost_fn,
        x0,
        meta,
    )

    grasp_axis = meta.get("grasp_axis_target")
    T_base_grasp = np.eye(4, dtype=np.float64)
    T_base_grasp[:3, :3] = solved["rotation_matrix"]
    T_base_grasp[:3, 3] = meta["keypoint_3d"]

    selected_tool_targets = build_panthera_tool_link_targets(T_base_grasp)
    primary_tool_target = selected_tool_targets[0]

    ranked_candidates = []
    candidate_solutions = sorted(
        meta["candidate_solutions"],
        key=lambda candidate: candidate["partkep_cost"],
    )
    for rank, candidate in enumerate(candidate_solutions, start=1):
        candidate_T_base_grasp = np.eye(4, dtype=np.float64)
        candidate_T_base_grasp[:3, :3] = CoordinateTransformer.rodrigues(
            candidate["x_solution"][3:]
        )
        candidate_T_base_grasp[:3, 3] = candidate["keypoint_3d"]
        ranked_candidates.append({
            "rank": rank,
            "part_name": candidate["part_name"],
            "partkep_cost": float(candidate["partkep_cost"]),
            "keypoint_p_base_m": candidate["keypoint_3d"].tolist(),
            "approach_direction_base": candidate["approach_dir"].tolist(),
            "grasp_axis_target_base": candidate["grasp_axis"].tolist(),
            "body_strategy": candidate["body_strategy"],
            "T_base_grasp": candidate_T_base_grasp.tolist(),
            "tool_link_targets": build_panthera_tool_link_targets(
                candidate_T_base_grasp
            ),
        })

    return {
        "success": bool(solved["success"]),
        "message": str(solved["message"]),
        "frame": "Panthera base",
        "grasp_target": meta["grasp_target"],
        "target_keypoint_p_base_m": meta["keypoint_3d"].tolist(),
        "approach_direction_base": meta["approach_direction"].tolist(),
        "grasp_axis_target_base": (
            grasp_axis.tolist() if grasp_axis is not None else None
        ),
        "body_strategy": meta.get("body_strategy"),
        "candidate_best_cost": float(meta["candidate_best_cost"]),
        "ranked_candidates": ranked_candidates,
        "vlm_decision": {
            "w_grasp_axis": float(decision.w_grasp_axis),
            "w_safety": float(decision.w_safety),
            "confidence": float(decision.confidence),
            "reasoning": decision.reasoning,
            "is_fallback": bool(decision.is_fallback),
        },
        "pose_solver": {
            "position_base_m": solved["position"].tolist(),
            "rotation_matrix": solved["rotation_matrix"].tolist(),
            "rvec_rad": solved["rvec"].tolist(),
            "T_pick_legacy_solver": solved["T"].tolist(),
            "final_cost": float(solved["final_cost"]),
            "cost_breakdown": solved["cost_breakdown"],
            "iterations": int(solved["n_iter"]),
        },
        "semantic_grasp": {
            "frame": "Panthera base",
            "origin_definition": "selected semantic part keypoint",
            "axis_convention": {
                "+x": "+Y_grasp cross +Z_grasp",
                "+y": "gripper opening direction",
                "+z": "approach direction",
            },
            "T_base_grasp": T_base_grasp.tolist(),
        },
        "panthera_tool_link_adapter": {
            "origin_convention": "tool_link origin is treated as grasp center",
            "translation_tool_link_to_grasp_m": [0.0, 0.0, 0.0],
            "tool_link_axis_convention": {
                "+x": "link6 toward the fingertips / tool longitudinal axis",
                "+y": "gripper opening direction",
                "+z": "+x cross +y (right-handed)",
            },
            "axis_mapping": {
                "grasp_+x": "tool_link_-z",
                "grasp_+y": "tool_link_+y",
                "grasp_+z": "tool_link_+x",
            },
            "orientation_branch": primary_tool_target["orientation_branch"],
            "T_tool_link_grasp": primary_tool_target["T_tool_link_grasp"],
            "T_base_tool_link_target": primary_tool_target[
                "T_base_tool_link_target"
            ],
            "equivalent_orientation_branches": selected_tool_targets,
            "reported_physical_finger_length_m": 0.09,
            "finger_length_used_as_pose_offset": False,
            "validation": primary_tool_target["validation"],
            "ik_checked": False,
            "direct_robot_execution_allowed": False,
        },
        "legacy_semantics": {
            "finger_length_m": float(FINGER_LENGTH),
            "description": (
                "T_pick_legacy_solver retains the Panda hand-origin/fingertip "
                "offset and is not directly executable by Panthera."
            ),
            "T_current_source": "identity placeholder; unused by the pick branch",
            "direct_panthera_execution_allowed": False,
        },
    }


def run_pipeline(
    fixture_dir: Path,
    calibration_path: Path,
    output_dir: Path,
    device: str = "auto",
    detection_prompt: str = "plastic bottle",
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    sam_threshold: float = 0.25,
    require_all_parts: bool = False,
    solve_grasp: bool = False,
) -> Path:
    """Run detection, crop, part segmentation, annotation and 3D conversion."""
    import torch

    from modules.groundingdino import GroundingDINODetector
    from modules.imageprocessor import ImageProcessor
    from modules.sam3segmenter import SAM3Segmenter
    from modules.vlmDecider import build_annotated_image

    fixture_dir = Path(fixture_dir)
    calibration_path = Path(calibration_path)
    output_dir = Path(output_dir)
    sam_threshold = float(sam_threshold)
    if not np.isfinite(sam_threshold) or not 0.0 <= sam_threshold <= 1.0:
        raise ValueError(
            f"sam_threshold must be finite and within [0, 1], got {sam_threshold}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    image, depth_raw, camera_info = load_sdk_fixture(fixture_dir)
    with calibration_path.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    camera_config = build_camera_config(camera_info, calibration)
    expected_size = (
        camera_info["color_intrinsics"]["width"],
        camera_info["color_intrinsics"]["height"],
    )
    if image.size != expected_size or depth_raw.shape[::-1] != expected_size:
        raise ValueError(
            f"fixture resolution mismatch: image={image.size}, depth={depth_raw.shape[::-1]}, "
            f"metadata={expected_size}"
        )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be auto, cpu or cuda, got {device!r}")
    print(f"Input fixture: {fixture_dir}")
    print(f"Image: {image.size}, depth: {depth_raw.shape} {depth_raw.dtype}")
    print(f"Device: {device}; depth is aligned to color by the SDK capture script")
    print(
        f"SAM3 thresholds: instance={sam_threshold:.2f}, "
        f"mask={SAM_MASK_THRESHOLD:.2f}"
    )

    detector = GroundingDINODetector(device=device)
    detections = detector.detect(
        image,
        text_prompt=detection_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if not detections:
        raise RuntimeError(f"GroundingDINO found no object for prompt {detection_prompt!r}")
    print(f"Detections: {detections}")

    processor = ImageProcessor(output_dir=output_dir / "crops")
    crops = processor.crop_objects_batch(
        image=image, detection_results=detections[:1], padding=10, save_image=True
    )
    if not crops:
        raise RuntimeError("object crop failed")
    crop = crops[0]
    print(f"Crop: {crop['bbox_pixel']} size={crop['crop_size']}")

    segmenter = SAM3Segmenter(device=device)
    parts = segmenter.segment_parts(
        cropped_image=crop["cropped_image"],
        label="bottle",
        crop_bbox=crop["bbox_pixel"],
        confidence_threshold=sam_threshold,
        mask_threshold=SAM_MASK_THRESHOLD,
    )
    if not parts:
        raise RuntimeError("SAM3 returned no bottle parts")
    keypoints_2d = {
        part["part_name"]: tuple(float(value) for value in part["keypoint"])
        for part in parts
    }
    expected_parts = {"cap", "neck", "body"}
    missing_parts = sorted(expected_parts.difference(keypoints_2d))
    if missing_parts and require_all_parts:
        raise RuntimeError(f"SAM3 did not return required bottle parts: {sorted(missing_parts)}")
    if missing_parts:
        print(
            "WARNING: SAM3 did not return "
            f"{missing_parts}; preserving available parts for visual debugging."
        )

    for part in parts:
        mask_path = output_dir / "masks" / f"{part['part_name']}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), part["mask"])
    annotated = build_annotated_image(image, keypoints_2d)
    annotated_path = output_dir / "bottle_annotated_rgb.png"
    annotated.save(annotated_path)

    keypoints_3d = convert_keypoints_to_base(
        keypoints_2d=keypoints_2d,
        depth_raw=depth_raw,
        depth_scale_m_per_unit=camera_info["depth_scale_m_per_unit"],
        camera_config=camera_config,
    )
    for part_name, result in keypoints_3d.items():
        print(
            f"{part_name}: pixel={result['pixel']} depth_m={result['depth_m']:.4f} "
            f"camera={result['p_camera_m']} base={result['p_base_m']}"
        )

    partkep_grasp = None
    if solve_grasp:
        print("Running offline PartKep SAP/PoseSolver candidate selection...")
        partkep_grasp = solve_partkep_grasp(keypoints_3d)
        print(
            f"PartKep grasp: target={partkep_grasp['grasp_target']} "
            f"success={partkep_grasp['success']} "
            f"cost={partkep_grasp['pose_solver']['final_cost']:.6f}"
        )
        print(
            "WARNING: T_pick_legacy_solver retains the Panda FINGER_LENGTH "
            "offset and must not be sent to Panthera."
        )

    result = {
        "object": "bottle",
        "fixture_dir": str(fixture_dir),
        "calibration_path": str(calibration_path),
        "color_source": "bgr8 written by OpenCV; converted to RGB before inference",
        "depth_alignment": "depth_to_color",
        "detection_prompt": detection_prompt,
        "sam_instance_threshold": sam_threshold,
        "sam_mask_threshold": SAM_MASK_THRESHOLD,
        "detections": detections,
        "crop_bbox": crop["bbox_pixel"],
        "keypoints_2d": {name: list(point) for name, point in keypoints_2d.items()},
        "keypoints_3d": keypoints_3d,
        "expected_parts": sorted(expected_parts),
        "available_parts": sorted(keypoints_2d),
        "missing_parts": missing_parts,
        "complete_part_segmentation": not missing_parts,
        "annotated_path": str(annotated_path),
        "calibration_validation": calibration.get("validation"),
        "partkep_grasp": partkep_grasp,
    }
    result_path = output_dir / "bottle_pipeline_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"Saved: {result_path}")
    print("No robot motion or grasp execution was performed.")
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--detection-prompt", default="plastic bottle")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument(
        "--sam-threshold",
        type=float,
        default=0.25,
        help="SAM3 instance confidence threshold (mask threshold remains 0.50)",
    )
    parser.add_argument(
        "--require-all-parts",
        action="store_true",
        help="fail when SAM3 does not return cap, neck, and body",
    )
    parser.add_argument(
        "--solve-grasp",
        action="store_true",
        help="run offline SAP/PoseSolver selection; never runs IK or robot motion",
    )
    args = parser.parse_args()
    run_pipeline(
        fixture_dir=args.fixture_dir,
        calibration_path=args.calibration,
        output_dir=args.output_dir,
        device=args.device,
        detection_prompt=args.detection_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        sam_threshold=args.sam_threshold,
        require_all_parts=args.require_all_parts,
        solve_grasp=args.solve_grasp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
