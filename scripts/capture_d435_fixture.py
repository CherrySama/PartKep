#!/usr/bin/env python3
"""Preview the D435 stream and save one aligned RGB/depth fixture on request.

Press S or Space to overwrite ``results/d435_fixture`` with the currently
displayed synchronized color/depth pair. Press Q or Escape to quit. This script
does not initialize or move the robot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "d435_fixture"


def _intrinsics_dict(intrinsics) -> dict:
    return {
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.ppx),
        "cy": float(intrinsics.ppy),
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
    }


def save_fixture(
    output_dir: Path,
    color_frame,
    depth_frame,
    color_intrinsics,
    depth_intrinsics,
    depth_scale: float,
) -> Path:
    """Save the displayed aligned pair in the bottle-pipeline fixture format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    color_bgr = np.asanyarray(color_frame.get_data()).copy()
    depth_raw = np.asanyarray(depth_frame.get_data()).copy()

    if not cv2.imwrite(str(output_dir / "color.png"), color_bgr):
        raise RuntimeError(f"failed to save color image to {output_dir}")
    np.save(output_dir / "depth_raw.npy", depth_raw, allow_pickle=False)

    camera_info = {
        "color_intrinsics": _intrinsics_dict(color_intrinsics),
        "depth_intrinsics": _intrinsics_dict(depth_intrinsics),
        "depth_scale_m_per_unit": float(depth_scale),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_ns": time.time_ns(),
        "color_frame_number": int(color_frame.get_frame_number()),
        "depth_frame_number": int(depth_frame.get_frame_number()),
        "color_device_timestamp_ms": float(color_frame.get_timestamp()),
        "depth_device_timestamp_ms": float(depth_frame.get_timestamp()),
        "color_format": "bgr8",
        "depth_alignment": "depth_to_color",
    }
    with (output_dir / "camera_info.json").open("w", encoding="utf-8") as handle:
        json.dump(camera_info, handle, indent=2)
        handle.write("\n")
    return output_dir


def preview_and_capture(
    output_dir: Path,
    width: int,
    height: int,
    fps: int,
    warmup: int,
    exit_after_save: bool = False,
) -> None:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is required for D435 capture"
        ) from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    color_intrinsics = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    depth_intrinsics = (
        profile.get_stream(rs.stream.depth)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    align = rs.align(rs.stream.color)
    window_name = "D435 live preview"

    print("D435 live preview started.")
    print(f"S/Space: save current aligned fixture to {output_dir}")
    print("Q/Esc: quit")
    try:
        for _ in range(max(0, warmup)):
            align.process(pipeline.wait_for_frames())
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            preview = np.asanyarray(color_frame.get_data()).copy()
            cv2.putText(
                preview,
                "S/Space: save  |  Q/Esc: quit",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S"), 32):
                saved = save_fixture(
                    output_dir=output_dir,
                    color_frame=color_frame,
                    depth_frame=depth_frame,
                    color_intrinsics=color_intrinsics,
                    depth_intrinsics=depth_intrinsics,
                    depth_scale=depth_scale,
                )
                print(f"Saved D435 fixture: {saved}")
                if exit_after_save:
                    break
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"fixture directory to overwrite on save (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument(
        "--exit-after-save",
        action="store_true",
        help="exit immediately after S/Space saves one fixture",
    )
    args = parser.parse_args()

    try:
        preview_and_capture(
            args.output_dir,
            args.width,
            args.height,
            args.fps,
            args.warmup,
            args.exit_after_save,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print("No robot motion or grasp execution was requested by this command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
