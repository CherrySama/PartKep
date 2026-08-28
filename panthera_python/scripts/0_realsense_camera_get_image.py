import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pyrealsense2 as rs


SCRIPT_DIR = Path(__file__).resolve().parent


def _intrinsics_dict(intrinsics):
    return {
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.ppx),
        "cy": float(intrinsics.ppy),
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
    }


def save_fixture(
    out_dir: Path,
    color_frame,
    depth_frame,
    color_intr,
    depth_intr,
    depth_scale: float,
    stats: dict | None = None,
) -> None:
    """Save the currently displayed aligned RGB/depth pair."""
    out_dir.mkdir(parents=True, exist_ok=True)
    color = np.asanyarray(color_frame.get_data()).copy()
    depth = np.asanyarray(depth_frame.get_data()).copy()

    cv2.imwrite(str(out_dir / "color.png"), color)
    np.save(out_dir / "depth_raw.npy", depth)
    info = {
        "color_intrinsics": _intrinsics_dict(color_intr),
        "depth_intrinsics": _intrinsics_dict(depth_intr),
        "depth_scale_m_per_unit": float(depth_scale),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "color_frame_number": int(color_frame.get_frame_number()),
        "depth_frame_number": int(depth_frame.get_frame_number()),
        "color_device_timestamp_ms": float(color_frame.get_timestamp()),
        "depth_device_timestamp_ms": float(depth_frame.get_timestamp()),
        "depth_alignment": "depth_to_color",
    }
    if stats is not None:
        info["transport_stats"] = stats
    with (out_dir / "camera_info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    print(f"Saved fixture: {out_dir}")
    print(json.dumps(info, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show a live aligned D435 stream and save a frame on request."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "realsense_fixture",
        help="fixture directory written by S/Space (default: SDK scripts/realsense_fixture)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()
    depth_intr = depth_stream.get_intrinsics()
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    align = rs.align(rs.stream.color)

    print("D435 live preview started.")
    print("S/Space: save current aligned RGB/depth fixture")
    print("Q/Esc: quit")

    try:
        for _ in range(max(0, args.warmup)):
            align.process(pipeline.wait_for_frames())

        window_name = "D435 live preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        total_frames = 0
        valid_depth_frames = 0
        dropped_color_frames = 0
        dropped_depth_frames = 0
        timestamp_delta_max_ms = 0.0
        first_color_number = None
        first_depth_number = None
        last_color_number = None
        last_depth_number = None
        start_time = time.perf_counter()
        last_report_time = start_time
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data())
            total_frames += 1

            color_number = int(color_frame.get_frame_number())
            depth_number = int(depth_frame.get_frame_number())
            if first_color_number is None:
                first_color_number = color_number
                first_depth_number = depth_number
            if last_color_number is not None and color_number != last_color_number + 1:
                dropped_color_frames += max(1, color_number - last_color_number - 1)
            if last_depth_number is not None and depth_number != last_depth_number + 1:
                dropped_depth_frames += max(1, depth_number - last_depth_number - 1)
            last_color_number = color_number
            last_depth_number = depth_number

            valid_mask = depth > 0
            valid_pixels = int(np.count_nonzero(valid_mask))
            valid_ratio = valid_pixels / depth.size if depth.size else 0.0
            if valid_pixels:
                valid_depth_frames += 1
                depth_values_m = depth[valid_mask].astype(np.float32) * depth_scale
                depth_min_m = float(depth_values_m.min())
                depth_max_m = float(depth_values_m.max())
                depth_mean_m = float(depth_values_m.mean())
            else:
                depth_min_m = depth_max_m = depth_mean_m = 0.0

            timestamp_delta_ms = abs(
                float(color_frame.get_timestamp()) - float(depth_frame.get_timestamp())
            )
            timestamp_delta_max_ms = max(timestamp_delta_max_ms, timestamp_delta_ms)

            depth_display = cv2.convertScaleAbs(
                depth, alpha=255.0 / max(1.0, 4.0 / depth_scale)
            )
            depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            if depth_display.shape[:2] != color.shape[:2]:
                depth_display = cv2.resize(
                    depth_display, (color.shape[1], color.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            combined = np.hstack((color, depth_display))
            preview = color.copy()
            center_y, center_x = depth.shape[0] // 2, depth.shape[1] // 2
            center_raw = int(depth[center_y, center_x])
            center_depth_m = center_raw * depth_scale
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            actual_fps = total_frames / elapsed
            cv2.putText(
                combined,
                f"center depth: {center_depth_m:.3f} m | valid: {valid_ratio * 100:.1f}%",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                combined,
                f"FPS: {actual_fps:.1f} | ts delta max: {timestamp_delta_max_ms:.2f} ms",
                (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                combined,
                "S/Space: save  |  Q/Esc: quit",
                (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S"), 32):
                stats = {
                    "frames_seen": total_frames,
                    "actual_fps": round(actual_fps, 3),
                    "color_frame_gaps": dropped_color_frames,
                    "depth_frame_gaps": dropped_depth_frames,
                    "depth_valid_ratio_current": round(valid_ratio, 6),
                    "depth_valid_frame_ratio": round(valid_depth_frames / total_frames, 6),
                    "center_depth_m": round(center_depth_m, 6),
                    "depth_min_m_current": round(depth_min_m, 6),
                    "depth_max_m_current": round(depth_max_m, 6),
                    "depth_mean_m_current": round(depth_mean_m, 6),
                    "timestamp_delta_max_ms": round(timestamp_delta_max_ms, 6),
                }
                save_fixture(
                    args.output_dir,
                    color_frame,
                    depth_frame,
                    color_intr,
                    depth_intr,
                    depth_scale,
                    stats,
                )
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        if 'total_frames' in locals() and total_frames:
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            print(
                "传输统计："
                f"frames={total_frames}, FPS={total_frames / elapsed:.2f}, "
                f"color_gaps={dropped_color_frames}, depth_gaps={dropped_depth_frames}, "
                f"max_timestamp_delta={timestamp_delta_max_ms:.2f} ms"
            )
        cv2.destroyAllWindows()
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
