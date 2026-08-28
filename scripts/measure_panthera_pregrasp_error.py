#!/usr/bin/env python3
"""Measure the difference between computed and manually corrected pregrasp.

Normal mode runs the existing D435 + PartKep pipeline.  The Panthera child
stage moves only to the computed pregrasp, waits for one ``a`` key press to
start low-stiffness gravity-assisted manual alignment, and records the
manually corrected pose on one ``s`` key press.  It never performs grasp or
lift motion.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import termios
import time
import tty
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_CALIBRATION = PROJECT_ROOT / "configs" / "eye_to_hand_calibration.json"
DEFAULT_ROBOT_PYTHON = Path("/home/sunteng/.conda/envs/panthera/bin/python")
DEFAULT_SDK_PYTHON = Path(
    "/home/sunteng/Desktop/Panthera-HT/Panthera-HT_SDK/panthera_python"
)
DEFAULT_SDK_SCRIPTS = DEFAULT_SDK_PYTHON / "scripts"
DEFAULT_CONFIG = DEFAULT_SDK_PYTHON / "robot_param" / "Follower.yaml"
DEFAULT_URDF = (
    DEFAULT_SDK_PYTHON
    / "Panthera-HT_description/urdf/Panthera-HT_description_follower.urdf"
)


def default_output_dir() -> Path:
    """Create a unique timestamp-based output path for a new measurement run."""
    root = PROJECT_ROOT / "data"
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}"
    candidate = root / f"pregrasp_error_measurement_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"pregrasp_error_measurement_{stamp}_{suffix}"
        suffix += 1
    return candidate


def save_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


@contextmanager
def raw_terminal() -> Iterator[Optional[int]]:
    """Yield a terminal fd in cbreak mode, restoring it on every exit path."""
    if not sys.stdin.isatty():
        yield None
        return
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def read_key(fd: Optional[int], timeout_s: float) -> Optional[str]:
    if fd is None:
        return None
    ready, _, _ = select.select([fd], [], [], max(0.0, timeout_s))
    if not ready:
        return None
    value = os.read(fd, 1)
    if not value:
        return None
    return value.decode("utf-8", errors="ignore").lower()


def child_environment() -> Dict[str, str]:
    env = os.environ.copy()
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not previous
        else str(PROJECT_ROOT) + os.pathsep + previous
    )
    return env


def clean_sdk_environment() -> Dict[str, str]:
    env = child_environment()
    env.pop("PYTHONPATH", None)
    return env


def run_child(command: List[str], env: Dict[str, str]) -> int:
    print(" ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    return int(completed.returncode)


def run_robot_child(command: List[str], env: Dict[str, str]) -> int:
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env)
    previous_handler = signal.getsignal(signal.SIGINT)

    def parent_handles_interrupt(_signum: int, _frame: Any) -> None:
        print(
            "\n收到 Ctrl+C；交由 Panthera 子进程关闭重力补偿并回零。",
            flush=True,
        )

    signal.signal(signal.SIGINT, parent_handles_interrupt)
    try:
        return int(process.wait())
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def require_fixture(fixture_dir: Path) -> None:
    required = (
        fixture_dir / "color.png",
        fixture_dir / "depth_raw.npy",
        fixture_dir / "camera_info.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("D435 fixture files are missing: " + ", ".join(missing))


def positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the normal and Panthera child-stage argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-stage", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pipeline-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="输出目录；省略时自动使用 data/pregrasp_error_measurement_时间戳",
    )
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--robot-python", type=Path, default=DEFAULT_ROBOT_PYTHON)
    parser.add_argument("--sdk-scripts", type=Path, default=DEFAULT_SDK_SCRIPTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--detection-prompt", default="the yellow bottle")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--sam-threshold", type=float, default=0.25)
    parser.add_argument("--pregrasp-mm", type=float, default=50.0)
    parser.add_argument("--ik-max-evaluations", type=int, default=1000)
    parser.add_argument("--position-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--joint-limit-warning-deg", type=float, default=5.0)
    parser.add_argument("--state-samples", type=int, default=5)
    parser.add_argument("--sample-delay", type=float, default=0.05)
    parser.add_argument("--max-state-speed-rad-s", type=float, default=0.05)
    parser.add_argument("--max-saved-state-delta-deg", type=float, default=5.0)
    parser.add_argument("--max-command-speed-rad-s", type=float, default=0.15)
    parser.add_argument("--pregrasp-duration", type=float, default=16.0)
    parser.add_argument("--position-tolerance-deg", type=float, default=1.7)
    parser.add_argument("--timeout-margin-s", type=float, default=8.0)
    parser.add_argument("--gravity-control-dt", type=float, default=0.01)
    parser.add_argument(
        "--gravity-hold-kp",
        type=float,
        default=5.0,
        help="A 后手动微调模式的机械臂位置刚度（MIT Kp）。",
    )
    parser.add_argument(
        "--gravity-hold-kd",
        type=float,
        default=0.5,
        help="A 后手动微调模式的机械臂速度阻尼（MIT Kd）。",
    )
    parser.add_argument(
        "--gravity-follow-threshold-deg",
        type=float,
        default=1.0,
        help="手动移动超过该关节角阈值后，更新保持姿态（度）。",
    )
    parser.add_argument(
        "--gravity-hold-settle-s",
        type=float,
        default=0.5,
        help="按 A 后固定保持初始姿态、禁止跟随的稳定时间（秒）。",
    )
    parser.set_defaults(
        check_fk_after_motion=False,
        fk_position_tolerance_mm=3.0,
        fk_rotation_tolerance_deg=2.0,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="授权真实机械臂移动到 pregrasp；不执行抓取。",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "position-tolerance-mm",
        "rotation-tolerance-deg",
        "max-state-speed-rad-s",
        "max-command-speed-rad-s",
        "pregrasp-duration",
        "gravity-control-dt",
        "gravity-hold-kp",
        "gravity-hold-kd",
        "gravity-follow-threshold-deg",
        "gravity-hold-settle-s",
    ):
        positive_finite(getattr(args, name.replace("-", "_")), name)
    if args.state_samples < 1:
        raise ValueError("state-samples must be positive")
    if args.ik_max_evaluations < 1:
        raise ValueError("ik-max-evaluations must be positive")
    if args.robot_stage:
        if args.pipeline_result is None or args.output is None:
            raise ValueError("robot stage requires --pipeline-result and --output")
        if not args.pipeline_result.is_file():
            raise FileNotFoundError(f"pipeline result not found: {args.pipeline_result}")
        return
    if args.fixture_dir is None or args.output_dir is None:
        raise ValueError("normal mode requires --fixture-dir and --output-dir")
    for name, path in (
        ("calibration", args.calibration),
        ("robot-python", args.robot_python),
        ("sdk-scripts", args.sdk_scripts),
        ("config", args.config),
        ("urdf", args.urdf),
    ):
        if not path.expanduser().exists():
            raise FileNotFoundError(f"{name} path not found: {path}")
    for name, value in (
        ("sam-threshold", args.sam_threshold),
        ("box-threshold", args.box_threshold),
        ("text-threshold", args.text_threshold),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")


def read_current_state(robot: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Refresh and read the six arm joints once."""
    from run_panthera_bottle_fast import finite_vector

    for _ in range(4):
        robot.send_get_motor_state_cmd()
        robot.motor_send_cmd()
    q = finite_vector(robot.get_current_pos(), "Panthera current q", 6)
    dq = finite_vector(robot.get_current_vel(), "Panthera current dq", 6)
    return q, dq


def gravity_teach_until_save(
    robot: Any,
    gravity_control_dt: float,
    fd: int,
    hold_kp: float,
    hold_kd: float,
    follow_threshold_deg: float,
    hold_settle_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hold the A-pose softly while allowing small manual corrections.

    Pure gravity compensation uses zero position gains and therefore cannot
    preserve the pregrasp pose.  Here the pose read immediately after ``a``
    is used as a low-stiffness MIT target.  Once the operator deliberately
    moves farther than the follow threshold, the target follows that new pose
    so the operator can make a small correction without fighting a spring back
    to the original computed pregrasp.
    """
    from execute_panthera_pregrasp import (
        GRIPPER_MIT_KD,
        GRIPPER_MIT_KP,
        GRIPPER_MIT_TORQUE,
        GRIPPER_MIT_VELOCITY,
        GRIPPER_OPEN_POS,
    )

    zero = np.zeros(6, dtype=np.float64)
    hold_kp_vector = np.full(6, hold_kp, dtype=np.float64)
    hold_kd_vector = np.full(6, hold_kd, dtype=np.float64)
    max_torque = np.asarray(robot.max_torque, dtype=np.float64)
    if max_torque.shape != (6,) or not np.all(np.isfinite(max_torque)):
        raise RuntimeError("Panthera maximum torque limits are unavailable")
    q_hold, _ = read_current_state(robot)
    initial_hold = q_hold.copy()
    follow_threshold_rad = np.deg2rad(follow_threshold_deg)
    follow_enabled_at = time.perf_counter() + hold_settle_s
    next_tick = time.perf_counter()
    print(
        "低刚度重力补偿已启动；当前姿态已锁定，可手动微调。"
        "按一次 S 保存，Q/Esc 取消。",
        flush=True,
    )
    while True:
        key = read_key(fd, 0.0)
        if key in ("q", "\x1b"):
            raise KeyboardInterrupt("operator cancelled manual pregrasp alignment")

        q, dq = read_current_state(robot)
        if key == "s":
            return q, dq

        if (
            time.perf_counter() >= follow_enabled_at
            and float(np.max(np.abs(q - q_hold))) > follow_threshold_rad
        ):
            q_hold = q.copy()
            print(
                "手动微调已接管保持目标："
                f"最大偏移={np.rad2deg(np.max(np.abs(q_hold - initial_hold))):.2f} deg",
                flush=True,
            )

        gravity_torque = np.asarray(robot.get_Gravity(q), dtype=np.float64)
        if gravity_torque.shape != (6,) or not np.all(np.isfinite(gravity_torque)):
            raise RuntimeError("Panthera gravity compensation returned an invalid torque")
        gravity_torque = np.clip(gravity_torque, -max_torque, max_torque)
        if not robot.pos_vel_tqe_kp_kd(
            q_hold,
            zero,
            gravity_torque,
            hold_kp_vector,
            hold_kd_vector,
        ):
            raise RuntimeError("gravity compensation command was rejected")
        # Keep the already-open gripper stationary without changing the arm pose.
        robot.gripper_control_MIT(
            GRIPPER_OPEN_POS,
            GRIPPER_MIT_VELOCITY,
            GRIPPER_MIT_TORQUE,
            GRIPPER_MIT_KP,
            GRIPPER_MIT_KD,
        )
        next_tick += gravity_control_dt
        remaining = next_tick - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        else:
            next_tick = time.perf_counter()


def pose_check_transform(selected_path: Dict[str, Any], label: str) -> np.ndarray:
    from run_panthera_bottle_fast import validate_transform

    checks = selected_path.get("pose_checks")
    if not isinstance(checks, list):
        raise ValueError("selected path has no pose checks")
    for item in checks:
        if isinstance(item, dict) and item.get("label") == label:
            return validate_transform(
                item.get("T_base_tool_link_target"),
                f"{label} target transform",
            )
    raise ValueError(f"selected path has no {label} target transform")


def build_measurement_record(
    plan: Dict[str, Any],
    pregrasp_motion: Dict[str, Any],
    q_manual: np.ndarray,
    dq_manual: np.ndarray,
    model: Any,
    frame_id: int,
) -> Dict[str, Any]:
    from run_panthera_bottle_fast import forward_kinematics, pose_errors

    selected_path = plan["selected_path"]
    T_computed = pose_check_transform(selected_path, "pregrasp")
    q_manual = np.asarray(q_manual, dtype=np.float64)
    dq_manual = np.asarray(dq_manual, dtype=np.float64)
    T_fk_manual = forward_kinematics(model, frame_id, q_manual)
    position_error_m, rotation_error_rad = pose_errors(T_fk_manual, T_computed)
    delta_base = T_fk_manual @ np.linalg.inv(T_computed)
    delta_tool = np.linalg.inv(T_computed) @ T_fk_manual
    return {
        "schema_version": "panthera-pregrasp-error-measurement-v1",
        "selected_target": selected_path["target"],
        "computed_pregrasp": {
            "q_commanded_rad": selected_path["q_pregrasp_rad"],
            "T_base_tool_link_target": T_computed.tolist(),
            "q_actual_after_move_rad": pregrasp_motion["q_post_rad"],
            "T_base_tool_link_fk_after_move": forward_kinematics(
                model,
                frame_id,
                np.asarray(pregrasp_motion["q_post_rad"], dtype=np.float64),
            ).tolist(),
        },
        "manual_pregrasp": {
            "q_rad": q_manual.tolist(),
            "dq_rad_s": dq_manual.tolist(),
            "T_base_tool_link_fk": T_fk_manual.tolist(),
        },
        "error_manual_minus_computed": {
            "position_vector_base_mm": (
                (T_fk_manual[:3, 3] - T_computed[:3, 3]) * 1000.0
            ).tolist(),
            "position_norm_mm": float(position_error_m * 1000.0),
            "rotation_error_deg": float(np.rad2deg(rotation_error_rad)),
            "T_delta_base_left": delta_base.tolist(),
            "T_delta_computed_tool_right": delta_tool.tolist(),
        },
        "interpretation": (
            "This is a manually defined pregrasp correction measurement. "
            "It is not an absolute ground-truth measurement unless the manual "
            "alignment rule is externally defined and repeatable."
        ),
    }


def build_pregrasp_only_plan(args: argparse.Namespace, robot: Any) -> Dict[str, Any]:
    """Solve only the measured pregrasp pose for every PartKep candidate."""
    import pinocchio as pin
    from run_panthera_bottle_fast import (
        check_model,
        finite_vector,
        load_robot_config,
        load_targets,
        read_state_samples,
        solve_ik_from_seed,
    )

    targets = load_targets(args.pipeline_result)
    joint_names, lower, upper, velocity_limits = load_robot_config(args.config)
    model = pin.buildModelFromUrdf(str(args.urdf.resolve()))
    frame_id = check_model(model, joint_names)

    print("Reading Panthera state for pregrasp-only IK; no motion API is called...")
    state_samples = read_state_samples(
        robot,
        sample_count=args.state_samples,
        sample_delay_s=args.sample_delay,
    )
    q_samples = np.asarray([sample["q_rad"] for sample in state_samples], dtype=np.float64)
    dq_samples = np.asarray([sample["dq_rad_s"] for sample in state_samples], dtype=np.float64)
    q_current = q_samples[-1]
    maximum_state_speed = float(np.max(np.abs(dq_samples)))
    if maximum_state_speed > args.max_state_speed_rad_s:
        raise RuntimeError(
            f"Panthera is not stationary: measured speed={maximum_state_speed:.6f} rad/s"
        )
    if np.any(q_current < lower) or np.any(q_current > upper):
        raise RuntimeError("fresh Panthera state violates Follower joint limits")

    position_tolerance_m = positive_finite(
        args.position_tolerance_mm, "position-tolerance-mm"
    ) / 1000.0
    rotation_tolerance_rad = np.deg2rad(
        positive_finite(args.rotation_tolerance_deg, "rotation-tolerance-deg")
    )
    joint_limit_warning_rad = np.deg2rad(
        positive_finite(args.joint_limit_warning_deg, "joint-limit-warning-deg")
    )

    paths: List[Dict[str, Any]] = []
    for index, candidate in enumerate(targets, start=1):
        pregrasp = candidate.transform.copy()
        pregrasp[:3, 3] -= (
            candidate.approach_direction_base * (float(args.pregrasp_mm) / 1000.0)
        )
        try:
            solved = solve_ik_from_seed(
                model=model,
                frame_id=frame_id,
                target=pregrasp,
                q_seed=q_current,
                lower=lower,
                upper=upper,
                max_evaluations=args.ik_max_evaluations,
            )
            margin_deg = np.rad2deg(solved.joint_margin_rad)
            reachable = bool(
                solved.optimizer_success
                and solved.position_error_m <= position_tolerance_m
                and solved.rotation_error_rad <= rotation_tolerance_rad
                and np.all(solved.q >= lower)
                and np.all(solved.q <= upper)
            )
            margin_safe = bool(reachable and np.min(solved.joint_margin_rad) >= joint_limit_warning_rad)
            pose_check = {
                "label": "pregrasp",
                "q_rad": solved.q.tolist(),
                "T_base_tool_link_target": pregrasp.tolist(),
                "position_error_mm": float(solved.position_error_m * 1000.0),
                "rotation_error_deg": float(np.rad2deg(solved.rotation_error_rad)),
                "optimizer_success": bool(solved.optimizer_success),
                "evaluations": int(solved.evaluations),
                "joint_limit_margin_deg": margin_deg.tolist(),
                "minimum_joint_limit_margin_deg": float(np.min(margin_deg)),
                "reachable": reachable,
                "margin_safe": margin_safe,
            }
            path = {
                "target": {
                    "rank": candidate.rank,
                    "part_name": candidate.part_name,
                    "partkep_cost": candidate.partkep_cost,
                    "orientation_branch": candidate.orientation_branch,
                    "approach_direction_base": candidate.approach_direction_base.tolist(),
                    "T_base_tool_link_grasp": candidate.transform.tolist(),
                },
                "path_type": "pregrasp_only",
                "path_reachable": reachable,
                "margin_safe": margin_safe,
                "minimum_joint_limit_margin_deg": float(np.min(margin_deg)),
                "max_current_to_pregrasp_joint_travel_deg": float(
                    np.max(np.abs(np.rad2deg(solved.q - q_current)))
                ),
                "pose_checks": [pose_check],
                "q_current_rad": q_current.tolist(),
                "q_pregrasp_rad": solved.q.tolist(),
            }
            print(
                f"  branch {index}/{len(targets)}: {candidate.part_name}/"
                f"{candidate.orientation_branch}; pregrasp only -> "
                f"reachable={reachable} margin_safe={margin_safe} "
                f"position={pose_check['position_error_mm']:.3f} mm "
                f"rotation={pose_check['rotation_error_deg']:.3f} deg",
                flush=True,
            )
        except Exception as exc:
            path = {
                "target": {
                    "rank": candidate.rank,
                    "part_name": candidate.part_name,
                    "partkep_cost": candidate.partkep_cost,
                    "orientation_branch": candidate.orientation_branch,
                    "approach_direction_base": candidate.approach_direction_base.tolist(),
                    "T_base_tool_link_grasp": candidate.transform.tolist(),
                },
                "path_type": "pregrasp_only",
                "path_reachable": False,
                "margin_safe": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"  branch {index}/{len(targets)}: {candidate.part_name}/"
                f"{candidate.orientation_branch}; pregrasp solve failed: {exc}",
                flush=True,
            )
        paths.append(path)

    margin_safe = [path for path in paths if path.get("margin_safe")]
    reachable = [path for path in paths if path.get("path_reachable")]
    if not margin_safe:
        details = []
        for path in paths:
            target = path["target"]
            if "error" in path:
                details.append(
                    f"{target['part_name']}/{target['orientation_branch']}: {path['error']}"
                )
            else:
                details.append(
                    f"{target['part_name']}/{target['orientation_branch']}: "
                    f"position={path['pose_checks'][0]['position_error_mm']:.3f} mm, "
                    f"rotation={path['pose_checks'][0]['rotation_error_deg']:.3f} deg"
                )
        if reachable:
            reason = "pregrasp branches are reachable but all violate the joint-limit safety margin"
        else:
            reason = "no pregrasp-only branch is reachable from current Panthera state"
        raise RuntimeError(reason + "; " + "; ".join(details))
    selected = min(
        margin_safe,
        key=lambda path: (
            path["target"]["rank"],
            path["target"]["partkep_cost"],
            path["max_current_to_pregrasp_joint_travel_deg"],
        ),
    )
    print(
        "Pregrasp-only IK selected "
        f"{selected['target']['part_name']}/{selected['target']['orientation_branch']} "
        f"(min margin={selected['minimum_joint_limit_margin_deg']:.2f} deg)",
        flush=True,
    )
    return {
        "schema_version": "panthera-pregrasp-only-plan-v1",
        "check_type": "current-seeded Panthera pregrasp-only IK",
        "hardware_access": True,
        "robot_motion_commanded": False,
        "direct_robot_execution_allowed": bool(args.execute),
        "pipeline_result": str(args.pipeline_result.resolve()),
        "urdf": str(args.urdf.resolve()),
        "config": str(args.config.resolve()),
        "sdk_scripts": str(args.sdk_scripts.resolve()),
        "joint_names": joint_names,
        "joint_limits_rad": {"lower": lower.tolist(), "upper": upper.tolist()},
        "velocity_limits_rad_s": velocity_limits.tolist(),
        "current_state_samples": state_samples,
        "q_current_rad": q_current.tolist(),
        "maximum_state_speed_rad_s": maximum_state_speed,
        "selected_path": selected,
        "candidate_paths": paths,
        "parameters": {
            "pregrasp_mm": float(args.pregrasp_mm),
            "position_tolerance_mm": float(args.position_tolerance_mm),
            "rotation_tolerance_deg": float(args.rotation_tolerance_deg),
            "joint_limit_warning_deg": float(args.joint_limit_warning_deg),
            "ik_max_evaluations": int(args.ik_max_evaluations),
        },
        "limitations": [
            "this diagnostic does not solve or execute grasp/lift",
            "no collision geometry check is available in the Panthera SDK path",
            "the hand-eye absolute error is not corrected by this measurement",
        ],
    }


def execute_measurement(args: argparse.Namespace, robot: Any, plan: Dict[str, Any]) -> int:
    from execute_panthera_pregrasp import (
        return_to_zero,
        load_robot_limits,
    )
    from run_panthera_bottle_fast import (
        check_model,
        duration_for_joint_move,
        finite_vector,
        load_robot_config,
        move_and_verify,
        read_fresh_q,
    )
    import pinocchio as pin

    joint_names, lower, upper, velocity_limits = load_robot_config(args.config)
    model = pin.buildModelFromUrdf(str(args.urdf.resolve()))
    frame_id = check_model(model, joint_names)
    selected = plan["selected_path"]
    q_plan = finite_vector(plan["q_current_rad"], "planned current q", 6)
    q_pregrasp = finite_vector(selected["q_pregrasp_rad"], "q_pregrasp", 6)

    q_actual, speed, fresh_samples = read_fresh_q(
        robot, args.state_samples, args.sample_delay
    )
    saved_delta_deg = float(np.rad2deg(np.max(np.abs(q_actual - q_plan))))
    if speed > args.max_state_speed_rad_s:
        raise RuntimeError(f"Panthera is moving before pregrasp: {speed:.6f} rad/s")
    if saved_delta_deg > args.max_saved_state_delta_deg:
        raise RuntimeError(
            f"current state changed by {saved_delta_deg:.3f} deg since IK; replan required"
        )
    if np.any(q_actual < lower) or np.any(q_actual > upper):
        raise RuntimeError("fresh Panthera state violates Follower joint limits")

    pregrasp_duration = duration_for_joint_move(
        q_actual,
        q_pregrasp,
        args.pregrasp_duration,
        args.max_command_speed_rad_s,
        velocity_limits,
    )
    limits = load_robot_limits(args.config)
    execution: Dict[str, Any] = dict(plan)
    execution.update({
        "schema_version": "panthera-pregrasp-error-measurement-v1",
        "hardware_access": True,
        "robot_motion_commanded": False,
        "status": "ready_for_pregrasp_measurement",
        "no_grasp_or_lift": True,
        "fresh_state_samples_before_motion": fresh_samples,
        "q_fresh_current_rad": q_actual.tolist(),
        "fresh_state_speed_rad_s": speed,
        "saved_state_delta_deg": saved_delta_deg,
        "pregrasp_duration_s": pregrasp_duration,
        "gravity_control_dt_s": args.gravity_control_dt,
        "gravity_control_mode": "gravity_feedforward_with_low_stiffness_hold",
        "gravity_hold_kp": args.gravity_hold_kp,
        "gravity_hold_kd": args.gravity_hold_kd,
        "gravity_follow_threshold_deg": args.gravity_follow_threshold_deg,
        "gravity_hold_settle_s": args.gravity_hold_settle_s,
        "manual_key_protocol": {
            "start_gravity_compensation": "a",
            "save_manual_pregrasp": "s",
            "abort": "q or Esc",
            "keys_are_single_presses": True,
        },
    })
    save_json(args.output, execution)

    print(f"Fresh q_current={np.round(q_actual, 6).tolist()} rad")
    print(
        f"Moving current -> computed pregrasp over {pregrasp_duration:.2f}s. "
        "No grasp or lift will be executed."
    )
    print("No collision checking is provided by the Panthera SDK; keep the E-stop reachable.")

    motion_started = False
    returned_to_zero = False
    try:
        motion_started = True
        execution["robot_motion_commanded"] = True
        execution["status"] = "executing_pregrasp"
        save_json(args.output, execution)
        execution["pregrasp_motion"] = move_and_verify(
            robot,
            q_pregrasp,
            pregrasp_duration,
            args.position_tolerance_deg,
            args.timeout_margin_s,
            args.state_samples,
            args.sample_delay,
        )
        execution["status"] = "computed_pregrasp_reached_waiting_for_a"
        save_json(args.output, execution)
        print(
            "已到达计算出的 pregrasp。按一次 A 开启重力补偿；"
            f"手动对准真实目标后按一次 S 保存。"
            f"[Kp={args.gravity_hold_kp:.2f}, Kd={args.gravity_hold_kd:.2f}, "
            f"锁定={args.gravity_hold_settle_s:.2f}s, "
            f"跟随阈值={args.gravity_follow_threshold_deg:.2f}°]",
            flush=True,
        )

        with raw_terminal() as fd:
            if fd is None:
                raise RuntimeError("manual pregrasp calibration requires an interactive terminal")
            while True:
                key = read_key(fd, 0.1)
                if key == "a":
                    break
                if key in ("q", "\x1b"):
                    raise KeyboardInterrupt("operator cancelled before gravity compensation")

            execution["status"] = "gravity_compensation_manual_alignment"
            execution["gravity_compensation_started_at_unix_s"] = time.time()
            save_json(args.output, execution)
            q_manual, dq_manual = gravity_teach_until_save(
                robot,
                args.gravity_control_dt,
                fd,
                args.gravity_hold_kp,
                args.gravity_hold_kd,
                args.gravity_follow_threshold_deg,
                args.gravity_hold_settle_s,
            )

        measurement = build_measurement_record(
            plan=plan,
            pregrasp_motion=execution["pregrasp_motion"],
            q_manual=q_manual,
            dq_manual=dq_manual,
            model=model,
            frame_id=frame_id,
        )
        execution["manual_measurement"] = measurement
        execution["status"] = "manual_pregrasp_saved_returning_to_zero"
        save_json(args.output, execution)
        error = measurement["error_manual_minus_computed"]
        print(
            "Manual pregrasp saved: "
            f"position_delta_base={np.round(error['position_vector_base_mm'], 3).tolist()} mm, "
            f"norm={error['position_norm_mm']:.3f} mm, "
            f"rotation={error['rotation_error_deg']:.3f} deg",
            flush=True,
        )

        execution["return_to_zero"] = return_to_zero(robot, limits)
        returned_to_zero = True
        execution["status"] = "completed_returned_to_zero"
        save_json(args.output, execution)
        print("Pregrasp measurement completed; no grasp or lift was executed.")
        return 0
    except KeyboardInterrupt as interruption:
        execution["status"] = "operator_abort_returning_to_zero"
        execution["error"] = f"{type(interruption).__name__}: {interruption}"
        save_json(args.output, execution)
        if motion_started and not returned_to_zero:
            try:
                execution["return_to_zero"] = return_to_zero(robot, limits)
                returned_to_zero = True
                execution["status"] = "operator_abort_returned_to_zero"
                save_json(args.output, execution)
                print("Operator abort handled; returned to q=0.")
                return 130
            except KeyboardInterrupt:
                robot.set_stop()
                execution["status"] = "second_interrupt_stopped"
                save_json(args.output, execution)
                raise
            except Exception as return_error:
                robot.set_stop()
                execution["status"] = "operator_abort_return_failed_stopped"
                execution["return_error"] = (
                    f"{type(return_error).__name__}: {return_error}"
                )
                save_json(args.output, execution)
                raise RuntimeError(
                    f"operator abort return to zero failed: {return_error}"
                ) from return_error
        return 130
    except Exception as exc:
        robot.set_stop()
        execution["status"] = "execution_failed_stopped"
        execution["error"] = f"{type(exc).__name__}: {exc}"
        save_json(args.output, execution)
        print("Motor stop was requested; automatic return was not attempted after an execution error.")
        raise


def run_robot_stage(args: argparse.Namespace) -> int:
    from run_panthera_bottle_fast import connect_panthera

    robot = connect_panthera(args.sdk_scripts.resolve(), args.config.resolve())
    try:
        plan = build_pregrasp_only_plan(args, robot)
        save_json(args.output, plan)
        print(f"Saved pregrasp-only plan: {args.output}")
        if not args.execute:
            print("No --execute flag: no motion or gravity compensation was commanded.")
            return 0
        return execute_measurement(args, robot, plan)
    finally:
        if not args.execute:
            del robot


def build_robot_child_command(
    args: argparse.Namespace,
    pipeline_result: Path,
    output: Path,
) -> List[str]:
    command = [
        str(args.robot_python.resolve()),
        str(Path(__file__).resolve()),
        "--robot-stage",
        "--pipeline-result",
        str(pipeline_result.resolve()),
        "--output",
        str(output.resolve()),
        "--urdf",
        str(args.urdf.resolve()),
        "--config",
        str(args.config.resolve()),
        "--sdk-scripts",
        str(args.sdk_scripts.resolve()),
        "--pregrasp-mm",
        str(args.pregrasp_mm),
        "--ik-max-evaluations",
        str(args.ik_max_evaluations),
        "--position-tolerance-mm",
        str(args.position_tolerance_mm),
        "--rotation-tolerance-deg",
        str(args.rotation_tolerance_deg),
        "--joint-limit-warning-deg",
        str(args.joint_limit_warning_deg),
        "--state-samples",
        str(args.state_samples),
        "--sample-delay",
        str(args.sample_delay),
        "--max-state-speed-rad-s",
        str(args.max_state_speed_rad_s),
        "--max-saved-state-delta-deg",
        str(args.max_saved_state_delta_deg),
        "--max-command-speed-rad-s",
        str(args.max_command_speed_rad_s),
        "--pregrasp-duration",
        str(args.pregrasp_duration),
        "--position-tolerance-deg",
        str(args.position_tolerance_deg),
        "--timeout-margin-s",
        str(args.timeout_margin_s),
        "--gravity-control-dt",
        str(args.gravity_control_dt),
        "--gravity-hold-kp",
        str(args.gravity_hold_kp),
        "--gravity-hold-kd",
        str(args.gravity_hold_kd),
        "--gravity-follow-threshold-deg",
        str(args.gravity_follow_threshold_deg),
        "--gravity-hold-settle-s",
        str(args.gravity_hold_settle_s),
    ]
    if args.execute:
        command.append("--execute")
    return command


def run_normal_stage(args: argparse.Namespace) -> int:
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new one: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fixture_dir = args.fixture_dir.resolve()
    vision_dir = args.output_dir / "vision"
    measurement_path = args.output_dir / "pregrasp_error_measurement.json"
    status_path = args.output_dir / "session_status.json"
    status: Dict[str, Any] = {
        "schema_version": "panthera-pregrasp-error-session-v1",
        "status": "running",
        "started_at_unix_s": time.time(),
        "output_dir": str(args.output_dir.resolve()),
        "execute": bool(args.execute),
        "no_grasp_or_lift": True,
    }
    save_json(status_path, status)

    if args.capture:
        capture_command = [
            sys.executable,
            str(SCRIPTS_DIR / "capture_d435_fixture.py"),
            "--output-dir",
            str(fixture_dir),
            "--exit-after-save",
        ]
        print("\n[1/3] D435 synchronized fixture capture", flush=True)
        code = run_child(capture_command, child_environment())
        if code != 0:
            raise RuntimeError(f"D435 capture failed with exit code {code}")
    else:
        print("\n[1/3] Using existing D435 fixture", flush=True)
    require_fixture(fixture_dir)

    print("\n[2/3] Bottle vision and PartKep fixed grasp pose", flush=True)
    vision_command = [
        sys.executable,
        str(SCRIPTS_DIR / "run_d435_bottle_pipeline.py"),
        "--fixture-dir",
        str(fixture_dir),
        "--calibration",
        str(args.calibration.resolve()),
        "--output-dir",
        str(vision_dir),
        "--device",
        args.device,
        "--detection-prompt",
        args.detection_prompt,
        "--box-threshold",
        str(args.box_threshold),
        "--text-threshold",
        str(args.text_threshold),
        "--sam-threshold",
        str(args.sam_threshold),
        "--solve-grasp",
    ]
    code = run_child(vision_command, child_environment())
    if code != 0:
        raise RuntimeError(f"bottle vision failed with exit code {code}")
    pipeline_result = vision_dir / "bottle_pipeline_results.json"
    if not pipeline_result.is_file():
        raise RuntimeError(f"vision output is missing: {pipeline_result}")

    print("\n[3/3] Panthera pregrasp measurement; no grasp or lift", flush=True)
    robot_command = build_robot_child_command(args, pipeline_result, measurement_path)
    code = run_robot_child(robot_command, clean_sdk_environment())
    if code not in (0, 130):
        raise RuntimeError(f"Panthera pregrasp measurement failed with exit code {code}")
    status["status"] = "completed" if code == 0 else "operator_shutdown_complete"
    status["finished_at_unix_s"] = time.time()
    status["robot_stage_return_code"] = code
    save_json(status_path, status)
    print(f"Saved pregrasp measurement session: {args.output_dir}")
    return code


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if not args.robot_stage and args.output_dir is None:
            args.output_dir = default_output_dir()
        validate_args(args)
        if args.robot_stage:
            return run_robot_stage(args)
        return run_normal_stage(args)
    except KeyboardInterrupt:
        print("\nPregrasp measurement interrupted.", file=sys.stderr)
        return 130
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Pregrasp measurement failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
