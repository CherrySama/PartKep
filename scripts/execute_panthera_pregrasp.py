#!/usr/bin/env python3
"""Plan or execute the guarded Panthera bottle grasp test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import yaml


DEFAULT_SDK_SCRIPTS = Path(
    "/home/sunteng/Desktop/Panthera-HT/Panthera-HT_SDK/panthera_python/scripts"
)
DEFAULT_CONFIG = Path(
    "/home/sunteng/Desktop/Panthera-HT/Panthera-HT_SDK/"
    "panthera_python/robot_param/Follower.yaml"
)
RETURN_TO_ZERO_DURATION_S = 5.0
RETURN_CONTROL_DT_S = 0.01
RETURN_HOLD_DURATION_S = 0.2
GRASP_APPROACH_DURATION_S = 5.0
GRIPPER_SETTLE_DURATION_S = 1.0
LIFT_DURATION_S = 5.0
GRIPPER_OPEN_POS = 1.6
GRIPPER_CLOSE_POS = 0.0
GRIPPER_MIT_VELOCITY = 0.0
GRIPPER_MIT_TORQUE = 0.0
GRIPPER_MIT_KP = 5.0
GRIPPER_MIT_KD = 0.5
GRIPPER_POSITION_TOLERANCE = 0.05
GRIPPER_OPEN_TIMEOUT_S = 10.0
GRIPPER_CLOSE_TIMEOUT_S = 10.0
GRIPPER_GRASP_CLOSE_DURATION_S = 3.0
GRIPPER_GRASP_HOLD_AFTER_CLOSE_S = 1.0
GRIPPER_FINAL_CLOSE_DURATION_S = 3.0


class OperatorReturnRequested(RuntimeError):
    """The operator stopped staged motion and requested the guarded return."""


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def save_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def finite_six_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite six-vector")
    return vector


def load_robot_limits(config_path: Path) -> Dict[str, np.ndarray]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    robot = config.get("robot", {}) if isinstance(config, dict) else {}
    lower = finite_six_vector(robot.get("joint_limits", {}).get("lower"), "lower limits")
    upper = finite_six_vector(robot.get("joint_limits", {}).get("upper"), "upper limits")
    velocity = finite_six_vector(robot.get("velocity_limits"), "velocity limits")
    if np.any(lower >= upper) or np.any(velocity <= 0.0):
        raise ValueError("Follower joint or velocity limits are invalid")
    return {"lower": lower, "upper": upper, "velocity": velocity}


def build_plan(
    selection_path: Path,
    config_path: Path,
    segment_count: int,
    segment_duration_s: float,
    minimum_margin_deg: float,
    maximum_command_speed_rad_s: float,
) -> Dict[str, Any]:
    if segment_count < 2:
        raise ValueError("segments must be at least 2")
    segment_duration_s = positive_finite(segment_duration_s, "segment-duration")
    minimum_margin_rad = np.deg2rad(positive_finite(
        minimum_margin_deg, "minimum-margin-deg"
    ))
    maximum_command_speed_rad_s = positive_finite(
        maximum_command_speed_rad_s, "maximum-command-speed-rad-s"
    )

    selection = load_json(selection_path)
    if selection.get("selection_status") != "margin_safe_solution_found":
        raise ValueError("selection is not marked as a margin-safe solution")
    selected = selection.get("selected_solution")
    if not isinstance(selected, dict) or not isinstance(selected.get("path"), dict):
        raise ValueError("selection has no selected pregrasp path")
    path = selected["path"]
    if not path.get("path_reachable") or not path.get("path_has_warning_margin"):
        raise ValueError("selected pregrasp path is not reachable with safe margin")

    q_saved = finite_six_vector(selection.get("q_current_rad"), "saved current q")
    q_pregrasp = finite_six_vector(path.get("q_pregrasp_rad"), "pregrasp q")
    q_grasp = finite_six_vector(
        path.get("q_grasp_after_path_rad"), "grasp q"
    )
    lift_waypoints = [
        waypoint
        for waypoint in path.get("waypoint_checks", [])
        if waypoint.get("label") == "lift"
    ]
    if len(lift_waypoints) != 1:
        raise ValueError(f"expected one final lift waypoint; got {len(lift_waypoints)}")
    q_lift = finite_six_vector(lift_waypoints[0].get("q_rad"), "lift q")
    path_margin_rad = positive_finite(
        path.get("path_minimum_joint_margin_rad"), "path minimum joint margin"
    )
    if path_margin_rad < minimum_margin_rad:
        raise ValueError(
            "selected path margin is below the requested minimum: "
            f"{np.rad2deg(path_margin_rad):.3f} deg"
        )

    selected_part = str(selection.get("target", {}).get("part_name", "unknown"))

    limits = load_robot_limits(config_path)
    if np.any(q_saved < limits["lower"]) or np.any(q_saved > limits["upper"]):
        raise ValueError("saved current q violates Follower joint limits")
    for name, q in (
        ("pregrasp", q_pregrasp),
        ("grasp", q_grasp),
        ("lift", q_lift),
    ):
        if np.any(q < limits["lower"]) or np.any(q > limits["upper"]):
            raise ValueError(f"{name} q violates Follower joint limits")

    waypoints: List[Dict[str, Any]] = []
    previous = q_saved
    for index in range(1, segment_count + 1):
        fraction = index / segment_count
        target = q_saved + fraction * (q_pregrasp - q_saved)
        joint_margin = np.minimum(target - limits["lower"], limits["upper"] - target)
        command_speed = np.abs(target - previous) / segment_duration_s
        if np.any(joint_margin < minimum_margin_rad):
            raise ValueError(f"segment {index} violates the requested joint margin")
        if np.any(command_speed > maximum_command_speed_rad_s + 1e-12):
            raise ValueError(
                f"segment {index} exceeds maximum command speed: "
                f"{float(np.max(command_speed)):.6f} rad/s"
            )
        if np.any(command_speed > limits["velocity"] + 1e-12):
            raise ValueError(f"segment {index} exceeds Follower velocity limits")
        waypoints.append({
            "segment": index,
            "fraction": fraction,
            "confirmation": f"MOVE_{index}",
            "q_target_rad": target.tolist(),
            "maximum_command_speed_rad_s": float(np.max(command_speed)),
            "minimum_joint_margin_deg": float(np.rad2deg(np.min(joint_margin))),
        })
        previous = target

    return {
        "plan_type": "guarded Panthera bottle grasp test",
        "selection": str(selection_path.resolve()),
        "config": str(config_path.resolve()),
        "selected_part": selected_part,
        "selected_cap_yaw_deg": float(selected["cap_yaw_deg"]),
        "q_saved_current_rad": q_saved.tolist(),
        "q_pregrasp_rad": q_pregrasp.tolist(),
        "q_grasp_rad": q_grasp.tolist(),
        "q_lift_rad": q_lift.tolist(),
        "path_minimum_joint_margin_deg": float(np.rad2deg(path_margin_rad)),
        "segment_count": segment_count,
        "segment_duration_s": segment_duration_s,
        "minimum_joint_margin_required_deg": minimum_margin_deg,
        "maximum_command_speed_limit_rad_s": maximum_command_speed_rad_s,
        "waypoints": waypoints,
        "hardware_access": False,
        "robot_motion_commanded": False,
        "gripper_commanded": True,
        "direct_robot_execution_allowed": False,
        "return_to_zero_policy": {
            "normal_completion": False,
            "first_keyboard_interrupt": True,
            "second_keyboard_interrupt": "set_stop",
            "other_exception": "set_stop",
            "target_q_rad": [0.0] * 6,
            "duration_s": RETURN_TO_ZERO_DURATION_S,
            "release_before_return": True,
            "close_after_return": True,
        },
        "gripper_sequence": {
            "open_pos": GRIPPER_OPEN_POS,
            "close_pos": GRIPPER_CLOSE_POS,
            "mode": "MIT",
            "mit_velocity": GRIPPER_MIT_VELOCITY,
            "mit_torque": GRIPPER_MIT_TORQUE,
            "kp": GRIPPER_MIT_KP,
            "kd": GRIPPER_MIT_KD,
            "grasp_close_duration_s": GRIPPER_GRASP_CLOSE_DURATION_S,
            "grasp_hold_after_close_duration_s": GRIPPER_GRASP_HOLD_AFTER_CLOSE_S,
            "final_close_duration_s": GRIPPER_FINAL_CLOSE_DURATION_S,
            "settle_duration_s": GRIPPER_SETTLE_DURATION_S,
            "open_position_tolerance": GRIPPER_POSITION_TOLERANCE,
        },
    }


def read_state_samples(
    robot: Any,
    sample_count: int,
    sample_delay_s: float,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for sample_index in range(sample_count):
        for _ in range(4):
            robot.send_get_motor_state_cmd()
            robot.motor_send_cmd()
        q = finite_six_vector(robot.get_current_pos(), "Panthera current q")
        dq = finite_six_vector(robot.get_current_vel(), "Panthera current dq")
        records.append({
            "sample_index": sample_index,
            "timestamp_unix_s": time.time(),
            "q_rad": q.tolist(),
            "dq_rad_s": dq.tolist(),
        })
        if sample_index + 1 < sample_count:
            time.sleep(sample_delay_s)
    return records


def read_gripper_position(robot: Any) -> float:
    """Refresh and read the gripper position without commanding motion."""
    for _ in range(4):
        robot.send_get_motor_state_cmd()
        robot.motor_send_cmd()
    position = float(robot.get_current_pos_gripper())
    if not np.isfinite(position):
        raise RuntimeError("Panthera returned a non-finite gripper position")
    return position


def command_gripper_mit(robot: Any, target_pos: float) -> None:
    accepted = robot.gripper_control_MIT(
        float(target_pos),
        GRIPPER_MIT_VELOCITY,
        GRIPPER_MIT_TORQUE,
        GRIPPER_MIT_KP,
        GRIPPER_MIT_KD,
    )
    if not accepted:
        raise RuntimeError(f"gripper MIT target {target_pos:.3f} rad was rejected")


def open_gripper_and_wait(robot: Any, target_pos: float) -> Dict[str, Any]:
    """Repeatedly command MIT opening and verify the measured position."""
    deadline = time.monotonic() + GRIPPER_OPEN_TIMEOUT_S
    samples = []
    while True:
        command_gripper_mit(robot, target_pos)
        time.sleep(0.02)
        position = read_gripper_position(robot)
        samples.append({"timestamp_unix_s": time.time(), "position": position})
        if abs(position - target_pos) <= GRIPPER_POSITION_TOLERANCE:
            return {
                "commanded": True,
                "target_pos": target_pos,
                "mit_velocity": GRIPPER_MIT_VELOCITY,
                "mit_torque": GRIPPER_MIT_TORQUE,
                "kp": GRIPPER_MIT_KP,
                "kd": GRIPPER_MIT_KD,
                "final_position": position,
                "position_tolerance": GRIPPER_POSITION_TOLERANCE,
                "samples": samples,
            }
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "gripper did not reach fully-open position: "
                f"target={target_pos:.3f}, final={position:.3f}"
            )
        time.sleep(0.05)


def smoothstep_position(start: float, end: float, elapsed: float, duration: float) -> float:
    """Quintic smoothstep used by the SDK replay for gradual gripper motion."""
    if duration <= 0.0:
        return float(end)
    ratio = float(np.clip(elapsed / duration, 0.0, 1.0))
    blend = ratio ** 3 * (10.0 - 15.0 * ratio + 6.0 * ratio ** 2)
    return float(start + (end - start) * blend)


def smooth_close_gripper_mit(
    robot: Any,
    start_pos: float,
    target_pos: float,
    duration: float,
    hold_after_s: float,
    require_target: bool,
) -> Dict[str, Any]:
    """Continuously command one MIT target, then verify the final position."""
    start_time = time.monotonic()
    samples = []
    deadline = start_time + duration
    while time.monotonic() < deadline:
        command_gripper_mit(robot, target_pos)
        time.sleep(0.02)

    hold_deadline = time.monotonic() + hold_after_s
    final_position = float("nan")
    while True:
        command_gripper_mit(robot, target_pos)
        time.sleep(0.02)
        final_position = read_gripper_position(robot)
        samples.append({
            "timestamp_unix_s": time.time(),
            "target_position": target_pos,
            "position": final_position,
        })
        if require_target and abs(final_position - target_pos) <= GRIPPER_POSITION_TOLERANCE:
            break
        if time.monotonic() >= hold_deadline:
            if require_target:
                raise RuntimeError(
                    "gripper did not reach closed position: "
                    f"target={target_pos:.3f}, final={final_position:.3f}"
                )
            break
    return {
        "commanded": True,
        "target_pos": target_pos,
        "mit_velocity": GRIPPER_MIT_VELOCITY,
        "mit_torque": GRIPPER_MIT_TORQUE,
        "kp": GRIPPER_MIT_KP,
        "kd": GRIPPER_MIT_KD,
        "start_position": start_pos,
        "close_duration_s": duration,
        "hold_after_close_duration_s": hold_after_s,
        "require_target": require_target,
        "final_position": final_position,
        "samples": samples,
    }


def build_return_to_zero_trajectory(
    robot: Any,
    start_q: np.ndarray,
    limits: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    zero_q = np.zeros(robot.motor_count, dtype=np.float64)
    if np.any(zero_q < limits["lower"]) or np.any(zero_q > limits["upper"]):
        raise RuntimeError("q=0 is outside the configured Follower joint limits")
    step_count = int(round(RETURN_TO_ZERO_DURATION_S / RETURN_CONTROL_DT_S))
    sample_times = np.linspace(0.0, RETURN_TO_ZERO_DURATION_S, step_count + 1)
    positions = []
    velocities = []
    accelerations = []
    for elapsed in sample_times:
        position, velocity, acceleration = robot.septic_interpolation(
            start_q,
            zero_q,
            RETURN_TO_ZERO_DURATION_S,
            float(elapsed),
        )
        positions.append(position)
        velocities.append(velocity)
        accelerations.append(acceleration)
    positions_array = np.asarray(positions, dtype=np.float64)
    velocities_array = np.asarray(velocities, dtype=np.float64)
    accelerations_array = np.asarray(accelerations, dtype=np.float64)
    if np.any(positions_array < limits["lower"] - 1e-9) or np.any(
        positions_array > limits["upper"] + 1e-9
    ):
        raise RuntimeError("return-to-zero trajectory violates joint limits")

    velocity_limits = finite_six_vector(robot.velocity_limits, "SDK velocity limits")
    acceleration_limits = finite_six_vector(
        robot.acceleration_limits, "SDK acceleration limits"
    )
    peak_velocity = np.max(np.abs(velocities_array), axis=0)
    peak_acceleration = np.max(np.abs(accelerations_array), axis=0)
    if np.any(peak_velocity > velocity_limits + 1e-9):
        raise RuntimeError("return-to-zero trajectory exceeds SDK velocity limits")
    if np.any(peak_acceleration > acceleration_limits + 1e-9):
        raise RuntimeError("return-to-zero trajectory exceeds SDK acceleration limits")
    return {
        "sample_times_s": sample_times,
        "positions_rad": positions_array,
        "velocities_rad_s": velocities_array,
        "peak_velocity_rad_s": peak_velocity,
        "peak_acceleration_rad_s2": peak_acceleration,
    }


def return_to_zero(
    robot: Any,
    limits: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    state_samples = read_state_samples(robot, sample_count=2, sample_delay_s=0.05)
    start_q = finite_six_vector(state_samples[-1]["q_rad"], "return start q")
    trajectory = build_return_to_zero_trajectory(robot, start_q, limits)
    max_torque = finite_six_vector(robot.max_torque, "SDK maximum torque")
    zero_velocity = np.zeros(robot.motor_count, dtype=np.float64)
    if not robot.Joint_Pos_Vel(
        start_q,
        zero_velocity,
        max_torque,
        iswait=False,
    ):
        raise RuntimeError("current-position hold was rejected before return to zero")
    time.sleep(RETURN_HOLD_DURATION_S)

    print(
        f"Returning to q=0 over {RETURN_TO_ZERO_DURATION_S:.1f}s at 100 Hz; "
        f"peak velocity={float(np.max(trajectory['peak_velocity_rad_s'])):.3f} rad/s"
    )
    sample_times = trajectory["sample_times_s"]
    positions = trajectory["positions_rad"]
    velocities = trajectory["velocities_rad_s"]
    last_command_time = time.perf_counter()
    for sample_index in range(1, len(sample_times)):
        sample_dt = float(sample_times[sample_index] - sample_times[sample_index - 1])
        remaining = sample_dt - (time.perf_counter() - last_command_time)
        if remaining > 0.0:
            time.sleep(remaining)
        if not robot.Joint_Pos_Vel(
            positions[sample_index],
            velocities[sample_index],
            max_torque,
            iswait=False,
        ):
            raise RuntimeError(
                f"return-to-zero command rejected at point {sample_index}"
            )
        last_command_time = time.perf_counter()

    zero_q = np.zeros(robot.motor_count, dtype=np.float64)
    if not robot.wait_for_position(zero_q, tolerance=0.05, timeout=5.0):
        raise RuntimeError("robot did not reach q=0 within 0.05 rad tolerance")
    final_samples = read_state_samples(robot, sample_count=2, sample_delay_s=0.05)
    final_q = finite_six_vector(final_samples[-1]["q_rad"], "return final q")
    final_error_deg = float(np.rad2deg(np.max(np.abs(final_q))))
    print(f"Returned to q=0; maximum final error={final_error_deg:.3f} deg")
    return {
        "start_q_rad": start_q.tolist(),
        "target_q_rad": zero_q.tolist(),
        "duration_s": RETURN_TO_ZERO_DURATION_S,
        "control_dt_s": RETURN_CONTROL_DT_S,
        "peak_velocity_rad_s": trajectory["peak_velocity_rad_s"].tolist(),
        "peak_acceleration_rad_s2": trajectory["peak_acceleration_rad_s2"].tolist(),
        "final_state_samples": final_samples,
        "final_q_rad": final_q.tolist(),
        "maximum_final_error_deg": final_error_deg,
        "gripper_commanded": False,
    }


def execute_plan(args: argparse.Namespace, plan: Dict[str, Any]) -> int:
    sdk_scripts = args.sdk_scripts.resolve()
    panthera_module = sdk_scripts / "Panthera_lib/Panthera.py"
    if not panthera_module.is_file():
        raise FileNotFoundError(f"Panthera SDK module not found: {panthera_module}")

    print("\nREAL ROBOT BOTTLE GRASP TEST REQUESTED")
    print("No automatic collision checking is available.")
    print("Clear the complete arm sweep, keep the E-stop reachable, and do not touch the robot.")
    print("--execute is the only motion authorization; connecting to Panthera now.")

    if str(sdk_scripts) not in sys.path:
        sys.path.insert(0, str(sdk_scripts))
    from Panthera_lib import Panthera

    robot = Panthera(config_path=str(args.config.resolve()))
    if robot.motor_count != 6:
        raise RuntimeError(f"expected six arm motors; got {robot.motor_count}")
    if not isinstance(robot.gripper_limits, dict):
        raise RuntimeError("Panthera gripper limits are unavailable")
    gripper_open_pos = GRIPPER_OPEN_POS
    gripper_lower = float(robot.gripper_limits["lower"])
    gripper_upper = float(robot.gripper_limits["upper"])
    if not np.isfinite(gripper_upper) or not (gripper_lower <= gripper_open_pos <= gripper_upper):
        raise RuntimeError(
            "MIT gripper open target is outside configured limits: "
            f"target={gripper_open_pos:.3f}, limits=[{gripper_lower:.3f}, {gripper_upper:.3f}]"
        )

    state_samples = read_state_samples(
        robot=robot,
        sample_count=args.state_samples,
        sample_delay_s=args.sample_delay,
    )
    q_samples = np.asarray([sample["q_rad"] for sample in state_samples])
    dq_samples = np.asarray([sample["dq_rad_s"] for sample in state_samples])
    q_actual = q_samples[-1]
    maximum_state_speed = float(np.max(np.abs(dq_samples)))
    q_pregrasp = finite_six_vector(plan["q_pregrasp_rad"], "pregrasp q")
    q_grasp = finite_six_vector(plan["q_grasp_rad"], "grasp q")
    q_lift = finite_six_vector(plan["q_lift_rad"], "lift q")
    if maximum_state_speed > args.max_state_speed_rad_s:
        raise RuntimeError(
            f"robot is not stationary: {maximum_state_speed:.6f} rad/s"
        )

    limits = load_robot_limits(args.config)
    if np.any(q_actual < limits["lower"]) or np.any(q_actual > limits["upper"]):
        raise RuntimeError("fresh Panthera state violates Follower joint limits")

    minimum_margin_rad = np.deg2rad(plan["minimum_joint_margin_required_deg"])
    direct_duration_s = plan["segment_duration_s"] * plan["segment_count"]
    joint_margin = np.minimum(
        q_pregrasp - limits["lower"], limits["upper"] - q_pregrasp
    )
    command_speed = np.abs(q_pregrasp - q_actual) / direct_duration_s
    if np.any(joint_margin < minimum_margin_rad):
        raise RuntimeError("q_pregrasp violates the requested joint margin")
    if np.any(command_speed > plan["maximum_command_speed_limit_rad_s"] + 1e-12):
        raise RuntimeError(
            "direct current-to-pregrasp motion exceeds the command speed limit"
        )
    if np.any(command_speed > limits["velocity"] + 1e-12):
        raise RuntimeError(
            "direct current-to-pregrasp motion exceeds Follower velocity limits"
        )
    for name, start, target, duration in (
        ("pregrasp-to-grasp", q_pregrasp, q_grasp, GRASP_APPROACH_DURATION_S),
        ("grasp-to-lift", q_grasp, q_lift, LIFT_DURATION_S),
    ):
        transition_speed = np.abs(target - start) / duration
        if np.any(transition_speed > plan["maximum_command_speed_limit_rad_s"] + 1e-12):
            raise RuntimeError(f"{name} exceeds the command speed limit")
        if np.any(transition_speed > limits["velocity"] + 1e-12):
            raise RuntimeError(f"{name} exceeds Follower velocity limits")
    direct_waypoint = {
        "label": "pregrasp",
        "q_target_rad": q_pregrasp.tolist(),
        "duration_s": direct_duration_s,
        "maximum_command_speed_rad_s": float(np.max(command_speed)),
        "minimum_joint_margin_deg": float(np.rad2deg(np.min(joint_margin))),
    }

    execution_log = dict(plan)
    execution_log.update({
        "hardware_access": True,
        "robot_motion_commanded": False,
        "direct_robot_execution_allowed": True,
        "status": "ready_for_pregrasp_motion",
        "fresh_state_samples": state_samples,
        "q_fresh_current_rad": q_actual.tolist(),
        "maximum_fresh_state_speed_rad_s": maximum_state_speed,
        "saved_state_delta_deg": float(
            np.rad2deg(np.max(np.abs(q_actual - plan["q_saved_current_rad"])))
        ),
        "direct_pregrasp_motion": direct_waypoint,
        "gripper_open_before_move": None,
        "grasp_motion": {
            "q_grasp_rad": q_grasp.tolist(),
            "duration_s": GRASP_APPROACH_DURATION_S,
            "gripper_close": False,
        },
        "lift_motion": {
            "q_lift_rad": q_lift.tolist(),
            "duration_s": LIFT_DURATION_S,
        },
        "executed_segments": [],
    })
    save_json(args.output, execution_log)

    print(f"Fresh q_current={np.round(q_actual, 6).tolist()} rad")
    print(
        "Saved-state delta is recorded for diagnostics only: "
        f"{execution_log['saved_state_delta_deg']:.3f} deg"
    )
    print(f"Measured speed={maximum_state_speed:.6f} rad/s")
    print(f"Target q_pregrasp={np.round(q_pregrasp, 6).tolist()} rad")
    returning_to_zero = False
    grasped = False
    try:
        execution_log["status"] = "motion_authorized_by_execute_flag"
        save_json(args.output, execution_log)

        print(
            f"Opening gripper fully before arm motion: target pos={gripper_open_pos:.3f}."
        )
        execution_log["status"] = "opening_gripper_before_pregrasp"
        save_json(args.output, execution_log)
        execution_log["gripper_open_before_move"] = open_gripper_and_wait(
            robot, gripper_open_pos
        )
        execution_log["status"] = "gripper_open_before_pregrasp_confirmed"
        save_json(args.output, execution_log)

        target = q_pregrasp
        print(
            f"\nMoving directly to q_pregrasp: duration={direct_duration_s:.1f}s, "
            f"speed<={direct_waypoint['maximum_command_speed_rad_s']:.6f} rad/s"
        )
        print(f"q_target={np.round(target, 6).tolist()} rad")
        execution_log["robot_motion_commanded"] = True
        execution_log["status"] = "executing_pregrasp"
        save_json(args.output, execution_log)
        success = robot.moveJ(
            target,
            duration=direct_duration_s,
            iswait=True,
            tolerance=np.deg2rad(args.position_tolerance_deg),
            timeout=direct_duration_s + args.timeout_margin_s,
        )
        time.sleep(args.settle_seconds)
        post_samples = read_state_samples(
            robot=robot,
            sample_count=args.state_samples,
            sample_delay_s=args.sample_delay,
        )
        post_q = np.asarray(post_samples[-1]["q_rad"], dtype=np.float64)
        post_dq = np.asarray([sample["dq_rad_s"] for sample in post_samples])
        position_error_deg = float(np.rad2deg(np.max(np.abs(post_q - target))))
        post_speed = float(np.max(np.abs(post_dq)))
        execution_log["direct_pregrasp_motion"].update({
            "sdk_moveJ_success": bool(success),
            "post_state_samples": post_samples,
            "q_post_rad": post_q.tolist(),
            "maximum_position_error_deg": position_error_deg,
            "maximum_post_speed_rad_s": post_speed,
        })
        save_json(args.output, execution_log)
        if not success:
            raise RuntimeError("SDK moveJ failed or timed out before q_pregrasp")
        if position_error_deg > args.position_tolerance_deg:
            raise RuntimeError(
                f"q_pregrasp position error is {position_error_deg:.3f} deg"
            )
        execution_log["status"] = "pregrasp_reached_approaching_grasp"
        save_json(args.output, execution_log)
        print(
            f"q_pregrasp reached: position error={position_error_deg:.3f} deg; "
            f"measured speed={post_speed:.6f} rad/s"
        )

        print(
            f"Moving q_pregrasp -> q_grasp over {GRASP_APPROACH_DURATION_S:.1f}s."
        )
        execution_log["robot_motion_commanded"] = True
        execution_log["status"] = "executing_grasp_approach"
        save_json(args.output, execution_log)
        grasp_success = robot.moveJ(
            q_grasp,
            duration=GRASP_APPROACH_DURATION_S,
            iswait=True,
            tolerance=np.deg2rad(args.position_tolerance_deg),
            timeout=GRASP_APPROACH_DURATION_S + args.timeout_margin_s,
        )
        grasp_samples = read_state_samples(
            robot=robot,
            sample_count=args.state_samples,
            sample_delay_s=args.sample_delay,
        )
        grasp_q_post = np.asarray(grasp_samples[-1]["q_rad"], dtype=np.float64)
        grasp_error_deg = float(np.rad2deg(np.max(np.abs(grasp_q_post - q_grasp))))
        execution_log["grasp_motion"].update({
            "sdk_moveJ_success": bool(grasp_success),
            "post_state_samples": grasp_samples,
            "q_post_rad": grasp_q_post.tolist(),
            "maximum_position_error_deg": grasp_error_deg,
        })
        save_json(args.output, execution_log)
        if not grasp_success:
            raise RuntimeError("SDK moveJ failed or timed out before q_grasp")
        if grasp_error_deg > args.position_tolerance_deg:
            raise RuntimeError(
                f"q_grasp position error is {grasp_error_deg:.3f} deg"
            )

        print(
            "q_grasp reached; smoothly closing gripper with MIT over "
            f"{GRIPPER_GRASP_CLOSE_DURATION_S:.1f}s."
        )
        # From the first close command onward, treat the object as potentially held
        # so Ctrl+C always performs the release-before-return sequence.
        grasped = True
        gripper_start_before_close = read_gripper_position(robot)
        execution_log["grasp_motion"]["gripper_close"] = smooth_close_gripper_mit(
            robot,
            start_pos=gripper_start_before_close,
            target_pos=GRIPPER_CLOSE_POS,
            duration=GRIPPER_GRASP_CLOSE_DURATION_S,
            hold_after_s=GRIPPER_GRASP_HOLD_AFTER_CLOSE_S,
            require_target=False,
        )
        execution_log["status"] = "grasp_closed_lifting"
        save_json(args.output, execution_log)

        print(f"Lifting q_grasp -> q_lift over {LIFT_DURATION_S:.1f}s.")
        execution_log["status"] = "executing_lift"
        save_json(args.output, execution_log)
        lift_success = robot.moveJ(
            q_lift,
            duration=LIFT_DURATION_S,
            iswait=True,
            tolerance=np.deg2rad(args.position_tolerance_deg),
            timeout=LIFT_DURATION_S + args.timeout_margin_s,
        )
        lift_samples = read_state_samples(
            robot=robot,
            sample_count=args.state_samples,
            sample_delay_s=args.sample_delay,
        )
        lift_q_post = np.asarray(lift_samples[-1]["q_rad"], dtype=np.float64)
        lift_error_deg = float(np.rad2deg(np.max(np.abs(lift_q_post - q_lift))))
        execution_log["lift_motion"].update({
            "sdk_moveJ_success": bool(lift_success),
            "post_state_samples": lift_samples,
            "q_post_rad": lift_q_post.tolist(),
            "maximum_position_error_deg": lift_error_deg,
        })
        save_json(args.output, execution_log)
        if not lift_success:
            raise RuntimeError("SDK moveJ failed or timed out before q_lift")
        if lift_error_deg > args.position_tolerance_deg:
            raise RuntimeError(
                f"q_lift position error is {lift_error_deg:.3f} deg"
            )

        execution_log["status"] = "holding_lifted_object"
        save_json(args.output, execution_log)
        print("Object lifted; holding indefinitely. Press Ctrl+C to release and return.")
        while True:
            time.sleep(1.0)
    except (KeyboardInterrupt, OperatorReturnRequested) as interruption:
        if returning_to_zero:
            robot.set_stop()
            execution_log["status"] = "return_to_zero_interrupted_stopped"
            execution_log["error"] = f"{type(interruption).__name__}: {interruption}"
            save_json(args.output, execution_log)
            print("\nReturn to zero was interrupted; motor stop was requested.")
            if isinstance(interruption, KeyboardInterrupt):
                raise
            return 1

        was_grasped = grasped
        if was_grasped:
            print("\nCtrl+C received: opening gripper before returning to q=0.")
            execution_log["status"] = "opening_gripper_before_return"
            save_json(args.output, execution_log)
            try:
                execution_log["gripper_release"] = open_gripper_and_wait(
                    robot, gripper_open_pos
                )
            except Exception as release_error:
                robot.set_stop()
                execution_log["status"] = "gripper_release_failed_stopped"
                execution_log["release_error"] = (
                    f"{type(release_error).__name__}: {release_error}"
                )
                save_json(args.output, execution_log)
                raise RuntimeError(
                    f"gripper was not fully released; return was blocked: {release_error}"
                ) from release_error
            save_json(args.output, execution_log)
        print("Returning from the actual position to q=0.")
        execution_log["status"] = "interrupted_returning_to_zero"
        execution_log["error"] = f"{type(interruption).__name__}: {interruption}"
        save_json(args.output, execution_log)
        try:
            returning_to_zero = True
            execution_log["return_to_zero"] = return_to_zero(robot, limits)
            returning_to_zero = False
            if was_grasped:
                print(
                    "Returned to q=0; smoothly closing gripper as the final "
                    f"shutdown action over {GRIPPER_FINAL_CLOSE_DURATION_S:.1f}s."
                )
                gripper_start_after_return = read_gripper_position(robot)
                execution_log["gripper_close_after_return"] = smooth_close_gripper_mit(
                    robot,
                    start_pos=gripper_start_after_return,
                    target_pos=GRIPPER_CLOSE_POS,
                    duration=GRIPPER_FINAL_CLOSE_DURATION_S,
                    hold_after_s=GRIPPER_CLOSE_TIMEOUT_S,
                    require_target=True,
                )
            execution_log["status"] = "interrupted_returned_to_zero"
            save_json(args.output, execution_log)
        except KeyboardInterrupt:
            robot.set_stop()
            execution_log["status"] = "second_interrupt_stopped"
            save_json(args.output, execution_log)
            print("\nSecond interrupt received; motor stop was requested.")
            raise
        except Exception as return_error:
            robot.set_stop()
            execution_log["status"] = "interrupted_return_failed_stopped"
            execution_log["return_error"] = (
                f"{type(return_error).__name__}: {return_error}"
            )
            save_json(args.output, execution_log)
            raise RuntimeError(
                f"interrupted return to zero failed: {return_error}"
            ) from return_error
        if isinstance(interruption, KeyboardInterrupt):
            raise
        return 1
    except Exception as exc:
        robot.set_stop()
        execution_log["status"] = "execution_failed_stopped"
        execution_log["error"] = f"{type(exc).__name__}: {exc}"
        save_json(args.output, execution_log)
        print("Motor stop was requested; no automatic return after SDK/path failure.")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sdk-scripts", type=Path, default=DEFAULT_SDK_SCRIPTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--segment-duration", type=float, default=4.0)
    parser.add_argument("--minimum-margin-deg", type=float, default=10.0)
    parser.add_argument("--maximum-command-speed-rad-s", type=float, default=0.15)
    parser.add_argument("--state-samples", type=int, default=5)
    parser.add_argument("--sample-delay", type=float, default=0.05)
    parser.add_argument("--max-state-speed-rad-s", type=float, default=0.05)
    parser.add_argument("--max-saved-state-delta-deg", type=float, default=5.0)
    parser.add_argument("--position-tolerance-deg", type=float, default=1.7)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-margin-s", type=float, default=8.0)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(
        selection_path=args.selection,
        config_path=args.config,
        segment_count=args.segments,
        segment_duration_s=args.segment_duration,
        minimum_margin_deg=args.minimum_margin_deg,
        maximum_command_speed_rad_s=args.maximum_command_speed_rad_s,
    )
    if args.state_samples < 2:
        raise ValueError("state-samples must be at least 2")
    args.sample_delay = positive_finite(args.sample_delay, "sample-delay")
    args.max_state_speed_rad_s = positive_finite(
        args.max_state_speed_rad_s, "max-state-speed-rad-s"
    )
    args.max_saved_state_delta_deg = positive_finite(
        args.max_saved_state_delta_deg, "max-saved-state-delta-deg"
    )
    args.position_tolerance_deg = positive_finite(
        args.position_tolerance_deg, "position-tolerance-deg"
    )
    args.settle_seconds = positive_finite(args.settle_seconds, "settle-seconds")
    args.timeout_margin_s = positive_finite(args.timeout_margin_s, "timeout-margin-s")
    if args.execute:
        return execute_plan(args, plan)
    save_json(args.output, plan)
    print("Panthera bottle grasp-test DRY RUN")
    print(
        f"  selected part={plan['selected_part']}, "
        f"yaw={plan['selected_cap_yaw_deg']:.3f} deg"
    )
    print(f"  path margin={plan['path_minimum_joint_margin_deg']:.3f} deg")
    direct_duration_s = plan["segment_duration_s"] * plan["segment_count"]
    q_saved = np.asarray(plan["q_saved_current_rad"], dtype=np.float64)
    q_pregrasp = np.asarray(plan["q_pregrasp_rad"], dtype=np.float64)
    direct_speed = float(np.max(np.abs(q_pregrasp - q_saved) / direct_duration_s))
    print(
        f"  direct current->q_pregrasp: duration={direct_duration_s:.1f}s, "
        f"speed={direct_speed:.6f} rad/s"
    )
    print(
        f"  q_pregrasp->q_grasp: duration={GRASP_APPROACH_DURATION_S:.1f}s; "
        f"gripper MIT close: pos={GRIPPER_CLOSE_POS:.1f}, "
        f"smooth={GRIPPER_GRASP_CLOSE_DURATION_S:.1f}s, "
        f"kp={GRIPPER_MIT_KP:.1f}, kd={GRIPPER_MIT_KD:.1f}"
    )
    print(f"  q_grasp->q_lift: duration={LIFT_DURATION_S:.1f}s")
    print(
        "  hold lifted object until Ctrl+C; then open -> return q=0 -> "
        f"smooth close ({GRIPPER_FINAL_CLOSE_DURATION_S:.1f}s)"
    )
    print(f"  saved: {args.output}")
    print("After q_lift there is no normal completion; Ctrl+C performs open -> return q=0 -> close.")
    print("No SDK module was imported and no hardware access occurred.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Pregrasp execution interrupted; guarded return handling has ended.")
        raise SystemExit(130)
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Pregrasp command rejected: {exc}", file=sys.stderr)
        raise SystemExit(2)
