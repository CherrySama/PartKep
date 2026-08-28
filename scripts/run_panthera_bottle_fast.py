#!/usr/bin/env python3
"""Fast real Panthera bottle deployment entry point.

Runtime path:

    D435 fixture -> bottle vision + PartKep -> fixed tool pose
        -> current-state seeded Panthera IK -> SDK pregrasp/grasp/lift

This entry point deliberately does not perform the exhaustive current-aware
72-yaw x 32-multistart branch search.  It tries the finite tool-link branches
already emitted by PartKep, using the measured Panthera state as the IK seed.

The script has two interpreter modes.  The normal mode runs camera/vision in
the PartKep environment and launches the same file in the Panthera SDK
environment for state, IK, and optional motion.  ``--execute`` is the only
authorization for robot motion; without it the SDK child only reads state and
writes a plan.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "panthera_bottle_fast"
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
    """Return a unique timestamped output directory for one deployment run."""
    root = PROJECT_ROOT / "data"
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}"
    candidate = root / f"real_deployment_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"real_deployment_{stamp}_{suffix}"
        suffix += 1
    return candidate


@dataclass(frozen=True)
class Target:
    rank: int
    part_name: str
    partkep_cost: float
    orientation_branch: str
    approach_direction_base: np.ndarray
    transform: np.ndarray


@dataclass(frozen=True)
class PoseSolution:
    label: str
    target: np.ndarray
    q: np.ndarray
    position_error_m: float
    rotation_error_rad: float
    evaluations: int
    optimizer_success: bool
    joint_margin_rad: np.ndarray


def save_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite_vector(value: Any, name: str, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return vector


def validate_transform(value: Any, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
        raise ValueError(f"{name} rotation is not right-handed")
    return transform


def load_targets(result_path: Path) -> List[Target]:
    """Read PartKep's authoritative Panthera tool-link targets.

    ``T_pick_legacy_solver`` is intentionally ignored because it contains the
    old Panda hand/finger offset.  The deployment target is always
    ``T_base_tool_link_target`` under a ranked PartKep candidate.
    """
    result = load_json(result_path)
    grasp = result.get("partkep_grasp")
    if not isinstance(grasp, dict) or not grasp.get("success"):
        raise ValueError("pipeline result has no successful PartKep grasp")
    candidates = grasp.get("ranked_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("pipeline result has no ranked PartKep candidates")

    targets: List[Target] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        part_name = str(candidate.get("part_name", "unknown"))
        rank = int(candidate.get("rank", 10**9))
        cost = float(candidate.get("partkep_cost", float("inf")))
        approach = finite_vector(
            candidate.get("approach_direction_base"),
            f"{part_name} approach direction",
            3,
        )
        norm = float(np.linalg.norm(approach))
        if not np.isfinite(cost) or norm < 1e-9:
            raise ValueError(f"invalid PartKep candidate {part_name}")
        approach = approach / norm
        branches = candidate.get("tool_link_targets")
        if not isinstance(branches, list) or not branches:
            raise ValueError(f"candidate {part_name} has no tool_link targets")
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            targets.append(Target(
                rank=rank,
                part_name=part_name,
                partkep_cost=cost,
                orientation_branch=str(branch.get("orientation_branch", "unknown")),
                approach_direction_base=approach.copy(),
                transform=validate_transform(
                    branch.get("T_base_tool_link_target"),
                    f"{part_name} tool-link target",
                ),
            ))
    if not targets:
        raise ValueError("PartKep emitted no valid Panthera tool-link targets")
    return sorted(targets, key=lambda item: (item.rank, item.partkep_cost))


def load_robot_config(config_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    try:
        names = list(config["kinematics"]["joint_names"])
        lower = np.asarray(config["robot"]["joint_limits"]["lower"], dtype=np.float64)
        upper = np.asarray(config["robot"]["joint_limits"]["upper"], dtype=np.float64)
        velocity = np.asarray(config["robot"]["velocity_limits"], dtype=np.float64)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid Panthera Follower config: {config_path}") from exc
    if (
        len(names) != 6
        or lower.shape != (6,)
        or upper.shape != (6,)
        or velocity.shape != (6,)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or not np.all(np.isfinite(velocity))
        or np.any(lower >= upper)
        or np.any(velocity <= 0.0)
    ):
        raise ValueError("Follower config must contain six valid joint limits and velocities")
    return names, lower, upper, velocity


def check_model(model: Any, joint_names: Sequence[str]) -> int:
    if model.nq != len(joint_names) or model.nv != len(joint_names):
        raise ValueError(
            f"Panthera URDF has nq={model.nq}, nv={model.nv}; expected six joints"
        )
    if list(model.names[1:]) != list(joint_names):
        raise ValueError(
            f"URDF/config joint order mismatch: URDF={list(model.names[1:])}, "
            f"config={list(joint_names)}"
        )
    frame_id = int(model.getFrameId("tool_link"))
    if frame_id >= len(model.frames) or model.frames[frame_id].name != "tool_link":
        raise ValueError("tool_link frame is missing from Panthera URDF")
    return frame_id


def forward_kinematics(model: Any, frame_id: int, q: np.ndarray) -> np.ndarray:
    import pinocchio as pin

    data = model.createData()
    pin.framesForwardKinematics(model, data, q)
    placement = data.oMf[frame_id]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = placement.rotation
    transform[:3, 3] = placement.translation
    return transform


def pose_errors(actual: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    import pinocchio as pin

    position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    rotation_error = float(np.linalg.norm(pin.log3(target[:3, :3].T @ actual[:3, :3])))
    return position_error, rotation_error


def solve_ik_from_seed(
    model: Any,
    frame_id: int,
    target: np.ndarray,
    q_seed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_evaluations: int,
) -> PoseSolution:
    """Solve one pose from the preceding joint state to preserve continuity."""
    from scipy.optimize import least_squares
    import pinocchio as pin

    q_seed = np.clip(finite_vector(q_seed, "IK seed", 6), lower, upper)

    def residual(q: np.ndarray) -> np.ndarray:
        actual = forward_kinematics(model, frame_id, q)
        return np.concatenate((
            actual[:3, 3] - target[:3, 3],
            pin.log3(target[:3, :3].T @ actual[:3, :3]),
        ))

    optimized = least_squares(
        residual,
        q_seed,
        bounds=(lower, upper),
        method="trf",
        max_nfev=max_evaluations,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    q = optimized.x.astype(np.float64, copy=True)
    actual = forward_kinematics(model, frame_id, q)
    position_error, rotation_error = pose_errors(actual, target)
    return PoseSolution(
        label="",
        target=target.copy(),
        q=q,
        position_error_m=position_error,
        rotation_error_rad=rotation_error,
        evaluations=int(optimized.nfev),
        optimizer_success=bool(optimized.success),
        joint_margin_rad=np.minimum(q - lower, upper - q),
    )


def make_target_poses(
    grasp: np.ndarray,
    approach_direction_base: np.ndarray,
    pregrasp_mm: float,
    lift_mm: float,
) -> Dict[str, np.ndarray]:
    pregrasp = grasp.copy()
    pregrasp[:3, 3] -= approach_direction_base * (float(pregrasp_mm) / 1000.0)
    lift = grasp.copy()
    lift[:3, 3] += np.array([0.0, 0.0, float(lift_mm) / 1000.0])
    return {"pregrasp": pregrasp, "grasp": grasp.copy(), "lift": lift}


def read_state_samples(
    robot: Any,
    sample_count: int,
    sample_delay_s: float,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for sample_index in range(sample_count):
        # This is the SDK state-query sequence used by the existing read-only
        # Panthera scripts.  It does not queue arm or gripper motion.
        for _ in range(4):
            robot.send_get_motor_state_cmd()
            robot.motor_send_cmd()
        q = finite_vector(robot.get_current_pos(), "Panthera current q", 6)
        dq = finite_vector(robot.get_current_vel(), "Panthera current dq", 6)
        records.append({
            "sample_index": sample_index,
            "timestamp_unix_s": time.time(),
            "q_rad": q.tolist(),
            "dq_rad_s": dq.tolist(),
        })
        if sample_index + 1 < sample_count:
            time.sleep(sample_delay_s)
    return records


def serialize_pose_solution(solution: PoseSolution) -> Dict[str, Any]:
    return {
        "label": solution.label,
        "q_rad": solution.q.tolist(),
        "T_base_tool_link_target": solution.target.tolist(),
        "position_error_mm": solution.position_error_m * 1000.0,
        "rotation_error_deg": float(np.rad2deg(solution.rotation_error_rad)),
        "optimizer_success": solution.optimizer_success,
        "evaluations": solution.evaluations,
        "joint_limit_margin_deg": np.rad2deg(solution.joint_margin_rad).tolist(),
        "minimum_joint_limit_margin_deg": float(
            np.rad2deg(np.min(solution.joint_margin_rad))
        ),
    }


def solve_target_path(
    model: Any,
    frame_id: int,
    candidate: Target,
    q_current: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    pregrasp_mm: float,
    lift_mm: float,
    max_evaluations: int,
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
    joint_limit_warning_rad: float,
) -> Dict[str, Any]:
    poses = make_target_poses(
        grasp=candidate.transform,
        approach_direction_base=candidate.approach_direction_base,
        pregrasp_mm=pregrasp_mm,
        lift_mm=lift_mm,
    )
    solutions: List[PoseSolution] = []
    q_seed = q_current.copy()
    for label in ("pregrasp", "grasp", "lift"):
        solved = solve_ik_from_seed(
            model=model,
            frame_id=frame_id,
            target=poses[label],
            q_seed=q_seed,
            lower=lower,
            upper=upper,
            max_evaluations=max_evaluations,
        )
        solved = PoseSolution(
            label=label,
            target=solved.target,
            q=solved.q,
            position_error_m=solved.position_error_m,
            rotation_error_rad=solved.rotation_error_rad,
            evaluations=solved.evaluations,
            optimizer_success=solved.optimizer_success,
            joint_margin_rad=solved.joint_margin_rad,
        )
        solutions.append(solved)
        q_seed = solved.q

    pose_checks = [
        {
            **serialize_pose_solution(solution),
            "reachable": bool(
                solution.optimizer_success
                and solution.position_error_m <= position_tolerance_m
                and solution.rotation_error_rad <= rotation_tolerance_rad
                and np.all(solution.q >= lower)
                and np.all(solution.q <= upper)
            ),
            "margin_safe": bool(
                np.min(solution.joint_margin_rad) >= joint_limit_warning_rad
            ),
        }
        for solution in solutions
    ]
    path_reachable = bool(all(item["reachable"] for item in pose_checks))
    margin_safe = bool(path_reachable and all(item["margin_safe"] for item in pose_checks))
    current_to_pregrasp = solutions[0].q - q_current
    return {
        "target": {
            "rank": candidate.rank,
            "part_name": candidate.part_name,
            "partkep_cost": candidate.partkep_cost,
            "orientation_branch": candidate.orientation_branch,
            "approach_direction_base": candidate.approach_direction_base.tolist(),
            "T_base_tool_link_grasp": candidate.transform.tolist(),
        },
        "path_reachable": path_reachable,
        "margin_safe": margin_safe,
        "minimum_joint_limit_margin_deg": float(
            min(item["minimum_joint_limit_margin_deg"] for item in pose_checks)
        ),
        "max_current_to_pregrasp_joint_travel_deg": float(
            np.max(np.abs(np.rad2deg(current_to_pregrasp)))
        ),
        "pose_checks": pose_checks,
        "q_current_rad": q_current.tolist(),
        "q_pregrasp_rad": solutions[0].q.tolist(),
        "q_grasp_rad": solutions[1].q.tolist(),
        "q_lift_rad": solutions[2].q.tolist(),
    }


def choose_path(paths: List[Dict[str, Any]]) -> Dict[str, Any]:
    margin_safe = [path for path in paths if path["margin_safe"]]
    if not margin_safe:
        reachable = [path for path in paths if path["path_reachable"]]
        if not reachable:
            raise RuntimeError("no PartKep tool-link branch is reachable from current Panthera state")
        best_risk = max(
            reachable,
            key=lambda path: (
                path["minimum_joint_limit_margin_deg"],
                -path["target"]["rank"],
                -path["target"]["partkep_cost"],
            ),
        )
        raise RuntimeError(
            "reachable branches are below the configured joint-limit warning: "
            f"closest margin={best_risk['minimum_joint_limit_margin_deg']:.3f} deg"
        )
    # Preserve PartKep's semantic ranking; only use joint travel to break
    # ties between the two equivalent tool-link opening-axis branches.
    return min(
        margin_safe,
        key=lambda path: (
            path["target"]["rank"],
            path["target"]["partkep_cost"],
            path["max_current_to_pregrasp_joint_travel_deg"],
        ),
    )


def positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def duration_for_joint_move(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    requested_s: float,
    command_speed_limit: float,
    sdk_velocity_limits: np.ndarray,
) -> float:
    allowed = np.minimum(
        sdk_velocity_limits,
        np.full(6, command_speed_limit, dtype=np.float64),
    )
    required = float(np.max(np.abs(q_goal - q_start) / allowed))
    return max(float(requested_s), required * 1.05, 0.5)


def connect_panthera(sdk_scripts: Path, config_path: Path) -> Any:
    module_path = sdk_scripts / "Panthera_lib" / "Panthera.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Panthera SDK module not found: {module_path}")
    if str(sdk_scripts) not in sys.path:
        sys.path.insert(0, str(sdk_scripts))
    from Panthera_lib import Panthera

    robot = Panthera(config_path=str(config_path.resolve()))
    if robot.motor_count != 6:
        raise RuntimeError(f"expected six Panthera arm motors; got {robot.motor_count}")
    return robot


def build_robot_plan(args: argparse.Namespace, robot: Any) -> Dict[str, Any]:
    """Read state and solve the fixed PartKep pose without commanding motion."""
    import pinocchio as pin

    targets = load_targets(args.pipeline_result)
    joint_names, lower, upper, velocity_limits = load_robot_config(args.config)
    model = pin.buildModelFromUrdf(str(args.urdf.resolve()))
    frame_id = check_model(model, joint_names)

    print("Reading Panthera state for current-seeded IK; no motion API is called...")
    state_samples = read_state_samples(
        robot=robot,
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
    for index, target in enumerate(targets, start=1):
        print(
            f"  branch {index}/{len(targets)}: {target.part_name}/"
            f"{target.orientation_branch}; solving pregrasp -> grasp -> lift...",
            flush=True,
        )
        path = solve_target_path(
            model=model,
            frame_id=frame_id,
            candidate=target,
            q_current=q_current,
            lower=lower,
            upper=upper,
            pregrasp_mm=args.pregrasp_mm,
            lift_mm=args.lift_mm,
            max_evaluations=args.ik_max_evaluations,
            position_tolerance_m=position_tolerance_m,
            rotation_tolerance_rad=rotation_tolerance_rad,
            joint_limit_warning_rad=joint_limit_warning_rad,
        )
        paths.append(path)
        print(
            f"    reachable={path['path_reachable']} "
            f"margin_safe={path['margin_safe']} "
            f"min_margin={path['minimum_joint_limit_margin_deg']:.2f} deg",
            flush=True,
        )

    selected = choose_path(paths)
    print(
        "Fast Panthera IK selected "
        f"{selected['target']['part_name']}/"
        f"{selected['target']['orientation_branch']} "
        f"(min margin={selected['minimum_joint_limit_margin_deg']:.2f} deg)",
        flush=True,
    )
    return {
        "schema_version": "panthera-bottle-fast-plan-v1",
        "check_type": "current-seeded Panthera fixed-pose IK",
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
            "lift_mm": float(args.lift_mm),
            "position_tolerance_mm": float(args.position_tolerance_mm),
            "rotation_tolerance_deg": float(args.rotation_tolerance_deg),
            "joint_limit_warning_deg": float(args.joint_limit_warning_deg),
            "ik_max_evaluations": int(args.ik_max_evaluations),
            "max_command_speed_rad_s": float(args.max_command_speed_rad_s),
            "check_fk_after_motion": bool(args.check_fk_after_motion),
            "fk_position_tolerance_mm": float(args.fk_position_tolerance_mm),
            "fk_rotation_tolerance_deg": float(args.fk_rotation_tolerance_deg),
        },
        "limitations": [
            "no collision geometry check is available in the Panthera SDK path",
            "the hand-eye absolute error is not corrected by URDF FK",
            "only PartKep-emitted tool-link opening-axis branches are tried",
            "the current state is a snapshot and is rechecked before motion",
        ],
    }


def read_fresh_q(robot: Any, sample_count: int, sample_delay: float) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
    samples = read_state_samples(robot, sample_count, sample_delay)
    dq = np.asarray([sample["dq_rad_s"] for sample in samples], dtype=np.float64)
    return (
        np.asarray(samples[-1]["q_rad"], dtype=np.float64),
        float(np.max(np.abs(dq))),
        samples,
    )


def move_and_verify(
    robot: Any,
    target: np.ndarray,
    duration: float,
    tolerance_deg: float,
    timeout_margin_s: float,
    sample_count: int,
    sample_delay: float,
    gripper_target=None,
) -> Dict[str, Any]:
    """Move the arm with repeated position-velocity commands and verify it."""
    target = finite_vector(target, "position-velocity target", 6)
    if duration <= 0.0 or not np.isfinite(duration):
        raise ValueError(f"motion duration must be positive and finite, got {duration}")

    q_start, _, _ = read_fresh_q(robot, sample_count, sample_delay)
    command_period = 0.01
    command_count = max(1, int(np.ceil(duration / command_period)))
    command_dt = float(duration) / command_count
    delta = target - q_start
    max_torque = finite_vector(robot.max_torque, "Panthera maximum torque", 6)
    if gripper_target is not None:
        from execute_panthera_pregrasp import command_gripper_mit

    motion_start = time.perf_counter()

    for command_index in range(command_count + 1):
        elapsed = min(float(duration), command_index * command_dt)
        ratio = elapsed / float(duration)
        blend = ratio ** 3 * (10.0 - 15.0 * ratio + 6.0 * ratio ** 2)
        blend_rate = (
            30.0 * ratio ** 2
            - 60.0 * ratio ** 3
            + 30.0 * ratio ** 4
        ) / float(duration)
        position = q_start + delta * blend
        velocity = delta * blend_rate
        accepted = robot.Joint_Pos_Vel(
            position.tolist(),
            velocity.tolist(),
            max_torque,
            iswait=False,
        )
        if not accepted:
            raise RuntimeError(
                f"Panthera repeated Joint_Pos_Vel rejected at {elapsed:.2f}s"
            )
        if gripper_target is not None:
            command_gripper_mit(robot, gripper_target)
        if command_index < command_count:
            deadline = motion_start + (command_index + 1) * command_dt
            remaining = deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)

    # Keep the final arm position commanded briefly so the last target is not
    # lost before the endpoint state is sampled.  Keep the requested gripper
    # target in the same control loop when one is active.
    hold_deadline = time.perf_counter() + max(0.2, min(0.5, timeout_margin_s))
    zero_velocity = np.zeros(6, dtype=np.float64)
    while time.perf_counter() < hold_deadline:
        accepted = robot.Joint_Pos_Vel(
            target.tolist(),
            zero_velocity.tolist(),
            max_torque,
            iswait=False,
        )
        if not accepted:
            raise RuntimeError("Panthera final Joint_Pos_Vel hold was rejected")
        if gripper_target is not None:
            command_gripper_mit(robot, gripper_target)
        time.sleep(command_period)

    q_post, speed, samples = read_fresh_q(robot, sample_count, sample_delay)
    error_deg = float(np.rad2deg(np.max(np.abs(q_post - target))))
    if error_deg > tolerance_deg:
        raise RuntimeError(
            f"Panthera Joint_Pos_Vel endpoint error is {error_deg:.3f} deg "
            f"> {tolerance_deg:.3f} deg"
        )
    return {
        "sdk_moveJ_success": True,
        "control_mode": "repeated_Joint_Pos_Vel",
        "q_target_rad": target.tolist(),
        "q_post_rad": q_post.tolist(),
        "maximum_position_error_deg": error_deg,
        "maximum_post_speed_rad_s": speed,
        "post_state_samples": samples,
        "duration_s": duration,
    }


def hold_gripper_mit(robot: Any, target_pos: float, duration: float) -> Dict[str, Any]:
    """Send one MIT gripper target continuously for a fixed time.

    This deliberately does not read gripper feedback or gate the next stage on
    the reported position.  The deployment task only requires a timed MIT
    close/open command.
    """
    from execute_panthera_pregrasp import (
        GRIPPER_MIT_KD,
        GRIPPER_MIT_KP,
        GRIPPER_MIT_TORQUE,
        GRIPPER_MIT_VELOCITY,
        command_gripper_mit,
    )

    if duration <= 0.0 or not np.isfinite(duration):
        raise ValueError(f"gripper command duration must be positive and finite, got {duration}")
    deadline = time.monotonic() + duration
    command_count = 0
    while time.monotonic() < deadline:
        command_gripper_mit(robot, target_pos)
        command_count += 1
        time.sleep(0.02)
    command_gripper_mit(robot, target_pos)
    command_count += 1
    return {
        "commanded": True,
        "target_pos": float(target_pos),
        "duration_s": float(duration),
        "command_period_s": 0.02,
        "command_count": command_count,
        "feedback_verification": "not_required",
        "mit_velocity": GRIPPER_MIT_VELOCITY,
        "mit_torque": GRIPPER_MIT_TORQUE,
        "kp": GRIPPER_MIT_KP,
        "kd": GRIPPER_MIT_KD,
    }


def check_tool_link_fk_error(
    model: Any,
    frame_id: int,
    q_actual: np.ndarray,
    T_target: np.ndarray,
    position_tolerance_mm: float,
    rotation_tolerance_deg: float,
) -> Dict[str, Any]:
    """Compare measured-joint FK with a commanded base-frame tool pose.

    This measures execution/model consistency only.  It is not an absolute
    measurement of the bottle or camera target; that still requires an
    external reference such as a ruler or a calibrated marker.
    """
    q_actual = finite_vector(q_actual, "FK-check q_actual", 6)
    T_target = validate_transform(T_target, "FK-check target transform")
    T_actual = forward_kinematics(model, frame_id, q_actual)
    position_error_m, rotation_error_rad = pose_errors(T_actual, T_target)
    error_vector_mm = (T_actual[:3, 3] - T_target[:3, 3]) * 1000.0
    position_tolerance_mm = positive_finite(
        position_tolerance_mm, "fk-position-tolerance-mm"
    )
    rotation_tolerance_deg = positive_finite(
        rotation_tolerance_deg, "fk-rotation-tolerance-deg"
    )
    position_error_mm = position_error_m * 1000.0
    rotation_error_deg = float(np.rad2deg(rotation_error_rad))
    return {
        "enabled": True,
        "q_actual_rad": q_actual.tolist(),
        "T_target_base_tool_link": T_target.tolist(),
        "T_fk_base_tool_link": T_actual.tolist(),
        "p_target_base_m": T_target[:3, 3].tolist(),
        "p_fk_base_m": T_actual[:3, 3].tolist(),
        "position_error_vector_base_mm": error_vector_mm.tolist(),
        "position_error_norm_mm": float(position_error_mm),
        "rotation_error_deg": rotation_error_deg,
        "position_tolerance_mm": position_tolerance_mm,
        "rotation_tolerance_deg": rotation_tolerance_deg,
        "passed": bool(
            position_error_mm <= position_tolerance_mm
            and rotation_error_deg <= rotation_tolerance_deg
        ),
        "interpretation": (
            "FK-vs-command consistency only; it does not measure absolute "
            "vision-to-object error"
        ),
    }


def execute_robot_plan(args: argparse.Namespace, robot: Any, plan: Dict[str, Any]) -> int:
    """Execute the already validated q path and guard Ctrl+C recovery."""
    from execute_panthera_pregrasp import (
        GRIPPER_CLOSE_POS,
        GRIPPER_MIT_KD,
        GRIPPER_MIT_KP,
        GRIPPER_MIT_TORQUE,
        GRIPPER_MIT_VELOCITY,
        GRIPPER_OPEN_POS,
        GRIPPER_POSITION_TOLERANCE,
        LIFT_DURATION_S,
        RETURN_TO_ZERO_DURATION_S,
        command_gripper_mit,
        return_to_zero,
    )

    _, lower, upper, velocity_limits = load_robot_config(args.config)
    selected = plan["selected_path"]
    q_plan = np.asarray(plan["q_current_rad"], dtype=np.float64)
    q_pregrasp = finite_vector(selected["q_pregrasp_rad"], "q_pregrasp", 6)
    q_grasp = finite_vector(selected["q_grasp_rad"], "q_grasp", 6)
    q_lift = finite_vector(selected["q_lift_rad"], "q_lift", 6)

    fk_model = None
    fk_frame_id = None
    fk_targets: Dict[str, np.ndarray] = {}
    if args.check_fk_after_motion:
        import pinocchio as pin

        joint_names, _, _, _ = load_robot_config(args.config)
        fk_model = pin.buildModelFromUrdf(str(args.urdf.resolve()))
        fk_frame_id = check_model(fk_model, joint_names)
        pose_checks = selected.get("pose_checks")
        if not isinstance(pose_checks, list):
            raise ValueError("selected path has no pose_checks for optional FK validation")
        for pose in pose_checks:
            if not isinstance(pose, dict) or "label" not in pose:
                raise ValueError("selected path contains an invalid pose check")
            label = str(pose["label"])
            fk_targets[label] = validate_transform(
                pose.get("T_base_tool_link_target"),
                f"{label} FK-check target transform",
            )
        missing_targets = {"pregrasp", "grasp", "lift"}.difference(fk_targets)
        if missing_targets:
            raise ValueError(
                "selected path is missing FK-check targets: "
                + ", ".join(sorted(missing_targets))
            )
        fk_position_tolerance_mm = positive_finite(
            args.fk_position_tolerance_mm, "fk-position-tolerance-mm"
        )
        fk_rotation_tolerance_deg = positive_finite(
            args.fk_rotation_tolerance_deg, "fk-rotation-tolerance-deg"
        )
        print(
            "Optional FK-after-motion check enabled: "
            f"position <= {fk_position_tolerance_mm:.2f} mm, "
            f"rotation <= {fk_rotation_tolerance_deg:.2f} deg"
        )

    q_actual, speed, fresh_samples = read_fresh_q(
        robot, args.state_samples, args.sample_delay
    )
    saved_delta_deg = float(np.rad2deg(np.max(np.abs(q_actual - q_plan))))
    if speed > args.max_state_speed_rad_s:
        raise RuntimeError(f"Panthera is moving before execution: {speed:.6f} rad/s")
    if saved_delta_deg > args.max_saved_state_delta_deg:
        raise RuntimeError(
            f"current state changed by {saved_delta_deg:.3f} deg since IK; replan required"
        )

    pregrasp_duration = duration_for_joint_move(
        q_actual, q_pregrasp, args.pregrasp_duration,
        args.max_command_speed_rad_s, velocity_limits,
    )
    grasp_duration = duration_for_joint_move(
        q_pregrasp, q_grasp, args.grasp_duration,
        args.max_command_speed_rad_s, velocity_limits,
    )
    lift_duration = duration_for_joint_move(
        q_grasp, q_lift, args.lift_duration,
        args.max_command_speed_rad_s, velocity_limits,
    )
    execution: Dict[str, Any] = dict(plan)
    execution.update({
        "hardware_access": True,
        "robot_motion_commanded": False,
        "status": "ready_for_motion",
        "fresh_state_samples_before_motion": fresh_samples,
        "q_fresh_current_rad": q_actual.tolist(),
        "fresh_state_speed_rad_s": speed,
        "saved_state_delta_deg": saved_delta_deg,
        "durations_s": {
            "current_to_pregrasp": pregrasp_duration,
            "pregrasp_to_grasp": grasp_duration,
            "grasp_to_lift": lift_duration,
        },
        "gripper_parameters": {
            "mode": "MIT",
            "open_pos": GRIPPER_OPEN_POS,
            "close_pos": GRIPPER_CLOSE_POS,
            "velocity": GRIPPER_MIT_VELOCITY,
            "torque": GRIPPER_MIT_TORQUE,
            "kp": GRIPPER_MIT_KP,
            "kd": GRIPPER_MIT_KD,
            "position_tolerance": GRIPPER_POSITION_TOLERANCE,
        },
        "return_to_zero_duration_s": RETURN_TO_ZERO_DURATION_S,
        "fk_check": {
            "enabled": bool(args.check_fk_after_motion),
            "position_tolerance_mm": float(args.fk_position_tolerance_mm),
            "rotation_tolerance_deg": float(args.fk_rotation_tolerance_deg),
            "stages": {},
            "interpretation": (
                "FK-vs-command consistency only; it does not measure absolute "
                "vision-to-object error"
            ),
        },
    })
    save_json(args.output, execution)

    print(f"Fresh q_current={np.round(q_actual, 6).tolist()} rad")
    print(
        f"Moving current -> pregrasp over {pregrasp_duration:.2f}s, "
        f"then grasp over {grasp_duration:.2f}s and lift over {lift_duration:.2f}s."
    )
    print("No collision checking is provided by the Panthera SDK; clear the sweep and keep the E-stop reachable.")

    grasp_started = False

    def record_fk_check(stage: str) -> None:
        if not args.check_fk_after_motion:
            return
        if fk_model is None or fk_frame_id is None:
            raise RuntimeError("FK check was enabled but the Panthera model was not loaded")
        motion_key = f"{stage}_motion"
        motion = execution.get(motion_key)
        if not isinstance(motion, dict):
            raise RuntimeError(f"missing motion result for FK check: {motion_key}")
        result = check_tool_link_fk_error(
            model=fk_model,
            frame_id=fk_frame_id,
            q_actual=motion["q_post_rad"],
            T_target=fk_targets[stage],
            position_tolerance_mm=args.fk_position_tolerance_mm,
            rotation_tolerance_deg=args.fk_rotation_tolerance_deg,
        )
        motion["fk_check"] = result
        execution["fk_check"]["stages"][stage] = result
        vector = np.round(result["position_error_vector_base_mm"], 3).tolist()
        print(
            f"FK check {stage}: error_vector_base={vector} mm, "
            f"norm={result['position_error_norm_mm']:.3f} mm, "
            f"rotation={result['rotation_error_deg']:.3f} deg, "
            f"passed={result['passed']}",
            flush=True,
        )
        if not result["passed"]:
            print(
                "WARNING: FK-vs-command error exceeds the diagnostic tolerance; "
                "motion will continue because this check is non-blocking.",
                flush=True,
            )
        save_json(args.output, execution)

    try:
        execution["status"] = "opening_gripper_before_motion"
        save_json(args.output, execution)
        open_deadline = time.monotonic() + 2.0
        while time.monotonic() < open_deadline:
            command_gripper_mit(robot, GRIPPER_OPEN_POS)
            time.sleep(0.02)
        execution["gripper_open_before_motion"] = {
            "commanded": True,
            "target_pos": GRIPPER_OPEN_POS,
            "verification": "not_required",
            "duration_s": 2.0,
            "mit_velocity": GRIPPER_MIT_VELOCITY,
            "mit_torque": GRIPPER_MIT_TORQUE,
            "kp": GRIPPER_MIT_KP,
            "kd": GRIPPER_MIT_KD,
        }
        save_json(args.output, execution)

        execution["robot_motion_commanded"] = True
        execution["status"] = "executing_pregrasp"
        save_json(args.output, execution)
        execution["pregrasp_motion"] = move_and_verify(
            robot, q_pregrasp, pregrasp_duration, args.position_tolerance_deg,
            args.timeout_margin_s, args.state_samples, args.sample_delay,
            gripper_target=GRIPPER_OPEN_POS,
        )
        record_fk_check("pregrasp")
        execution["status"] = "pregrasp_reached"
        save_json(args.output, execution)

        execution["status"] = "executing_grasp"
        save_json(args.output, execution)
        execution["grasp_motion"] = move_and_verify(
            robot, q_grasp, grasp_duration, args.position_tolerance_deg,
            args.timeout_margin_s, args.state_samples, args.sample_delay,
            gripper_target=GRIPPER_OPEN_POS,
        )
        record_fk_check("grasp")

        execution["status"] = "closing_gripper"
        save_json(args.output, execution)
        grasp_started = True
        execution["gripper_close"] = hold_gripper_mit(
            robot, GRIPPER_CLOSE_POS, duration=2.0
        )
        execution["status"] = "executing_lift"
        save_json(args.output, execution)
        execution["lift_motion"] = move_and_verify(
            robot, q_lift, lift_duration, args.position_tolerance_deg,
            args.timeout_margin_s, args.state_samples, args.sample_delay,
            gripper_target=GRIPPER_CLOSE_POS,
        )
        record_fk_check("lift")

        execution["status"] = "holding_lifted_object"
        save_json(args.output, execution)
        print("Object lifted; holding indefinitely. Press Ctrl+C to release and return.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nCtrl+C received: releasing gripper before returning to q=0.")
        execution["status"] = "interrupt_release_before_return"
        save_json(args.output, execution)
        try:
            if grasp_started:
                release_deadline = time.monotonic() + 2.0
                while time.monotonic() < release_deadline:
                    command_gripper_mit(robot, GRIPPER_OPEN_POS)
                    time.sleep(0.02)
                execution["gripper_release"] = {
                    "commanded": True,
                    "target_pos": GRIPPER_OPEN_POS,
                    "verification": "not_required",
                    "duration_s": 2.0,
                    "mit_velocity": GRIPPER_MIT_VELOCITY,
                    "mit_torque": GRIPPER_MIT_TORQUE,
                    "kp": GRIPPER_MIT_KP,
                    "kd": GRIPPER_MIT_KD,
                }
            execution["status"] = "interrupt_returning_to_zero"
            save_json(args.output, execution)
            execution["return_to_zero"] = return_to_zero(robot, {
                "lower": lower,
                "upper": upper,
            })
            if grasp_started:
                execution["gripper_close_after_return"] = hold_gripper_mit(
                    robot, GRIPPER_CLOSE_POS, duration=2.0
                )
            execution["status"] = "interrupted_returned_to_zero"
            save_json(args.output, execution)
            print("Returned to q=0; final gripper close completed.")
            return 130
        except KeyboardInterrupt:
            robot.set_stop()
            execution["status"] = "second_interrupt_stopped"
            save_json(args.output, execution)
            print("\nSecond Ctrl+C received during recovery; motor stop was requested.")
            raise
        except Exception as recovery_error:
            robot.set_stop()
            execution["status"] = "recovery_failed_stopped"
            execution["recovery_error"] = (
                f"{type(recovery_error).__name__}: {recovery_error}"
            )
            save_json(args.output, execution)
            raise RuntimeError(
                f"guarded recovery failed: {recovery_error}"
            ) from recovery_error
    except Exception as exc:
        robot.set_stop()
        execution["status"] = "execution_failed_stopped"
        execution["error"] = f"{type(exc).__name__}: {exc}"
        save_json(args.output, execution)
        print("Motor stop was requested; automatic return was not attempted after an execution error.")
        raise


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


def require_fixture(fixture_dir: Path) -> None:
    required = [
        fixture_dir / "color.png",
        fixture_dir / "depth_raw.npy",
        fixture_dir / "camera_info.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("D435 fixture files are missing: " + ", ".join(missing))


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
            "\n收到 Ctrl+C；交由 Panthera 子进程执行松开夹爪和回零。",
            flush=True,
        )

    signal.signal(signal.SIGINT, parent_handles_interrupt)
    try:
        return int(process.wait())
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def build_robot_child_command(args: argparse.Namespace, pipeline_result: Path, output: Path) -> List[str]:
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
        "--lift-mm",
        str(args.lift_mm),
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
        "--grasp-duration",
        str(args.grasp_duration),
        "--lift-duration",
        str(args.lift_duration),
        "--position-tolerance-deg",
        str(args.position_tolerance_deg),
        "--timeout-margin-s",
        str(args.timeout_margin_s),
        "--fk-position-tolerance-mm",
        str(args.fk_position_tolerance_mm),
        "--fk-rotation-tolerance-deg",
        str(args.fk_rotation_tolerance_deg),
    ]
    if args.execute:
        command.append("--execute")
    if args.check_fk_after_motion:
        command.append("--check-fk-after-motion")
    return command


def run_robot_stage(args: argparse.Namespace) -> int:
    robot = connect_panthera(args.sdk_scripts.resolve(), args.config.resolve())
    try:
        plan = build_robot_plan(args, robot)
        save_json(args.output, plan)
        print(f"Saved fast Panthera plan: {args.output}")
        if not args.execute:
            print("No --execute flag: no arm or gripper motion was commanded.")
            return 0
        print("--execute is present; real Panthera motion is authorized.")
        return execute_robot_plan(args, robot, plan)
    finally:
        if not args.execute:
            del robot


def run_normal_stage(args: argparse.Namespace) -> int:
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new one: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fixture_dir = args.fixture_dir.resolve()
    vision_dir = args.output_dir / "vision"
    plan_path = args.output_dir / "panthera_fast_plan.json"
    status_path = args.output_dir / "session_status.json"
    status: Dict[str, Any] = {
        "schema_version": "panthera-bottle-fast-session-v1",
        "status": "running",
        "started_at_unix_s": time.time(),
        "output_dir": str(args.output_dir.resolve()),
        "execute": bool(args.execute),
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

    print("\n[3/3] Panthera current-seeded IK and optional SDK execution", flush=True)
    robot_command = build_robot_child_command(args, pipeline_result, plan_path)
    code = run_robot_child(robot_command, clean_sdk_environment())
    if code not in (0, 130):
        raise RuntimeError(f"Panthera fast stage failed with exit code {code}")
    status["status"] = "completed" if code == 0 else "operator_shutdown_complete"
    status["finished_at_unix_s"] = time.time()
    status["robot_stage_return_code"] = code
    save_json(status_path, status)
    print(f"Fast Panthera deployment session: {args.output_dir}")
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-stage", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pipeline-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="本次运行输出目录；省略时自动创建带时间戳的唯一目录",
    )
    parser.add_argument(
        "--no-capture",
        action="store_false",
        dest="capture",
        help="跳过自动 D435 采集，直接使用 fixture-dir 中已有数据",
    )
    parser.set_defaults(capture=True)
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
    parser.add_argument("--lift-mm", type=float, default=50.0)
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
    parser.add_argument("--grasp-duration", type=float, default=5.0)
    parser.add_argument("--lift-duration", type=float, default=5.0)
    parser.add_argument("--position-tolerance-deg", type=float, default=1.7)
    parser.add_argument("--timeout-margin-s", type=float, default=8.0)
    parser.add_argument(
        "--check-fk-after-motion",
        action="store_true",
        help="after each move, compare measured-q FK tool_link pose with the commanded target",
    )
    parser.add_argument("--fk-position-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--fk-rotation-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true", help="authorize real arm and gripper motion")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive_finite(args.fk_position_tolerance_mm, "fk-position-tolerance-mm")
    positive_finite(args.fk_rotation_tolerance_deg, "fk-rotation-tolerance-deg")
    if args.robot_stage:
        required = {
            "pipeline-result": args.pipeline_result,
            "output": args.output,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("robot stage is missing: " + ", ".join(missing))
        if not args.pipeline_result.is_file():
            raise FileNotFoundError(f"pipeline result not found: {args.pipeline_result}")
        return
    if args.fixture_dir is None or args.output_dir is None:
        raise ValueError("normal mode requires --fixture-dir")
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
    if args.ik_max_evaluations < 1:
        raise ValueError("ik-max-evaluations must be positive")


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
        print("\nFast Panthera deployment interrupted.", file=sys.stderr)
        return 130
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Fast Panthera deployment failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
