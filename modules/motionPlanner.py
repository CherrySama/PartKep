"""
Created by Yinghao Ho on 2026-2-23

Trajectory planner: SE3 interpolation + MuJoCo-native IK.

plan_pick_place returns a list of segment dicts, each with:
    'label'       : str, segment name
    'waypoints'   : List[np.ndarray], joint angle sequence
    'post_actions': List[str], actions triggered at segment end

Supported post_actions:
    'close_gripper'   -> MuJoCoEnv._set_gripper(False)
    'activate_weld'   -> MuJoCoEnv._set_weld(True)
    'open_gripper'    -> MuJoCoEnv._set_gripper(True)
    'deactivate_weld' -> MuJoCoEnv._set_weld(False)

Full pick-place motion sequence:
    HOME -> pick_above -> pick   [close_gripper, activate_weld]
    pick -> pick_above -> place_above -> place   [open_gripper, deactivate_weld]
    place -> HOME
"""

from typing import List, Dict, Optional

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp

from modules.IKSolver import IKSolver, Q_HOME


N_STEPS_PER_SEGMENT = 100   # interpolation steps per segment
LIFT_HEIGHT         = 0.15  # lift height above pick/place target (metres)
EE_BODY_NAME        = "hand"


def _compute_prepose(T: np.ndarray, approach_dir: np.ndarray, distance: float) -> np.ndarray:
    """
    Compute pre-pose by retreating along approach_dir by distance, rotation unchanged.

    For top-down parts (approach=[0,0,-1]): equivalent to lifting by distance along +Z.
    For lateral parts (approach=[0,-1,0]):  retreats outward along +Y, avoiding obstacles.
    """
    T_pre = T.copy()
    T_pre[:3, 3] = T[:3, 3] - approach_dir * distance
    return T_pre


def _interpolate_poses(T_start: np.ndarray,
                       T_end:   np.ndarray,
                       n_steps: int) -> List[np.ndarray]:
    """
    SE3 interpolation: linear position + SLERP rotation.
    Returns n_steps poses at t in (0, 1] (start excluded, end included).
    """
    p_start = T_start[:3, 3]
    p_end   = T_end[:3, 3]
    R_start = Rotation.from_matrix(T_start[:3, :3])
    R_end   = Rotation.from_matrix(T_end[:3, :3])
    slerp   = Slerp([0.0, 1.0], Rotation.concatenate([R_start, R_end]))

    poses = []
    for i in range(1, n_steps + 1):
        t = i / n_steps
        p = (1.0 - t) * p_start + t * p_end
        R = slerp(t).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = p
        poses.append(T)
    return poses


class MotionPlanner:
    """
    Trajectory planner: SE3 interpolation + MuJoCo-native IK.

    plan_to_pose    -> List[np.ndarray], flat waypoint list for a single segment.
    plan_pick_place -> List[Dict], full segment structure for pick-place.
    """

    def __init__(
        self,
        model:       mujoco.MjModel,
        n_steps:     int   = N_STEPS_PER_SEGMENT,
        lift_height: float = LIFT_HEIGHT,
        verbose:     bool  = True,
    ):
        self.model       = model
        self.n_steps     = n_steps
        self.lift_height = lift_height
        self.verbose     = verbose
        self._ik         = IKSolver(model, verbose=False)

        if verbose:
            print(f"[MotionPlanner] ee={EE_BODY_NAME}  "
                  f"steps/seg={n_steps}  lift={lift_height * 100:.0f}cm")

    def plan_to_pose(
        self,
        q_start:    np.ndarray,
        T_target:   np.ndarray,
        n_restarts: int = 1,
        q_target:   Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Joint angle waypoints from q_start to T_target.

        If q_target is provided, uses joint-space linear interpolation between
        q_start and q_target (guaranteed smooth and correct endpoint).
        Otherwise, uses SE3 Cartesian interpolation with per-step IK.

        If an IK step fails (SE3 path), the previous q is reused.
        """
        if q_target is not None:
            # joint-space linear interpolation: smooth and exact at both ends
            waypoints = []
            for i in range(1, self.n_steps + 1):
                t = i / self.n_steps
                waypoints.append((1.0 - t) * q_start + t * q_target)
            return waypoints

        T_start   = self._ik.forward_kinematics(q_start)
        poses     = _interpolate_poses(T_start, T_target, self.n_steps)
        waypoints = []
        q_current = q_start.copy()
        n_failures = 0

        for T_interp in poses:
            result = self._ik.solve(T_interp, q_init=q_current, n_restarts=n_restarts)
            if result['success']:  # use 'success', not 'within_limits'
                q_current = result['q']
            else:
                n_failures += 1
            waypoints.append(q_current.copy())

        if n_failures > 0:
            print(f"[MotionPlanner] plan_to_pose: {n_failures}/{self.n_steps} IK steps failed, "
                  f"previous q reused")
        return waypoints

    def plan_pick_place(
        self,
        T_pick:             np.ndarray,
        T_place:            np.ndarray,
        approach_dir_pick:  np.ndarray,
        approach_dir_place: np.ndarray = np.array([0.0, 0.0, -1.0]),
        q_home:             Optional[np.ndarray] = None,
        n_restarts:         int = 10,
    ) -> List[Dict]:
        """
        Full pick-place segment structure.
        Motion sequence: HOME -> pick_above -> pick -> pick_above -> place_above -> place -> HOME.

        pick_above / place_above are computed by retreating along the respective approach
        direction, so the pre-pose is always outside the object regardless of approach axis.

        Args:
            approach_dir_pick:  from meta_pick['approach_direction']
            approach_dir_place: from meta_place['approach_direction'], defaults to [0,0,-1]

        Returns List[Dict] with keys: label, waypoints, post_actions.
        """
        if q_home is None:
            q_home = Q_HOME.copy()

        T_pick_above  = _compute_prepose(T_pick,  approach_dir_pick,  self.lift_height)
        T_place_above = _compute_prepose(T_place, approach_dir_place, self.lift_height)
        T_home        = self._ik.forward_kinematics(q_home)

        # safe_above: pick_above position but at home Z height.
        # Ensures HOME -> safe_above only changes XY/rotation while staying high,
        # then safe_above -> pick_above descends vertically outside the object.
        safe_z        = T_home[2, 3]
        T_safe_above  = T_pick_above.copy()
        T_safe_above[2, 3] = max(T_pick_above[2, 3], safe_z)

        if self.verbose:
            print(f"[MotionPlanner] pick={np.round(T_pick[:3, 3], 3)}  "
                  f"place={np.round(T_place[:3, 3], 3)}")

        segment_defs = [
            ("home->safe_above",       T_safe_above,  []),
            ("safe_above->pick_above", T_pick_above,  []),
            ("pick_above->pick",       T_pick,        ['close_gripper', 'activate_weld']),
            ("pick->pick_above",       T_pick_above,  []),
            ("pick_above->place_above",T_place_above, []),
            ("place_above->place",     T_place,       ['open_gripper', 'deactivate_weld']),
            ("place->home",            T_home,        []),
        ]

        segments  = []
        q_current = q_home.copy()

        for label, T_target, post_actions in segment_defs:
            wps = self.plan_to_pose(q_current, T_target, n_restarts=n_restarts)
            segments.append({
                'label':        label,
                'waypoints':    wps,
                'post_actions': post_actions,
            })
            q_current = wps[-1]

            if self.verbose:
                pos_final = self._ik.forward_kinematics(q_current)[:3, 3]
                pa_str    = f"  -> {post_actions}" if post_actions else ""
                print(f"[MotionPlanner] [{label}] tcp={np.round(pos_final, 3)}{pa_str}")

        total = sum(len(s['waypoints']) for s in segments)
        if self.verbose:
            print(f"[MotionPlanner] total waypoints={total} ({len(segments)} segments)")

        return segments