"""
Created by Yinghao Ho on 2026-2-23

Franka Panda IK solver (MuJoCo-native).
Uses scipy L-BFGS-B + mujoco.mj_forward with random restarts.
Two-stage solve: position-only warmup first, then full pos+rot optimisation.

End-effector frame: MuJoCo body "hand".

qpos layout (scene.xml):
    [0:7]  -> joint1~joint7 (arm)
    [7:9]  -> finger_joint1, finger_joint2
    [9:16] -> cup_free (freejoint)
"""

import sys
from typing import Optional, Dict

import numpy as np
import mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


PANDA_JOINT_LIMITS = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
])

Q_HOME = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853])

_W_POS = 1000.0  # position error weight (dominant to prevent rotation gradient from pulling position off)
_W_ROT = 1.0     # rotation Frobenius error weight


class IKSolver:
    """MuJoCo-native IK solver for Franka Panda. Holds a private MjData to avoid polluting env state."""

    def __init__(self, model: mujoco.MjModel, verbose: bool = True):
        self._model  = model
        self._data   = mujoco.MjData(model)
        self.verbose = verbose

        # initialise from keyframe 0 so finger and freejoint qpos are valid
        mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        mujoco.mj_forward(self._model, self._data)

        self._hand_id = self._model.body('hand').id

        if verbose:
            T0 = self.forward_kinematics(Q_HOME)
            print(f"[IKSolver] hand id={self._hand_id}  home TCP={np.round(T0[:3, 3], 4)}")

    def forward_kinematics(self, q7: np.ndarray) -> np.ndarray:
        """7 joint angles -> 4x4 homogeneous transform of hand body."""
        self._data.qpos[:7] = q7
        mujoco.mj_forward(self._model, self._data)
        pos = self._data.xpos[self._hand_id].copy()
        R   = self._data.xmat[self._hand_id].reshape(3, 3).copy()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = pos
        return T

    def solve(self,
              target_T:   np.ndarray,
              q_init:     Optional[np.ndarray] = None,
              n_restarts: int = 10) -> Dict:
        """
        IK solve: target 4x4 T -> 7 joint angles.

        Stage 1: position-only warmup from q_init to get q_warm aligned to target position.
        Stage 2: full pos+rot optimisation with n_restarts random perturbations around q_warm.
        Returns the lowest-error successful result, or the best failed result if none succeed.

        Returns dict: success, q, position_error, within_limits, fk_position.
        """
        if q_init is None:
            q_init = Q_HOME.copy()

        target_T = np.array(target_T, dtype=np.float64)
        p_target = target_T[:3, 3]
        R_target = target_T[:3, :3]
        bounds   = [(lo, hi) for lo, hi in PANDA_JOINT_LIMITS]

        if self.verbose:
            print(f"[IKSolver] target pos={np.round(p_target, 4)}  max_restarts={n_restarts}")

        def _cost_pos_only(q7: np.ndarray) -> float:
            self._data.qpos[:7] = q7
            mujoco.mj_forward(self._model, self._data)
            p = self._data.xpos[self._hand_id]
            return float(np.sum((p - p_target) ** 2))

        def _cost(q7: np.ndarray) -> float:
            self._data.qpos[:7] = q7
            mujoco.mj_forward(self._model, self._data)
            p = self._data.xpos[self._hand_id]
            R = self._data.xmat[self._hand_id].reshape(3, 3)
            return _W_POS * np.sum((p - p_target) ** 2) + _W_ROT * np.sum((R - R_target) ** 2)

        # stage 1: position-only warmup
        res_warm = minimize(
            fun     = _cost_pos_only,
            x0      = q_init.copy(),
            method  = 'L-BFGS-B',
            bounds  = bounds,
            options = {'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10},
        )
        q_warm = res_warm.x
        print(f"[IKSolver] warm pos error: "
              f"{np.linalg.norm(self.forward_kinematics(q_warm)[:3, 3] - p_target) * 1000:.2f} mm")

        T_warm         = self.forward_kinematics(q_warm)
        err_warm       = float(np.linalg.norm(T_warm[:3, 3] - p_target))
        in_limits_warm = bool(np.all(
            (q_warm >= PANDA_JOINT_LIMITS[:, 0]) &
            (q_warm <= PANDA_JOINT_LIMITS[:, 1])
        ))
        best = {
            'success':        (err_warm < 0.005) and in_limits_warm,
            'q':              q_warm,
            'position_error': err_warm,
            'within_limits':  in_limits_warm,
            'fk_position':    T_warm[:3, 3],
        }

        # stage 2: full optimisation with restarts around q_warm
        for attempt in range(n_restarts):
            if attempt == 0:
                q0 = q_warm.copy()
            else:
                q0 = q_warm.copy()
                q0 += np.random.normal(0, 0.3, size=7)
                q0[3] = np.clip(q0[3], -3.0, -0.1)  # joint4 must stay negative
                q0 = np.clip(q0, PANDA_JOINT_LIMITS[:, 0], PANDA_JOINT_LIMITS[:, 1])

            res = minimize(
                fun     = _cost,
                x0      = q0,
                method  = 'L-BFGS-B',
                bounds  = bounds,
                options = {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8},
            )

            q7_result = res.x
            T_fk      = self.forward_kinematics(q7_result)
            pos_err   = float(np.linalg.norm(T_fk[:3, 3] - p_target))
            in_limits = bool(np.all(
                (q7_result >= PANDA_JOINT_LIMITS[:, 0]) &
                (q7_result <= PANDA_JOINT_LIMITS[:, 1])
            ))
            success = (pos_err < 0.005) and in_limits

            candidate = {
                'success':        success,
                'q':              q7_result,
                'position_error': pos_err,
                'within_limits':  in_limits,
                'fk_position':    T_fk[:3, 3],
            }

            if best is None or pos_err < best['position_error']:
                best = candidate

            if success:
                if self.verbose:
                    print(f"[IKSolver] attempt {attempt + 1} succeeded, "
                          f"error={pos_err * 1000:.2f} mm")
                return best

        if self.verbose:
            print(f"[IKSolver] all attempts failed, "
                  f"best error={best['position_error'] * 1000:.2f} mm  "
                  f"within_limits={best['within_limits']}")

        return best


if __name__ == "__main__":
    sys.path.insert(0, '.')

    SCENE_XML = "assets/franka_emika_panda/scene.xml"
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    ik    = IKSolver(model, verbose=True)

    T0 = ik.forward_kinematics(Q_HOME)
    print(f"home TCP: {np.round(T0[:3, 3], 4)}")

    result = ik.solve(T0, q_init=Q_HOME, n_restarts=10)
    assert result['position_error'] < 0.001, \
        f"round-trip error too large: {result['position_error'] * 1000:.2f} mm"
    print(f"round-trip error: {result['position_error'] * 1000:.3f} mm")

    R_pick = Rotation.from_rotvec([1.057, 1.4206, 1.4206]).as_matrix()
    T_pick = np.eye(4)
    T_pick[:3, :3] = R_pick
    T_pick[:3,  3] = [0.4713, 0.0204, 0.45]
    result2 = ik.solve(T_pick, q_init=Q_HOME, n_restarts=10)
    print(f"T_pick error={result2['position_error'] * 1000:.2f} mm  "
          f"success={result2['success']}  within_limits={result2['within_limits']}")