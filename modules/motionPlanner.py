"""
modules/motionPlanner.py

轨迹规划器（SE3 插值 + MuJoCo-native IK）

plan_pick_place 返回段结构（List[Dict]），每段含：
    'label'       : str，段名称
    'waypoints'   : List[np.ndarray]，关节角序列
    'post_actions': List[str]，段结束后触发的动作

支持的 post_actions：
    'close_gripper'   → MuJoCoEnv._set_gripper(False)
    'activate_weld'   → MuJoCoEnv._set_weld(True)
    'open_gripper'    → MuJoCoEnv._set_gripper(True)
    'deactivate_weld' → MuJoCoEnv._set_weld(False)

完整 pick-place 运动序列：
    HOME → pick_above → pick
        [close_gripper, activate_weld]（weld 激活后物理稳定 50 步）
    pick → pick_above → place_above → place
        [open_gripper, deactivate_weld]
    place → HOME
"""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from typing import List, Dict, Optional

from modules.IKSolver import IKSolver, Q_HOME


# ── 常量 ──────────────────────────────────────────────────────────────────────

N_STEPS_PER_SEGMENT = 100   # 每段插值步数
LIFT_HEIGHT         = 0.15  # pick/place 上方抬升高度（米）
EE_BODY_NAME        = "hand"


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _lift_T(T: np.ndarray, dz: float) -> np.ndarray:
    """世界坐标系 Z 方向平移，旋转不变"""
    T_lifted      = T.copy()
    T_lifted[2, 3] += dz
    return T_lifted


def _interpolate_poses(T_start: np.ndarray,
                       T_end:   np.ndarray,
                       n_steps: int) -> List[np.ndarray]:
    """
    SE3 插值：位置线性 + 旋转 SLERP

    返回 n_steps 个位姿，t ∈ (0, 1]（不含起点，含终点）
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
        T       = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = p
        poses.append(T)
    return poses


# ── 主类 ──────────────────────────────────────────────────────────────────────

class MotionPlanner:
    """
    轨迹规划器（SE3 插值 + MuJoCo-native IK）

    plan_to_pose   → List[np.ndarray]（平铺 waypoints，用于单段）
    plan_pick_place → List[Dict]（段结构，用于完整 pick-place）
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
            print("=" * 55)
            print("初始化 MotionPlanner（SE3 插值 + MuJoCo IK）")
            print(f"  末端帧  : {EE_BODY_NAME} (body)")
            print(f"  每段步数: {n_steps}")
            print(f"  抬升高度: {lift_height * 100:.0f} cm")
            print("=" * 55)

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def plan_to_pose(
        self,
        q_start:    np.ndarray,
        T_target:   np.ndarray,
        n_restarts: int = 1,
    ) -> List[np.ndarray]:
        """
        从 q_start 到 T_target 的关节角 waypoints

        Args:
            q_start:    shape=(7,) 起始关节角
            T_target:   shape=(4,4) 目标末端位姿
            n_restarts: IK 随机重启次数

        Returns:
            List[np.ndarray]，每个 shape=(7,)
        """
        T_start    = self._ik.forward_kinematics(q_start)
        poses      = _interpolate_poses(T_start, T_target, self.n_steps)
        waypoints  = []
        q_current  = q_start.copy()
        n_failures = 0

        for T_interp in poses:
            result = self._ik.solve(T_interp, q_init=q_current,
                                    n_restarts=n_restarts)
            # ← 修复：用 success 而非 within_limits
            if result['success']:
                q_current = result['q']
            else:
                n_failures += 1
            waypoints.append(q_current.copy())

        if n_failures > 0:
            print(f"  ⚠️  plan_to_pose: {n_failures}/{self.n_steps} 步 IK 失败，"
                  f"已沿用上一步配置")
        return waypoints

    def plan_pick_place(
        self,
        T_pick:     np.ndarray,
        T_place:    np.ndarray,
        q_home:     Optional[np.ndarray] = None,
        n_restarts: int = 10,
    ) -> List[Dict]:
        """
        完整 pick-place 段结构

        运动序列：
            HOME → pick_above → pick
            pick → pick_above → place_above → place
            place → HOME

        Returns:
            List[Dict]，每个 Dict：
                'label'       : str
                'waypoints'   : List[np.ndarray shape=(7,)]
                'post_actions': List[str]
        """
        if q_home is None:
            q_home = Q_HOME.copy()

        T_pick_above  = _lift_T(T_pick,  self.lift_height)
        T_place_above = _lift_T(T_place, self.lift_height)
        T_home        = self._ik.forward_kinematics(q_home)

        if self.verbose:
            print("\n📐 MotionPlanner 规划 pick-place 序列")
            print(f"  pick  位置: {np.round(T_pick[:3, 3], 3)}")
            print(f"  place 位置: {np.round(T_place[:3, 3], 3)}")

        # 段定义：(label, T_target, post_actions)
        # q_start 由上一段末尾的 q 自动传递
        segment_defs = [
            ("home→pick_above",        T_pick_above,  []),
            ("pick_above→pick",         T_pick,        ['close_gripper', 'activate_weld']),
            ("pick→pick_above",         T_pick_above,  []),
            ("pick_above→place_above",  T_place_above, []),
            ("place_above→place",       T_place,       ['open_gripper', 'deactivate_weld']),
            ("place→home",              T_home,        []),
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
                pa_str    = f"  → {post_actions}" if post_actions else ""
                print(f"  ✅ [{label}]  末端: {np.round(pos_final, 3)}{pa_str}")

        total = sum(len(s['waypoints']) for s in segments)
        if self.verbose:
            print(f"  总 waypoints: {total}（{len(segments)} 段）")

        return segments