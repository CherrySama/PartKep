"""
modules/motionPlanner.py
Created by Yinghao Ho on 2026-04-19

轨迹规划器（基于 mink 微分 IK）

职责：
    接收 IKSolver 输出的末端位姿 T，生成从当前配置到目标位姿的
    平滑关节角轨迹（waypoints 列表），供 MuJoCo 执行层逐点执行。

当前实现：mink 微分 IK
    - 适用于简单桌面场景（无复杂障碍物）
    - pip install mink
    - 未来可替换为 OMPL + MuJoCo 碰撞检测（复杂障碍场景）

末端执行器帧：
    frame_name="hand", frame_type="body"
    与 panda.xml kinematic chain 末端完全一致，
    与 IKSolver 的 URDF 末端帧（hand_joint 固定关节后的 hand link）吻合。

完整 pick-place 运动序列（plan_pick_place）：
    HOME → T_pick_above → T_pick → T_pick_above → T_place_above → T_place → HOME
    其中 T_above = T + [0, 0, LIFT_HEIGHT]（末端在 Z 方向抬升）
"""

import numpy as np
import mujoco
import mink
from scipy.spatial.transform import Rotation
from typing import List, Optional

from modules.IKSolver import Q_HOME


# ── 常量 ──────────────────────────────────────────────────────────────────────

# 每段轨迹的 IK 迭代步数
# gain=1.0 + n_steps=100 → 约 100 步内收敛，速度快
N_STEPS_PER_SEGMENT = 100

# 微分 IK 的时间步长（影响每步的关节角增量大小，不影响仿真时间）
DT = 0.01

# pick / place 上方抬升高度（米）
LIFT_HEIGHT = 0.15

# 末端执行器 body 名称（与 panda.xml 完全对应）
EE_BODY_NAME = "hand"


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _matrix_to_mink_se3(T: np.ndarray) -> mink.SE3:
    """
    将 4×4 齐次变换矩阵转换为 mink SE3 对象

    使用 scipy 将旋转矩阵转为轴角，再用 SO3.exp 构造，
    避免依赖 SO3.from_matrix（该接口在部分 mink 版本中不存在）。

    Args:
        T: shape=(4,4) 齐次变换矩阵

    Returns:
        mink.SE3 对象
    """
    R = T[:3, :3]
    t = T[:3, 3]
    rotvec = Rotation.from_matrix(R).as_rotvec()  # 轴角向量
    so3 = mink.SO3.exp(rotvec)
    return mink.SE3.from_rotation_and_translation(so3, t)


def _lift_T(T: np.ndarray, dz: float) -> np.ndarray:
    """
    在世界坐标系 Z 方向对 T 的位置做偏移（旋转不变）

    Args:
        T:  shape=(4,4) 原始末端位姿
        dz: Z 方向偏移量（米），正值向上

    Returns:
        shape=(4,4) 抬升后的位姿矩阵
    """
    T_lifted = T.copy()
    T_lifted[2, 3] += dz
    return T_lifted


# ── 主类 ──────────────────────────────────────────────────────────────────────

class MotionPlanner:
    """
    轨迹规划器

    使用 mink 微分 IK 离线生成关节角轨迹，不依赖物理仿真。
    生成的 waypoints 列表交给 mujoco_env.execute_trajectory() 执行。

    使用示例：
        >>> planner = MotionPlanner(model)
        >>> waypoints = planner.plan_pick_place(T_pick, T_place)
        >>> env.execute_trajectory(waypoints)
    """

    def __init__(
        self,
        model:         mujoco.MjModel,
        n_steps:       int   = N_STEPS_PER_SEGMENT,
        dt:            float = DT,
        lift_height:   float = LIFT_HEIGHT,
        verbose:       bool  = True,
    ):
        """
        Args:
            model:       MuJoCo 模型（来自 scene.xml）
            n_steps:     每段轨迹的 IK 迭代步数
            dt:          微分 IK 时间步长
            lift_height: pick/place 抬升高度（米）
            verbose:     是否打印规划信息
        """
        self.model       = model
        self.n_steps     = n_steps
        self.dt          = dt
        self.lift_height = lift_height
        self.verbose     = verbose

        if verbose:
            print("=" * 55)
            print("初始化 MotionPlanner（mink 微分 IK）")
            print(f"  末端帧  : {EE_BODY_NAME} (body)")
            print(f"  每段步数: {n_steps}  dt={dt}")
            print(f"  抬升高度: {lift_height*100:.0f} cm")
            print("=" * 55)

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def plan_to_pose(
        self,
        q_start:  np.ndarray,
        T_target: np.ndarray,
    ) -> List[np.ndarray]:
        """
        从 q_start 出发，生成到达 T_target 的关节角轨迹

        Args:
            q_start:  shape=(7,) 起始关节角（rad）
            T_target: shape=(4,4) 目标末端位姿（hand body 坐标系）

        Returns:
            List[np.ndarray]: 关节角 waypoints，每个 shape=(7,)
        """
        # ── 初始化 mink Configuration ──
        # 从 keyframe "home" 重置完整状态（保证手指/杯子等 DOF 合法），
        # 再覆盖 arm 的 qpos，保证起始位形正确。
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, 0)
        data.qpos[:7] = q_start
        mujoco.mj_forward(self.model, data)
        configuration = mink.Configuration(self.model, data)

        # ── 构造目标 SE3 ──
        target_se3 = _matrix_to_mink_se3(T_target)

        # ── 创建末端执行器任务（6DOF，位置+姿态全约束）──
        task = mink.FrameTask(
            frame_name=EE_BODY_NAME,
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
            gain=1.0,            # 最快收敛
        )
        task.set_target(target_se3)

        # ── 添加关节限位约束 ──
        limits = [mink.ConfigurationLimit(self.model)]

        # ── 微分 IK 循环（离线，不推进物理）──
        waypoints = []
        for _ in range(self.n_steps):
            vel = mink.solve_ik(configuration, [task], self.dt, limits=limits)
            configuration.integrate_inplace(vel, self.dt)
            waypoints.append(configuration.q[:7].copy())

        return waypoints

    def plan_pick_place(
        self,
        T_pick:  np.ndarray,
        T_place: np.ndarray,
        q_home:  Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        生成完整 pick-place 运动序列的 waypoints

        运动序列：
            HOME → T_pick_above → T_pick
                 → T_pick_above → T_place_above → T_place
                 → HOME

        Args:
            T_pick:  shape=(4,4) pick 目标位姿
            T_place: shape=(4,4) place 目标位姿
            q_home:  shape=(7,) home 关节角，None 时使用 Q_HOME

        Returns:
            平铺的关节角 waypoints 列表（所有段连接在一起）
        """
        if q_home is None:
            q_home = Q_HOME.copy()

        T_pick_above  = _lift_T(T_pick,  self.lift_height)
        T_place_above = _lift_T(T_place, self.lift_height)

        # FK(Q_HOME) 作为 home 末端位姿
        from modules.IKSolver import IKSolver
        _ik = IKSolver(verbose=False)
        T_home = _ik.forward_kinematics(q_home)

        if self.verbose:
            print("\n📐 MotionPlanner 规划 pick-place 序列")
            print(f"  pick  位置: {np.round(T_pick[:3, 3], 3)}")
            print(f"  place 位置: {np.round(T_place[:3, 3], 3)}")

        segments = [
            ("home → pick_above",   q_home,                     T_pick_above),
            ("pick_above → pick",   None,                        T_pick),
            ("pick → pick_above",   None,                        T_pick_above),
            ("pick_above → place_above", None,                   T_place_above),
            ("place_above → place", None,                        T_place),
            ("place → home",        None,                        T_home),
        ]

        all_waypoints = []
        q_current = q_home.copy()

        for label, q_start_override, T_target in segments:
            q_start = q_start_override if q_start_override is not None else q_current
            wps = self.plan_to_pose(q_start, T_target)
            all_waypoints.extend(wps)
            q_current = wps[-1]  # 下一段从本段末尾出发

            if self.verbose:
                pos_final = _ik.forward_kinematics(q_current)[:3, 3]
                print(f"  ✅ [{label}]  末端: {np.round(pos_final, 3)}")

        if self.verbose:
            print(f"  总 waypoints: {len(all_waypoints)}")

        return all_waypoints
        