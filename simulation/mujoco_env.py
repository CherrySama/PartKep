"""
simulation/mujoco_env.py
Created by Yinghao Ho on 2026-04-19

MuJoCo 执行层（测试专用）

职责：
    接收 MotionPlanner 生成的关节角 waypoints，
    驱动 Franka Panda 在 MuJoCo 中逐点执行并渲染。

设计原则：
    - execute_trajectory: 逐 waypoint 写 data.ctrl + mj_step，无 sleep
    - move_to: 保留，用于单次关节空间插值（调试/reset 用）
    - 从 keyframe "home" 初始化，与 panda.xml 保持一致
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
from typing import List

# home 关节角，与 panda.xml keyframe 完全一致
Q_HOME = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853])

# 夹爪控制值（actuator8：0=关闭，255=全开）
GRIPPER_OPEN  = 255.0
GRIPPER_CLOSE = 0.0

# execute_trajectory 每个 waypoint 的物理步数
# 越多越平滑，但越慢；5步约等于 0.01s 仿真时间
STEPS_PER_WAYPOINT = 5

# execute_trajectory 每个 waypoint 后的 sleep 时间（秒）
# 这是控制真实播放速度的主要参数
# 0.01 → 约 1s/100waypoints，整段约 3s（n_steps=300 时）
SLEEP_PER_WAYPOINT = 0.01


class MuJoCoEnv:
    """
    Franka Panda MuJoCo 执行环境

    Args:
        scene_xml : scene.xml 路径（相对于项目根目录）
        n_steps   : move_to 插值步数（仅用于 reset/调试）
        step_dt   : move_to 每步 sleep 时间（秒）
    """

    def __init__(
        self,
        scene_xml: str = "assets/franka_emika_panda/scene.xml",
        n_steps:   int   = 150,
        step_dt:   float = 0.005,
    ):
        print("=" * 50)
        print("初始化 MuJoCoEnv")

        self.model   = mujoco.MjModel.from_xml_path(scene_xml)
        self.data    = mujoco.MjData(self.model)
        self.n_steps = n_steps
        self.step_dt = step_dt

        # 重置到 home keyframe + 恢复杯子位置
        self._reset_state()

        # 启动 passive viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance  = 2.0
        self.viewer.cam.azimuth   = 135
        self.viewer.cam.elevation = -20

        print(f"  home 位形: {np.round(Q_HOME, 4)}")
        print("✅ MuJoCoEnv 初始化完成")
        print("=" * 50)

    # ── 核心接口 ──────────────────────────────────────────────────────────────

    def execute_trajectory(self, waypoints: List[np.ndarray]):
        """
        执行 MotionPlanner 生成的轨迹

        每个 waypoint 写入 data.ctrl，推进 STEPS_PER_WAYPOINT 步物理仿真，
        然后同步 viewer。不加 sleep，速度由仿真本身决定。

        Args:
            waypoints: List[np.ndarray]，每个 shape=(7,)，来自 MotionPlanner
        """
        print(f"\n▶ 执行轨迹  ({len(waypoints)} waypoints)")
        for q in waypoints:
            self.data.ctrl[0:7] = q
            self.data.ctrl[7]   = GRIPPER_OPEN
            for _ in range(STEPS_PER_WAYPOINT):
                mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(SLEEP_PER_WAYPOINT)
        print("  ✅ 轨迹执行完成")

    def get_site_xpos(self, site_name: str) -> np.ndarray:
        """
        读取 site 的世界坐标

        Args:
            site_name: scene.xml 中定义的 site 名称

        Returns:
            np.ndarray shape=(3,)
        """
        mujoco.mj_forward(self.model, self.data)
        site_id = self.model.site(site_name).id
        return self.data.site_xpos[site_id].copy()

    def reset(self):
        """重置到 home 位形，恢复杯子位置"""
        self._reset_state()
        self.move_to(Q_HOME, label="home")

    def close(self):
        """关闭 viewer"""
        self.viewer.close()
        print("MuJoCoEnv 已关闭")

    # ── 调试接口 ──────────────────────────────────────────────────────────────

    def move_to(self, q_target: np.ndarray, label: str = ""):
        """
        关节空间线性插值运动到目标位形（调试/reset 用）

        Args:
            q_target: 目标关节角 shape=(7,)
            label:    打印标签
        """
        if label:
            print(f"\n▶ 运动到 [{label}]")

        q_start = self.data.qpos[0:7].copy()
        for i in range(self.n_steps + 1):
            alpha    = i / self.n_steps
            q_interp = (1 - alpha) * q_start + alpha * q_target
            self.data.ctrl[0:7] = q_interp
            self.data.ctrl[7]   = GRIPPER_OPEN
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(self.step_dt)

        print(f"  ✅ 到达 [{label}]")

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _reset_state(self):
        """
        重置仿真状态：
          - panda arm：从 keyframe "home" 读取
          - cup freejoint：恢复到 scene.xml 定义的初始位置
        """
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._reset_cup()
        mujoco.mj_forward(self.model, self.data)

    def _reset_cup(self):
        """
        恢复 cup freejoint 到 scene.xml 初始位置。
        cup body pos="0.5 0.05 0.45"，单位四元数。
        keyframe 只含 panda 的 qpos，不含 freejoint，需手动恢复。
        """
        try:
            jnt_id = self.model.joint('cup_free').id
            adr    = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[adr + 0] = 0.5
            self.data.qpos[adr + 1] = 0.05
            self.data.qpos[adr + 2] = 0.45
            self.data.qpos[adr + 3] = 1.0
            self.data.qpos[adr + 4] = 0.0
            self.data.qpos[adr + 5] = 0.0
            self.data.qpos[adr + 6] = 0.0
        except Exception:
            pass