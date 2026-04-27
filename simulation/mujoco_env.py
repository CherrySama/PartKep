"""
simulation/mujoco_env.py
Created by Yinghao Ho on 2026-04-19

MuJoCo 执行层

职责：
    接收 MotionPlanner 生成的段结构（List[Dict]），
    驱动 Franka Panda 在 MuJoCo 中逐段执行、渲染，
    并在正确时序下触发夹爪和 weld 约束。

主要接口：
    execute_pick_place(segments) : 执行完整 pick-place 段结构
    execute_trajectory(waypoints): 执行平铺 waypoints（向后兼容）
    get_site_xpos(site_name)     : 读取 site 世界坐标
    reset()                      : 重置到 home 位形
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
from typing import List, Dict

Q_HOME = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853])

GRIPPER_OPEN  = 255.0
GRIPPER_CLOSE = 0.0

STEPS_PER_WAYPOINT  = 5      # 每个 waypoint 推进的物理步数
SLEEP_PER_WAYPOINT  = 0.01   # 每个 waypoint 后 sleep（秒），控制播放速度
WELD_SETTLE_STEPS   = 50     # weld 激活后稳定物理的步数


class MuJoCoEnv:
    """
    Franka Panda MuJoCo 执行环境

    Args:
        scene_xml : scene.xml 路径
        n_steps   : move_to 插值步数（调试用）
        step_dt   : move_to 每步 sleep（秒）
    """

    def __init__(
        self,
        scene_xml: str   = "assets/franka_emika_panda/scene.xml",
        n_steps:   int   = 150,
        step_dt:   float = 0.005,
    ):
        print("=" * 50)
        print("初始化 MuJoCoEnv")

        self.model   = mujoco.MjModel.from_xml_path(scene_xml)
        self.data    = mujoco.MjData(self.model)
        self.n_steps = n_steps
        self.step_dt = step_dt

        self._reset_state()

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance  = 2.0
        self.viewer.cam.azimuth   = 135
        self.viewer.cam.elevation = -20

        print(f"  home 位形: {np.round(Q_HOME, 4)}")
        print("✅ MuJoCoEnv 初始化完成")
        print("=" * 50)

    # ── 核心接口 ──────────────────────────────────────────────────────────────

    def execute_pick_place(self, segments: List[Dict]):
        """
        按段结构执行完整 pick-place 动作序列

        每段执行完毕后，依次触发 post_actions：
            'close_gripper'   → 夹爪关闭
            'activate_weld'   → cup_weld 激活 + 物理稳定 WELD_SETTLE_STEPS 步
            'open_gripper'    → 夹爪张开
            'deactivate_weld' → cup_weld 解除

        Args:
            segments: MotionPlanner.plan_pick_place() 的返回值
                      List[Dict]，每个 Dict 含 'label','waypoints','post_actions'
        """
        total = sum(len(s['waypoints']) for s in segments)
        print(f"\n▶ 执行 pick-place 序列  ({len(segments)} 段，{total} waypoints)")

        for seg in segments:
            label       = seg['label']
            waypoints   = seg['waypoints']
            post_actions = seg['post_actions']

            print(f"  → [{label}]  ({len(waypoints)} waypoints)")
            for q in waypoints:
                self.data.ctrl[:7] = q
                for _ in range(STEPS_PER_WAYPOINT):
                    mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                time.sleep(SLEEP_PER_WAYPOINT)

            # 段结束后触发 post_actions
            for action in post_actions:
                if action == 'close_gripper':
                    self._set_gripper(open=False)
                    print(f"    🤏 夹爪关闭")
                elif action == 'activate_weld':
                    self._set_weld(active=True)
                    print(f"    🔗 cup_weld 激活，稳定 {WELD_SETTLE_STEPS} 步...")
                    self._settle(WELD_SETTLE_STEPS)
                elif action == 'open_gripper':
                    self._set_gripper(open=True)
                    print(f"    ✋ 夹爪张开")
                elif action == 'deactivate_weld':
                    self._set_weld(active=False)
                    print(f"    🔓 cup_weld 解除")
                else:
                    print(f"    ⚠️  未知 post_action: {action}")

        print("  ✅ pick-place 序列执行完成")

    def execute_trajectory(self, waypoints: List[np.ndarray]):
        """
        执行平铺 waypoints（向后兼容，夹爪全程保持张开）

        Args:
            waypoints: List[np.ndarray]，每个 shape=(7,)
        """
        print(f"\n▶ 执行轨迹  ({len(waypoints)} waypoints)")
        for q in waypoints:
            self.data.ctrl[:7] = q
            self.data.ctrl[7]  = GRIPPER_OPEN
            for _ in range(STEPS_PER_WAYPOINT):
                mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(SLEEP_PER_WAYPOINT)
        print("  ✅ 轨迹执行完成")

    def get_site_xpos(self, site_name: str) -> np.ndarray:
        """读取 site 世界坐标"""
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

    # ── 夹爪 & weld 接口 ──────────────────────────────────────────────────────

    def _set_gripper(self, open: bool):
        """
        设置夹爪开/关

        Args:
            open: True → 张开（GRIPPER_OPEN），False → 关闭（GRIPPER_CLOSE）
        """
        self.data.ctrl[7] = GRIPPER_OPEN if open else GRIPPER_CLOSE
        # 推进几步让夹爪物理到位
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    def _set_weld(self, active: bool):
        """
        激活/解除 cup_weld equality 约束

        使用 model.equality('cup_weld').id 查找，避免硬编码 index。
        panda.xml 包含手指 mimic 约束，cup_weld 不在 index 0。

        Args:
            active: True → 激活约束（杯子跟随手臂），False → 解除
        """
        weld_id = self.model.equality('cup_weld').id
        self.data.eq_active[weld_id] = active
        # 让约束求解器立即感知状态变化
        mujoco.mj_forward(self.model, self.data)

    def _settle(self, n_steps: int):
        """
        推进 n_steps 步物理仿真（ctrl 不变），让约束/接触稳定

        用于 weld 激活后消除初始抖动。

        Args:
            n_steps: 稳定步数（建议 50）
        """
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

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

        q_start = self.data.qpos[:7].copy()
        for i in range(self.n_steps + 1):
            alpha               = i / self.n_steps
            q_interp            = (1 - alpha) * q_start + alpha * q_target
            self.data.ctrl[:7]  = q_interp
            self.data.ctrl[7]   = GRIPPER_OPEN
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(self.step_dt)

        print(f"  ✅ 到达 [{label}]")

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _reset_state(self):
        """重置仿真状态：arm 从 keyframe 0，cup 恢复初始位置"""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._reset_cup()
        mujoco.mj_forward(self.model, self.data)

    def _reset_cup(self):
        """
        恢复 cup freejoint 到初始位置
        cup body pos="0.5 0.05 0.45"，单位四元数
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