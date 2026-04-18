"""
simulation/mujoco_env.py

轻量 MuJoCo 执行层（测试专用）
职责：接收关节角 q，驱动 Franka Panda 运动到目标位形并渲染。

设计原则：
    - 只做关节空间线性插值运动，不涉及 weld / 夹爪 / 状态机
    - 通过 data.ctrl 驱动 PD 控制器，mj_step 推进物理
    - 使用 passive viewer 渲染，move_to 阻塞直到运动完成
    - 从 keyframe "home" 初始化，与 panda.xml 保持一致
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

# home 关节角，与 panda.xml keyframe 完全一致
Q_HOME = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853])

# 夹爪 actuator8 控制值（0=关闭，255=全开）
GRIPPER_OPEN = 255.0


class MuJoCoEnv:
    """
    Franka Panda 轻量运动学可视化环境

    Args:
        scene_xml : scene.xml 路径（相对于项目根目录）
        n_steps   : 每次 move_to 的插值步数
        step_dt   : 每步渲染间隔（秒），影响动画速度
    """

    def __init__(
        self,
        scene_xml: str = "assets/franka_emika_panda/scene.xml",
        n_steps:   int   = 150,
        step_dt:   float = 0.01,
    ):
        print("=" * 50)
        print("初始化 MuJoCoEnv")

        self.model    = mujoco.MjModel.from_xml_path(scene_xml)
        self.data     = mujoco.MjData(self.model)
        self.n_steps  = n_steps
        self.step_dt  = step_dt

        # 重置到 home keyframe（索引 0）
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # keyframe 只含 panda 的 9 个 qpos（7关节+2手指），
        # cup 的 freejoint 不在其中，需要手动恢复初始位置，
        # 否则杯子会重置到世界原点 (0,0,0) 后掉落。
        self._reset_cup()
        mujoco.mj_forward(self.model, self.data)

        # 启动 passive viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance  = 2.0
        self.viewer.cam.azimuth   = 135
        self.viewer.cam.elevation = -20

        print(f"  home 位形: {np.round(Q_HOME, 4)}")
        print("✅ MuJoCoEnv 初始化完成")
        print("=" * 50)

    def _current_q(self) -> np.ndarray:
        """读取当前 joint1~7 的 qpos"""
        return self.data.qpos[0:7].copy()

    def move_to(self, q_target: np.ndarray, label: str = ""):
        """
        从当前位形线性插值到目标位形

        Args:
            q_target : 目标关节角 shape=(7,)
            label    : 打印标签
        """
        if label:
            print(f"\n▶ 运动到 [{label}]  目标: {np.round(q_target, 3)}")

        q_start = self._current_q()

        for i in range(self.n_steps + 1):
            alpha   = i / self.n_steps
            q_interp = (1 - alpha) * q_start + alpha * q_target

            # 写入控制目标（joint1~7）+ 保持夹爪全开
            self.data.ctrl[0:7] = q_interp
            self.data.ctrl[7]   = GRIPPER_OPEN

            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(self.step_dt)

        print(f"  ✅ 到达 [{label}]")

    def _reset_cup(self):
        """
        手动恢复 cup freejoint 到 scene.xml 中定义的初始位置。
        cup body pos="0.5 0.05 0.45"，quaternion 为单位四元数。
        """
        try:
            jnt_id  = self.model.joint('cup_free').id
            adr     = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[adr + 0] = 0.5    # x
            self.data.qpos[adr + 1] = 0.05   # y
            self.data.qpos[adr + 2] = 0.45   # z
            self.data.qpos[adr + 3] = 1.0    # qw
            self.data.qpos[adr + 4] = 0.0    # qx
            self.data.qpos[adr + 5] = 0.0    # qy
            self.data.qpos[adr + 6] = 0.0    # qz
        except Exception:
            pass  # scene 中无 cup 时静默跳过

    def reset(self):
        """回到 home 位形，同时恢复杯子位置"""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._reset_cup()
        mujoco.mj_forward(self.model, self.data)
        self.move_to(Q_HOME, label="home")

    def get_site_xpos(self, site_name: str) -> np.ndarray:
        """
        读取 site 的世界坐标（从仿真状态直接读）

        Args:
            site_name: scene.xml 中定义的 site 名称

        Returns:
            np.ndarray shape=(3,)，世界坐标（米）
        """
        mujoco.mj_forward(self.model, self.data)
        site_id = self.model.site(site_name).id
        return self.data.site_xpos[site_id].copy()

    def close(self):
        """关闭 viewer"""
        self.viewer.close()
        print("MuJoCoEnv 已关闭")