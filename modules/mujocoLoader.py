"""
modules/mujoco_env.py
MuJoCo 执行环境（A阶段：运动学验证）

职责：
    接收 IKSolver 输出的关节角，驱动 Franka Panda 在 MuJoCo 中运动。
    仅做运动学验证，无物体、无抓取物理。

使用示例：
    env = MuJoCoEnv()
    env.move_to(pick_q)
    env.move_to(place_q)
    env.close()
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

# Franka Panda 7 个关节的名称（与 scene.xml 一致）
PANDA_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
]

# Home 位形（与 IKSolver Q_HOME 一致）
Q_HOME = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])


class MuJoCoEnv:
    """
    Franka Panda MuJoCo 运动学环境

    Args:
        scene_xml : scene.xml 路径
        n_steps   : 每次插值运动的步数（越大越慢越平滑）
        step_dt   : 每步之间的等待时间（秒）
    """

    def __init__(
        self,
        scene_xml: str = "assets/franka_emika_panda/scene.xml",
        n_steps:   int   = 100,
        step_dt:   float = 0.02,
    ):
        print("=" * 50)
        print("初始化 MuJoCoEnv")

        self.model  = mujoco.MjModel.from_xml_path(scene_xml)
        self.data   = mujoco.MjData(self.model)
        self.n_steps = n_steps
        self.step_dt = step_dt

        # 获取关节 qpos 索引
        self.joint_ids = []
        for name in PANDA_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise RuntimeError(f"关节 '{name}' 在 scene.xml 中未找到")
            self.joint_ids.append(self.model.jnt_qposadr[jid])

        # 设置 home 位形
        self._set_qpos(Q_HOME)
        mujoco.mj_forward(self.model, self.data)

        # 启动 passive viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 2.0
        self.viewer.cam.azimuth  = 135
        self.viewer.cam.elevation = -20

        print("✅ MuJoCoEnv 初始化完成")
        print("=" * 50)

    def _set_qpos(self, q: np.ndarray):
        """直接设置关节角（无插值）"""
        for i, addr in enumerate(self.joint_ids):
            self.data.qpos[addr] = q[i]

    def move_to(self, q_target: np.ndarray, label: str = ""):
        """
        从当前关节角插值运动到目标关节角

        Args:
            q_target : 目标关节角 shape=(7,)
            label    : 打印标签（pick / place）
        """
        if label:
            print(f"\n▶ 运动到 [{label}] 位姿")

        q_start = np.array([self.data.qpos[addr] for addr in self.joint_ids])

        for i in range(self.n_steps + 1):
            alpha = i / self.n_steps
            q_interp = (1 - alpha) * q_start + alpha * q_target
            self._set_qpos(q_interp)
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
            time.sleep(self.step_dt)

        print(f"  ✅ 到达 [{label}] 位姿")

    def reset(self):
        """回到 home 位形"""
        self.move_to(Q_HOME, label="home")

    def close(self):
        """关闭 viewer"""
        self.viewer.close()
        print("MuJoCoEnv 已关闭")