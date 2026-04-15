"""
simulation/mujoco_env.py
Created by Yinghao Ho

MuJoCo 执行层
职责：
    接收 Pipeline 输出的 pick_q / place_q，
    驱动 Franka Panda 完成完整 pick-place 动作序列。

设计原则：
    - 步进式（B2）：step() 只推进一个仿真步，viewer 和主循环在外部
    - 不依赖 keyframe，直接写 data.ctrl 驱动关节
    - weld 约束在运行时动态创建/删除，不需要预定义在 XML 里
    - sim 模式下 get_keypoints() 直接读 data.site_xpos，不走视觉 pipeline

状态机：
    HOME → PRE_GRASP → GRASP → WELD →
    LIFT → PRE_PLACE → PLACE → UNWELD → DONE

使用示例（外部测试脚本）：
    env = MuJoCoEnv("assets/franka_emika_panda/scene.xml")
    sim_kps = env.get_keypoints()

    result = pipeline.run(
        instruction   = "pick up the cup and place it on the tray",
        rgb_image     = env.get_rgb(),
        depth_map     = None,
        sim_keypoints = sim_kps,
    )

    env.set_plan(result.pick_q, result.place_q)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while not env.is_done():
            env.step()
            viewer.sync()
"""

from __future__ import annotations

import enum
import numpy as np
import mujoco
from PIL import Image
from typing import Dict, Optional
from modules.IKSolver import IKSolver, Q_HOME as _IK_Q_HOME


# ── 常量 ──────────────────────────────────────────────────────────────────────

# home 关节角（rad），来自 panda.xml keyframe
HOME_Q = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853])

# 夹爪 actuator8 控制值：255=全开，0=全闭
GRIPPER_OPEN  = 255.0
GRIPPER_CLOSE = 0.0

# 收敛阈值（rad）：所有关节误差均小于此值视为到达目标
CONVERGE_THRESH = 0.01

# PRE_GRASP / PRE_PLACE 阶段的 z 轴预备偏移（米）
PREGRASP_OFFSET = 0.12   # 下降到抓取点前先停在上方
LIFT_OFFSET     = 0.15   # 抓住后抬高的距离

# 每个状态的最小停留步数（timestep=0.002s，500步=1秒仿真时间）
# 保证 viewer 能看到平滑的运动过渡，不会瞬间切换
MIN_STATE_STEPS = 500

# 相机渲染分辨率
CAM_WIDTH  = 640
CAM_HEIGHT = 480

# scene_cam 相机名称（与 scene.xml 一致）
CAM_NAME = "scene_cam"

# 需要读取的关键点 site 名称（与 scene.xml 一致）
SITE_NAMES = {
    "cup": ["kp_handle", "kp_body", "kp_rim"],
    "tray": ["kp_tray_surface"],
}


# ── 状态机定义 ─────────────────────────────────────────────────────────────────

class State(enum.Enum):
    HOME      = "HOME"
    PRE_GRASP = "PRE_GRASP"
    GRASP     = "GRASP"
    WELD      = "WELD"          # 激活 weld，关闭夹爪
    LIFT      = "LIFT"
    PRE_PLACE = "PRE_PLACE"
    PLACE     = "PLACE"
    UNWELD    = "UNWELD"        # 解除 weld，打开夹爪
    DONE      = "DONE"


# ── MuJoCoEnv ─────────────────────────────────────────────────────────────────

class MuJoCoEnv:
    """
    Franka Panda pick-place 仿真执行层

    Args:
        scene_xml : scene.xml 路径（相对于项目根目录）
        verbose   : 是否打印状态切换信息
    """

    def __init__(self, scene_xml: str, verbose: bool = True):
        self.verbose = verbose

        # ── 加载模型 ──
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data  = mujoco.MjData(self.model)

        # ── 渲染器（供 get_rgb 使用） ──
        self._renderer = mujoco.Renderer(self.model, CAM_HEIGHT, CAM_WIDTH)

        # ── 规划结果 ──
        self._pick_q  : Optional[np.ndarray] = None   # shape=(7,)
        self._place_q : Optional[np.ndarray] = None   # shape=(7,)

        # ── 状态机 ──
        self._state      = State.HOME
        self._target_q   = HOME_Q.copy()   # 当前控制目标关节角
        self._weld_id    = -1              # 动态 weld 的 equality id
        self._state_steps = 0               # 当前状态已执行的仿真步数

        # ── IKSolver（用于预计算 z 偏移关节角）──
        self._ik_solver = IKSolver(verbose=False)

        # ── 预计算的偏移关节角（由 set_plan 填入）──
        self._pregrasp_q : Optional[np.ndarray] = None
        self._lift_q     : Optional[np.ndarray] = None
        self._preplace_q : Optional[np.ndarray] = None

        # 初始化到 home
        self.reset()

        if verbose:
            print("✅ MuJoCoEnv 初始化完成")
            print(f"   场景：{scene_xml}")
            print(f"   状态：{self._state.value}")

    # ── 公开接口 ───────────────────────────────────────────────────────────────

    def reset(self):
        """重置仿真到 home 位形，清除所有动态 weld"""
        mujoco.mj_resetData(self.model, self.data)

        # 直接写 ctrl 和 qpos，不依赖 keyframe
        self.data.ctrl[0:7] = HOME_Q
        self.data.ctrl[7]   = GRIPPER_OPEN
        self.data.qpos[0:7] = HOME_Q
        self.data.qpos[7:9] = 0.04   # 手指初始位置（米）

        # 杯子初始位置（与 scene.xml 中 body pos 一致）
        # freejoint qpos: [x, y, z, qw, qx, qy, qz]
        # 杯子 freejoint 在 qpos[9:16]
        self.data.qpos[9:12]  = [0.5, 0.05, 0.45]   # 位置
        self.data.qpos[12:16] = [1.0, 0.0, 0.0, 0.0] # 姿态（单位四元数）

        mujoco.mj_forward(self.model, self.data)

        self._state       = State.HOME
        self._target_q    = HOME_Q.copy()
        self._weld_id     = -1
        self._state_steps = 0

        if self.verbose:
            print("🔄 环境已重置到 home")

    def set_plan(self, pick_q: np.ndarray, place_q: np.ndarray):
        """
        接收 Pipeline 输出的关节角，准备执行

        Args:
            pick_q  : pick 目标关节角，shape=(7,)
            place_q : place 目标关节角，shape=(7,)
        """
        assert pick_q.shape  == (7,), f"pick_q shape 错误：{pick_q.shape}"
        assert place_q.shape == (7,), f"place_q shape 错误：{place_q.shape}"

        self._pick_q  = pick_q.copy()
        self._place_q = place_q.copy()
        self._state   = State.HOME   # 从 HOME 开始执行

        # 预计算三个 z 偏移关节角，避免运行时动态 IK 拖慢仿真循环
        self._pregrasp_q = self._ik_with_z_offset(pick_q,  PREGRASP_OFFSET)
        self._lift_q     = self._ik_with_z_offset(pick_q,  LIFT_OFFSET)
        self._preplace_q = self._ik_with_z_offset(place_q, PREGRASP_OFFSET)

        if self.verbose:
            print("📋 规划已接收，准备执行")
            print(f"   pick_q      = {np.round(pick_q, 3)}")
            print(f"   place_q     = {np.round(place_q, 3)}")
            print(f"   pregrasp_q  = {np.round(self._pregrasp_q, 3)}")
            print(f"   lift_q      = {np.round(self._lift_q, 3)}")
            print(f"   preplace_q  = {np.round(self._preplace_q, 3)}")

    def step(self):
        """
        推进一个仿真步 + 状态机切换

        外部主循环每帧调用一次，配合 viewer.sync() 使用：
            while not env.is_done():
                env.step()
                viewer.sync()
        """
        if self._state == State.DONE:
            return

        # 写入当前目标控制量
        self.data.ctrl[0:7] = self._target_q

        # 推进物理仿真一步
        mujoco.mj_step(self.model, self.data)
        self._state_steps += 1

        # 状态机切换判断
        self._update_state()

    def is_done(self) -> bool:
        """是否已完成整个 pick-place 序列"""
        return self._state == State.DONE

    def get_keypoints(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        读取仿真中各关键点的世界坐标

        Returns:
            {
                "cup":  {"handle": np.array([x,y,z]),
                         "body":   np.array([x,y,z]),
                         "rim":    np.array([x,y,z])},
                "tray": {"surface": np.array([x,y,z])},
            }
        """
        mujoco.mj_forward(self.model, self.data)

        result = {}
        for obj, sites in SITE_NAMES.items():
            result[obj] = {}
            for site_name in sites:
                site_id = self.model.site(site_name).id
                # data.site_xpos shape: (n_sites, 3)，存储世界坐标
                result[obj][site_name.replace("kp_", "").replace(f"{obj}_", "")] = \
                    self.data.site_xpos[site_id].copy()

        return result

    def get_rgb(self) -> Image.Image:
        """
        渲染 scene_cam 相机图像

        Returns:
            PIL.Image，RGB，尺寸 (CAM_WIDTH x CAM_HEIGHT)
            用途：生成 VLMDecider 的标注图输入
        """
        self._renderer.update_scene(self.data, camera=CAM_NAME)
        pixels = self._renderer.render()   # (H, W, 3) uint8
        return Image.fromarray(pixels)

    def close(self):
        """释放渲染器资源"""
        self._renderer.close()

    # ── 内部状态机 ─────────────────────────────────────────────────────────────

    def _update_state(self):
        """根据当前状态和收敛情况决定是否切换到下一状态"""

        if self._state == State.HOME:
            self._target_q = HOME_Q.copy()
            if self._converged(HOME_Q):
                self._transition(State.PRE_GRASP)

        elif self._state == State.PRE_GRASP:
            # pick 位置上方：z 轴抬高 PREGRASP_OFFSET（预计算）
            self._target_q = self._pregrasp_q
            if self._converged(self._pregrasp_q):
                self._transition(State.GRASP)

        elif self._state == State.GRASP:
            self._target_q = self._pick_q.copy()
            if self._converged(self._pick_q):
                self._transition(State.WELD)

        elif self._state == State.WELD:
            # 关闭夹爪 + 激活 weld
            self.data.ctrl[7] = GRIPPER_CLOSE
            self._activate_weld()
            # weld 是瞬时操作，直接跳到下一状态
            self._transition(State.LIFT)

        elif self._state == State.LIFT:
            # pick 位置上方抬起（预计算）
            self._target_q = self._lift_q
            if self._converged(self._lift_q):
                self._transition(State.PRE_PLACE)

        elif self._state == State.PRE_PLACE:
            # place 位置上方（预计算）
            self._target_q = self._preplace_q
            if self._converged(self._preplace_q):
                self._transition(State.PLACE)

        elif self._state == State.PLACE:
            self._target_q = self._place_q.copy()
            if self._converged(self._place_q):
                self._transition(State.UNWELD)

        elif self._state == State.UNWELD:
            # 打开夹爪 + 解除 weld
            self.data.ctrl[7] = GRIPPER_OPEN
            self._deactivate_weld()
            self._transition(State.HOME)
            # HOME 阶段 set_plan 已执行完毕，标记 DONE
            self._state = State.DONE
            if self.verbose:
                print("✅ pick-place 完成")

    def _converged(self, target_q: np.ndarray) -> bool:
        """判断前7个关节是否已收敛到目标位形（需满足最小停留步数）"""
        if self._state_steps < MIN_STATE_STEPS:
            return False
        return np.max(np.abs(self.data.qpos[0:7] - target_q)) < CONVERGE_THRESH

    def _fk_from_q(self, q7: np.ndarray) -> np.ndarray:
        """
        用 ikpy FK 从7关节角求 4×4 TCP 世界位姿。
        ikpy 链长度=11，主动关节在索引1-7，其余固定关节置0。
        """
        full_q = np.zeros(11)
        full_q[1:8] = q7
        return self._ik_solver.chain.forward_kinematics(full_q)  # 4×4

    def _ik_with_z_offset(self, q7: np.ndarray, dz: float) -> np.ndarray:
        """
        在 q7 对应的末端位置上加世界坐标 z 偏移，返回新关节角。
        用于预计算 PRE_GRASP、LIFT、PRE_PLACE 的目标位形。

        Args:
            q7 : 原始7关节角
            dz : z 轴偏移量（米），正值=向上
        Returns:
            偏移后的7关节角；IK失败时退化为原 q7（安全保底）
        """
        T = self._fk_from_q(q7)
        T_offset = T.copy()
        T_offset[2, 3] += dz
        result = self._ik_solver.solve(T_offset, q_init=q7)
        if result['success']:
            return result['q']
        # IK 失败时保底：返回原关节角（不偏移总比崩溃好）
        if self.verbose:
            print(f"⚠️  _ik_with_z_offset IK 失败，dz={dz}，退化为原 q7")
        return q7.copy()

    def _activate_weld(self):
        """
        激活 scene.xml 中预定义的 cup_weld equality 约束。
        通过名字查找 equality id，切换 data.eq_active 为 1。
        """
        weld_id = self.model.equality("cup_weld").id
        self.data.eq_active[weld_id] = 1
        self._weld_id = weld_id
        if self.verbose:
            print("🔗 weld 已激活")

    def _deactivate_weld(self):
        """
        解除 cup_weld equality 约束。
        切换 data.eq_active 为 0，杯子恢复自由运动。
        """
        if self._weld_id >= 0:
            self.data.eq_active[self._weld_id] = 0
            self._weld_id = -1
        if self.verbose:
            print("🔓 weld 已解除")

    def _transition(self, next_state: State):
        """切换状态并打印日志"""
        if self.verbose:
            print(f"▶ {self._state.value} → {next_state.value}")
        self._state       = next_state
        self._state_steps = 0   # 重置步数计数器

    # ── 属性 ───────────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state