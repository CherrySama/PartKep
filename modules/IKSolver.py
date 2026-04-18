"""
Created by Yinghao Ho on 2026-2-24

Franka Panda IK 求解器（基于 ikpy）

职责：
    接收 PoseSolver 输出的 SE3 目标位姿，
    求解 Franka Panda 的 7 个关节角。

设计选择：
    - 使用 ikpy 纯运动学求解，完全独立于仿真平台
    - 同一份代码可直接用于 MuJoCo 仿真和真机部署
    - URDF 使用本地 assets/franka_emika_panda/panda.urdf
      命名与 MuJoCo Menagerie panda.xml 完全对应（无 panda_ 前缀）

链结构（来自 panda.urdf）：
    [0]  link0        fixed    ← 不参与 IK（基座）
    [1]  joint1       revolute ← 7 个主动关节
    [2]  joint2       revolute
    [3]  joint3       revolute
    [4]  joint4       revolute
    [5]  joint5       revolute
    [6]  joint6       revolute
    [7]  joint7       revolute
    [8]  hand_joint   fixed    ← 不参与 IK
    末端 link：hand

Franka Panda 关节限位（rad）：
    joint1: [-2.8973,  2.8973]
    joint2: [-1.7628,  1.7628]
    joint3: [-2.8973,  2.8973]
    joint4: [-3.0718, -0.0698]
    joint5: [-2.8973,  2.8973]
    joint6: [-0.0175,  3.7525]
    joint7: [-2.8973,  2.8973]
"""

import numpy as np
from typing import Optional, Dict
from pathlib import Path


# Franka Panda 关节限位（rad），与官方 URDF 一致
PANDA_JOINT_LIMITS = np.array([
    [-2.8973,  2.8973],   # joint1
    [-1.7628,  1.7628],   # joint2
    [-2.8973,  2.8973],   # joint3
    [-3.0718, -0.0698],   # joint4
    [-2.8973,  2.8973],   # joint5
    [-0.0175,  3.7525],   # joint6
    [-2.8973,  2.8973],   # joint7
])

# ikpy 链中主动关节的索引（对应上表）
# 链长度 9，索引 1-7 是 revolute 主动关节
ACTIVE_LINKS_MASK = [False, True, True, True, True, True, True, True, False]

# Panda 默认初始位形（接近竖直，关节限位内的安全位置）
# 可在 MuJoCo 中观察 home 位置得到
Q_HOME = np.array([0, 0, 0, -1.5708, 0, 1.5708, -0.7853])


class IKSolver:
    """
    Franka Panda IK 求解器

    输入：spatialmath.SE3 目标末端位姿（来自 PoseSolver）
    输出：7 个关节角（rad）

    使用示例：
        >>> from modules.solver import PoseSolver
        >>> from modules.ik_solver import IKSolver
        >>>
        >>> # 上游：PoseSolver 求解末端位姿
        >>> pose_result = pose_solver.solve(cost_fn, x0, meta)
        >>>
        >>> # IK 求解关节角
        >>> ik = IKSolver()
        >>> ik_result = ik.solve(pose_result['se3'])
        >>>
        >>> if ik_result['success']:
        ...     q = ik_result['q']   # shape=(7,)，直接发给 MuJoCo 或真机
    """

    def __init__(self,
                 urdf_path: Optional[str] = None,
                 verbose: bool = True):
        """
        初始化 IK 求解器，加载运动学链

        Args:
            urdf_path: URDF 文件路径。
                       None → 自动定位 assets/franka_emika_panda/panda.urdf
            verbose:   是否打印加载和求解信息
        """
        self.verbose = verbose

        if verbose:
            print("=" * 60)
            print("初始化 IKSolver（Franka Panda）")
            print("=" * 60)

        # ===== 1. 确定 URDF 路径 =====
        if urdf_path is None:
            urdf_path = self._get_urdf_path()
        else:
            urdf_path = str(urdf_path)

        if not Path(urdf_path).exists():
            raise FileNotFoundError(f"URDF 文件不存在: {urdf_path}")

        if verbose:
            print(f"✓ URDF: {urdf_path}")

        # ===== 2. 加载 ikpy 运动学链 =====
        try:
            import ikpy.chain as ikpy_chain
        except ImportError:
            raise ImportError(
                "需要安装 ikpy：pip install ikpy"
            )

        import warnings
        with warnings.catch_warnings():
            # 屏蔽 ikpy 对 fixed 关节被标记为 active 的警告（正常现象）
            warnings.simplefilter("ignore", UserWarning)
            self.chain = ikpy_chain.Chain.from_urdf_file(
                urdf_path,
                base_elements=['link0'],
                base_element_type='link',
                active_links_mask=ACTIVE_LINKS_MASK
            )

        if verbose:
            print(f"✓ 运动学链加载完成")
            print(f"  链长度: {len(self.chain.links)}")
            print(f"  主动关节: 7 个（joint1 ~ joint7）")

            # 打印零位 FK 验证
            q_full = self._q7_to_q_full(np.zeros(7))
            T_home = self.chain.forward_kinematics(q_full)
            print(f"  零位TCP位置: {np.round(T_home[:3, 3], 4)}")
            print("=" * 60)
            print()

    @staticmethod
    def _get_urdf_path() -> str:
        """返回本地 Panda URDF 路径（与 MuJoCo Menagerie panda.xml 同源）"""
        urdf_path = (
            Path(__file__).parent.parent
            / "assets" / "franka_emika_panda" / "panda.urdf"
        )
        if not urdf_path.exists():
            raise FileNotFoundError(
                f"未找到 URDF 文件: {urdf_path}\n"
                "请确认 assets/franka_emika_panda/panda.urdf 存在"
            )
        return str(urdf_path)

    @staticmethod
    def _q7_to_q_full(q7: np.ndarray) -> list:
        """
        将 7 维关节角扩展为 ikpy 所需的 11 维列表

        ikpy 的输入输出 q 向量长度等于链长度（11），
        非主动关节位置填 0.0。

        Args:
            q7: shape=(7,) 7 个主动关节角（rad）

        Returns:
            list[11]：ikpy 格式的完整关节角列表
        """
        q_full = [0.0] * 9
        # 主动关节在索引 1-7
        for i, val in enumerate(q7):
            q_full[i + 1] = float(val)
        return q_full

    @staticmethod
    def _q_full_to_q7(q_full: list) -> np.ndarray:
        """
        从 ikpy 的 11 维输出中提取 7 个主动关节角

        Args:
            q_full: ikpy 返回的完整关节角列表（长度 11）

        Returns:
            np.ndarray shape=(7,)：7 个主动关节角（rad）
        """
        return np.array(q_full[1:8], dtype=np.float64)

    def forward_kinematics(self, q7: np.ndarray) -> np.ndarray:
        """
        正向运动学：7 个关节角 → 末端 4×4 变换矩阵

        Args:
            q7: shape=(7,) 关节角（rad）

        Returns:
            np.ndarray shape=(4,4)：末端 TCP 齐次变换矩阵
        """
        q_full = self._q7_to_q_full(q7)
        return self.chain.forward_kinematics(q_full)

    def solve(self,
              target_T: np.ndarray,
              q_init: Optional[np.ndarray] = None) -> Dict:
        """
        IK 求解：目标末端位姿 → 7 个关节角

        Args:
            target_T: shape=(4,4) 齐次变换矩阵
                      直接来自 PoseSolver.solve()['T']
            q_init:   shape=(7,) 初始关节角猜测（rad）
                      None → 使用默认 home 位形

        Returns:
            Dict：
                'success':        bool，是否求解成功
                'q':              np.ndarray shape=(7,)，关节角（rad）
                'position_error': float，末端位置误差（米）
                'within_limits':  bool，是否在关节限位内
                'fk_position':    np.ndarray shape=(3,)，FK 验证末端位置
        """
        if q_init is None:
            q_init = Q_HOME.copy()

        target_T = np.array(target_T, dtype=np.float64)
        if target_T.shape != (4, 4):
            raise ValueError(f"target_T 必须 shape=(4,4)，当前: {target_T.shape}")

        if self.verbose:
            print(f"\n🦾 IK 求解")
            print(f"  目标位置: {np.round(target_T[:3, 3], 4)}")
            print(f"  初始位形: {np.round(q_init, 4)}")

        # ===== ikpy IK 求解 =====
        q_init_full = self._q7_to_q_full(q_init)

        # ikpy 接口：position 和 orientation 分开传
        # target_position: shape=(3,)
        # target_orientation: shape=(3,3) 旋转矩阵
        q_full_result = self.chain.inverse_kinematics(
            target_position=target_T[:3, 3],
            target_orientation=target_T[:3, :3],
            initial_position=q_init_full,
            orientation_mode="all"
        )

        q7_result = self._q_full_to_q7(q_full_result)

        # ===== 验证：FK 反算末端位置误差 =====
        T_fk = self.forward_kinematics(q7_result)
        fk_position     = T_fk[:3, 3]
        target_position = target_T[:3, 3]
        position_error  = float(np.linalg.norm(fk_position - target_position))

        # ===== 验证：关节限位检查 =====
        within_limits = bool(np.all(
            (q7_result >= PANDA_JOINT_LIMITS[:, 0]) &
            (q7_result <= PANDA_JOINT_LIMITS[:, 1])
        ))

        # 综合判定：位置误差 < 1cm 且在限位内视为成功
        success = (position_error < 0.01) and within_limits

        result = {
            'success':        success,
            'q':              q7_result,
            'position_error': position_error,
            'within_limits':  within_limits,
            'fk_position':    fk_position,
        }

        if self.verbose:
            status = "✅ 成功" if success else "⚠️  失败"
            print(f"\n  {status}")
            print(f"  关节角（rad）: {np.round(q7_result, 4)}")
            print(f"  FK 末端位置:  {np.round(fk_position, 4)}")
            print(f"  位置误差:     {position_error*1000:.2f} mm")
            print(f"  在关节限位内: {within_limits}")
            if not within_limits:
                # 打印超出限位的关节
                for i in range(7):
                    lo, hi = PANDA_JOINT_LIMITS[i]
                    q = q7_result[i]
                    if q < lo or q > hi:
                        print(f"    ⚠️  joint{i+1}: {q:.4f} 超出 [{lo:.4f}, {hi:.4f}]")

        return result


# ==================== 模块测试 ====================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from utils import CoordinateTransformer

    print("=" * 70)
    print("测试 IKSolver")
    print("=" * 70)

    ik = IKSolver(verbose=True)

    # ---------- 测试1：home 位形 FK → IK 反解 ----------
    print("\n【测试1】FK home位形 → IK 反解")
    print("-" * 70)

    # 用 Q_HOME 的 FK 结果作为 IK 目标（Q_HOME 在关节限位内，是合法初始猜测）
    T_home = ik.forward_kinematics(Q_HOME)
    target_T = CoordinateTransformer.rotation_to_matrix4x4(
        position=T_home[:3, 3],
        R=T_home[:3, :3]
    )
    print(f"  Q_HOME FK 末端位置: {np.round(T_home[:3, 3], 4)}")

    result = ik.solve(target_T, q_init=Q_HOME)
    print(f"\n  位置误差: {result['position_error']*1000:.3f} mm")
    assert result['position_error'] < 0.01, "home位形反解误差过大"
    print(f"  ✅ home位形反解成功")

    # ---------- 测试2：典型抓取位姿 ----------
    print("\n【测试2】典型抓取位姿（桌面杯子）")
    print("-" * 70)

    # 模拟抓取桌上杯子：位置约在机器人前方桌面
    target_pos = np.array([0.4, 0.1, 0.3])
    target_R   = CoordinateTransformer.rodrigues(
        np.array([0.0, np.pi, 0.0])
    )
    T_grasp = CoordinateTransformer.rotation_to_matrix4x4(target_pos, target_R)

    result2 = ik.solve(T_grasp, q_init=Q_HOME)
    print(f"\n  目标位置: {target_pos}")
    print(f"  FK 末端:  {np.round(result2['fk_position'], 4)}")
    print(f"  位置误差: {result2['position_error']*1000:.3f} mm")
    print(f"  限位检查: {result2['within_limits']}")
    if result2['success']:
        print(f"  ✅ 求解成功，关节角: {np.round(result2['q'], 3)}")
    else:
        print(f"  ⚠️  求解结果仅供参考（误差或超限位）")
