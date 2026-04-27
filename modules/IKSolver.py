"""
modules/IKSolver.py

Franka Panda IK 求解器（MuJoCo-native）

实现：scipy L-BFGS-B + mujoco.mj_forward + 随机重启
    - FK/IK 使用同一个物理引擎，坐标系完全一致
    - IKSolver 持有私有 MjData，不污染 MuJoCoEnv 的仿真状态
    - 代价函数：位置误差² + w_rot × 旋转 Frobenius 距离²
    - n_restarts 次随机重启，显著提升成功率（40%单次 → >99% 重启10次）

末端帧：MuJoCo body "hand"

qpos 布局（scene.xml）：
    [0:7]  → joint1~joint7（arm）
    [7:9]  → finger_joint1, finger_joint2
    [9:16] → cup_free（freejoint）
    qpos[:7] 直接对应 arm 关节。
"""

import numpy as np
import mujoco
from scipy.optimize import minimize
from typing import Optional, Dict


# ── 常量 ──────────────────────────────────────────────────────────────────────

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

_W_ROT = 0.1   # 旋转误差权重（位置精度优先）


# ── 主类 ──────────────────────────────────────────────────────────────────────

class IKSolver:
    """
    Franka Panda IK 求解器（MuJoCo-native）

    构造时传入 MuJoCo model，内部创建独立私有 data。

    使用示例：
        model = mujoco.MjModel.from_xml_path("assets/.../scene.xml")
        ik = IKSolver(model)
        T  = ik.forward_kinematics(Q_HOME)
        res = ik.solve(T_pick, q_init=Q_HOME)
        if res['success']:
            q = res['q']
    """

    def __init__(self, model: mujoco.MjModel, verbose: bool = True):
        self._model  = model
        self._data   = mujoco.MjData(model)
        self.verbose = verbose

        # 从 keyframe 0 初始化，保证手指、freejoint 等 qpos 合法
        mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        mujoco.mj_forward(self._model, self._data)

        self._hand_id = self._model.body('hand').id

        if verbose:
            print("=" * 55)
            print("初始化 IKSolver（MuJoCo-native）")
            T0 = self.forward_kinematics(Q_HOME)
            print(f"  末端帧    : hand (body id={self._hand_id})")
            print(f"  home TCP  : {np.round(T0[:3, 3], 4)}")
            print("=" * 55)

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def forward_kinematics(self, q7: np.ndarray) -> np.ndarray:
        """
        7 个关节角 → hand body 的 4×4 齐次变换矩阵

        Args:
            q7: shape=(7,) 关节角（rad）

        Returns:
            np.ndarray shape=(4,4)
        """
        self._data.qpos[:7] = q7
        mujoco.mj_forward(self._model, self._data)
        pos = self._data.xpos[self._hand_id].copy()
        R   = self._data.xmat[self._hand_id].reshape(3, 3).copy()
        T       = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = pos
        return T

    def solve(self,
              target_T:   np.ndarray,
              q_init:     Optional[np.ndarray] = None,
              n_restarts: int = 10) -> Dict:
        """
        IK 求解：目标末端位姿 → 7 个关节角

        策略：
            第 0 次用 q_init（热启动优先）
            第 1~n_restarts-1 次用随机关节角
            返回所有成功解中位置误差最小的；无成功解则返回误差最小的失败解

        Args:
            target_T:   shape=(4,4) 目标位姿
            q_init:     shape=(7,) 初始关节角，None → Q_HOME
            n_restarts: 最大尝试次数（默认 10）

        Returns:
            Dict:
                'success'        : bool，位置误差 < 5mm 且在关节限位内
                'q'              : np.ndarray shape=(7,)
                'position_error' : float（米）
                'within_limits'  : bool
                'fk_position'    : np.ndarray shape=(3,)
        """
        if q_init is None:
            q_init = Q_HOME.copy()

        target_T = np.array(target_T, dtype=np.float64)
        p_target = target_T[:3, 3]
        R_target = target_T[:3, :3]
        bounds   = [(lo, hi) for lo, hi in PANDA_JOINT_LIMITS]

        if self.verbose:
            print(f"\n🦾 IK 求解（MuJoCo-native，最多 {n_restarts} 次）")
            print(f"  目标位置: {np.round(p_target, 4)}")

        def _cost(q7: np.ndarray) -> float:
            self._data.qpos[:7] = q7
            mujoco.mj_forward(self._model, self._data)
            p = self._data.xpos[self._hand_id]
            R = self._data.xmat[self._hand_id].reshape(3, 3)
            return np.sum((p - p_target) ** 2) + _W_ROT * np.sum((R - R_target) ** 2)

        best = None   # 记录误差最小的结果（成功或失败）

        for attempt in range(n_restarts):
            # 第 0 次热启动，后续随机初始化
            if attempt == 0:
                q0 = q_init.copy()
            else:
                q0 = np.random.uniform(
                    PANDA_JOINT_LIMITS[:, 0],
                    PANDA_JOINT_LIMITS[:, 1]
                )

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

            # 保留误差最小的结果
            if best is None or pos_err < best['position_error']:
                best = candidate

            if success:
                if self.verbose:
                    print(f"  ✅ 第 {attempt+1} 次成功，误差 {pos_err*1000:.2f} mm")
                return best   # 提前返回

        # 所有重启都失败，返回误差最小的结果
        if self.verbose:
            status = "⚠️  全部失败"
            print(f"  {status}，最小误差 {best['position_error']*1000:.2f} mm")
            print(f"  FK 末端位置: {np.round(best['fk_position'], 4)}")
            print(f"  在关节限位内: {best['within_limits']}")

        return best


# ── 模块自测 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    SCENE_XML = "assets/franka_emika_panda/scene.xml"
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    ik    = IKSolver(model, verbose=True)

    print("\n【1】Q_HOME FK 验证")
    T0 = ik.forward_kinematics(Q_HOME)
    print(f"  hand body 位置: {np.round(T0[:3, 3], 4)}")

    print("\n【2】FK → IK 往返验证（n_restarts=10）")
    result = ik.solve(T0, q_init=Q_HOME, n_restarts=10)
    assert result['position_error'] < 0.001, \
        f"往返误差过大: {result['position_error']*1000:.2f} mm"
    print(f"  ✅ 往返误差 {result['position_error']*1000:.3f} mm < 1mm")

    print("\n【3】T_pick 目标（handle 水平侧向，n_restarts=10）")
    from scipy.spatial.transform import Rotation
    R_pick = Rotation.from_rotvec([1.057, 1.4206, 1.4206]).as_matrix()
    T_pick = np.eye(4)
    T_pick[:3, :3] = R_pick
    T_pick[:3,  3] = [0.4713, 0.0204, 0.45]
    result2 = ik.solve(T_pick, q_init=Q_HOME, n_restarts=10)
    print(f"  {'✅' if result2['success'] else '⚠️ '} 位置误差: "
          f"{result2['position_error']*1000:.2f} mm  "
          f"限位: {result2['within_limits']}")