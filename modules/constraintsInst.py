"""
Created by Yinghao Ho on 2026-2-23

代价函数实例化模块
核心职责：
    将语义关键点 + SAP 知识库 + VLMDecision → 可被 SLSQP 直接调用的数值代价函数

约束体系（共 4 个）：
    ┌────────┬──────────────────────────────┬──────────┬──────────────┐
    │ 编号   │ 含义                         │ 控制方   │ 适用模式     │
    ├────────┼──────────────────────────────┼──────────┼──────────────┤
    │ C1     │ 指尖位置对齐目标关键点       │ 固定     │ pick + place │
    │ C2     │ 夹爪 Z 轴对齐接近方向        │ 固定     │ pick + place │
    │ C3     │ 夹爪 Y 轴对齐 SAP grasp_axis │ VLM      │ pick 专用    │
    │ C4     │ 安全距离惩罚（指尖距离）     │ VLM      │ pick + place │
    └────────┴──────────────────────────────┴──────────┴──────────────┘

    C1/C2 属于 SAP 几何职责，始终激活，权重固定。
    C2 采用 (1-dot)² 形式：d=+1（正向对齐）→ 0，d=-1（wrist flip）→ 4，
    翻转保护由数学形式内建，不依赖语义权重控制，不能被 VLM 关掉。
    C3/C4 属于语义约束，权重由 VLMDecision 提供。
    P1 为几何正则项（ε·‖rvec‖²），防止轴角 Rodrigues 奇点，权重固定。

归一化说明：
    - C1：(||fingertip - keypoint|| / FINGER_LENGTH)²    → [0, ∞)
          fingertip = pos + (R@[0,0,1]) * FINGER_LENGTH
    - C2：(1 - dot(gripper_Z, approach_dir))²            → [0, 4]
          d=+1（正向对齐）→ 0，d=0（垂直）→ 1，d=-1（wrist flip）→ 4
    - C3：1 - dot(gripper_Y, grasp_axis)²                → [0, 1]
    - C4：Σ max(0, 1 - ||fingertip-p_avoid||/margin)²   → [0, N_avoid]
    - P1：ε · ‖rvec‖²                                    → [0, ∞)

最优交互点选择机制：
    Pick 模式对每个 grasp 候选关键点分别构建 cost_fn 并运行完整 SLSQP（200次），
    选取最终代价最低的候选作为最优交互点，返回其 cost_fn 和对应 x0 供
    PoseSolver 做最终精细优化（200次）。最优交互点由优化过程自然涌现，
    不由任何硬编码优先级决定。VLM 权重影响各候选的代价，间接决定涌现结果。

优化变量 x（6D）：
    [px, py, pz, ax, ay, az]
    - (px, py, pz)：末端执行器位置（世界坐标系，米）
    - (ax, ay, az)：末端执行器旋转（轴角表示，||a|| 为旋转角 rad）

夹爪局部坐标系约定：
    - 局部 Z 轴：夹爪接近轴（approach axis）
    - 局部 Y 轴：夹爪张开轴（grasp axis）
    - 局部 X 轴：由右手定则 X = Y × Z 得出
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional

from scipy.optimize import minimize

from configs.SAP import get_sap_strict, SAP
from modules.vlmDecider import VLMDecision
from utils import CoordinateTransformer


# ==================== 固定权重（C1、C2）====================
W_APPROACH_POS = 1.0    # C1 位置对齐（固定）
W_APPROACH_ROT = 1.0    # C2 接近方向旋转对齐（固定，新公式值域[0,4]，需重新验证）

# ==================== 几何正则项权重 ====================
EPS_RVEC_REG = 1e-6     # P1 轴角正则化（ε·‖rvec‖²），防止 Rodrigues 奇点

# ==================== 几何参数（来自 panda.xml 实测）====================
# hand link 原点 → 指尖 pad 中心（沿 hand Z 轴）
# 来源：finger pos=0.0584 + pad_1 center=0.0445 = 0.1029m ≈ 0.103m
FINGER_LENGTH  = 0.103


# ==================== 辅助函数 ====================

def _compute_actual_approach(
        part_name:     str,
        sap:           SAP,
        keypoint_3d:   np.ndarray,
        object_center: np.ndarray,
) -> np.ndarray:
    """
    运行时修正 approach_direction

    策略：
        - body / cap / surface（从上方接近）：直接使用 SAP 预定义方向 [0,0,-1]
        - handle / neck（侧向抓取）：SAP 存的是参考方向 [1,0,0]，
          运行时替换为关键点相对物体中心的水平方向向量，
          使接近方向真正从物体外部指向该关键点。

    Returns:
        shape=(3,) 归一化的接近方向向量
    """
    LATERAL_PARTS = {"handle", "neck", "body"}

    if part_name in LATERAL_PARTS:
        delta_xy = object_center[:2] - keypoint_3d[:2] # 从 handle 指向杯中心，方向向内（从外部接近）
        norm = np.linalg.norm(delta_xy)
        if norm < 1e-6:
            # 关键点与物体中心重合，退回 SAP 参考方向
            return sap.approach_direction.copy()
        return np.array([delta_xy[0] / norm, delta_xy[1] / norm, 0.0])
    else:
        return sap.approach_direction.copy()


def _compute_rvec_init(
        approach_dir: np.ndarray,
        grasp_axis:   np.ndarray,
) -> np.ndarray:
    """
    通过叉积法计算旋转初始值，修复 rvec=[0,0,0] 时的梯度死区问题

    构造目标旋转矩阵 R_init：
        - R_init @ [0,0,1] = approach_dir  （夹爪 Z 轴对齐接近方向）
        - R_init @ [0,1,0] = grasp_axis    （夹爪 Y 轴对齐抓取轴）
        - R_init @ [1,0,0] = grasp_axis × approach_dir（右手定则）

    前提：approach_dir ⊥ grasp_axis（SAP 设计保证）

    Args:
        approach_dir : 归一化接近方向
        grasp_axis   : 归一化夹爪张开轴

    Returns:
        shape=(3,) 轴角旋转向量，使优化从物理合理姿态出发
    """
    z_col = approach_dir.copy()                        # 夹爪 Z → 接近方向
    y_col = grasp_axis.copy()                          # 夹爪 Y → 抓取轴
    x_col = np.cross(grasp_axis, approach_dir)         # X = Y × Z

    # 正交化保障（数值误差兜底）
    norm_x = np.linalg.norm(x_col)
    if norm_x < 1e-8:
        # approach_dir 和 grasp_axis 近似平行，退回零向量
        return np.zeros(3)
    x_col = x_col / norm_x

    # 构造旋转矩阵（列向量 = 夹爪各轴在世界坐标系中的方向）
    R_init = np.column_stack([x_col, y_col, z_col])

    # 旋转矩阵 → 轴角向量（Rodrigues 逆变换）
    try:
        from scipy.spatial.transform import Rotation
        rvec = Rotation.from_matrix(R_init).as_rotvec()
    except Exception:
        # scipy 不可用时手动计算
        rvec = _rodrigues_inverse(R_init)

    return rvec


def _rodrigues_inverse(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 轴角向量（手动实现，供 scipy 不可用时使用）"""
    trace_val = (np.trace(R) - 1.0) / 2.0
    trace_val = float(np.clip(trace_val, -1.0, 1.0))
    theta = np.arccos(trace_val)
    if theta < 1e-10:
        return np.zeros(3)
    k = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))
    return k * theta


def _build_cost_fn(
        keypoint_3d:    np.ndarray,
        approach_dir:   np.ndarray,
        grasp_axis:     Optional[np.ndarray],
        avoid_kps:      Dict[str, Tuple[np.ndarray, float]],
        vlm_decision:   VLMDecision,
        mode:           str,
) -> Tuple[Callable, Callable]:
    """
    构建单个候选关键点的代价函数

    Args:
        keypoint_3d    : 目标关键点世界坐标（物体表面点）
        approach_dir   : 归一化接近方向（运行时修正后）
        grasp_axis     : 归一化夹爪张开轴（pick 专用，place 传 None）
        avoid_kps      : {part_name → (point_3d, margin)} 需回避的部件
        vlm_decision   : VLM 约束决策（C3/C4 权重）
        mode           : "pick" 或 "place"

    Returns:
        (cost_fn, cost_breakdown_fn) 元组
    """
    # ── 固定参数（闭包捕获） ──
    _kp    = keypoint_3d.copy()
    _app   = approach_dir.copy()
    _gavt  = grasp_axis.copy() if grasp_axis is not None else None
    _avd   = {k: (v[0].copy(), v[1]) for k, v in avoid_kps.items()}
    _L     = FINGER_LENGTH

    # ── 固定权重 ──
    _wc1 = W_APPROACH_POS
    _wc2 = W_APPROACH_ROT
    _eps  = EPS_RVEC_REG

    # ── VLM 权重（C3 pick 专用，place 强制 0） ──
    _wc3 = vlm_decision.w_grasp_axis if mode == "pick" else 0.0
    _wc4 = vlm_decision.w_safety

    # ── 中间量计算（Rodrigues 只算一次） ──
    def _compute_intermediates(x: np.ndarray):
        pos       = x[:3]
        rvec      = x[3:]
        R         = CoordinateTransformer.rodrigues(rvec)
        gripper_z = R @ np.array([0.0, 0.0, 1.0])
        gripper_y = R @ np.array([0.0, 1.0, 0.0])
        fingertip = pos + gripper_z * _L
        return gripper_z, gripper_y, fingertip, rvec

    # ── 独立约束函数（接收已算好的中间量） ──
    def _c1_position(fingertip: np.ndarray) -> float:
        """C1：指尖位置对齐代价"""
        return (float(np.linalg.norm(fingertip - _kp)) / _L) ** 2

    def _c2_approach(gripper_z: np.ndarray) -> float:
        """C2：接近方向对齐代价，(1-dot)²，d=-1时=4（翻转），d=+1时=0（对齐）"""
        d = float(np.clip(np.dot(gripper_z, _app), -1.0, 1.0))
        return (1.0 - d) ** 2

    def _c3_grasp_axis(gripper_y: np.ndarray) -> float:
        """C3：夹爪张开轴对齐代价，dot²保证对称夹爪Y/-Y等价"""
        if _gavt is None:
            return 0.0
        d = float(np.clip(np.dot(gripper_y, _gavt), -1.0, 1.0))
        return 1.0 - d ** 2

    def _c4_safety(fingertip: np.ndarray):
        """C4：安全距离惩罚，返回 (total, per_part_dict)"""
        total    = 0.0
        per_part = {}
        for pname, (p_avoid, margin) in _avd.items():
            dist = float(np.linalg.norm(fingertip - p_avoid))
            v    = max(0.0, 1.0 - dist / margin)
            val  = v * v
            per_part[pname] = val
            total += val
        return total, per_part

    def _p1_rvec_reg(rvec: np.ndarray) -> float:
        """P1：轴角正则化，防止 Rodrigues 奇点"""
        return float(np.dot(rvec, rvec))

    # ── 代价函数（SLSQP 调用） ──
    def cost_fn(x: np.ndarray) -> float:
        gz, gy, fingertip, rvec = _compute_intermediates(x)
        c1       = _c1_position(fingertip)
        c2       = _c2_approach(gz)
        c3       = _c3_grasp_axis(gy)
        c4, _    = _c4_safety(fingertip)
        p1       = _p1_rvec_reg(rvec)
        return float(_wc1*c1 + _wc2*c2 + _wc3*c3 + _wc4*c4 + _eps*p1)

    # ── 代价分解函数（调试用，与 cost_fn 共享中间量计算） ──
    def cost_breakdown_fn(x: np.ndarray) -> Dict:
        gz, gy, fingertip, rvec = _compute_intermediates(x)
        c1               = _c1_position(fingertip)
        c2               = _c2_approach(gz)
        c3               = _c3_grasp_axis(gy)
        c4, safety_parts = _c4_safety(fingertip)
        p1               = _p1_rvec_reg(rvec)
        total = _wc1*c1 + _wc2*c2 + _wc3*c3 + _wc4*c4 + _eps*p1
        return {
            'total':           float(total),
            'approach_pos':    float(_wc1 * c1),
            'approach_rot':    float(_wc2 * c2),
            'grasp_axis':      float(_wc3 * c3),
            'safety':          float(_wc4 * c4),
            'safety_per_part': safety_parts,
            'rvec_reg':        float(_eps  * p1),
        }

    return cost_fn, cost_breakdown_fn




# ==================== 主类 ====================

class ConstraintInstantiator:
    """
    代价函数实例化器

    根据输入关键点的 SAP contact_mode 自动选择 pick 或 place 模式，
    接收 VLMDecision 提供的语义约束权重，构建完整代价函数。

    Pick 模式：
        对所有 grasp 候选关键点分别构建代价函数并运行完整 SLSQP（200次），
        选取最终代价最低的候选（最优交互点自然涌现），
        返回其 cost_fn 和对应 x0 供 PoseSolver 精细优化。

    Place 模式：
        单一 surface 目标，构建含 C1/C2/C4 的代价函数。

    使用示例（pick）：
        >>> from modules.vlmDecider import VLMDecider
        >>> from modules.IKSolver import IKSolver
        >>>
        >>> ik = IKSolver(verbose=False)
        >>> T_current = ik.forward_kinematics(env.get_current_q())
        >>>
        >>> decision = vlm_decider.decide(image, kps_2d, instruction, "pick")
        >>> inst = ConstraintInstantiator()
        >>> cost_fn, x0, meta = inst.instantiate(keypoints_3d, decision, T_current)

    使用示例（place）：
        >>> decision_place = vlm_decider.decide(image, kps_2d, instruction, "place")
        >>> cost_fn, x0, meta = inst.instantiate(tray_keypoints_3d, decision_place, T_current)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # ==================== 公开接口 ====================

    def instantiate(
            self,
            keypoints_3d: Dict[str, np.ndarray],
            vlm_decision: VLMDecision,
            T_current:    np.ndarray,
    ) -> Tuple[Callable, np.ndarray, Dict]:
        """
        实例化代价函数（自动选择 pick / place 模式）

        Args:
            keypoints_3d : {part_name → 3D 世界坐标 shape=(3,)}
            vlm_decision : VLMDecision，来自 VLMDecider.decide()
            T_current    : shape=(4,4) 当前末端齐次变换矩阵（来自 IKSolver.forward_kinematics(q_current)）
                           用于初始化 x0，使优化从机械臂实际状态出发

        Returns:
            cost_fn : Callable，SLSQP 目标函数
            x0      : shape=(6,)，优化初始猜测（来自 T_current）
            meta    : Dict，调试信息

        Raises:
            ValueError : 语义冲突（同时含 grasp 和 place 部件）
            ValueError : 既无 grasp 也无 place 部件
            KeyError   : 部件名称不在 SAP 知识库中
        """
        # ── 1. 按 contact_mode 分类关键点 ──
        grasp_kps: Dict[str, np.ndarray]              = {}
        place_kps: Dict[str, np.ndarray]              = {}
        avoid_kps: Dict[str, Tuple[np.ndarray, float]] = {}

        for part_name, point in keypoints_3d.items():
            sap = get_sap_strict(part_name)
            pt  = np.array(point, dtype=np.float64)
            if sap.contact_mode == 'grasp':
                grasp_kps[part_name] = pt
            elif sap.contact_mode == 'place':
                place_kps[part_name] = pt
            else:
                avoid_kps[part_name] = (pt, sap.safety_margin)

        # ── 2. 模式判断 ──
        if grasp_kps and place_kps:
            raise ValueError(
                f"keypoints_3d 同时含有 grasp 部件 {list(grasp_kps.keys())} "
                f"和 place 部件 {list(place_kps.keys())}，语义冲突。\n"
                f"pick 目标和 place 目标应分别调用 instantiate()。"
            )
        if not grasp_kps and not place_kps:
            raise ValueError(
                f"keypoints_3d 中既无 grasp 也无 place 部件。\n"
                f"输入的部件: {list(keypoints_3d.keys())}"
            )

        if grasp_kps:
            return self._instantiate_pick(grasp_kps, avoid_kps, vlm_decision, T_current)
        else:
            return self._instantiate_place(place_kps, avoid_kps, vlm_decision, T_current)

    # ==================== 私有：pick 路径 ====================

    def _instantiate_pick(
            self,
            grasp_kps:    Dict[str, np.ndarray],
            avoid_kps:    Dict[str, Tuple[np.ndarray, float]],
            vlm_decision: VLMDecision,
            T_current:    np.ndarray,
    ) -> Tuple[Callable, np.ndarray, Dict]:
        """
        Pick 模式：对所有 grasp 候选关键点各自构建代价函数，
        运行完整 SLSQP（200次）筛选最优交互点，返回其代价函数和 x0。
        """
        # 物体中心（所有关键点均值，用于运行时修正接近方向）
        avoid_pts = [v[0] for v in avoid_kps.values()]
        if avoid_pts:
            object_center = np.mean(np.stack(avoid_pts), axis=0)
        else:
            # 没有 avoid 关键点时退回所有关键点均值
            object_center = np.mean(np.stack(list(grasp_kps.values())), axis=0)

        if self.verbose:
            print(f"\n📐 ConstraintInstantiator [pick 模式]")
            print(f"  grasp 候选: {list(grasp_kps.keys())}")
            print(f"  avoid 目标: {list(avoid_kps.keys())}")
            print(f"  VLM 决策:   {vlm_decision}")

        best_cost         = float('inf')
        best_cost_fn      = None
        best_breakdown_fn = None
        best_x0           = None
        best_meta_part    = None

        # ── 遍历所有 grasp 候选，完整优化筛选最优交互点 ──
        for part_name, keypoint_3d in grasp_kps.items():
            sap = get_sap_strict(part_name)

            # 运行时修正接近方向
            approach_dir = _compute_actual_approach(
                part_name, sap, keypoint_3d, object_center
            )

            # x0 位置：hand link 放在"指尖对准关键点"的位置
            x0_rot_per = _compute_rvec_init(approach_dir, sap.grasp_axis)
            x0_pos     = keypoint_3d - approach_dir * FINGER_LENGTH
            x0         = np.concatenate([x0_pos, x0_rot_per])

            # 构建代价函数
            cost_fn, breakdown_fn = _build_cost_fn(
                keypoint_3d  = keypoint_3d,
                approach_dir = approach_dir,
                grasp_axis   = sap.grasp_axis,
                avoid_kps    = avoid_kps,
                vlm_decision = vlm_decision,
                mode         = "pick",
            )

            # 完整 SLSQP（200次），结果可靠
            result = minimize(
                fun     = cost_fn,
                x0      = x0,
                method  = 'SLSQP',
                options = {'maxiter': 200, 'ftol': 1e-6, 'disp': False}
            )
            cost_val = float(result.fun)

            if self.verbose:
                print(f"  [{part_name}] approach={np.round(approach_dir, 3)} "
                      f"cost={cost_val:.4f}")

            if cost_val < best_cost:
                best_cost         = cost_val
                best_cost_fn      = cost_fn
                best_breakdown_fn = breakdown_fn
                best_x0           = result.x
                best_meta_part    = {
                    'part_name':    part_name,
                    'approach_dir': approach_dir,
                    'keypoint_3d':  keypoint_3d,
                    'grasp_axis':   sap.grasp_axis,
                }

        if self.verbose:
            print(f"  ✅ 最优交互点: {best_meta_part['part_name']} "
                  f"(cost={best_cost:.4f})")

        meta = {
            'mode':               'pick',
            'grasp_target':       best_meta_part['part_name'],
            'avoid_targets':      list(avoid_kps.keys()),
            'object_center':      object_center,
            'approach_direction': best_meta_part['approach_dir'],
            'keypoint_3d':        best_meta_part['keypoint_3d'],
            'grasp_axis_target':  best_meta_part['grasp_axis'],
            'vlm_decision':       vlm_decision,
            'cost_breakdown_fn':  best_breakdown_fn,
            'candidate_best_cost': best_cost,
        }

        return best_cost_fn, best_x0, meta

    # ==================== 私有：place 路径 ====================

    def _instantiate_place(
            self,
            place_kps:    Dict[str, np.ndarray],
            avoid_kps:    Dict[str, Tuple[np.ndarray, float]],
            vlm_decision: VLMDecision,
            T_current:    np.ndarray,
    ) -> Tuple[Callable, np.ndarray, Dict]:
        """
        Place 模式：单一 surface 目标，构建含 C1/C2/C4 的代价函数。
        末端从上方（[0,0,-1]）接近，物体平稳落到承载面。
        """
        place_target_name  = next(iter(place_kps))
        place_target_point = place_kps[place_target_name]
        place_sap          = get_sap_strict(place_target_name)

        # place 模式接近方向固定为 [0,0,-1]（始终从上方放下）
        approach_dir = place_sap.approach_direction.copy()  # [0,0,-1]

        # x0 位置：hand link 放在"指尖对准关键点"的位置
        x0_pos = place_target_point - approach_dir * FINGER_LENGTH
        # x0 旋转：从 T_current 提取，Q_HOME 时夹爪 Z ≈ [0,0,-1]，与 approach_dir 对齐
        R_current = T_current[:3, :3]
        try:
            from scipy.spatial.transform import Rotation
            x0_rot = Rotation.from_matrix(R_current).as_rotvec()
        except Exception:
            x0_rot = _rodrigues_inverse(R_current)

        x0 = np.concatenate([x0_pos, x0_rot])

        # 构建代价函数（place 模式无 C3，grasp_axis 传 None）
        cost_fn, breakdown_fn = _build_cost_fn(
            keypoint_3d  = place_target_point,
            approach_dir = approach_dir,
            grasp_axis   = None,
            avoid_kps    = avoid_kps,
            vlm_decision = vlm_decision,
            mode         = "place",
        )

        if self.verbose:
            print(f"\n📐 ConstraintInstantiator [place 模式]")
            print(f"  放置目标: {place_target_name}")
            print(f"  avoid 目标: {list(avoid_kps.keys())}")
            print(f"  surface 坐标: {np.round(place_target_point, 3)}")
            print(f"  VLM 决策:     {vlm_decision}")

        meta = {
            'mode':               'place',
            'place_target':       place_target_name,
            'avoid_targets':      list(avoid_kps.keys()),
            'approach_direction': approach_dir,
            'keypoint_3d':        place_target_point,
            'vlm_decision':       vlm_decision,
            'cost_breakdown_fn':  breakdown_fn,
        }

        return cost_fn, x0, meta

