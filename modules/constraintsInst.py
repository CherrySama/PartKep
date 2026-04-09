"""
Created by Yinghao Ho on 2026-2-23

代价函数实例化模块
核心职责：
    将语义关键点 + SAP 知识库 + VLMDecision → 可被 SLSQP 直接调用的数值代价函数

约束体系（共 5 个）：
    ┌────────┬──────────────────────────────┬──────────┬──────────────┐
    │ 编号   │ 含义                         │ 控制方   │ 适用模式     │
    ├────────┼──────────────────────────────┼──────────┼──────────────┤
    │ C1     │ 末端位置对齐目标接触点       │ 固定     │ pick + place │
    │ C2     │ 夹爪 Z 轴对齐接近方向        │ 固定     │ pick + place │
    │ C3     │ 夹爪 Y 轴对齐 SAP grasp_axis │ VLM      │ pick 专用    │
    │ C4     │ 安全距离惩罚                 │ VLM      │ pick + place │
    │ C5     │ 末端保持竖直（anti-tilt）    │ VLM      │ pick + place │
    └────────┴──────────────────────────────┴──────────┴──────────────┘

    C1/C2 属于 SAP 几何职责，始终激活，权重固定。
    C3/C4/C5 属于语义约束，权重由 VLMDecision 提供。

归一化说明：
    - C1：||pos - contact_point||²（米²，固定权重保证量纲一致）
    - C2：1 - dot(gripper_Z, approach_dir)²         → [0, 1]
    - C3：1 - dot(gripper_Y, grasp_axis)²           → [0, 1]
    - C4：Σ max(0, 1 - dist/margin)²                → [0, N_avoid]
    - C5：1 - gripper_Z_world[2]²                   → [0, 1]
          （惩罚夹爪 Z 轴偏离竖直方向，0 = 完全竖直）

最优交互点选择机制：
    Pick 模式对每个 grasp 候选关键点分别构建 cost_fn 并运行快速 SLSQP（50次），
    选取最终代价最低的候选作为最优交互点，返回其 cost_fn 和精化后的 x0 供
    PoseSolver 做最终精细优化（200次）。最优交互点由优化过程自然涌现，
    不由任何硬编码优先级决定。

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
W_APPROACH_ROT = 1.0    # C2 接近方向旋转对齐（固定）

# ==================== 几何参数 ====================
APPROACH_OFFSET    = 0.05   # 末端与接触点的偏移距离（米）
INIT_HEIGHT_OFFSET = 0.15   # 初始猜测额外高度偏移（米，仅 fallback 用）

# 快速候选筛选的 SLSQP 迭代次数（在 instantiate 内部使用）
CANDIDATE_MAX_ITER = 50


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
    LATERAL_PARTS = {"handle", "neck"}

    if part_name in LATERAL_PARTS:
        delta_xy = keypoint_3d[:2] - object_center[:2]
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
        contact_point:  np.ndarray,
        approach_dir:   np.ndarray,
        grasp_axis:     Optional[np.ndarray],
        avoid_kps:      Dict[str, Tuple[np.ndarray, float]],
        vlm_decision:   VLMDecision,
        mode:           str,
) -> Tuple[Callable, Callable]:
    """
    构建单个候选关键点的代价函数

    Args:
        contact_point  : 末端目标停靠点（关键点 - approach_dir * offset）
        approach_dir   : 归一化接近方向（运行时修正后）
        grasp_axis     : 归一化夹爪张开轴（pick 专用，place 传 None）
        avoid_kps      : {part_name → (point_3d, margin)} 需回避的部件
        vlm_decision   : VLM 约束决策（C3/C4/C5 权重）
        mode           : "pick" 或 "place"

    Returns:
        (cost_fn, cost_breakdown_fn) 元组
    """
    # ── 固定参数（闭包捕获） ──
    _cp    = contact_point.copy()
    _app   = approach_dir.copy()
    _gavt  = grasp_axis.copy() if grasp_axis is not None else None
    _avd   = {k: (v[0].copy(), v[1]) for k, v in avoid_kps.items()}
    _mode  = mode

    # ── 固定权重 ──
    _wc1 = W_APPROACH_POS
    _wc2 = W_APPROACH_ROT

    # ── VLM 权重（pick 下 C3 有效，place 下强制 0） ──
    _wc3 = vlm_decision.w_grasp_axis if _mode == "pick" else 0.0
    _wc4 = vlm_decision.w_safety
    _wc5 = vlm_decision.w_upright

    def cost_fn(x: np.ndarray) -> float:
        pos  = x[:3]
        rvec = x[3:]
        R    = CoordinateTransformer.rodrigues(rvec)

        gripper_z = R @ np.array([0.0, 0.0, 1.0])   # 夹爪接近轴（世界系）
        gripper_y = R @ np.array([0.0, 1.0, 0.0])   # 夹爪张开轴（世界系）

        # C1：位置对齐代价（米²）
        c1 = float(np.dot(pos - _cp, pos - _cp))

        # C2：接近方向旋转代价（固定，[0,1]）
        c2 = 1.0 - float(np.dot(gripper_z, _app)) ** 2

        # C3：夹爪张开轴对齐代价（pick 专用，[0,1]）
        c3 = 0.0
        if _gavt is not None:
            c3 = 1.0 - float(np.dot(gripper_y, _gavt)) ** 2

        # C4：安全距离惩罚（归一化，每项 [0,1]）
        c4 = 0.0
        for p_avoid, margin in _avd.values():
            dist = float(np.linalg.norm(pos - p_avoid))
            v    = max(0.0, 1.0 - dist / margin)
            c4  += v * v

        # C5：upright 约束，惩罚夹爪 Z 轴偏离竖直（[0,1]）
        c5 = 1.0 - gripper_z[2] ** 2

        return float(
            _wc1 * c1 +
            _wc2 * c2 +
            _wc3 * c3 +
            _wc4 * c4 +
            _wc5 * c5
        )

    def cost_breakdown_fn(x: np.ndarray) -> Dict:
        pos  = x[:3]
        rvec = x[3:]
        R    = CoordinateTransformer.rodrigues(rvec)

        gripper_z = R @ np.array([0.0, 0.0, 1.0])
        gripper_y = R @ np.array([0.0, 1.0, 0.0])

        c1 = float(np.dot(pos - _cp, pos - _cp))
        c2 = 1.0 - float(np.dot(gripper_z, _app)) ** 2
        c3 = 0.0
        if _gavt is not None:
            c3 = 1.0 - float(np.dot(gripper_y, _gavt)) ** 2

        c4             = 0.0
        safety_per_part = {}
        for pname, (p_avoid, margin) in _avd.items():
            dist = float(np.linalg.norm(pos - p_avoid))
            v    = max(0.0, 1.0 - dist / margin)
            val  = v * v
            safety_per_part[pname] = val
            c4 += val

        c5    = 1.0 - gripper_z[2] ** 2
        total = _wc1*c1 + _wc2*c2 + _wc3*c3 + _wc4*c4 + _wc5*c5

        return {
            'total':            float(total),
            'approach_pos':     float(_wc1 * c1),
            'approach_rot':     float(_wc2 * c2),
            'grasp_axis':       float(_wc3 * c3),
            'safety':           float(_wc4 * c4),
            'upright':          float(_wc5 * c5),
            'safety_per_part':  safety_per_part,
            # 向后兼容 PoseSolver 读取的字段名
            'approach':         float(_wc1 * c1),
            'grasp':            float(_wc3 * c3),
        }

    return cost_fn, cost_breakdown_fn


def _quick_slsqp(cost_fn: Callable, x0: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    快速 SLSQP（50次迭代），用于候选关键点筛选

    Returns:
        (x_opt, final_cost)
    """
    result = minimize(
        fun=cost_fn,
        x0=x0,
        method='SLSQP',
        options={'maxiter': CANDIDATE_MAX_ITER, 'ftol': 1e-4, 'disp': False}
    )
    return result.x, float(result.fun)


# ==================== 主类 ====================

class ConstraintInstantiator:
    """
    代价函数实例化器（重构版）

    根据输入关键点的 SAP contact_mode 自动选择 pick 或 place 模式，
    接收 VLMDecision 提供的语义约束权重，构建完整代价函数。

    Pick 模式：
        对所有 grasp 候选关键点分别构建代价函数并运行快速 SLSQP，
        选取最终代价最低的候选（最优交互点自然涌现），
        返回其 cost_fn 和精化后的 x0 供 PoseSolver 精细优化。

    Place 模式：
        单一 surface 目标，构建含 C1/C2/C4/C5 的代价函数。

    使用示例（pick）：
        >>> from modules.vlmDecider import VLMDecider
        >>> decision = vlm_decider.decide(image, kps_2d, instruction, "pick")
        >>>
        >>> inst = ConstraintInstantiator()
        >>> cost_fn, x0, meta = inst.instantiate(keypoints_3d, decision)

    使用示例（place）：
        >>> decision_place = vlm_decider.decide(image, kps_2d, instruction, "place")
        >>> cost_fn, x0, meta = inst.instantiate(tray_keypoints_3d, decision_place)
    """

    def __init__(self,
                 approach_offset: float = APPROACH_OFFSET,
                 verbose:         bool  = True):
        self.approach_offset = approach_offset
        self.verbose         = verbose

    # ==================== 公开接口 ====================

    def instantiate(
            self,
            keypoints_3d: Dict[str, np.ndarray],
            vlm_decision: VLMDecision,
    ) -> Tuple[Callable, np.ndarray, Dict]:
        """
        实例化代价函数（自动选择 pick / place 模式）

        Args:
            keypoints_3d : {part_name → 3D 世界坐标 shape=(3,)}
            vlm_decision : VLMDecision，来自 VLMDecider.decide()

        Returns:
            cost_fn : Callable，SLSQP 目标函数
            x0      : shape=(6,)，优化初始猜测（叉积法初始化）
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
            return self._instantiate_pick(grasp_kps, avoid_kps, vlm_decision)
        else:
            return self._instantiate_place(place_kps, avoid_kps, vlm_decision)

    # ==================== 私有：pick 路径 ====================

    def _instantiate_pick(
            self,
            grasp_kps:    Dict[str, np.ndarray],
            avoid_kps:    Dict[str, Tuple[np.ndarray, float]],
            vlm_decision: VLMDecision,
    ) -> Tuple[Callable, np.ndarray, Dict]:
        """
        Pick 模式：对所有 grasp 候选关键点各自构建代价函数，
        运行快速 SLSQP 筛选最优交互点，返回其代价函数和精化后的 x0。
        """
        # 物体中心（所有关键点均值，用于运行时修正接近方向）
        all_pts       = list(grasp_kps.values()) + [v[0] for v in avoid_kps.values()]
        object_center = np.mean(np.stack(all_pts), axis=0)

        if self.verbose:
            print(f"\n📐 ConstraintInstantiator [pick 模式]")
            print(f"  grasp 候选: {list(grasp_kps.keys())}")
            print(f"  avoid 目标: {list(avoid_kps.keys())}")
            print(f"  VLM 决策:   {vlm_decision}")

        best_cost     = float('inf')
        best_cost_fn  = None
        best_breakdown_fn = None
        best_x0       = None
        best_meta_part = None

        # ── 遍历所有 grasp 候选，筛选最优交互点 ──
        for part_name, keypoint_3d in grasp_kps.items():
            sap = get_sap_strict(part_name)

            # 运行时修正接近方向
            approach_dir = _compute_actual_approach(
                part_name, sap, keypoint_3d, object_center
            )

            # 目标接触点 = 关键点沿接近方向退后 offset
            contact_point = keypoint_3d - approach_dir * self.approach_offset

            # 叉积法计算 rvec 初始值（修复梯度死区 bug）
            rvec_init = _compute_rvec_init(approach_dir, sap.grasp_axis)

            # 初始 x0：接触点位置 + 叉积法旋转
            x0 = np.concatenate([contact_point, rvec_init])

            # 构建代价函数
            cost_fn, breakdown_fn = _build_cost_fn(
                contact_point = contact_point,
                approach_dir  = approach_dir,
                grasp_axis    = sap.grasp_axis,
                avoid_kps     = avoid_kps,
                vlm_decision  = vlm_decision,
                mode          = "pick",
            )

            # 快速 SLSQP 筛选
            x_quick, cost_quick = _quick_slsqp(cost_fn, x0)

            if self.verbose:
                print(f"  [{part_name}] approach={np.round(approach_dir,3)} "
                      f"quick_cost={cost_quick:.4f}")

            if cost_quick < best_cost:
                best_cost         = cost_quick
                best_cost_fn      = cost_fn
                best_breakdown_fn = breakdown_fn
                best_x0           = x_quick          # 精化后的 x0 给 PoseSolver
                best_meta_part    = {
                    'part_name':    part_name,
                    'approach_dir': approach_dir,
                    'contact_pt':   contact_point,
                    'grasp_axis':   sap.grasp_axis,
                    'keypoint_3d':  keypoint_3d,
                }

        if self.verbose:
            print(f"  ✅ 最优交互点: {best_meta_part['part_name']} "
                  f"(quick_cost={best_cost:.4f})")

        meta = {
            'mode':               'pick',
            'grasp_target':       best_meta_part['part_name'],
            'avoid_targets':      list(avoid_kps.keys()),
            'object_center':      object_center,
            'approach_direction': best_meta_part['approach_dir'],
            'contact_point':      best_meta_part['contact_pt'],
            'grasp_axis_target':  best_meta_part['grasp_axis'],
            'keypoint_3d':        best_meta_part['keypoint_3d'],
            'vlm_decision':       vlm_decision,
            'cost_breakdown_fn':  best_breakdown_fn,
            'candidate_quick_cost': best_cost,
        }

        return best_cost_fn, best_x0, meta

    # ==================== 私有：place 路径 ====================

    def _instantiate_place(
            self,
            place_kps:    Dict[str, np.ndarray],
            avoid_kps:    Dict[str, Tuple[np.ndarray, float]],
            vlm_decision: VLMDecision,
    ) -> Tuple[Callable, np.ndarray, Dict]:
        """
        Place 模式：单一 surface 目标，构建含 C1/C2/C4/C5 的代价函数。
        末端从上方（[0,0,-1]）接近，物体平稳落到承载面。
        """
        place_target_name  = next(iter(place_kps))
        place_target_point = place_kps[place_target_name]
        place_sap          = get_sap_strict(place_target_name)

        # place 模式接近方向固定为 [0,0,-1]（始终从上方放下）
        approach_dir  = place_sap.approach_direction.copy()  # [0,0,-1]

        # 目标停靠点：surface 正上方 offset 处
        contact_point = place_target_point - approach_dir * self.approach_offset

        # 旋转初始值：place 模式夹爪 Z 向下，Y 轴任意水平方向
        rvec_init = _compute_rvec_init(
            approach_dir = approach_dir,
            grasp_axis   = np.array([1.0, 0.0, 0.0]),  # place 模式 Y 轴取 X 方向
        )
        x0 = np.concatenate([contact_point, rvec_init])

        # 构建代价函数（place 模式无 C3，grasp_axis 传 None）
        cost_fn, breakdown_fn = _build_cost_fn(
            contact_point = contact_point,
            approach_dir  = approach_dir,
            grasp_axis    = None,
            avoid_kps     = avoid_kps,
            vlm_decision  = vlm_decision,
            mode          = "place",
        )

        if self.verbose:
            print(f"\n📐 ConstraintInstantiator [place 模式]")
            print(f"  放置目标: {place_target_name}")
            print(f"  avoid 目标: {list(avoid_kps.keys())}")
            print(f"  surface 坐标: {np.round(place_target_point, 3)}")
            print(f"  目标停靠点:   {np.round(contact_point, 3)}")
            print(f"  VLM 决策:     {vlm_decision}")

        meta = {
            'mode':               'place',
            'place_target':       place_target_name,
            'avoid_targets':      list(avoid_kps.keys()),
            'approach_direction': approach_dir,
            'target_point':       contact_point,
            'keypoint_3d':        place_target_point,
            'vlm_decision':       vlm_decision,
            'cost_breakdown_fn':  breakdown_fn,
        }

        return cost_fn, x0, meta


# ==================== 模块测试 ====================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from modules.vlmDecider import VLMDecision, FALLBACK_DECISION

    print("=" * 70)
    print("测试 ConstraintInstantiator（重构版）")
    print("=" * 70)

    fallback = FALLBACK_DECISION

    # ── 【1】cup — pick 模式，handle 侧向抓取 ──
    print("\n【1】cup — pick 模式，handle 优先")
    inst = ConstraintInstantiator(verbose=True)
    cost_fn, x0, meta = inst.instantiate(
        keypoints_3d = {
            'handle': np.array([0.45,  0.07,  0.10]),
            'rim':    np.array([0.45,  0.00,  0.16]),
            'body':   np.array([0.45,  0.00,  0.08]),
        },
        vlm_decision = fallback,
    )
    assert meta['mode'] == 'pick'
    assert meta['grasp_target'] in ['handle', 'body']
    bd = meta['cost_breakdown_fn'](x0)
    assert abs(bd['total'] - cost_fn(x0)) < 1e-8
    print(f"  ✅ mode=pick, grasp_target={meta['grasp_target']}, "
          f"cost={cost_fn(x0):.4f}")

    # ── 【2】C2 旋转代价验证（x0 初始代价应接近 0）──
    print("\n【2】C2 接近方向旋转代价（叉积初始化验证）")
    bd_x0 = meta['cost_breakdown_fn'](x0)
    print(f"  approach_rot(C2) at x0 = {bd_x0['approach_rot']:.6f}  （应接近 0）")
    print(f"  grasp_axis  (C3) at x0 = {bd_x0['grasp_axis']:.6f}   （应接近 0）")
    assert bd_x0['approach_rot'] < 0.01, "C2 初始代价过高，叉积初始化可能失败"
    assert bd_x0['grasp_axis']   < 0.01, "C3 初始代价过高"
    print("  ✅ 叉积法初始化有效，梯度死区修复验证通过")

    # ── 【3】bottle — neck 和 body 同时候选，优化选最优 ──
    print("\n【3】bottle — neck vs body 候选竞争")
    inst_b = ConstraintInstantiator(verbose=True)
    _, _, meta_b = inst_b.instantiate(
        keypoints_3d = {
            'neck': np.array([0.50, 0.00, 0.20]),
            'cap':  np.array([0.50, 0.00, 0.26]),
            'body': np.array([0.50, 0.00, 0.10]),
        },
        vlm_decision = fallback,
    )
    assert meta_b['mode'] == 'pick'
    print(f"  ✅ 最优交互点: {meta_b['grasp_target']}（由优化自然涌现）")

    # ── 【4】tray — place 模式 ──
    print("\n【4】tray — place 模式")
    inst_t = ConstraintInstantiator(verbose=True)
    place_fallback = VLMDecision(
        w_grasp_axis=0.0, w_safety=2.0, w_upright=1.5,
        confidence=0.0, reasoning="fallback", is_fallback=True
    )
    cost_fn_t, x0_t, meta_t = inst_t.instantiate(
        keypoints_3d = {
            'surface': np.array([0.60, 0.10, 0.02]),
            'rim':     np.array([0.60, 0.10, 0.05]),
        },
        vlm_decision = place_fallback,
    )
    assert meta_t['mode'] == 'place'
    assert meta_t['place_target'] == 'surface'
    bd_t = meta_t['cost_breakdown_fn'](x0_t)
    assert abs(bd_t['total'] - cost_fn_t(x0_t)) < 1e-8
    assert bd_t['grasp_axis'] == 0.0  # place 模式 C3 = 0
    print(f"  ✅ mode=place, place_target=surface, "
          f"cost={cost_fn_t(x0_t):.4f}")

    # ── 【5】错误处理 ──
    print("\n【5】grasp + place 同时存在 → 应报错")
    try:
        inst.instantiate(
            {'handle': np.array([0.1, 0.0, 0.1]),
             'surface': np.array([0.6, 0.1, 0.0])},
            fallback
        )
        print("  ❌ 未报错！")
    except ValueError as e:
        print(f"  ✅ 正确报错: {str(e).splitlines()[0]}")

    print("\n" + "=" * 70)
    print("✅ ConstraintInstantiator 所有测试通过！")
    print("=" * 70)
    