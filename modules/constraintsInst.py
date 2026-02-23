"""
Created by Yinghao Ho on 2026-2-23

代价函数实例化模块

核心职责：
    将语义关键点 + SAP 知识库 → 可被 SLSQP 直接调用的数值代价函数

优化变量 x（6D）：
    [px, py, pz, ax, ay, az]
    - (px, py, pz)：末端执行器位置（世界坐标系，单位：米）
    - (ax, ay, az)：末端执行器旋转（轴角表示，||a|| 为旋转角，rad）

夹爪局部坐标系约定：
    - 局部 Z 轴：夹爪接近轴（手指朝向目标的方向）
    - 局部 Y 轴：夹爪张开轴（两指分开的方向）

三个代价项：
    cost_approach：末端执行器位置与目标接触点的偏差
    cost_grasp：夹爪张开轴与目标方向的对齐偏差
    cost_safety：末端执行器与 avoid 部件关键点的安全距离惩罚

参考文献：
    - kPAM (ISRR 2019)：语义关键点上的几何代价和约束
    - ReKep (2024)：关键点映射到数值代价，SLSQP 求解 SE(3)
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional

from configs.sap_knowledge import get_sap_strict, SAP


# ==================== 超参数 ====================
W_APPROACH = 1.0    # 接近方向代价权重
W_GRASP    = 0.5    # 夹爪对齐代价权重
W_SAFETY   = 2.0    # 安全距离代价权重（惩罚更重，确保不碰 rim）

# 接触点偏移距离：末端执行器停在关键点前方的距离（米）
APPROACH_OFFSET = 0.05  # 5cm

# 初始猜测：末端执行器在物体正上方 offset 处
INIT_HEIGHT_OFFSET = 0.15   # 15cm


# ==================== 辅助函数 ====================

def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    """
    轴角向量 → 3×3 旋转矩阵（Rodrigues 公式）

    Args:
        rvec: shape=(3,) 轴角向量，方向为旋转轴，模长为旋转角（rad）
              零向量对应单位旋转矩阵

    Returns:
        np.ndarray shape=(3, 3)：旋转矩阵

    推导：
        设 θ = ||rvec||，k = rvec / θ（单位旋转轴）
        R = I + sin(θ)*K + (1-cos(θ))*K²
        其中 K 是 k 的反对称矩阵（叉积矩阵）
    """
    rvec = np.array(rvec, dtype=np.float64)
    theta = np.linalg.norm(rvec)

    if theta < 1e-10:
        return np.eye(3)

    k = rvec / theta

    K = np.array([
        [   0.0, -k[2],  k[1]],
        [ k[2],    0.0, -k[0]],
        [-k[1],  k[0],    0.0]
    ])

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def _compute_actual_approach(part_name: str,
                              sap: SAP,
                              keypoint_3d: np.ndarray,
                              object_center: np.ndarray) -> np.ndarray:
    """
    运行时修正 approach_direction

    策略：
        - body / cap（从上方抓取）：SAP 已定义 [0,0,-1]，直接使用
        - handle / neck（侧向抓取）：SAP 存的是参考方向 [1,0,0]，
          运行时用关键点相对物体中心的水平方向替换，
          使接近方向从物体外部指向关键点。

    Args:
        part_name: 部件名称
        sap: 对应的 SAP 实例
        keypoint_3d: 该部件的3D关键点（世界坐标）
        object_center: 物体中心（所有关键点均值）

    Returns:
        np.ndarray shape=(3,)：修正后的单位接近方向向量
    """
    LATERAL_PARTS = {"handle", "neck"}

    if part_name in LATERAL_PARTS:
        delta_xy = keypoint_3d[:2] - object_center[:2]
        norm = np.linalg.norm(delta_xy)

        if norm < 1e-6:
            return sap.approach_direction.copy()

        approach_horizontal = delta_xy / norm
        return np.array([approach_horizontal[0],
                         approach_horizontal[1],
                         0.0])
    else:
        return sap.approach_direction.copy()


# ==================== 主类 ====================

class ConstraintInstantiator:
    """
    代价函数实例化器

    输入：语义关键点的3D世界坐标字典
    输出：SLSQP 可直接调用的目标函数 + 初始猜测 + 调试信息

    使用示例：
        >>> keypoints_3d = {
        ...     'handle': np.array([0.15, 0.05, 0.08]),
        ...     'rim':    np.array([0.00, 0.00, 0.12]),
        ...     'body':   np.array([0.00, 0.00, 0.06]),
        ... }
        >>> instantiator = ConstraintInstantiator(object_label='cup')
        >>> cost_fn, x0, meta = instantiator.instantiate(keypoints_3d)
        >>>
        >>> from scipy.optimize import minimize
        >>> result = minimize(cost_fn, x0, method='SLSQP')
    """

    def __init__(self,
                 object_label: str,
                 w_approach: float = W_APPROACH,
                 w_grasp: float = W_GRASP,
                 w_safety: float = W_SAFETY,
                 approach_offset: float = APPROACH_OFFSET):
        """
        Args:
            object_label: 物体类别（仅用于日志）
            w_approach: 接近方向代价权重
            w_grasp: 夹爪对齐代价权重
            w_safety: 安全距离代价权重
            approach_offset: 末端执行器与接触点的偏移距离（米）
        """
        self.object_label    = object_label
        self.w_approach      = w_approach
        self.w_grasp         = w_grasp
        self.w_safety        = w_safety
        self.approach_offset = approach_offset

    def instantiate(
            self,
            keypoints_3d: Dict[str, np.ndarray]
    ) -> Tuple[Callable[[np.ndarray], float], np.ndarray, Dict]:
        """
        实例化代价函数

        Args:
            keypoints_3d: 部件名称 → 3D世界坐标
                {
                    'handle': np.array([x, y, z]),
                    'rim':    np.array([x, y, z]),
                    ...
                }

        Returns:
            cost_fn:  Callable[[np.ndarray], float]，目标函数
            x0:       np.ndarray shape=(6,)，优化初始猜测
            meta:     Dict，调试信息

        Raises:
            ValueError: keypoints_3d 中没有可抓取部件
            KeyError: 部件名称不在 SAP 知识库中
        """
        # ===== 1. 按 contact_mode 分类关键点 =====
        grasp_keypoints = {}
        avoid_keypoints = {}

        for part_name, point in keypoints_3d.items():
            sap = get_sap_strict(part_name)
            if sap.contact_mode == 'grasp':
                grasp_keypoints[part_name] = np.array(point, dtype=np.float64)
            else:
                avoid_keypoints[part_name] = (
                    np.array(point, dtype=np.float64),
                    sap.safety_margin
                )

        if len(grasp_keypoints) == 0:
            raise ValueError(
                f"keypoints_3d 中没有可抓取部件（contact_mode='grasp'）。\n"
                f"输入的部件: {list(keypoints_3d.keys())}"
            )

        # ===== 2. 按优先级选定主抓取目标 =====
        # handle > neck > cap > body
        PRIORITY = ["handle", "neck", "cap", "body"]
        grasp_target_name = None
        for candidate in PRIORITY:
            if candidate in grasp_keypoints:
                grasp_target_name = candidate
                break
        if grasp_target_name is None:
            grasp_target_name = next(iter(grasp_keypoints))

        grasp_target_point = grasp_keypoints[grasp_target_name]
        grasp_sap = get_sap_strict(grasp_target_name)

        # ===== 3. 估计物体中心（所有关键点均值）=====
        all_points = [np.array(p, dtype=np.float64)
                      for p in keypoints_3d.values()]
        object_center = np.mean(np.stack(all_points), axis=0)

        # ===== 4. 运行时修正 approach_direction =====
        approach_dir = _compute_actual_approach(
            part_name=grasp_target_name,
            sap=grasp_sap,
            keypoint_3d=grasp_target_point,
            object_center=object_center
        )

        # ===== 5. 计算目标接触点 =====
        # 末端执行器停在关键点沿接近方向反向 offset 处
        # approach_dir 指向目标，所以末端在目标的 approach_dir 反方向
        contact_point = grasp_target_point - approach_dir * self.approach_offset

        # ===== 6. 夹爪张开轴目标方向 =====
        grasp_axis_target = grasp_sap.grasp_axis.copy()

        # ===== 7. 构造初始猜测 x0 =====
        x0_pos = object_center.copy()
        x0_pos[2] += INIT_HEIGHT_OFFSET
        x0 = np.concatenate([x0_pos, np.zeros(3)])

        # ===== 8. 构造代价函数（闭包） =====
        # 用 .copy() 防止外部修改影响闭包内的捕获值
        _contact_point     = contact_point.copy()
        _grasp_axis_target = grasp_axis_target.copy()
        _avoid_keypoints   = {k: (v[0].copy(), v[1])
                              for k, v in avoid_keypoints.items()}
        _w_approach        = self.w_approach
        _w_grasp           = self.w_grasp
        _w_safety          = self.w_safety

        def cost_fn(x: np.ndarray) -> float:
            """
            总代价函数

            Args:
                x: shape=(6,) [px, py, pz, ax, ay, az]

            Returns:
                标量代价值
            """
            pos  = x[:3]
            rvec = x[3:]
            R    = _rodrigues(rvec)

            # 代价1：位置接近代价
            diff = pos - _contact_point
            c_approach = float(np.dot(diff, diff))

            # 代价2：夹爪对齐代价
            # 夹爪局部 Y 轴在世界坐标系下的方向
            gripper_open_axis = R @ np.array([0.0, 1.0, 0.0])
            cos_angle = np.dot(gripper_open_axis, _grasp_axis_target)
            c_grasp = 1.0 - cos_angle ** 2

            # 代价3：安全距离代价
            c_safety = 0.0
            for p_avoid, margin in _avoid_keypoints.values():
                dist = np.linalg.norm(pos - p_avoid)
                violation = margin - dist
                if violation > 0:
                    c_safety += violation ** 2

            return float(_w_approach * c_approach +
                         _w_grasp    * c_grasp +
                         _w_safety   * c_safety)

        def cost_breakdown_fn(x: np.ndarray) -> Dict:
            """
            返回各项代价的分解（调试用）

            Returns:
                {
                    'total': float,
                    'approach': float,
                    'grasp': float,
                    'safety': float,
                    'safety_per_part': {part_name: float}
                }
            """
            pos  = x[:3]
            rvec = x[3:]
            R    = _rodrigues(rvec)

            diff       = pos - _contact_point
            c_approach = float(np.dot(diff, diff))

            gripper_open_axis = R @ np.array([0.0, 1.0, 0.0])
            cos_angle  = np.dot(gripper_open_axis, _grasp_axis_target)
            c_grasp    = 1.0 - cos_angle ** 2

            c_safety = 0.0
            safety_per_part = {}
            for part, (p_avoid, margin) in _avoid_keypoints.items():
                dist      = np.linalg.norm(pos - p_avoid)
                violation = margin - dist
                c_part    = violation ** 2 if violation > 0 else 0.0
                safety_per_part[part] = c_part
                c_safety += c_part

            total = (_w_approach * c_approach +
                     _w_grasp    * c_grasp +
                     _w_safety   * c_safety)

            return {
                'total':           float(total),
                'approach':        float(_w_approach * c_approach),
                'grasp':           float(_w_grasp    * c_grasp),
                'safety':          float(_w_safety   * c_safety),
                'safety_per_part': safety_per_part
            }

        # ===== 9. 构造 meta =====
        meta = {
            'grasp_target':       grasp_target_name,
            'avoid_targets':      list(avoid_keypoints.keys()),
            'object_center':      object_center,
            'approach_direction': approach_dir,
            'contact_point':      contact_point,
            'grasp_axis_target':  grasp_axis_target,
            'cost_breakdown_fn':  cost_breakdown_fn
        }

        print(f"\n📐 代价函数实例化完成 [{self.object_label}]")
        print(f"  主抓取目标: {grasp_target_name}")
        print(f"  回避目标:   {list(avoid_keypoints.keys())}")
        print(f"  物体中心:   {np.round(object_center, 3)}")
        print(f"  接近方向:   {np.round(approach_dir, 3)}")
        print(f"  目标接触点: {np.round(contact_point, 3)}")
        print(f"  夹爪张开轴: {np.round(grasp_axis_target, 3)}")
        print(f"  初始猜测x0: {np.round(x0, 3)}")

        return cost_fn, x0, meta


# ==================== 模块测试代码 ====================
if __name__ == "__main__":
    print("=" * 70)
    print("测试 ConstraintInstantiator")
    print("=" * 70)

    # ---------- 测试1：cup（有 handle）----------
    print("\n【测试1】cup - handle 侧向抓取")
    print("-" * 70)

    keypoints_cup = {
        'handle': np.array([0.08,  0.00, 0.06]),
        'rim':    np.array([0.00,  0.00, 0.10]),
        'body':   np.array([0.00,  0.00, 0.05]),
    }

    inst = ConstraintInstantiator(object_label='cup')
    cost_fn, x0, meta = inst.instantiate(keypoints_cup)

    assert meta['grasp_target'] == 'handle'
    assert meta['approach_direction'][0] > 0.9, "handle 接近方向应朝 +X"
    print(f"  ✅ 主抓取目标: {meta['grasp_target']}")
    print(f"  ✅ 接近方向修正: {np.round(meta['approach_direction'], 3)}")

    cost_val = cost_fn(x0)
    breakdown = meta['cost_breakdown_fn'](x0)
    assert abs(breakdown['total'] - cost_val) < 1e-10
    print(f"  ✅ 代价函数可调用，x0处代价={cost_val:.4f}")
    print(f"     breakdown: approach={breakdown['approach']:.4f}, "
          f"grasp={breakdown['grasp']:.4f}, "
          f"safety={breakdown['safety']:.4f}")

    # ---------- 测试2：bottle（neck 优先）----------
    print("\n【测试2】bottle - neck 侧向抓取")
    print("-" * 70)

    keypoints_bottle = {
        'neck':   np.array([0.02, 0.00, 0.18]),
        'cap':    np.array([0.00, 0.00, 0.22]),
        'body':   np.array([0.00, 0.00, 0.10]),
    }

    inst_b = ConstraintInstantiator(object_label='bottle')
    cost_fn_b, x0_b, meta_b = inst_b.instantiate(keypoints_bottle)

    assert meta_b['grasp_target'] == 'neck'
    print(f"  ✅ 优先级验证: grasp_target={meta_b['grasp_target']}")
    print(f"  ✅ 初始代价={cost_fn_b(x0_b):.4f}")

    # ---------- 测试3：bowl（body 兜底 + rim 安全约束）----------
    print("\n【测试3】bowl - body 兜底 + rim 安全约束")
    print("-" * 70)

    keypoints_bowl = {
        'rim':  np.array([0.00, 0.00, 0.08]),
        'body': np.array([0.00, 0.00, 0.04]),
    }

    inst_bowl = ConstraintInstantiator(object_label='bowl')
    cost_fn_bowl, x0_bowl, meta_bowl = inst_bowl.instantiate(keypoints_bowl)

    assert meta_bowl['grasp_target'] == 'body'
    assert 'rim' in meta_bowl['avoid_targets']

    # 在 rim 处安全代价应 > 0
    x_at_rim = x0_bowl.copy()
    x_at_rim[:3] = keypoints_bowl['rim']
    bd = meta_bowl['cost_breakdown_fn'](x_at_rim)
    assert bd['safety'] > 0
    print(f"  ✅ grasp_target=body, avoid=[rim]")
    print(f"  ✅ 在rim处安全代价={bd['safety']:.4f} > 0")

    # ---------- 测试4：Rodrigues 验证 ----------
    print("\n【测试4】Rodrigues 公式验证")
    print("-" * 70)

    assert np.allclose(_rodrigues(np.zeros(3)), np.eye(3))
    print(f"  ✅ 零旋转 → 单位矩阵")

    R_90z = _rodrigues(np.array([0.0, 0.0, np.pi / 2]))
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert np.allclose(R_90z, expected, atol=1e-6)
    print(f"  ✅ 绕Z轴90°旋转正确")

    R_rand = _rodrigues(np.array([0.3, -0.5, 0.8]))
    assert np.allclose(R_rand.T @ R_rand, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R_rand), 1.0, atol=1e-10)
    print(f"  ✅ 任意旋转满足正交性（det=1，R^T R=I）")

    # ---------- 测试5：无 graspable 部件报错 ----------
    print("\n【测试5】无可抓取部件时报错")
    print("-" * 70)

    try:
        inst.instantiate({'rim': np.array([0.0, 0.0, 0.1])})
        print("  ❌ 应该报错但没有！")
    except ValueError:
        print(f"  ✅ 正确抛出 ValueError")

    print()
    print("=" * 70)
    print("✅ ConstraintInstantiator 所有测试通过！")
    print("=" * 70)