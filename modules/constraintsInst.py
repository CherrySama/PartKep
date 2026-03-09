"""
Created by Yinghao Ho on 2026-2-23

代价函数实例化模块
核心职责：
    将语义关键点 + SAP 知识库 → 可被 SLSQP 直接调用的数值代价函数

支持两种模式，由输入关键点的 contact_mode 自动判断：

    【pick 模式】keypoints_3d 含 grasp 部件（handle/neck/cap/body）
        代价项：
            cost_approach  末端执行器位置与目标接触点的偏差
            cost_grasp     夹爪张开轴与目标方向的对齐偏差
            cost_safety    末端执行器与 avoid 部件的安全距离惩罚

    【place 模式】keypoints_3d 含 place 部件（surface）
        代价项：
            cost_place     末端执行器位置与放置目标点的偏差
            cost_safety    末端执行器与 avoid 部件的安全距离惩罚（tray rim）
        注：place 模式无 grasp axis 对齐项

优化变量 x（6D）：
    [px, py, pz, ax, ay, az]
    - (px, py, pz)：末端执行器位置（世界坐标系，单位：米）
    - (ax, ay, az)：末端执行器旋转（轴角表示，||a|| 为旋转角，rad）

夹爪局部坐标系约定（pick 模式）：
    - 局部 Z 轴：夹爪接近轴
    - 局部 Y 轴：夹爪张开轴
"""

import numpy as np
from typing import Dict, Tuple, Callable

from configs.SAP import get_sap_strict, SAP
from utils import CoordinateTransformer


# ==================== 超参数 ====================
W_APPROACH = 1.0    # 接近方向代价权重（pick + place）
W_GRASP    = 0.5    # 夹爪对齐代价权重（pick 专用）
W_SAFETY   = 2.0    # 安全距离代价权重（pick + place）

APPROACH_OFFSET    = 0.05   # 末端执行器与接触点的偏移距离（米）
INIT_HEIGHT_OFFSET = 0.15   # 初始猜测：末端在物体正上方的高度偏移（米）


# ==================== 辅助函数 ====================

def _compute_actual_approach(part_name: str,
                              sap: SAP,
                              keypoint_3d: np.ndarray,
                              object_center: np.ndarray) -> np.ndarray:
    """
    运行时修正 approach_direction

    策略：
        - body / cap / surface（从上方接近）：SAP 已定义 [0,0,-1]，直接使用
        - handle / neck（侧向抓取）：SAP 存的是参考方向 [1,0,0]，
          运行时用关键点相对物体中心的水平方向替换，
          使接近方向从物体外部指向关键点。

    Args:
        part_name:     部件名称
        sap:           对应的 SAP 实例
        keypoint_3d:   该部件的3D关键点（世界坐标）
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
        return np.array([approach_horizontal[0], approach_horizontal[1], 0.0])
    else:
        return sap.approach_direction.copy()


# ==================== 主类 ====================

class ConstraintInstantiator:
    """
    代价函数实例化器

    根据输入关键点的 SAP contact_mode 自动选择 pick 或 place 模式：
        - 含 grasp 部件 → pick 模式（抓取约束）
        - 含 place 部件 → place 模式（放置约束）
        - 两者同时存在 → 抛出 ValueError（语义冲突）

    对外接口不变，下游 PoseSolver 无需修改。

    使用示例（pick）：
        >>> inst = ConstraintInstantiator(object_label='cup')
        >>> cost_fn, x0, meta = inst.instantiate({
        ...     'handle': np.array([0.45, 0.07, 0.10]),
        ...     'rim':    np.array([0.45, 0.00, 0.16]),
        ...     'body':   np.array([0.45, 0.00, 0.08]),
        ... })

    使用示例（place）：
        >>> inst = ConstraintInstantiator(object_label='tray')
        >>> cost_fn, x0, meta = inst.instantiate({
        ...     'surface': np.array([0.60, 0.10, 0.02]),
        ...     'rim':     np.array([0.60, 0.10, 0.05]),
        ... })
    """

    def __init__(self,
                 object_label:   str,
                 w_approach:     float = W_APPROACH,
                 w_grasp:        float = W_GRASP,
                 w_safety:       float = W_SAFETY,
                 approach_offset: float = APPROACH_OFFSET):
        self.object_label    = object_label
        self.w_approach      = w_approach
        self.w_grasp         = w_grasp
        self.w_safety        = w_safety
        self.approach_offset = approach_offset

    # ==================== 公开接口 ====================

    def instantiate(
            self,
            keypoints_3d: Dict[str, np.ndarray]
    ) -> Tuple[Callable[[np.ndarray], float], np.ndarray, Dict]:
        """
        实例化代价函数（自动选择 pick / place 模式）

        Args:
            keypoints_3d: {part_name → 3D世界坐标 np.ndarray shape=(3,)}

        Returns:
            cost_fn : Callable，SLSQP 目标函数
            x0      : np.ndarray shape=(6,)，优化初始猜测
            meta    : Dict，调试信息（含 'mode' 字段区分 pick/place）

        Raises:
            ValueError: 同时含有 grasp 和 place 部件（语义冲突）
            ValueError: 既无 grasp 也无 place 部件
            KeyError:   部件名称不在 SAP 知识库中
        """
        # ===== 1. 分类关键点 =====
        grasp_kps = {}
        place_kps = {}
        avoid_kps = {}

        for part_name, point in keypoints_3d.items():
            sap = get_sap_strict(part_name)
            pt  = np.array(point, dtype=np.float64)
            if sap.contact_mode == 'grasp':
                grasp_kps[part_name] = pt
            elif sap.contact_mode == 'place':
                place_kps[part_name] = pt
            else:  # 'avoid'
                avoid_kps[part_name] = (pt, sap.safety_margin)

        # ===== 2. 模式判断 =====
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

        # ===== 3. 分发到对应路径 =====
        if grasp_kps:
            return self._instantiate_pick(grasp_kps, avoid_kps)
        else:
            return self._instantiate_place(place_kps, avoid_kps)

    # ==================== 私有：pick 路径 ====================

    def _instantiate_pick(
            self,
            grasp_kps: Dict[str, np.ndarray],
            avoid_kps: Dict[str, Tuple[np.ndarray, float]]
    ) -> Tuple[Callable, np.ndarray, Dict]:
        """pick 模式：生成抓取代价函数"""

        # 按优先级选定主抓取目标
        PRIORITY = ["handle", "neck", "cap", "body"]
        grasp_target_name = next(
            (c for c in PRIORITY if c in grasp_kps),
            next(iter(grasp_kps))
        )
        grasp_target_point = grasp_kps[grasp_target_name]
        grasp_sap          = get_sap_strict(grasp_target_name)

        # 物体中心（所有 grasp + avoid 关键点均值）
        all_points    = list(grasp_kps.values()) + [v[0] for v in avoid_kps.values()]
        object_center = np.mean(np.stack(all_points), axis=0)

        # 运行时修正接近方向
        approach_dir = _compute_actual_approach(
            grasp_target_name, grasp_sap, grasp_target_point, object_center
        )

        # 目标接触点 = 关键点沿接近方向退后 offset
        contact_point     = grasp_target_point - approach_dir * self.approach_offset
        grasp_axis_target = grasp_sap.grasp_axis.copy()

        # 初始猜测：物体中心正上方
        x0_pos    = object_center.copy()
        x0_pos[2] += INIT_HEIGHT_OFFSET
        x0        = np.concatenate([x0_pos, np.zeros(3)])

        # 代价函数闭包（固化所有参数）
        _cp   = contact_point.copy()
        _gat  = grasp_axis_target.copy()
        _avd  = {k: (v[0].copy(), v[1]) for k, v in avoid_kps.items()}
        _wa, _wg, _ws = self.w_approach, self.w_grasp, self.w_safety

        def cost_fn(x: np.ndarray) -> float:
            pos  = x[:3]
            rvec = x[3:]
            R    = CoordinateTransformer.rodrigues(rvec)

            c_approach = float(np.dot(pos - _cp, pos - _cp))

            gripper_open_axis = R @ np.array([0.0, 1.0, 0.0])
            c_grasp = 1.0 - np.dot(gripper_open_axis, _gat) ** 2

            c_safety = 0.0
            for p_avoid, margin in _avd.values():
                v = margin - np.linalg.norm(pos - p_avoid)
                if v > 0:
                    c_safety += v ** 2

            return float(_wa * c_approach + _wg * c_grasp + _ws * c_safety)

        def cost_breakdown_fn(x: np.ndarray) -> Dict:
            pos  = x[:3]
            rvec = x[3:]
            R    = CoordinateTransformer.rodrigues(rvec)

            c_approach = float(np.dot(pos - _cp, pos - _cp))
            c_grasp    = 1.0 - np.dot(R @ np.array([0., 1., 0.]), _gat) ** 2
            c_safety, safety_per_part = 0.0, {}
            for part, (p_avoid, margin) in _avd.items():
                v = margin - np.linalg.norm(pos - p_avoid)
                c = v ** 2 if v > 0 else 0.0
                safety_per_part[part] = c
                c_safety += c

            total = _wa * c_approach + _wg * c_grasp + _ws * c_safety
            return {
                'total':           float(total),
                'approach':        float(_wa * c_approach),
                'grasp':           float(_wg * c_grasp),
                'safety':          float(_ws * c_safety),
                'safety_per_part': safety_per_part,
            }

        meta = {
            'mode':               'pick',
            'grasp_target':       grasp_target_name,
            'avoid_targets':      list(avoid_kps.keys()),
            'object_center':      object_center,
            'approach_direction': approach_dir,
            'contact_point':      contact_point,
            'grasp_axis_target':  grasp_axis_target,
            'cost_breakdown_fn':  cost_breakdown_fn,
        }

        print(f"\n📐 代价函数实例化完成 [{self.object_label}]  mode=pick")
        print(f"  主抓取目标: {grasp_target_name}")
        print(f"  回避目标:   {list(avoid_kps.keys())}")
        print(f"  物体中心:   {np.round(object_center, 3)}")
        print(f"  接近方向:   {np.round(approach_dir, 3)}")
        print(f"  目标接触点: {np.round(contact_point, 3)}")
        print(f"  夹爪张开轴: {np.round(grasp_axis_target, 3)}")
        print(f"  初始猜测x0: {np.round(x0, 3)}")

        return cost_fn, x0, meta

    # ==================== 私有：place 路径 ====================

    def _instantiate_place(
            self,
            place_kps: Dict[str, np.ndarray],
            avoid_kps: Dict[str, Tuple[np.ndarray, float]]
    ) -> Tuple[Callable, np.ndarray, Dict]:
        """
        place 模式：生成放置代价函数

        末端执行器（带着已抓取的物体）从上方接近 surface 关键点，
        在 surface 正上方 approach_offset 处停止，松开夹爪完成放置。

        代价函数：
            cost_place  = ||pos - (surface_point - approach_dir * offset)||²
            cost_safety = Σ max(0, margin - dist)²   （tray rim 回避）
        """
        # place 目标只取第一个 surface（通常只有一个）
        place_target_name  = next(iter(place_kps))
        place_target_point = place_kps[place_target_name]
        place_sap          = get_sap_strict(place_target_name)

        # surface 的接近方向固定为 [0,0,-1]（从上方），无需运行时修正
        approach_dir = place_sap.approach_direction.copy()

        # 目标停靠点：surface 正上方 offset 处
        target_point = place_target_point - approach_dir * self.approach_offset

        # 初始猜测：目标停靠点再上方 INIT_HEIGHT_OFFSET
        x0_pos    = target_point.copy()
        x0_pos[2] += INIT_HEIGHT_OFFSET
        x0        = np.concatenate([x0_pos, np.zeros(3)])

        # 代价函数闭包
        _tp  = target_point.copy()
        _avd = {k: (v[0].copy(), v[1]) for k, v in avoid_kps.items()}
        _wa, _ws = self.w_approach, self.w_safety

        def cost_fn(x: np.ndarray) -> float:
            pos = x[:3]
            # place 模式不使用旋转变量（x[3:] 保持为零）

            c_place = float(np.dot(pos - _tp, pos - _tp))

            c_safety = 0.0
            for p_avoid, margin in _avd.values():
                v = margin - np.linalg.norm(pos - p_avoid)
                if v > 0:
                    c_safety += v ** 2

            return float(_wa * c_place + _ws * c_safety)

        def cost_breakdown_fn(x: np.ndarray) -> Dict:
            pos      = x[:3]
            c_place  = float(np.dot(pos - _tp, pos - _tp))
            c_safety, safety_per_part = 0.0, {}
            for part, (p_avoid, margin) in _avd.items():
                v = margin - np.linalg.norm(pos - p_avoid)
                c = v ** 2 if v > 0 else 0.0
                safety_per_part[part] = c
                c_safety += c

            total = _wa * c_place + _ws * c_safety
            return {
                'total':           float(total),
                'place':           float(_wa * c_place),
                'grasp':           0.0,         # place 模式无此项
                'approach':        float(_wa * c_place),  # 统一字段名供 PoseSolver 读取
                'safety':          float(_ws * c_safety),
                'safety_per_part': safety_per_part,
            }

        meta = {
            'mode':               'place',
            'place_target':       place_target_name,
            'avoid_targets':      list(avoid_kps.keys()),
            'approach_direction': approach_dir,
            'target_point':       target_point,
            'cost_breakdown_fn':  cost_breakdown_fn,
        }

        print(f"\n📐 代价函数实例化完成 [{self.object_label}]  mode=place")
        print(f"  放置目标部件: {place_target_name}")
        print(f"  回避目标:     {list(avoid_kps.keys())}")
        print(f"  surface 坐标: {np.round(place_target_point, 3)}")
        print(f"  接近方向:     {np.round(approach_dir, 3)}")
        print(f"  目标停靠点:   {np.round(target_point, 3)}")
        print(f"  初始猜测x0:   {np.round(x0, 3)}")

        return cost_fn, x0, meta


# ==================== 模块测试 ====================
if __name__ == "__main__":
    print("=" * 70)
    print("测试 ConstraintInstantiator（pick + place）")
    print("=" * 70)

    # ===== pick 测试（原有，回归验证） =====
    print("\n【1】cup — pick 模式，handle 侧向抓取")
    inst = ConstraintInstantiator(object_label='cup')
    cost_fn, x0, meta = inst.instantiate({
        'handle': np.array([0.08,  0.00, 0.06]),
        'rim':    np.array([0.00,  0.00, 0.10]),
        'body':   np.array([0.00,  0.00, 0.05]),
    })
    assert meta['mode'] == 'pick'
    assert meta['grasp_target'] == 'handle'
    bd = meta['cost_breakdown_fn'](x0)
    assert abs(bd['total'] - cost_fn(x0)) < 1e-10
    print(f"  ✅ mode=pick, grasp_target=handle, cost={cost_fn(x0):.4f}")

    print("\n【2】bottle — pick 模式，neck 优先")
    inst_b = ConstraintInstantiator(object_label='bottle')
    _, x0_b, meta_b = inst_b.instantiate({
        'neck': np.array([0.02, 0.00, 0.18]),
        'cap':  np.array([0.00, 0.00, 0.22]),
        'body': np.array([0.00, 0.00, 0.10]),
    })
    assert meta_b['mode'] == 'pick'
    assert meta_b['grasp_target'] == 'neck'
    print(f"  ✅ mode=pick, grasp_target=neck")

    print("\n【3】bowl — pick 模式，body + rim 安全约束")
    inst_bowl = ConstraintInstantiator(object_label='bowl')
    _, _, meta_bowl = inst_bowl.instantiate({
        'rim':  np.array([0.00, 0.00, 0.08]),
        'body': np.array([0.00, 0.00, 0.04]),
    })
    assert meta_bowl['mode'] == 'pick'
    assert meta_bowl['grasp_target'] == 'body'
    assert 'rim' in meta_bowl['avoid_targets']
    print(f"  ✅ mode=pick, grasp_target=body, avoid=['rim']")

    # ===== place 测试（新增） =====
    print("\n【4】tray — place 模式，surface + rim 回避")
    inst_tray = ConstraintInstantiator(object_label='tray')
    cost_fn_t, x0_t, meta_t = inst_tray.instantiate({
        'surface': np.array([0.60, 0.10, 0.02]),
        'rim':     np.array([0.60, 0.10, 0.05]),
    })
    assert meta_t['mode'] == 'place'
    assert meta_t['place_target'] == 'surface'
    assert 'rim' in meta_t['avoid_targets']
    bd_t = meta_t['cost_breakdown_fn'](x0_t)
    assert abs(bd_t['total'] - cost_fn_t(x0_t)) < 1e-10
    print(f"  ✅ mode=place, place_target=surface, avoid=['rim']")
    print(f"     目标停靠点: {np.round(meta_t['target_point'], 3)}")
    print(f"     初始代价:   {cost_fn_t(x0_t):.4f}")

    print("\n【5】table — place 模式，仅 surface（无回避）")
    inst_table = ConstraintInstantiator(object_label='table')
    cost_fn_tb, x0_tb, meta_tb = inst_table.instantiate({
        'surface': np.array([0.55, 0.00, 0.00]),
    })
    assert meta_tb['mode'] == 'place'
    assert meta_tb['avoid_targets'] == []
    print(f"  ✅ mode=place, avoid=[]")
    print(f"     目标停靠点: {np.round(meta_tb['target_point'], 3)}")

    # ===== 错误处理测试 =====
    print("\n【6】grasp + place 同时存在 → 应报错")
    try:
        inst.instantiate({
            'handle':  np.array([0.1, 0.0, 0.1]),
            'surface': np.array([0.6, 0.1, 0.0]),
        })
        print("  ❌ 未报错！")
    except ValueError as e:
        print(f"  ✅ 正确抛出 ValueError: {str(e).splitlines()[0]}")

    print("\n【7】只有 avoid 部件 → 应报错")
    try:
        inst.instantiate({'rim': np.array([0.0, 0.0, 0.1])})
        print("  ❌ 未报错！")
    except ValueError as e:
        print(f"  ✅ 正确抛出 ValueError: {str(e).splitlines()[0]}")

    print("\n" + "=" * 70)
    print("✅ ConstraintInstantiator 所有测试通过！")
    print("=" * 70)
    