"""
Created by Yinghao Ho on 2026-2-23

SAP (Semantic Affordance Primitive) 知识库

核心理论：语义部件标签是操作先验的符号编码。
每个语义部件名称唯一对应一组几何操作属性（SAP），
无需VLM推理，直接通过符号查询获得。

覆盖的物体：cup / bottle / bowl（三种本质不同的抓取模式）
覆盖的部件：handle / rim / body / neck / cap（五个）

SAP属性定义：
    approach_direction: np.ndarray shape=(3,) 或 None
        末端执行器的接近方向向量（归一化，世界坐标系下的参考方向）
        None 表示该部件为 avoid 部件，不作为接触点
    grasp_axis: np.ndarray shape=(3,) 或 None
        夹爪张开轴的目标方向（归一化）
        与 approach_direction 为 None 时同步为 None
    safety_margin: float
        该部件周围需要保持的安全距离（单位：米）
        avoid 部件的 safety_margin > 0，接触部件为 0.0

注意：approach_direction 是参考方向，约束实例化时会根据
关键点的实际位置在水平面内旋转到正确朝向。
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np


@dataclass(frozen=True)
class SAP:
    """
    单个部件的 Semantic Affordance Primitive

    frozen=True：实例创建后不可修改，知识库条目是常量。

    Attributes:
        part_name: 部件名称（与 PartConfig 中的名称一致）
        approach_direction: 接近方向参考向量（归一化），None 表示 avoid 部件
        grasp_axis: 夹爪张开轴方向（归一化），None 表示 avoid 部件
        safety_margin: 安全距离（米），avoid 部件 > 0
        contact_mode: 'grasp'（主动抓取）或 'avoid'（需要绕开）
    """
    part_name: str
    approach_direction: Optional[np.ndarray]
    grasp_axis: Optional[np.ndarray]
    safety_margin: float
    contact_mode: str  # 'grasp' | 'avoid'

    def __post_init__(self):
        # 验证 contact_mode 与其他属性的一致性
        if self.contact_mode == 'avoid':
            if self.approach_direction is not None or self.grasp_axis is not None:
                raise ValueError(
                    f"avoid部件 '{self.part_name}' 的 approach_direction "
                    f"和 grasp_axis 必须为 None"
                )
            if self.safety_margin <= 0:
                raise ValueError(
                    f"avoid部件 '{self.part_name}' 的 safety_margin 必须 > 0，"
                    f"当前: {self.safety_margin}"
                )
        elif self.contact_mode == 'grasp':
            if self.approach_direction is None or self.grasp_axis is None:
                raise ValueError(
                    f"grasp部件 '{self.part_name}' 的 approach_direction "
                    f"和 grasp_axis 不能为 None"
                )
        else:
            raise ValueError(
                f"contact_mode 必须是 'grasp' 或 'avoid'，"
                f"当前: '{self.contact_mode}'"
            )

    def is_graspable(self) -> bool:
        """是否为可抓取部件"""
        return self.contact_mode == 'grasp'

    def __repr__(self) -> str:
        if self.contact_mode == 'grasp':
            return (
                f"SAP(part='{self.part_name}', mode=grasp, "
                f"approach={np.round(self.approach_direction, 2)}, "
                f"grasp_axis={np.round(self.grasp_axis, 2)}, "
                f"margin={self.safety_margin})"
            )
        else:
            return (
                f"SAP(part='{self.part_name}', mode=avoid, "
                f"margin={self.safety_margin})"
            )


# ==================== SAP 知识库 ====================
# 键为部件名称（与 PartConfig.PART_MAP 中的字符串严格一致）
# 值为对应的 SAP 实例

SAP_KNOWLEDGE_BASE: Dict[str, SAP] = {

    # ------------------------------------------------------------------
    # handle：侧向抓取（cup/mug/kettle 的标志性操作）
    #
    # approach_direction: [1, 0, 0] 水平侧向（+X 方向作为参考）
    #   实际运行时，约束实例化模块会根据 handle 关键点相对于
    #   物体中心的方向，在水平面内旋转此向量到正确朝向。
    #
    # grasp_axis: [0, 0, 1] 夹爪张开轴沿 Z（竖直方向）
    #   handle 通常是竖直的圆柱/环形结构，夹爪需要从侧面张开
    #   垂直于 handle 长轴（长轴≈Z轴），所以张开轴取 Z 轴。
    # ------------------------------------------------------------------
    "handle": SAP(
        part_name="handle",
        approach_direction=np.array([1.0, 0.0, 0.0]),
        grasp_axis=np.array([0.0, 0.0, 1.0]),
        safety_margin=0.0,
        contact_mode='grasp'
    ),

    # ------------------------------------------------------------------
    # rim：杯口/碗口，avoid 部件
    #
    # 对于 cup/bowl，rim 是液体接触区域，夹爪不能碰触。
    # safety_margin=0.025：保持 25mm 安全距离，
    #   大于平行夹爪手指厚度（典型 8-15mm），留有余量。
    # ------------------------------------------------------------------
    "rim": SAP(
        part_name="rim",
        approach_direction=None,
        grasp_axis=None,
        safety_margin=0.025,
        contact_mode='avoid'
    ),

    # ------------------------------------------------------------------
    # body：物体主体，从上方抓取
    #
    # approach_direction: [0, 0, -1] 从正上方向下接近
    #   当 handle 不可见或物体没有 handle 时（如 bowl），
    #   退化为从上方抓取 body。
    #
    # grasp_axis: [1, 0, 0] 夹爪张开轴沿 X（水平方向）
    #   从上方抓取时，夹爪需要水平张开夹住侧面。
    # ------------------------------------------------------------------
    "body": SAP(
        part_name="body",
        approach_direction=np.array([0.0, 0.0, -1.0]),
        grasp_axis=np.array([1.0, 0.0, 0.0]),
        safety_margin=0.0,
        contact_mode='grasp'
    ),

    # ------------------------------------------------------------------
    # neck：瓶颈，侧向抓取（bottle 的最优抓取点）
    #
    # approach_direction: [1, 0, 0] 侧向（与 handle 相同逻辑）
    #   neck 是瓶子最细处，适合侧向抓取，接触面积小且稳定。
    #
    # grasp_axis: [0, 0, 1] 夹爪张开轴沿 Z
    #   neck 为竖直圆柱，夹爪需垂直于其轴线（即竖直张开）。
    # ------------------------------------------------------------------
    "neck": SAP(
        part_name="neck",
        approach_direction=np.array([1.0, 0.0, 0.0]),
        grasp_axis=np.array([0.0, 0.0, 1.0]),
        safety_margin=0.0,
        contact_mode='grasp'
    ),

    # ------------------------------------------------------------------
    # cap：瓶盖，从上方抓取
    #
    # approach_direction: [0, 0, -1] 从上方向下
    #   cap 在瓶子顶部，最自然的接近方式是从上方。
    #
    # grasp_axis: [1, 0, 0] 夹爪张开轴沿 X
    #   从上方抓取时，夹爪水平张开夹住 cap 侧面。
    # ------------------------------------------------------------------
    "cap": SAP(
        part_name="cap",
        approach_direction=np.array([0.0, 0.0, -1.0]),
        grasp_axis=np.array([1.0, 0.0, 0.0]),
        safety_margin=0.0,
        contact_mode='grasp'
    ),
}


# ==================== 查询接口 ====================

def get_sap(part_name: str) -> Optional[SAP]:
    """
    根据部件名称查询对应的 SAP

    Args:
        part_name: 部件名称（大小写不敏感）

    Returns:
        SAP 实例，如果不存在则返回 None

    Example:
        >>> sap = get_sap("handle")
        >>> print(sap.approach_direction)
        [1. 0. 0.]
        >>> sap = get_sap("rim")
        >>> print(sap.contact_mode)
        'avoid'
    """
    return SAP_KNOWLEDGE_BASE.get(part_name.lower().strip())


def get_sap_strict(part_name: str) -> SAP:
    """
    严格模式查询，部件不存在时抛出异常

    Args:
        part_name: 部件名称

    Returns:
        SAP 实例

    Raises:
        KeyError: 如果部件名称不在知识库中
    """
    sap = get_sap(part_name)
    if sap is None:
        available = list(SAP_KNOWLEDGE_BASE.keys())
        raise KeyError(
            f"部件 '{part_name}' 不在SAP知识库中。\n"
            f"当前可用部件: {available}"
        )
    return sap


def get_graspable_parts() -> Dict[str, SAP]:
    """返回所有可抓取部件（contact_mode='grasp'）"""
    return {k: v for k, v in SAP_KNOWLEDGE_BASE.items()
            if v.contact_mode == 'grasp'}


def get_avoid_parts() -> Dict[str, SAP]:
    """返回所有需要回避的部件（contact_mode='avoid'）"""
    return {k: v for k, v in SAP_KNOWLEDGE_BASE.items()
            if v.contact_mode == 'avoid'}


def print_knowledge_base():
    """打印完整知识库（用于调试）"""
    print("=" * 60)
    print("SAP 知识库")
    print("=" * 60)
    for part_name, sap in SAP_KNOWLEDGE_BASE.items():
        print(f"\n  {sap}")
    print()
    graspable = get_graspable_parts()
    avoid = get_avoid_parts()
    print(f"  可抓取部件: {list(graspable.keys())}")
    print(f"  回避部件:   {list(avoid.keys())}")
    print("=" * 60)


# ==================== 模块测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 SAP 知识库")
    print("=" * 60)
    print()

    # 测试1：打印完整知识库
    print("【测试1】完整知识库")
    print_knowledge_base()
    print()

    # 测试2：查询接口
    print("【测试2】查询接口")
    print("-" * 60)

    for part in ["handle", "rim", "body", "neck", "cap"]:
        sap = get_sap(part)
        print(f"  {part:8s} → mode={sap.contact_mode:5s}, "
              f"graspable={sap.is_graspable()}, "
              f"margin={sap.safety_margin}")

    print()

    # 测试3：大小写不敏感
    print("【测试3】大小写不敏感")
    print("-" * 60)
    assert get_sap("Handle") == get_sap("handle")
    assert get_sap("RIM") == get_sap("rim")
    print("  ✅ 大小写不敏感验证通过")
    print()

    # 测试4：不存在的部件
    print("【测试4】不存在的部件")
    print("-" * 60)
    result = get_sap("unknown_part")
    assert result is None
    print("  ✅ get_sap('unknown') 返回 None")

    try:
        get_sap_strict("unknown_part")
    except KeyError as e:
        print(f"  ✅ get_sap_strict 正确抛出 KeyError")
    print()

    # 测试5：验证向量已归一化
    print("【测试5】方向向量归一化验证")
    print("-" * 60)
    for part_name, sap in SAP_KNOWLEDGE_BASE.items():
        if sap.approach_direction is not None:
            norm_approach = np.linalg.norm(sap.approach_direction)
            norm_grasp = np.linalg.norm(sap.grasp_axis)
            assert np.isclose(norm_approach, 1.0), \
                f"{part_name}.approach_direction 未归一化: {norm_approach}"
            assert np.isclose(norm_grasp, 1.0), \
                f"{part_name}.grasp_axis 未归一化: {norm_grasp}"
            print(f"  {part_name:8s}: |approach|={norm_approach:.4f}, "
                  f"|grasp_axis|={norm_grasp:.4f} ✅")
    print()

    # 测试6：frozen dataclass 不可修改
    print("【测试6】SAP 不可修改性验证")
    print("-" * 60)
    sap_handle = get_sap("handle")
    try:
        sap_handle.safety_margin = 0.1  # 应该抛出异常
        print("  ❌ 未抛出异常，frozen失效！")
    except Exception:
        print("  ✅ frozen=True 生效，SAP 实例不可修改")
    print()

    print("=" * 60)
    print("✅ SAP 知识库所有测试通过！")
    print("=" * 60)