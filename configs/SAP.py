"""
Created by Yinghao Ho on 2026-2-23

SAP (Semantic Affordance Primitive) 知识库
核心理论：语义部件标签是操作先验的符号编码。
每个语义部件名称唯一对应一组几何操作属性（SAP），
无需VLM推理，直接通过符号查询获得。

覆盖的物体：
    pick 目标：cup / mug / bottle / bowl
    place 目标：tray / table

覆盖的部件（part_name）：
    handle   — 侧向抓取（cup/mug）
    rim      — 避开部件（cup/bowl/tray 的边缘）
    body     — 从上方抓取（bowl/bottle 备用，pick 目标）
    neck     — 侧向抓取（bottle）
    cap      — 从上方抓取（bottle 顶部）
    surface  — 放置目标承载面（tray/table，place 专用）

contact_mode 三种取值：
    'grasp'  — 主动抓取部件，需要 approach_direction + grasp_axis
    'avoid'  — 需要回避的部件，需要 safety_margin > 0
    'place'  — 放置目标部件，需要 approach_direction，grasp_axis=None

description 字段说明：
    每条 SAP entry 包含一段自然语言语义描述，覆盖：
        - 部件的功能用途（如何用于操作）
        - 部件的几何外观（形状、位置、大小关系）
        - 对应的抓取或回避方式
    该描述不参与任何数值计算，仅供 VLM 在两个场景下读取：
        1. 约束决策：VLM 读取所有已检测部件的描述，理解场景语义
        2. 泛化映射：VLM 将未知物体部件映射到语义最接近的 SAP entry

注意：approach_direction 是参考方向，约束实例化时会根据
关键点实际位置动态修正（侧向部件）或直接使用（竖向部件）。
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
        part_name          : 部件名称（与 PartConfig 中的名称一致）
        approach_direction : 接近方向参考向量（归一化）
                             grasp/place 部件必须提供，avoid 部件为 None
        grasp_axis         : 夹爪张开轴方向（归一化）
                             仅 grasp 部件提供，avoid/place 部件为 None
        safety_margin      : 安全距离（米），avoid 部件 > 0，其余为 0.0
        contact_mode       : 'grasp' | 'avoid' | 'place'
        description        : 自然语言语义描述，供 VLM 语义理解与泛化映射使用
    """
    part_name:          str
    approach_direction: Optional[np.ndarray]
    grasp_axis:         Optional[np.ndarray]
    safety_margin:      float
    contact_mode:       str          # 'grasp' | 'avoid' | 'place'
    description:        str          # 自然语言语义描述

    def __post_init__(self):
        """校验 contact_mode 与属性的一致性（description 无需校验）"""
        if self.contact_mode == 'avoid':
            if self.approach_direction is not None or self.grasp_axis is not None:
                raise ValueError(
                    f"avoid 部件 '{self.part_name}' 的 "
                    f"approach_direction 和 grasp_axis 必须为 None"
                )
            if self.safety_margin <= 0:
                raise ValueError(
                    f"avoid 部件 '{self.part_name}' 的 safety_margin 必须 > 0，"
                    f"当前: {self.safety_margin}"
                )

        elif self.contact_mode == 'grasp':
            if self.approach_direction is None or self.grasp_axis is None:
                raise ValueError(
                    f"grasp 部件 '{self.part_name}' 的 "
                    f"approach_direction 和 grasp_axis 不能为 None"
                )

        elif self.contact_mode == 'place':
            if self.approach_direction is None:
                raise ValueError(
                    f"place 部件 '{self.part_name}' 的 "
                    f"approach_direction 不能为 None"
                )
            if self.grasp_axis is not None:
                raise ValueError(
                    f"place 部件 '{self.part_name}' 的 "
                    f"grasp_axis 必须为 None（放置不需要夹爪对齐轴）"
                )
            if self.safety_margin != 0.0:
                raise ValueError(
                    f"place 部件 '{self.part_name}' 的 "
                    f"safety_margin 必须为 0.0，当前: {self.safety_margin}"
                )

        else:
            raise ValueError(
                f"contact_mode 必须是 'grasp' / 'avoid' / 'place'，"
                f"当前: '{self.contact_mode}'"
            )

    def is_graspable(self) -> bool:
        """是否为可抓取部件"""
        return self.contact_mode == 'grasp'

    def is_placeable(self) -> bool:
        """是否为放置目标部件"""
        return self.contact_mode == 'place'

    def __repr__(self) -> str:
        if self.contact_mode == 'grasp':
            return (
                f"SAP(part='{self.part_name}', mode=grasp, "
                f"approach={np.round(self.approach_direction, 2)}, "
                f"grasp_axis={np.round(self.grasp_axis, 2)}, "
                f"margin={self.safety_margin})"
            )
        elif self.contact_mode == 'place':
            return (
                f"SAP(part='{self.part_name}', mode=place, "
                f"approach={np.round(self.approach_direction, 2)})"
            )
        else:
            return (
                f"SAP(part='{self.part_name}', mode=avoid, "
                f"margin={self.safety_margin})"
            )


# ==================== SAP 知识库 ====================
# 键为 part_name（全局唯一，与 PartConfig.PART_MAP 严格对应）
# 值为对应的 SAP 实例

SAP_KNOWLEDGE_BASE: Dict[str, 'SAP'] = {

    # ----------------------------------------------------------------
    # handle：侧向抓取（cup/mug 的标志性操作）
    #
    # approach_direction: [1,0,0] 水平侧向参考方向
    #   运行时由 ConstraintInstantiator 根据关键点实际位置修正
    # grasp_axis: [0,0,1] 夹爪张开轴沿 Z（竖直方向）
    #   handle 为竖直圆柱/环形结构，夹爪从侧面张开
    # ----------------------------------------------------------------
    "handle": SAP(
        part_name="handle",
        approach_direction=np.array([1.0, 0.0, 0.0]),
        grasp_axis=np.array([0.0, 0.0, 1.0]),
        safety_margin=0.0,
        contact_mode='grasp',
        description=(
            "A laterally protruding loop or grip structure rigidly attached "
            "to the outer side wall of a container. Typically D-shaped or "
            "cylindrical, with a hollow interior large enough for fingers. "
            "Located at mid-height on the object exterior. Affords lateral "
            "grasping: gripper approaches horizontally from outside and closes "
            "vertically, gripping the handle along its vertical axis from "
            "both sides."
        )
    ),

    # ----------------------------------------------------------------
    # rim：杯口/碗口/托盘边缘，avoid 部件
    #
    # 液体接触区域或精密边缘，夹爪不能触碰。
    # safety_margin=0.025（25mm）> 平行夹爪手指厚度（8-15mm）
    # ----------------------------------------------------------------
    "rim": SAP(
        part_name="rim",
        approach_direction=None,
        grasp_axis=None,
        safety_margin=0.025,
        contact_mode='avoid',
        description=(
            "The uppermost continuous circular edge or lip defining the "
            "opening of a container. A thin boundary ring at the top of the "
            "object, often slightly flared or rounded inward. Borders the "
            "functional interior such as liquid or contents. Must be avoided "
            "entirely during grasping to prevent contamination, spillage, or "
            "structural damage to the object."
        )
    ),

    # ----------------------------------------------------------------
    # body：物体主体，从上方抓取（pick 目标专用）
    #
    # approach_direction: [0,0,-1] 从正上方向下接近
    # grasp_axis: [1,0,0] 夹爪水平张开夹住侧面
    # 用于：bowl（无 handle）、bottle 备用
    # ----------------------------------------------------------------
    "body": SAP(
        part_name="body",
        approach_direction=np.array([0.0, 0.0, -1.0]),
        grasp_axis=np.array([1.0, 0.0, 0.0]),
        safety_margin=0.0,
        contact_mode='grasp',
        description=(
            "The main cylindrical or rounded bulk section of an object, "
            "forming the largest and most structurally stable region. Wider "
            "than the neck, located below the rim, with a relatively uniform "
            "cross-section throughout. Affords top-down grasping: gripper "
            "descends vertically from above and closes horizontally around "
            "the widest cross-section to achieve a stable, symmetric grip."
        )
    ),

    # ----------------------------------------------------------------
    # neck：瓶颈，侧向抓取（bottle 最优抓取点）
    #
    # approach_direction: [1,0,0] 水平侧向参考方向（同 handle 逻辑）
    # grasp_axis: [0,0,1] 夹爪张开轴沿 Z
    # ----------------------------------------------------------------
    "neck": SAP(
        part_name="neck",
        approach_direction=np.array([1.0, 0.0, 0.0]),
        grasp_axis=np.array([0.0, 0.0, 1.0]),
        safety_margin=0.0,
        contact_mode='grasp',
        description=(
            "A narrowed, constricted cylindrical section that connects the "
            "wider body to the cap or mouth of a bottle-type object. Smaller "
            "in diameter than the body, providing a natural grip point with "
            "firm and stable contact. Affords lateral grasping at this narrow "
            "region: gripper approaches horizontally from outside and closes "
            "vertically around the neck circumference."
        )
    ),

    # ----------------------------------------------------------------
    # cap：瓶盖，从上方抓取
    #
    # approach_direction: [0,0,-1] 从上方向下
    # grasp_axis: [1,0,0] 夹爪水平张开夹住 cap 侧面
    # ----------------------------------------------------------------
    "cap": SAP(
        part_name="cap",
        approach_direction=np.array([0.0, 0.0, -1.0]),
        grasp_axis=np.array([1.0, 0.0, 0.0]),
        safety_margin=0.0,
        contact_mode='grasp',
        description=(
            "The topmost flat or dome-shaped closure seal at the apex of an "
            "object, typically circular and slightly raised above the neck. "
            "Smaller in diameter than the body. Affords top-down grasping "
            "directly from above: gripper descends vertically and closes "
            "horizontally around the cap perimeter from both sides."
        )
    ),

    # ----------------------------------------------------------------
    # surface：放置目标承载面（tray/table 专用）
    #
    # contact_mode='place'：末端执行器从上方接近，将物体放下
    # approach_direction: [0,0,-1] 从上方竖直向下接近承载面
    # grasp_axis: None（放置动作不需要夹爪对齐轴）
    # ----------------------------------------------------------------
    "surface": SAP(
        part_name="surface",
        approach_direction=np.array([0.0, 0.0, -1.0]),
        grasp_axis=None,
        safety_margin=0.0,
        contact_mode='place',
        description=(
            "A broad, flat, horizontal load-bearing area designed to receive "
            "and stably support objects placed upon it. Open, wide, and level. "
            "Found on trays, tables, shelves, and other placement platforms. "
            "Used exclusively as a placement target: end-effector descends "
            "vertically toward the surface centre and releases the held object "
            "onto it."
        )
    ),
}


# ==================== 查询接口 ====================

def get_sap(part_name: str) -> Optional['SAP']:
    """
    根据部件名称查询对应的 SAP（大小写不敏感）

    Returns:
        SAP 实例，部件不存在时返回 None
    """
    return SAP_KNOWLEDGE_BASE.get(part_name.lower().strip())


def get_sap_strict(part_name: str) -> 'SAP':
    """
    严格模式查询，部件不存在时抛出 KeyError

    Raises:
        KeyError: 部件名称不在知识库中
    """
    sap = get_sap(part_name)
    if sap is None:
        available = list(SAP_KNOWLEDGE_BASE.keys())
        raise KeyError(
            f"部件 '{part_name}' 不在 SAP 知识库中。\n"
            f"当前可用部件: {available}"
        )
    return sap


def get_graspable_parts() -> Dict[str, 'SAP']:
    """返回所有可抓取部件（contact_mode='grasp'）"""
    return {k: v for k, v in SAP_KNOWLEDGE_BASE.items()
            if v.contact_mode == 'grasp'}


def get_avoid_parts() -> Dict[str, 'SAP']:
    """返回所有需要回避的部件（contact_mode='avoid'）"""
    return {k: v for k, v in SAP_KNOWLEDGE_BASE.items()
            if v.contact_mode == 'avoid'}


def get_place_parts() -> Dict[str, 'SAP']:
    """返回所有放置目标部件（contact_mode='place'）"""
    return {k: v for k, v in SAP_KNOWLEDGE_BASE.items()
            if v.contact_mode == 'place'}


def get_sap_descriptions() -> Dict[str, str]:
    """
    返回所有 SAP entry 的 description 字典

    主要供 VLMDecider 构建 prompt 时使用：
        descriptions = get_sap_descriptions()
        for part_name, desc in descriptions.items():
            prompt += f"[{part_name}]: {desc}\n"

    Returns:
        {part_name: description} 字典
    """
    return {k: v.description for k, v in SAP_KNOWLEDGE_BASE.items()}


def print_knowledge_base():
    """打印完整知识库（调试用）"""
    print("=" * 65)
    print("SAP 知识库")
    print("=" * 65)
    for part_name, sap in SAP_KNOWLEDGE_BASE.items():
        print(f"  {sap}")
        print(f"    desc: {sap.description[:80]}...")
    print()
    print(f"  grasp 部件 : {list(get_graspable_parts().keys())}")
    print(f"  avoid 部件 : {list(get_avoid_parts().keys())}")
    print(f"  place 部件 : {list(get_place_parts().keys())}")
    print("=" * 65)


# ==================== 模块测试 ====================
if __name__ == "__main__":
    print("=" * 65)
    print("测试 SAP 知识库")
    print("=" * 65)

    # 【1】打印完整知识库
    print("\n【1】完整知识库")
    print_knowledge_base()

    # 【2】查询接口
    print("\n【2】查询接口")
    for part in ["handle", "rim", "body", "neck", "cap", "surface"]:
        sap = get_sap(part)
        print(f"  {part:8s} → mode={sap.contact_mode:6s}  "
              f"graspable={sap.is_graspable()}  "
              f"placeable={sap.is_placeable()}  "
              f"margin={sap.safety_margin}")

    # 【3】description 字段验证
    print("\n【3】description 字段验证")
    for part in ["handle", "rim", "body", "neck", "cap", "surface"]:
        sap = get_sap(part)
        assert isinstance(sap.description, str), f"{part}: description 不是 str"
        assert len(sap.description) > 0, f"{part}: description 为空"
        print(f"  {part:8s}: ✅ ({len(sap.description)} chars)")

    # 【4】get_sap_descriptions() 接口
    print("\n【4】get_sap_descriptions() 接口")
    descs = get_sap_descriptions()
    assert set(descs.keys()) == set(SAP_KNOWLEDGE_BASE.keys())
    print(f"  ✅ 返回 {len(descs)} 条描述，键集合与知识库一致")

    # 【5】place 部件校验
    print("\n【5】place 部件属性验证")
    sap_surface = get_sap("surface")
    assert sap_surface.contact_mode == 'place'
    assert sap_surface.approach_direction is not None
    assert sap_surface.grasp_axis is None
    assert sap_surface.safety_margin == 0.0
    assert sap_surface.is_placeable()
    assert not sap_surface.is_graspable()
    print("  ✅ surface 部件属性全部正确")

    # 【6】大小写不敏感
    print("\n【6】大小写不敏感")
    assert get_sap("SURFACE") == get_sap("surface")
    assert get_sap("Handle") == get_sap("handle")
    print("  ✅ 通过")

    # 【7】归一化验证
    print("\n【7】方向向量归一化验证")
    for pname, sap in SAP_KNOWLEDGE_BASE.items():
        if sap.approach_direction is not None:
            norm = np.linalg.norm(sap.approach_direction)
            assert np.isclose(norm, 1.0), f"{pname}.approach_direction 未归一化"
            print(f"  {pname:8s}: |approach|={norm:.4f} ✅")

    # 【8】frozen 不可修改
    print("\n【8】frozen 不可修改")
    try:
        get_sap("surface").safety_margin = 0.1
        print("  ❌ frozen 失效！")
    except Exception:
        print("  ✅ frozen=True 生效")

    # 【9】非法构造应报错
    print("\n【9】非法 place 构造 → 应抛出 ValueError")
    try:
        SAP(part_name="bad", approach_direction=None,
            grasp_axis=None, safety_margin=0.0,
            contact_mode='place', description="test")
    except ValueError:
        print("  ✅ approach_direction=None 时正确报错")

    print("\n" + "=" * 65)
    print("✅ SAP 所有测试通过！")
    print("=" * 65)
    