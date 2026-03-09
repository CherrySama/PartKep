"""
Created by Yinghao Ho on 2026-2-23

SAP (Semantic Affordance Primitive) 知识库
核心理论：语义部件标签是操作先验的符号编码。
每个语义部件名称唯一对应一组几何操作属性（SAP），
无需VLM推理，直接通过符号查询获得。

覆盖的物体：
    pick 目标：cup / bottle / bowl
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
    """
    part_name:          str
    approach_direction: Optional[np.ndarray]
    grasp_axis:         Optional[np.ndarray]
    safety_margin:      float
    contact_mode:       str   # 'grasp' | 'avoid' | 'place'

    def __post_init__(self):
        """校验 contact_mode 与属性的一致性"""
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
            # place 部件：需要接近方向，不需要夹爪对齐轴
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

SAP_KNOWLEDGE_BASE: Dict[str, SAP] = {

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
        contact_mode='grasp'
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
        contact_mode='avoid'
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
        contact_mode='grasp'
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
        contact_mode='grasp'
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
        contact_mode='grasp'
    ),

    # ----------------------------------------------------------------
    # surface：放置目标承载面（tray/table 专用）
    #
    # contact_mode='place'：末端执行器从上方接近，将物体放下
    # approach_direction: [0,0,-1] 从上方竖直向下接近承载面
    # grasp_axis: None（放置动作不需要夹爪对齐轴）
    #
    # 语义说明：
    #   放置时末端执行器带着已抓取的物体，从上方对准 surface
    #   关键点下降，直到物体底部接触承载面后释放夹爪。
    # ----------------------------------------------------------------
    "surface": SAP(
        part_name="surface",
        approach_direction=np.array([0.0, 0.0, -1.0]),
        grasp_axis=None,
        safety_margin=0.0,
        contact_mode='place'
    ),
}


# ==================== 查询接口 ====================

def get_sap(part_name: str) -> Optional[SAP]:
    """
    根据部件名称查询对应的 SAP（大小写不敏感）

    Returns:
        SAP 实例，部件不存在时返回 None
    """
    return SAP_KNOWLEDGE_BASE.get(part_name.lower().strip())


def get_sap_strict(part_name: str) -> SAP:
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


def get_graspable_parts() -> Dict[str, SAP]:
    """返回所有可抓取部件（contact_mode='grasp'）"""
    return {k: v for k, v in SAP_KNOWLEDGE_BASE.items()
            if v.contact_mode == 'grasp'}


def get_avoid_parts() -> Dict[str, SAP]:
    """返回所有需要回避的部件（contact_mode='avoid'）"""
    return {k: v for k, v in SAP_KNOWLEDGE_BASE.items()
            if v.contact_mode == 'avoid'}


def get_place_parts() -> Dict[str, SAP]:
    """返回所有放置目标部件（contact_mode='place'）"""
    return {k: v for k, v in SAP_KNOWLEDGE_BASE.items()
            if v.contact_mode == 'place'}


def print_knowledge_base():
    """打印完整知识库（调试用）"""
    print("=" * 65)
    print("SAP 知识库")
    print("=" * 65)
    for part_name, sap in SAP_KNOWLEDGE_BASE.items():
        print(f"  {sap}")
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

    # 【3】place 部件校验
    print("\n【3】place 部件属性验证")
    sap_surface = get_sap("surface")
    assert sap_surface.contact_mode == 'place'
    assert sap_surface.approach_direction is not None
    assert sap_surface.grasp_axis is None
    assert sap_surface.safety_margin == 0.0
    assert sap_surface.is_placeable()
    assert not sap_surface.is_graspable()
    print("  ✅ surface 部件属性全部正确")

    # 【4】非法 place 构造应报错
    print("\n【4】非法 place 构造 → 应抛出 ValueError")
    try:
        SAP(part_name="bad", approach_direction=None,
            grasp_axis=None, safety_margin=0.0, contact_mode='place')
    except ValueError:
        print("  ✅ approach_direction=None 时正确报错")
    try:
        SAP(part_name="bad",
            approach_direction=np.array([0., 0., -1.]),
            grasp_axis=np.array([1., 0., 0.]),
            safety_margin=0.0, contact_mode='place')
    except ValueError:
        print("  ✅ grasp_axis 非 None 时正确报错")

    # 【5】大小写不敏感
    print("\n【5】大小写不敏感")
    assert get_sap("SURFACE") == get_sap("surface")
    assert get_sap("Handle") == get_sap("handle")
    print("  ✅ 通过")

    # 【6】归一化验证（含 surface）
    print("\n【6】方向向量归一化验证")
    for pname, sap in SAP_KNOWLEDGE_BASE.items():
        if sap.approach_direction is not None:
            norm = np.linalg.norm(sap.approach_direction)
            assert np.isclose(norm, 1.0), f"{pname}.approach_direction 未归一化"
            print(f"  {pname:8s}: |approach|={norm:.4f} ✅")

    # 【7】frozen 不可修改
    print("\n【7】frozen 不可修改")
    try:
        get_sap("surface").safety_margin = 0.1
        print("  ❌ frozen 失效！")
    except Exception:
        print("  ✅ frozen=True 生效")

    print("\n" + "=" * 65)
    print("✅ SAP 所有测试通过！")
    print("=" * 65)
    