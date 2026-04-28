"""
modules/task_parser.py
Created by Yinghao Ho on 2026-03

任务指令解析模块

支持的句式（大小写不敏感）：
    1. "pick up the [modifier] {object}"
    2. "pick up the [modifier] {object} and place it on the {target}"
    3. "move the [modifier] {object} to the {target}"

可选空间修饰词（modifier）：
    leftmost, rightmost, largest, smallest, left, right, center, middle
    有修饰词时，get_detection_prompt() 返回 "{modifier} {object}"
    供 GroundingDINO 进行空间关系推理（如 "leftmost cup"）
"""

import re
from dataclasses import dataclass
from typing import Optional

from configs.part_config import PartConfig


# 支持的空间修饰词（用于正则 alternation）
_SPATIAL_MODIFIERS = (
    "leftmost|rightmost|largest|smallest|left|right|center|middle"
)


@dataclass
class TaskSpec:
    """
    解析结果

    Attributes:
        action            : "pick_only" | "pick_and_place"
        object_label      : 操作对象（与 PartConfig 对应）
        target_label      : 放置目标（pick_only 时为 None）
        raw               : 原始指令字符串
        spatial_modifier  : 可选的空间修饰词，如 "leftmost"（无修饰词时为 None）
    """
    action:           str
    object_label:     str
    target_label:     Optional[str]
    raw:              str
    spatial_modifier: Optional[str] = None   # 末尾，有默认值

    def get_detection_prompt(self) -> str:
        """
        返回供 GroundingDINO 使用的检测文本

        有修饰词时返回 "{modifier} {object}"（如 "leftmost cup"），
        无修饰词时返回 "{object}"（如 "cup"）。
        """
        if self.spatial_modifier:
            return f"{self.spatial_modifier} {self.object_label}"
        return self.object_label

    def __repr__(self) -> str:
        mod_str = f", modifier='{self.spatial_modifier}'" if self.spatial_modifier else ""
        if self.action == "pick_only":
            return (f"TaskSpec(action=pick_only, "
                    f"object='{self.object_label}'{mod_str})")
        else:
            return (f"TaskSpec(action=pick_and_place, "
                    f"object='{self.object_label}', "
                    f"target='{self.target_label}'{mod_str})")


# ──────────────────────────────────────────────────────────────
# 句式模板：(compiled_pattern, action, has_target)
#
# 正则使用具名分组：
#   ?P<modifier>  — 可选空间修饰词（未出现时 .group("modifier") 返回 None）
#   ?P<object>    — 操作物体（必选）
#   ?P<target>    — 放置目标（has_target=True 时必选）
#
# 注意：pick_and_place 句式必须排在 pick_only 之前，防止前缀被短路匹配。
# ──────────────────────────────────────────────────────────────
_PATTERNS = [
    # 1. "pick up the [modifier] {object} and place it on the {target}"
    (
        re.compile(
            rf"^pick\s+up\s+the\s+(?:(?P<modifier>{_SPATIAL_MODIFIERS})\s+)?"
            rf"(?P<object>\w+)\s+and\s+place\s+it\s+on\s+the\s+(?P<target>\w+)$",
            re.IGNORECASE,
        ),
        "pick_and_place",
        True,   # has_target
    ),
    # 2. "move the [modifier] {object} to the {target}"
    (
        re.compile(
            rf"^move\s+the\s+(?:(?P<modifier>{_SPATIAL_MODIFIERS})\s+)?"
            rf"(?P<object>\w+)\s+to\s+the\s+(?P<target>\w+)$",
            re.IGNORECASE,
        ),
        "pick_and_place",
        True,
    ),
    # 3. "pick up the [modifier] {object}"
    (
        re.compile(
            rf"^pick\s+up\s+the\s+(?:(?P<modifier>{_SPATIAL_MODIFIERS})\s+)?"
            rf"(?P<object>\w+)$",
            re.IGNORECASE,
        ),
        "pick_only",
        False,  # has_target
    ),
]


class TaskParser:
    """
    固定句式任务指令解析器（支持可选空间修饰词）

    使用示例：
        >>> parser = TaskParser()
        >>> spec = parser.parse("pick up the leftmost cup and place it on the tray")
        >>> spec.action              # "pick_and_place"
        >>> spec.object_label        # "cup"
        >>> spec.target_label        # "tray"
        >>> spec.spatial_modifier    # "leftmost"
        >>> spec.get_detection_prompt()  # "leftmost cup"
    """

    def parse(self, instruction: str) -> TaskSpec:
        """
        解析任务指令

        Raises:
            ValueError: 不匹配任何已知句式
            ValueError: object_label 不在 PartConfig 支持列表中
        """
        cleaned = instruction.strip()

        for pattern, action, has_target in _PATTERNS:
            m = pattern.match(cleaned)
            if m is None:
                continue

            # 具名分组提取（modifier 可能为 None）
            spatial_modifier = m.group("modifier")
            if spatial_modifier is not None:
                spatial_modifier = spatial_modifier.lower()

            object_label = m.group("object").lower()
            target_label = m.group("target").lower() if has_target else None

            if not PartConfig.has_parts(object_label):
                supported = PartConfig.get_supported_objects()
                raise ValueError(
                    f"物体 '{object_label}' 不在支持列表中。\n"
                    f"当前支持: {supported}\n"
                    f"原始指令: '{cleaned}'"
                )

            return TaskSpec(
                action=action,
                object_label=object_label,
                target_label=target_label,
                raw=cleaned,
                spatial_modifier=spatial_modifier,
            )

        raise ValueError(
            f"指令不匹配任何已知句式: '{cleaned}'\n"
            f"支持的句式：\n"
            f"  1. pick up the [modifier] {{object}}\n"
            f"  2. pick up the [modifier] {{object}} and place it on the {{target}}\n"
            f"  3. move the [modifier] {{object}} to the {{target}}\n"
            f"可选修饰词: leftmost, rightmost, largest, smallest, "
            f"left, right, center, middle"
        )


# ==================== 模块测试 ====================
if __name__ == "__main__":
    parser = TaskParser()

    print("=" * 65)
    print("测试 TaskParser")
    print("=" * 65)

    # ── 原有正常用例（验证向后兼容） ──
    print("\n【1】原有句式（无修饰词，向后兼容）")
    cases_basic = [
        "pick up the cup",
        "pick up the Cup",
        "pick up the mug and place it on the tray",
        "move the bottle to the tray",
        "Pick Up The Cup And Place It On The Table",
        "  pick up the cup  ",
    ]
    for inst in cases_basic:
        spec = parser.parse(inst)
        print(f"  {inst!r:<55} → {spec}")
        assert spec.spatial_modifier is None, "无修饰词时 spatial_modifier 应为 None"
        assert spec.get_detection_prompt() == spec.object_label

    # ── 空间修饰词用例 ──
    print("\n【2】空间修饰词句式")
    cases_modifier = [
        ("pick up the leftmost cup",
         "pick_only", "cup", None, "leftmost", "leftmost cup"),
        ("pick up the rightmost cup and place it on the tray",
         "pick_and_place", "cup", "tray", "rightmost", "rightmost cup"),
        ("move the largest bottle to the tray",
         "pick_and_place", "bottle", "tray", "largest", "largest bottle"),
        ("Pick Up The Left Cup",
         "pick_only", "cup", None, "left", "left cup"),
        ("pick up the center bowl",
         "pick_only", "bowl", None, "center", "center bowl"),
    ]
    for inst, exp_action, exp_obj, exp_tgt, exp_mod, exp_prompt in cases_modifier:
        spec = parser.parse(inst)
        print(f"  {inst!r:<55} → {spec}")
        assert spec.action == exp_action,           f"action 错误: {spec.action}"
        assert spec.object_label == exp_obj,        f"object 错误: {spec.object_label}"
        assert spec.target_label == exp_tgt,        f"target 错误: {spec.target_label}"
        assert spec.spatial_modifier == exp_mod,    f"modifier 错误: {spec.spatial_modifier}"
        assert spec.get_detection_prompt() == exp_prompt, \
            f"detection_prompt 错误: {spec.get_detection_prompt()}"
    print("  ✅ 空间修饰词用例全部通过")

    # ── 错误用例 ──
    print("\n【3】未知物体 → 应抛出 ValueError")
    for bad_inst in ["pick up the robot", "pick up the leftmost robot"]:
        try:
            parser.parse(bad_inst)
            print(f"  ❌ 未报错: {bad_inst!r}")
        except ValueError as e:
            print(f"  ✅ {bad_inst!r} → {str(e).splitlines()[0]}")

    print("\n【4】不匹配句式 → 应抛出 ValueError")
    for bad_inst in ["grab the cup", "take the leftmost cup"]:
        try:
            parser.parse(bad_inst)
            print(f"  ❌ 未报错: {bad_inst!r}")
        except ValueError as e:
            print(f"  ✅ {bad_inst!r} → {str(e).splitlines()[0]}")

    print("\n" + "=" * 65)
    print("✅ TaskParser 所有测试通过！")
    print("=" * 65)