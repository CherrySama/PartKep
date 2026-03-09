"""
modules/task_parser.py
Created by Yinghao Ho on 2026-03

任务指令解析模块

支持的句式（大小写不敏感）：
    1. "pick up the {object}"
    2. "pick up the {object} and place it on the {target}"
    3. "move the {object} to the {target}"
"""

import re
from dataclasses import dataclass
from typing import Optional

from configs.part_config import PartConfig


@dataclass
class TaskSpec:
    """
    解析结果

    Attributes:
        action       : "pick_only" | "pick_and_place"
        object_label : 操作对象（与 PartConfig 对应）
        target_label : 放置目标（pick_only 时为 None）
        raw          : 原始指令字符串
    """
    action:       str
    object_label: str
    target_label: Optional[str]
    raw:          str

    def __repr__(self) -> str:
        if self.action == "pick_only":
            return (f"TaskSpec(action=pick_only, "
                    f"object='{self.object_label}')")
        else:
            return (f"TaskSpec(action=pick_and_place, "
                    f"object='{self.object_label}', "
                    f"target='{self.target_label}')")


# 句式模板：(正则, action, group_mapping)
# group_mapping：{组编号: 字段名}，2 in group_map 表示有 target
_PATTERNS = [
    # "pick up the {object} and place it on the {target}"  ← 必须先于 pick_only
    (
        re.compile(
            r"^pick\s+up\s+the\s+(\w+)\s+and\s+place\s+it\s+on\s+the\s+(\w+)$",
            re.IGNORECASE
        ),
        "pick_and_place",
        {1: "object_label", 2: "target_label"},
    ),
    # "move the {object} to the {target}"
    (
        re.compile(
            r"^move\s+the\s+(\w+)\s+to\s+the\s+(\w+)$",
            re.IGNORECASE
        ),
        "pick_and_place",
        {1: "object_label", 2: "target_label"},
    ),
    # "pick up the {object}"
    (
        re.compile(
            r"^pick\s+up\s+the\s+(\w+)$",
            re.IGNORECASE
        ),
        "pick_only",
        {1: "object_label"},
    ),
]


class TaskParser:
    """
    固定句式任务指令解析器

    使用示例：
        >>> parser = TaskParser()
        >>> spec = parser.parse("pick up the cup and place it on the tray")
        >>> spec.action        # "pick_and_place"
        >>> spec.object_label  # "cup"
        >>> spec.target_label  # "tray"
    """

    def parse(self, instruction: str) -> TaskSpec:
        """
        解析任务指令

        Raises:
            ValueError: 不匹配任何已知句式
            ValueError: object_label 不在 PartConfig 支持列表中
        """
        cleaned = instruction.strip()

        for pattern, action, group_map in _PATTERNS:
            m = pattern.match(cleaned)
            if m is None:
                continue

            object_label = m.group(1).lower()
            target_label = m.group(2).lower() if 2 in group_map else None

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
            )

        raise ValueError(
            f"指令不匹配任何已知句式: '{cleaned}'\n"
            f"支持的句式：\n"
            f"  1. pick up the {{object}}\n"
            f"  2. pick up the {{object}} and place it on the {{target}}\n"
            f"  3. move the {{object}} to the {{target}}"
        )


# ==================== 模块测试 ====================
if __name__ == "__main__":
    parser = TaskParser()

    print("=" * 60)
    print("测试 TaskParser")
    print("=" * 60)

    cases = [
        "pick up the cup",
        "pick up the Cup",
        "pick up the mug and place it on the tray",
        "move the bottle to the bowl",
        "Pick Up The Cup And Place It On The Table",
        "  pick up the cup  ",
    ]

    print("\n【正常用例】")
    for inst in cases:
        spec = parser.parse(inst)
        print(f"  {inst!r:<55} → {spec}")

    print("\n【未知物体 → 应抛出 ValueError】")
    try:
        parser.parse("pick up the robot")
    except ValueError as e:
        print(f"  ✅ {str(e).splitlines()[0]}")

    print("\n【不匹配句式 → 应抛出 ValueError】")
    try:
        parser.parse("grab the cup")
    except ValueError as e:
        print(f"  ✅ {str(e).splitlines()[0]}")

    print("\n✅ TaskParser 测试完成")
    