"""
Created by Yinghao Ho on 2026-03

Task instruction parser.

Supported patterns (case-insensitive):
    1. "pick up the [modifier] {object}"
    2. "pick up the [modifier] {object} and place it on the {target}"
    3. "move the [modifier] {object} to the {target}"

Optional spatial modifiers: leftmost, rightmost, largest, smallest, left, right, center, middle
When a modifier is present, get_detection_prompt() returns "{modifier} {object}"
for GroundingDINO spatial reasoning (e.g. "leftmost cup").
"""

import re
from dataclasses import dataclass
from typing import Optional

from configs.part_config import PartConfig


_SPATIAL_MODIFIERS = (
    "leftmost|rightmost|largest|smallest|left|right|center|middle"
)


@dataclass
class TaskSpec:
    """Parsed task instruction."""
    action:           str            # "pick_only" | "pick_and_place"
    object_label:     str
    target_label:     Optional[str]  # None for pick_only
    raw:              str
    spatial_modifier: Optional[str] = None

    def get_detection_prompt(self) -> str:
        """Returns GroundingDINO detection prompt: "{modifier} {object}" or "{object}"."""
        if self.spatial_modifier:
            return f"{self.spatial_modifier} {self.object_label}"
        return self.object_label

    def __repr__(self) -> str:
        mod_str = f", modifier='{self.spatial_modifier}'" if self.spatial_modifier else ""
        if self.action == "pick_only":
            return f"TaskSpec(action=pick_only, object='{self.object_label}'{mod_str})"
        else:
            return (f"TaskSpec(action=pick_and_place, "
                    f"object='{self.object_label}', "
                    f"target='{self.target_label}'{mod_str})")


# Regex named groups: ?P<modifier> (optional), ?P<object> (required), ?P<target> (pick_and_place only).
# pick_and_place patterns must come before pick_only to avoid prefix short-circuit matching.
_PATTERNS = [
    (
        re.compile(
            rf"^pick\s+up\s+the\s+(?:(?P<modifier>{_SPATIAL_MODIFIERS})\s+)?"
            rf"(?P<object>\w+)\s+and\s+place\s+it\s+on\s+the\s+(?P<target>\w+)$",
            re.IGNORECASE,
        ),
        "pick_and_place",
        True,
    ),
    (
        re.compile(
            rf"^move\s+the\s+(?:(?P<modifier>{_SPATIAL_MODIFIERS})\s+)?"
            rf"(?P<object>\w+)\s+to\s+the\s+(?P<target>\w+)$",
            re.IGNORECASE,
        ),
        "pick_and_place",
        True,
    ),
    (
        re.compile(
            rf"^pick\s+up\s+the\s+(?:(?P<modifier>{_SPATIAL_MODIFIERS})\s+)?"
            rf"(?P<object>\w+)$",
            re.IGNORECASE,
        ),
        "pick_only",
        False,
    ),
]


class TaskParser:
    """Fixed-pattern task instruction parser with optional spatial modifier support."""

    def parse(self, instruction: str) -> TaskSpec:
        """
        Parse a task instruction string into a TaskSpec.

        Raises:
            ValueError: instruction does not match any known pattern
            ValueError: object_label not in PartConfig supported list
        """
        cleaned = instruction.strip()

        for pattern, action, has_target in _PATTERNS:
            m = pattern.match(cleaned)
            if m is None:
                continue

            spatial_modifier = m.group("modifier")
            if spatial_modifier is not None:
                spatial_modifier = spatial_modifier.lower()

            object_label = m.group("object").lower()
            target_label = m.group("target").lower() if has_target else None

            if not PartConfig.has_parts(object_label):
                supported = PartConfig.get_supported_objects()
                raise ValueError(
                    f"Object '{object_label}' not in supported list: {supported}. "
                    f"Instruction: '{cleaned}'"
                )

            return TaskSpec(
                action=action,
                object_label=object_label,
                target_label=target_label,
                raw=cleaned,
                spatial_modifier=spatial_modifier,
            )

        raise ValueError(
            f"Instruction does not match any known pattern: '{cleaned}'\n"
            f"Supported patterns:\n"
            f"  1. pick up the [modifier] {{object}}\n"
            f"  2. pick up the [modifier] {{object}} and place it on the {{target}}\n"
            f"  3. move the [modifier] {{object}} to the {{target}}\n"
            f"Optional modifiers: leftmost, rightmost, largest, smallest, "
            f"left, right, center, middle"
        )


if __name__ == "__main__":
    parser = TaskParser()

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
        print(spec)
        assert spec.spatial_modifier is None
        assert spec.get_detection_prompt() == spec.object_label

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
        print(spec)
        assert spec.action == exp_action
        assert spec.object_label == exp_obj
        assert spec.target_label == exp_tgt
        assert spec.spatial_modifier == exp_mod
        assert spec.get_detection_prompt() == exp_prompt

    for bad_inst in ["pick up the robot", "pick up the leftmost robot"]:
        try:
            parser.parse(bad_inst)
            assert False, f"should have raised: {bad_inst!r}"
        except ValueError:
            pass

    for bad_inst in ["grab the cup", "take the leftmost cup"]:
        try:
            parser.parse(bad_inst)
            assert False, f"should have raised: {bad_inst!r}"
        except ValueError:
            pass

    print("all checks passed")