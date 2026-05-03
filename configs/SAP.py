"""
Created by Yinghao Ho on 2026-2-23

SAP (Semantic Affordance Primitive) knowledge base.
Each part name maps to a fixed set of geometric manipulation attributes.
contact_mode: 'grasp' | 'avoid' | 'place'
approach_direction for lateral parts (handle/neck) is a reference vector,
corrected at runtime by ConstraintInstantiator._compute_actual_approach.
"""

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np


@dataclass(frozen=True)
class SAP:
    """Semantic Affordance Primitive for a single object part. Immutable at runtime."""
    part_name:          str
    approach_direction: Optional[np.ndarray]
    grasp_axis:         Optional[np.ndarray]
    safety_margin:      float
    contact_mode:       str   # 'grasp' | 'avoid' | 'place'
    description:        str   # natural language description for VLM

    def __post_init__(self):
        if self.contact_mode == 'avoid':
            if self.approach_direction is not None or self.grasp_axis is not None:
                raise ValueError(
                    f"avoid part '{self.part_name}': approach_direction and grasp_axis must be None"
                )
            if self.safety_margin <= 0:
                raise ValueError(
                    f"avoid part '{self.part_name}': safety_margin must be > 0, got {self.safety_margin}"
                )
        elif self.contact_mode == 'grasp':
            if self.approach_direction is None or self.grasp_axis is None:
                raise ValueError(
                    f"grasp part '{self.part_name}': approach_direction and grasp_axis cannot be None"
                )
        elif self.contact_mode == 'place':
            if self.approach_direction is None:
                raise ValueError(
                    f"place part '{self.part_name}': approach_direction cannot be None"
                )
            if self.grasp_axis is not None:
                raise ValueError(
                    f"place part '{self.part_name}': grasp_axis must be None"
                )
            if self.safety_margin != 0.0:
                raise ValueError(
                    f"place part '{self.part_name}': safety_margin must be 0.0, got {self.safety_margin}"
                )
        else:
            raise ValueError(
                f"contact_mode must be 'grasp' / 'avoid' / 'place', got '{self.contact_mode}'"
            )

    def is_graspable(self) -> bool:
        return self.contact_mode == 'grasp'

    def is_placeable(self) -> bool:
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
            return f"SAP(part='{self.part_name}', mode=avoid, margin={self.safety_margin})"


SAP_KNOWLEDGE_BASE: Dict[str, 'SAP'] = {

    # handle: lateral grasp (cup/mug). approach [1,0,0] corrected at runtime.
    # grasp_axis [0,1,0]: gripper opens along Y to grip the handle from both sides.
    "handle": SAP(
        part_name="handle",
        approach_direction=np.array([1.0, 0.0, 0.0]),
        grasp_axis=np.array([1.0, 0.0, 0.0]),
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

    # rim: avoid part. safety_margin=0.025m > gripper finger thickness (8-15mm).
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

    # body: top-down grasp (bowl/bottle fallback).
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

    # neck: lateral grasp (bottle). Same runtime approach correction as handle.
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

    # cap: top-down grasp (bottle cap).
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

    # surface: placement target (tray/table). grasp_axis=None, no gripper alignment needed.
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


def get_sap(part_name: str) -> Optional['SAP']:
    """Returns SAP for part_name (case-insensitive), or None if not found."""
    return SAP_KNOWLEDGE_BASE.get(part_name.lower().strip())


def get_sap_strict(part_name: str) -> 'SAP':
    """Returns SAP for part_name, raises KeyError if not found."""
    sap = get_sap(part_name)
    if sap is None:
        available = list(SAP_KNOWLEDGE_BASE.keys())
        raise KeyError(f"Part '{part_name}' not in SAP knowledge base. Available: {available}")
    return sap


def get_graspable_parts() -> Dict[str, 'SAP']:
    return {k: v for k, v in SAP_KNOWLEDGE_BASE.items() if v.contact_mode == 'grasp'}


def get_avoid_parts() -> Dict[str, 'SAP']:
    return {k: v for k, v in SAP_KNOWLEDGE_BASE.items() if v.contact_mode == 'avoid'}


def get_place_parts() -> Dict[str, 'SAP']:
    return {k: v for k, v in SAP_KNOWLEDGE_BASE.items() if v.contact_mode == 'place'}


def get_sap_descriptions() -> Dict[str, str]:
    """Returns {part_name: description} for all entries. Used by VLMDecider to build prompts."""
    return {k: v.description for k, v in SAP_KNOWLEDGE_BASE.items()}


def print_knowledge_base():
    print("=" * 65)
    print("SAP Knowledge Base")
    print("=" * 65)
    for part_name, sap in SAP_KNOWLEDGE_BASE.items():
        print(f"  {sap}")
        print(f"    desc: {sap.description[:80]}...")
    print()
    print(f"  grasp parts : {list(get_graspable_parts().keys())}")
    print(f"  avoid parts : {list(get_avoid_parts().keys())}")
    print(f"  place parts : {list(get_place_parts().keys())}")
    print("=" * 65)


if __name__ == "__main__":
    print_knowledge_base()

    for part in ["handle", "rim", "body", "neck", "cap", "surface"]:
        sap = get_sap(part)
        print(f"  {part:8s} -> mode={sap.contact_mode:6s}  "
              f"graspable={sap.is_graspable()}  "
              f"placeable={sap.is_placeable()}  "
              f"margin={sap.safety_margin}")

    descs = get_sap_descriptions()
    assert set(descs.keys()) == set(SAP_KNOWLEDGE_BASE.keys())

    sap_surface = get_sap("surface")
    assert sap_surface.contact_mode == 'place'
    assert sap_surface.approach_direction is not None
    assert sap_surface.grasp_axis is None
    assert sap_surface.safety_margin == 0.0
    assert sap_surface.is_placeable()
    assert not sap_surface.is_graspable()

    assert get_sap("SURFACE") == get_sap("surface")
    assert get_sap("Handle") == get_sap("handle")

    for pname, sap in SAP_KNOWLEDGE_BASE.items():
        if sap.approach_direction is not None:
            norm = np.linalg.norm(sap.approach_direction)
            assert np.isclose(norm, 1.0), f"{pname}.approach_direction not normalised"

    try:
        get_sap("surface").safety_margin = 0.1
        print("frozen check failed")
    except Exception:
        pass

    try:
        SAP(part_name="bad", approach_direction=None,
            grasp_axis=None, safety_margin=0.0,
            contact_mode='place', description="test")
    except ValueError:
        pass

    print("all checks passed")