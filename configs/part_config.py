"""
Created by Yinghao Ho on 2026-1-23

Object part configuration.
Each entry has two fields:
    part_name : key in SAP knowledge base (must match SAP.py exactly)
    prompt    : text prompt passed to SAM3 (prefixed with object name for accuracy)

Usage flow:
    1. TaskParser extracts object_label (pick) / target_label (place)
    2. GroundingDINO detects object bbox
    3. get_parts(label) returns PartEntry list
    4. Each entry["prompt"] is passed to SAM3 for segmentation
    5. Each entry["part_name"] is used to query SAP knowledge base for constraints
"""

from typing import Dict, List

PartEntry = Dict[str, str]


class PartConfig:
    """
    Object part configuration.
    PART_MAP maps each object label to a list of PartEntry dicts
    with keys 'part_name' (SAP key) and 'prompt' (SAM3 text).
    """

    PART_MAP: Dict[str, List[PartEntry]] = {

        # pick targets
        "cup": [
            {"part_name": "handle",  "prompt": "cup handle"},
            {"part_name": "rim",     "prompt": "cup rim"},
            {"part_name": "body",    "prompt": "cup body"},
        ],
        "mug": [
            {"part_name": "handle",  "prompt": "mug handle"},
            {"part_name": "rim",     "prompt": "mug rim"},
            {"part_name": "body",    "prompt": "mug body"},
        ],
        "bottle": [
            {"part_name": "cap",     "prompt": "bottle cap"},
            {"part_name": "neck",    "prompt": "bottle neck"},
            {"part_name": "body",    "prompt": "bottle body"},
        ],
        "bowl": [
            {"part_name": "rim",     "prompt": "bowl rim"},
            {"part_name": "body",    "prompt": "bowl body"},
        ],

        # place targets: surface -> contact_mode='place', rim -> contact_mode='avoid'
        "tray": [
            {"part_name": "surface", "prompt": "tray surface"},
            {"part_name": "rim",     "prompt": "tray rim"},
        ],
        "table": [
            {"part_name": "surface", "prompt": "table surface"},
        ],
    }

    @classmethod
    def get_parts(cls, object_label: str) -> List[PartEntry]:
        """Returns PartEntry list for object_label, or [] if not found."""
        return cls.PART_MAP.get(object_label.strip().lower(), [])

    @classmethod
    def has_parts(cls, object_label: str) -> bool:
        return object_label.strip().lower() in cls.PART_MAP

    @classmethod
    def get_part_names(cls, object_label: str) -> List[str]:
        """Returns list of part_name strings for SAP queries."""
        return [e["part_name"] for e in cls.get_parts(object_label)]

    @classmethod
    def get_prompts(cls, object_label: str) -> List[str]:
        """Returns list of prompt strings for SAM3 calls."""
        return [e["prompt"] for e in cls.get_parts(object_label)]

    @classmethod
    def get_supported_objects(cls) -> List[str]:
        return list(cls.PART_MAP.keys())

    @classmethod
    def get_pick_objects(cls) -> List[str]:
        return ["cup", "mug", "bottle", "bowl"]

    @classmethod
    def get_place_objects(cls) -> List[str]:
        return ["tray", "table"]

    @classmethod
    def print_config(cls):
        print("=" * 60)
        print("PartConfig")
        print("=" * 60)
        print("\n  [pick targets]")
        for obj in cls.get_pick_objects():
            print(f"    {obj}:")
            for e in cls.get_parts(obj):
                print(f"      part_name={e['part_name']:8s}  prompt='{e['prompt']}'")
        print("\n  [place targets]")
        for obj in cls.get_place_objects():
            print(f"    {obj}:")
            for e in cls.get_parts(obj):
                print(f"      part_name={e['part_name']:8s}  prompt='{e['prompt']}'")
        print("=" * 60)


if __name__ == "__main__":
    PartConfig.print_config()

    for obj in ["cup", "mug", "bottle", "bowl", "tray", "table"]:
        names   = PartConfig.get_part_names(obj)
        prompts = PartConfig.get_prompts(obj)
        print(f"  {obj:8s}: {names}  prompts={prompts}")

    print(f"  pick  targets: {PartConfig.get_pick_objects()}")
    print(f"  place targets: {PartConfig.get_place_objects()}")

    assert PartConfig.get_part_names("tray")  == ["surface", "rim"]
    assert PartConfig.get_part_names("table") == ["surface"]
    assert PartConfig.get_parts("spaceship")  == []
    assert not PartConfig.has_parts("spaceship")
    assert PartConfig.get_parts("CUP")  == PartConfig.get_parts("cup")
    assert PartConfig.get_parts("TRAY") == PartConfig.get_parts("tray")

    all_objs = PartConfig.get_supported_objects()
    assert "tray" in all_objs and "table" in all_objs

    print("all checks passed")