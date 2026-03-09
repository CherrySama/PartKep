"""
Created by Yinghao Ho on 2026-1-23

物体部件配置文件
每个部件条目包含两个字段：
    part_name : SAP 知识库的 key（不可更改，与 SAP.py 严格对应）
    prompt    : 传给 SAM3 的实际文本（加物体名前缀以提升分割准确性）

物体分两类：
    pick 目标（被抓取）：cup / mug / bottle / bowl
        — 每个物体有多个部件，用于约束抓取姿态
    place 目标（被放置到）：tray / table
        — 只需识别承载面（surface）和边缘（rim，若有）
        — surface 对应 SAP 的 contact_mode='place'

使用场景：
    1. TaskParser 解析出 object_label（pick）/ target_label（place）
    2. GroundingDINO 检测到物体 bbox
    3. get_parts(label) 返回 PartEntry 列表
    4. 用每条 entry["prompt"] 调用 SAM3 分割
    5. 用每条 entry["part_name"] 查询 SAP 知识库生成约束
"""

from typing import Dict, List

PartEntry = Dict[str, str]


class PartConfig:
    """
    物体部件配置类

    PART_MAP 中每个物体对应一个 PartEntry 列表，每条记录包含：
        part_name : 对应 SAP 知识库的 key（如 "handle"、"surface"）
        prompt    : 传给 SAM3 的文本 prompt（如 "cup handle"）
    """

    PART_MAP: Dict[str, List[PartEntry]] = {

        # ==================== pick 目标 ====================

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

        # ==================== place 目标 ====================
        # surface → SAP contact_mode='place'，放置目标承载面
        # rim     → SAP contact_mode='avoid'，边缘回避（tray 有边框）

        "tray": [
            {"part_name": "surface", "prompt": "tray surface"},
            {"part_name": "rim",     "prompt": "tray rim"},
        ],
        "table": [
            {"part_name": "surface", "prompt": "table surface"},
        ],
    }

    # ==================== 查询接口 ====================

    @classmethod
    def get_parts(cls, object_label: str) -> List[PartEntry]:
        """
        获取物体的部件列表

        Returns:
            List[PartEntry]，物体不存在时返回空列表
        """
        return cls.PART_MAP.get(object_label.strip().lower(), [])

    @classmethod
    def has_parts(cls, object_label: str) -> bool:
        """物体是否在配置中"""
        return object_label.strip().lower() in cls.PART_MAP

    @classmethod
    def get_part_names(cls, object_label: str) -> List[str]:
        """只返回 part_name 列表（用于 SAP 查询）"""
        return [e["part_name"] for e in cls.get_parts(object_label)]

    @classmethod
    def get_prompts(cls, object_label: str) -> List[str]:
        """只返回 prompt 列表（用于 SAM3 调用）"""
        return [e["prompt"] for e in cls.get_parts(object_label)]

    @classmethod
    def get_supported_objects(cls) -> List[str]:
        """返回所有支持的物体类别"""
        return list(cls.PART_MAP.keys())

    @classmethod
    def get_pick_objects(cls) -> List[str]:
        """返回所有 pick 目标物体"""
        return ["cup", "mug", "bottle", "bowl"]

    @classmethod
    def get_place_objects(cls) -> List[str]:
        """返回所有 place 目标物体"""
        return ["tray", "table"]

    @classmethod
    def print_config(cls):
        """打印完整配置（调试用）"""
        print("=" * 60)
        print("PartConfig 当前配置")
        print("=" * 60)
        print("\n  [pick 目标]")
        for obj in cls.get_pick_objects():
            entries = cls.get_parts(obj)
            print(f"    {obj}:")
            for e in entries:
                print(f"      part_name={e['part_name']:8s}  "
                      f"prompt='{e['prompt']}'")
        print("\n  [place 目标]")
        for obj in cls.get_place_objects():
            entries = cls.get_parts(obj)
            print(f"    {obj}:")
            for e in entries:
                print(f"      part_name={e['part_name']:8s}  "
                      f"prompt='{e['prompt']}'")
        print("=" * 60)


# ==================== 模块测试 ====================
if __name__ == "__main__":
    print("测试 PartConfig\n")

    # 【1】打印完整配置
    PartConfig.print_config()
    print()

    # 【2】基本查询
    print("【2】基本查询")
    for obj in ["cup", "mug", "bottle", "bowl", "tray", "table"]:
        names   = PartConfig.get_part_names(obj)
        prompts = PartConfig.get_prompts(obj)
        print(f"  {obj:8s}: {names}")
        print(f"  {'':8s}  prompts={prompts}")
    print()

    # 【3】pick / place 分类
    print("【3】pick / place 分类")
    print(f"  pick  目标: {PartConfig.get_pick_objects()}")
    print(f"  place 目标: {PartConfig.get_place_objects()}")
    print()

    # 【4】tray 包含 surface + rim
    assert PartConfig.get_part_names("tray") == ["surface", "rim"]
    print("  ✅ tray 部件顺序正确: ['surface', 'rim']")

    # 【5】table 只有 surface
    assert PartConfig.get_part_names("table") == ["surface"]
    print("  ✅ table 部件正确: ['surface']")

    # 【6】不存在物体返回空列表
    assert PartConfig.get_parts("spaceship") == []
    assert not PartConfig.has_parts("spaceship")
    print("  ✅ 不存在物体返回空列表")

    # 【7】大小写不敏感
    assert PartConfig.get_parts("CUP") == PartConfig.get_parts("cup")
    assert PartConfig.get_parts("TRAY") == PartConfig.get_parts("tray")
    print("  ✅ 大小写不敏感")

    # 【8】TaskParser 支持列表包含 place 目标
    all_objs = PartConfig.get_supported_objects()
    assert "tray" in all_objs and "table" in all_objs
    print(f"  ✅ supported_objects = {all_objs}")

    print("\n✅ PartConfig 所有测试通过！")
    