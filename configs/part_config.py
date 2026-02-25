"""
Created by Yinghao Ho on 2026-1-23
"""

from typing import Dict, List

PartEntry = Dict[str, str]

class PartConfig:
    """
    物体部件配置类

    PART_MAP 中每个物体对应一个 PartEntry 列表，每条记录包含：
        part_name : 对应 SAP 知识库的 key（如 "handle"）
        prompt    : 传给 SAM3 的文本 prompt（如 "cup handle"）

    使用场景：
        1. GroundingDINO 检测到 "cup"
        2. get_parts("cup") 返回 PartEntry 列表
        3. 用每条 entry["prompt"] 调用 SAM3 分割
        4. 用每条 entry["part_name"] 查询 SAP 知识库
    """
    
    # ==================== 核心配置：物体-部件映射 ====================
    PART_MAP: Dict[str, List[PartEntry]] = {
        "cup": [
            {"part_name": "handle", "prompt": "cup handle"},
            {"part_name": "rim",    "prompt": "cup rim"},
            {"part_name": "body",   "prompt": "cup body"},
        ],
        "mug": [
            {"part_name": "handle", "prompt": "mug handle"},
            {"part_name": "rim",    "prompt": "mug rim"},
            {"part_name": "body",   "prompt": "mug body"},
        ],

        "bottle": [
            {"part_name": "cap",    "prompt": "bottle cap"},
            {"part_name": "neck",   "prompt": "bottle neck"},
            {"part_name": "body",   "prompt": "bottle body"},
        ],

        "bowl": [
            {"part_name": "rim",    "prompt": "bowl rim"},
            {"part_name": "body",   "prompt": "bowl body"},
        ],
    }
    
    @classmethod
    def get_parts(cls, object_label: str) -> List[PartEntry]:
        """
        根据物体类别获取需要分割的部件列表
        
        Args:
            object_label: 物体类别标签（如 "cup", "bottle"）
        
        Returns:
            List[PartEntry]: 部件列表，每条含 part_name 和 prompt
                            物体不在配置中时返回空列表
        """
        key = object_label.strip().lower()
        return cls.PART_MAP.get(key, [])
    
    @classmethod
    def has_parts(cls, object_label: str) -> bool:
        """物体是否在配置中"""
        return object_label.strip().lower() in cls.PART_MAP

    @classmethod
    def get_part_names(cls, object_label: str) -> List[str]:
        """
        只返回 part_name 列表（用于 SAP 查询）

        Examples:
            >>> PartConfig.get_part_names("cup")
            ["handle", "rim", "body"]
        """
        return [e["part_name"] for e in cls.get_parts(object_label)]

    @classmethod
    def get_prompts(cls, object_label: str) -> List[str]:
        """
        只返回 prompt 列表（用于 SAM3 调用）

        Examples:
            >>> PartConfig.get_prompts("cup")
            ["cup handle", "cup rim", "cup body"]
        """
        return [e["prompt"] for e in cls.get_parts(object_label)]

    @classmethod
    def get_supported_objects(cls) -> List[str]:
        """返回所有支持的物体类别"""
        return list(cls.PART_MAP.keys())

    @classmethod
    def print_config(cls):
        """打印完整配置（调试用）"""
        print("=" * 60)
        print("PartConfig 当前配置")
        print("=" * 60)
        for obj, entries in cls.PART_MAP.items():
            print(f"\n  {obj}:")
            for e in entries:
                print(f"    part_name={e['part_name']:8s}  "
                      f"prompt='{e['prompt']}'")
        print("=" * 60)


# ==================== 模块测试代码 ====================
if __name__ == "__main__":
    print("测试 PartConfig\n")

    # 测试1：基本查询
    for obj in ["cup", "mug", "bottle", "bowl"]:
        entries = PartConfig.get_parts(obj)
        print(f"{obj}: {[(e['part_name'], e['prompt']) for e in entries]}")

    print()

    # 测试2：不存在的物体
    assert PartConfig.get_parts("spaceship") == []
    assert not PartConfig.has_parts("spaceship")
    print("✅ 不存在物体返回空列表")

    # 测试3：大小写不敏感
    assert PartConfig.get_parts("CUP") == PartConfig.get_parts("cup")
    assert PartConfig.get_parts("  Cup  ") == PartConfig.get_parts("cup")
    print("✅ 大小写不敏感")

    # 测试4：part_name 和 prompt 分离获取
    assert PartConfig.get_part_names("cup") == ["handle", "rim", "body"]
    assert PartConfig.get_prompts("cup") == ["cup handle", "cup rim", "cup body"]
    print("✅ part_name / prompt 分离获取正确")

    # 测试5：打印完整配置
    print()
    PartConfig.print_config()

    print("\n✅ 所有测试通过！")
    