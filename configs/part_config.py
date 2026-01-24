"""
物体部件配置文件

定义每种物体需要分割的部件列表。
这个配置用于指导SAM3进行部件级分割。

Author: PartKep Project
Date: 2026-01
"""

from typing import Dict, List


class PartConfig:
    """
    物体部件配置类
    
    定义了不同物体类别及其对应的部件列表。
    当检测到某个物体后，根据其label查询对应的部件，
    然后传给SAM3进行逐个分割。
    
    使用场景：
        1. Grounding DINO检测到 "cup"
        2. 查询 PART_MAP["cup"] 得到 ["handle", "rim", "body"]
        3. 依次调用 SAM3 分割这些部件
    """
    
    # ==================== 核心配置：物体-部件映射 ====================
    PART_MAP: Dict[str, List[str]] = {
        # 杯子类
        "cup": ["handle", "rim", "body"],
        "mug": ["handle", "rim", "body"],
        "teacup": ["handle", "rim", "body", "saucer"],
        "kettel": ["handle", "rim", "body", "spout"],
        
        # 瓶子类
        "bottle": ["cap", "neck", "body", "label"],
        "water bottle": ["cap", "body"],
        
        # 碗类
        "bowl": ["rim", "body", "base"],
        
        # 盘子类
        "plate": ["rim", "center"],
        "dish": ["rim", "center"],
        
        # 锅类
        "pot": ["lid", "handle", "body"],
        "pan": ["handle", "body"],
        
        # 工具类
        "spoon": ["bowl", "handle"],
        "fork": ["tines", "handle"],
        "knife": ["blade", "handle"],
        
        # 其他常用物体
        "basket": ["handle", "body"],
        "bag": ["handle", "body"],
        "box": ["lid", "body"],
    }
    
    # ==================== 部件别名映射（可选）====================
    # 某些部件可能有多种叫法，这里定义别名映射
    PART_ALIASES: Dict[str, List[str]] = {
        "handle": ["grip", "holder"],
        "rim": ["edge", "lip"],
        "body": ["main", "container"],
        "cap": ["lid", "top", "cover"],
        "base": ["bottom", "foot"],
    }
    
    @classmethod
    def get_parts(cls, object_label: str) -> List[str]:
        """
        根据物体类别获取需要分割的部件列表
        
        Args:
            object_label: 物体类别标签（如 "cup", "bottle"）
        
        Returns:
            List[str]: 部件列表（如 ["handle", "rim", "body"]）
                      如果物体不在配置中，返回空列表
        
        Examples:
            >>> PartConfig.get_parts("cup")
            ['handle', 'rim', 'body']
            >>> PartConfig.get_parts("unknown_object")
            []
        """
        # 转换为小写，确保匹配
        label_lower = object_label.lower().strip()
        
        # 查询配置
        parts = cls.PART_MAP.get(label_lower, [])
        
        return parts.copy()  # 返回副本，避免外部修改
    
    @classmethod
    def has_parts(cls, object_label: str) -> bool:
        """
        检查某个物体是否有预定义的部件
        
        Args:
            object_label: 物体类别标签
        
        Returns:
            bool: 是否有预定义部件
        """
        label_lower = object_label.lower().strip()
        return label_lower in cls.PART_MAP
    
    @classmethod
    def add_object_parts(cls, object_label: str, parts: List[str]):
        """
        动态添加新物体的部件配置
        
        Args:
            object_label: 物体类别标签
            parts: 部件列表
        
        Examples:
            >>> PartConfig.add_object_parts("lamp", ["base", "shade", "bulb"])
        """
        label_lower = object_label.lower().strip()
        cls.PART_MAP[label_lower] = parts
        print(f"✓ 已添加 '{object_label}' 的部件配置: {parts}")
    
    @classmethod
    def get_all_objects(cls) -> List[str]:
        """
        获取所有已配置的物体类别
        
        Returns:
            List[str]: 物体类别列表
        """
        return list(cls.PART_MAP.keys())
    
    @classmethod
    def print_config(cls):
        """打印当前配置（用于调试）"""
        print("=" * 60)
        print("物体部件配置")
        print("=" * 60)
        print(f"已配置物体数量: {len(cls.PART_MAP)}")
        print()
        
        for obj, parts in sorted(cls.PART_MAP.items()):
            print(f"{obj:20s} → {', '.join(parts)}")
        
        print("=" * 60)
    
    @classmethod
    def validate_parts(cls, object_label: str, detected_parts: List[str]) -> Dict:
        """
        验证检测到的部件是否符合配置
        
        Args:
            object_label: 物体类别
            detected_parts: 实际检测到的部件列表
        
        Returns:
            Dict: 验证结果
            {
                'expected': [...],      # 期望的部件
                'detected': [...],      # 实际检测的部件
                'missing': [...],       # 缺失的部件
                'extra': [...],         # 多余的部件
                'match_rate': 0.8       # 匹配率
            }
        """
        expected = cls.get_parts(object_label)
        detected_set = set(detected_parts)
        expected_set = set(expected)
        
        missing = list(expected_set - detected_set)
        extra = list(detected_set - expected_set)
        
        if len(expected) > 0:
            match_rate = len(expected_set & detected_set) / len(expected)
        else:
            match_rate = 0.0
        
        return {
            'expected': expected,
            'detected': detected_parts,
            'missing': missing,
            'extra': extra,
            'match_rate': match_rate
        }


# ==================== 快捷访问函数 ====================

def get_parts_for_object(object_label: str) -> List[str]:
    """
    快捷函数：获取物体的部件列表
    
    Args:
        object_label: 物体类别
    
    Returns:
        List[str]: 部件列表
    """
    return PartConfig.get_parts(object_label)


# ==================== 模块测试代码 ====================
if __name__ == "__main__":
    """
    测试PartConfig配置
    运行方式: python configs/part_config.py
    """
    print("=" * 60)
    print("测试 PartConfig - 物体部件配置")
    print("=" * 60)
    print()
    
    # 测试1：查询预定义物体
    print("【测试1】查询预定义物体的部件")
    print("-" * 60)
    
    test_objects = ["cup", "bottle", "bowl", "fork"]
    for obj in test_objects:
        parts = PartConfig.get_parts(obj)
        print(f"{obj:15s} → {parts}")
    print()
    
    # 测试2：查询不存在的物体
    print("【测试2】查询不存在的物体")
    print("-" * 60)
    unknown_obj = "spaceship"
    parts = PartConfig.get_parts(unknown_obj)
    has_parts = PartConfig.has_parts(unknown_obj)
    print(f"{unknown_obj:15s} → {parts}")
    print(f"是否有预定义部件: {has_parts}")
    print()
    
    # 测试3：动态添加新物体
    print("【测试3】动态添加新物体")
    print("-" * 60)
    PartConfig.add_object_parts("teapot", ["spout", "handle", "lid", "body"])
    parts = PartConfig.get_parts("teapot")
    print(f"teapot → {parts}")
    print()
    
    # 测试4：大小写不敏感
    print("【测试4】大小写不敏感测试")
    print("-" * 60)
    test_cases = ["Cup", "CUP", "  cup  ", "BOTTLE"]
    for test in test_cases:
        parts = PartConfig.get_parts(test)
        print(f"'{test}' → {parts}")
    print()
    
    # 测试5：验证部件
    print("【测试5】验证检测到的部件")
    print("-" * 60)
    validation = PartConfig.validate_parts(
        "cup", 
        ["handle", "rim"]  # 缺少 "body"
    )
    print(f"期望部件: {validation['expected']}")
    print(f"检测部件: {validation['detected']}")
    print(f"缺失部件: {validation['missing']}")
    print(f"多余部件: {validation['extra']}")
    print(f"匹配率: {validation['match_rate']:.1%}")
    print()
    
    # 测试6：打印完整配置
    print("【测试6】打印完整配置")
    print()
    PartConfig.print_config()
    print()
    
    print("=" * 60)
    print("✅ PartConfig 测试完成！")
    print("=" * 60)
    