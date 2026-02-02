"""
下载HF模型到本地 - 简化版

将模型下载到 /workspace/PartKep/models/ 目录
运行方式: python download_models.py
"""

from transformers import (
    Sam3Model, 
    Sam3Processor,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor
)

# 本地路径
SAM3_PATH = "models/sam3"
GROUNDING_DINO_PATH = "models/grounding-dino-base"

print("=" * 70)
print("下载 HF 模型到本地")
print("=" * 70)
print()

# ==================== 下载 SAM3 ====================
print("【1/2】下载 SAM3 模型")
print("-" * 70)
print(f"模型ID: facebook/sam3")
print(f"保存到: {SAM3_PATH}")
print(f"大小: ~3.5GB")
print()

try:
    print("正在下载...")
    model = Sam3Model.from_pretrained("facebook/sam3")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    
    print("正在保存到本地...")
    model.save_pretrained(SAM3_PATH)
    processor.save_pretrained(SAM3_PATH)
    
    print("✅ SAM3 下载完成！")
except Exception as e:
    print(f"❌ SAM3 下载失败: {e}")
    exit(1)

print()

# ==================== 下载 GroundingDINO ====================
print("【2/2】下载 GroundingDINO 模型")
print("-" * 70)
print(f"模型ID: IDEA-Research/grounding-dino-base")
print(f"保存到: {GROUNDING_DINO_PATH}")
print(f"大小: ~1.5GB")
print()

try:
    print("正在下载...")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    )
    processor = AutoProcessor.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    )
    
    print("正在保存到本地...")
    model.save_pretrained(GROUNDING_DINO_PATH)
    processor.save_pretrained(GROUNDING_DINO_PATH)
    
    print("✅ GroundingDINO 下载完成！")
except Exception as e:
    print(f"❌ GroundingDINO 下载失败: {e}")
    exit(1)

print()
print("=" * 70)
print("✅ 所有模型下载完成！")
print("=" * 70)
print()
print("模型位置:")
print(f"  SAM3: {SAM3_PATH}")
print(f"  GroundingDINO: {GROUNDING_DINO_PATH}")
print()
print("下一步: 运行你的代码，模块会自动使用本地模型")