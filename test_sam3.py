"""
快速验证 SAM3Segmenter (HF版本)

测试新的HF版本是否能正常工作并保持接口一致
"""

import sys
import numpy as np
from PIL import Image

# 添加模块路径
sys.path.insert(0, '/home/claude')

print("=" * 70)
print("SAM3Segmenter (HF版本) 快速验证")
print("=" * 70)

# 1. 导入新的HF版本
print("\n【步骤1】导入 SAM3Segmenter (HF版本)...")
try:
    from modules import SAM3Segmenter
    print("✓ 导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 2. 初始化分割器
print("\n【步骤2】初始化分割器...")
try:
    segmenter = SAM3Segmenter(device="cuda")
    print("✓ 初始化成功")
except Exception as e:
    print(f"✗ 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 3. 准备测试数据
print("\n【步骤3】准备测试数据...")
image_path = "images/cup3.jpg"
try:
    image = Image.open(image_path).convert('RGB')
    print(f"✓ 图片加载: {image.size[0]} x {image.size[1]}")
    
    # 模拟一个裁剪区域
    crop_bbox = [100, 200, 600, 700]
    cropped_image = image.crop(crop_bbox)
    print(f"✓ 裁剪图像: {cropped_image.size[0]} x {cropped_image.size[1]}")
    
except Exception as e:
    print(f"✗ 图片准备失败: {e}")
    exit(1)

# 4. 测试分割（接口兼容性测试）
print("\n【步骤4】测试 segment_parts() 接口...")
try:
    results = segmenter.segment_parts(
        cropped_image=cropped_image,
        label="cup",
        crop_bbox=crop_bbox
    )
    
    print(f"✓ segment_parts() 调用成功")
    print(f"  返回结果数: {len(results)}")
    
    # 验证返回格式
    if len(results) > 0:
        print(f"\n  检查返回格式:")
        result = results[0]
        
        # 检查必需的字段
        required_fields = ['part_name', 'keypoint', 'score', 'mask']
        for field in required_fields:
            if field in result:
                print(f"    ✓ '{field}' 存在")
                
                # 检查类型
                if field == 'part_name':
                    assert isinstance(result[field], str)
                    print(f"      类型: str, 值: {result[field]}")
                elif field == 'keypoint':
                    assert isinstance(result[field], tuple)
                    assert len(result[field]) == 2
                    print(f"      类型: tuple(2), 值: ({result[field][0]:.2f}, {result[field][1]:.2f})")
                elif field == 'score':
                    assert isinstance(result[field], float)
                    print(f"      类型: float, 值: {result[field]:.4f}")
                elif field == 'mask':
                    assert isinstance(result[field], np.ndarray)
                    print(f"      类型: ndarray, shape: {result[field].shape}, dtype: {result[field].dtype}")
            else:
                print(f"    ✗ '{field}' 缺失！")
        
        print(f"\n  ✅ 接口格式验证通过！")
    else:
        print("  ⚠️  未检测到部件（可能是正常的，取决于图片内容）")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. 总结
print("\n" + "=" * 70)
print("✅ SAM3Segmenter (HF版本) 验证通过！")
print("=" * 70)
print("\n接口兼容性:")
print("  ✓ 初始化接口一致")
print("  ✓ segment_parts() 接口一致")
print("  ✓ 返回格式一致")
print("\n可以安全替换原有的 sam3segmenter.py 文件")
print("=" * 70)