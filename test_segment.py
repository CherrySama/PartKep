"""
测试SAM3Segmenter - 完整pipeline测试

流程：
    1. GroundingDINO检测物体
    2. ImageProcessor裁剪物体
    3. SAM3Segmenter分割部件+提取关键点
    4. 可视化结果

运行方式：
    cd /workspace/PartKep
    python test_sam3segmenter.py

文件放置说明：
    方案1（推荐）：将sam3segmenter.py放在modules/目录下
        from modules.sam3segmenter import SAM3Segmenter
    
    方案2：将sam3segmenter.py放在项目根目录
        from sam3segmenter import SAM3Segmenter
"""

import numpy as np
from PIL import Image, ImageDraw

from modules import GroundingDINODetector, ImageProcessor

# 尝试从modules导入（如果sam3segmenter.py在modules/目录下）
try:
    from modules.sam3segmenter import SAM3Segmenter
except ImportError:
    # 如果不在modules目录，尝试从根目录导入
    try:
        from sam3segmenter import SAM3Segmenter
    except ImportError:
        print("❌ 无法导入SAM3Segmenter")
        print("请确保sam3segmenter.py在以下位置之一：")
        print("  1. /workspace/PartKep/modules/sam3segmenter.py")
        print("  2. /workspace/PartKep/sam3segmenter.py")
        raise


def visualize_keypoints(image, keypoints_results, save_path="test_keypoints_result.jpg"):
    """
    可视化关键点在原图上
    
    Args:
        image: 原始图像（PIL Image）
        keypoints_results: SAM3Segmenter返回的结果列表
        save_path: 保存路径
    """
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image.copy()
    
    draw = ImageDraw.Draw(image_pil)
    
    # 定义颜色
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
    print(f"\n🎨 可视化 {len(keypoints_results)} 个部件的关键点:")
    
    for idx, result in enumerate(keypoints_results):
        part_name = result['part_name']
        keypoint = result['keypoint']  # (x, y) 原图坐标
        score = result['score']
        
        x, y = keypoint
        color = colors[idx % len(colors)]
        
        print(f"  [{idx+1}] {part_name}: ({x:.1f}, {y:.1f}), score={score:.3f}, color={color}")
        
        # 画关键点（大圆点）
        radius = 8
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline='white',
            width=2
        )
        
        # 画标签
        text = f"{part_name}"
        draw.text((x + 15, y - 10), text, fill=color)
    
    # 保存
    image_pil.save(save_path)
    print(f"\n💾 可视化结果已保存: {save_path}")
    
    return image_pil


def test_full_pipeline():
    """测试完整的pipeline"""
    
    print("=" * 80)
    print("测试 SAM3Segmenter - 完整Pipeline")
    print("=" * 80)
    print()
    
    # ==================== 第1步：初始化模块 ====================
    print("【第1步】初始化模块")
    print("-" * 80)
    
    detector = GroundingDINODetector()
    processor = ImageProcessor(output_dir="images/objectlist")
    segmenter = SAM3Segmenter()
    
    print()
    
    # ==================== 第2步：读取测试图片 ====================
    print("【第2步】读取测试图片")
    print("-" * 80)
    
    image_path = "images/cup3.jpg"
    print(f"📁 图片路径: {image_path}")
    
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"✓ 图片加载成功: {image.size[0]}x{image.size[1]}")
    except Exception as e:
        print(f"❌ 图片加载失败: {e}")
        return
    
    print()
    
    # ==================== 第3步：GroundingDINO检测 ====================
    print("【第3步】GroundingDINO检测物体")
    print("-" * 80)
    
    detection_results = detector.detect(
        image=image,
        text_prompt="cup",
        box_threshold=0.35,
        text_threshold=0.25
    )
    
    if len(detection_results) == 0:
        print("❌ 未检测到物体")
        return
    
    print(f"✓ 检测到 {len(detection_results)} 个物体")
    print()
    
    # ==================== 第4步：裁剪物体 ====================
    print("【第4步】裁剪物体")
    print("-" * 80)
    
    crop_results = processor.crop_objects_batch(
        image=image,
        detection_results=detection_results,
        padding=10
    )
    
    if len(crop_results) == 0:
        print("❌ 裁剪失败")
        return
    
    # 使用第一个裁剪结果
    crop_result = crop_results[0]
    print(f"\n✓ 使用第一个检测结果: {crop_result['label']}")
    print(f"  裁剪尺寸: {crop_result['crop_size']}")
    print(f"  bbox_pixel: {crop_result['bbox_pixel']}")
    print()
    
    # ==================== 第5步：SAM3分割部件 ====================
    print("【第5步】SAM3分割部件并提取关键点")
    print("-" * 80)
    
    segmentation_results = segmenter.segment_parts(
        cropped_image=crop_result['cropped_image'],
        label=crop_result['label'],
        crop_bbox=crop_result['bbox_pixel']
    )
    
    print()
    
    # ==================== 第6步：可视化结果 ====================
    print("【第6步】可视化结果")
    print("-" * 80)
    
    if len(segmentation_results) > 0:
        visualize_keypoints(
            image=image,
            keypoints_results=segmentation_results,
            save_path="images/cup3_keypoints_result.jpg"
        )
    else:
        print("⚠️  没有分割结果，跳过可视化")
    
    print()
    
    # ==================== 总结 ====================
    print("=" * 80)
    print("✅ 完整Pipeline测试完成！")
    print("=" * 80)
    print(f"\n📊 结果汇总:")
    print(f"  检测到物体: {len(detection_results)}")
    print(f"  成功分割部件: {len(segmentation_results)}")
    
    if len(segmentation_results) > 0:
        print(f"\n  部件关键点详情:")
        for result in segmentation_results:
            print(f"    - {result['part_name']}: "
                  f"({result['keypoint'][0]:.2f}, {result['keypoint'][1]:.2f})")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        test_full_pipeline()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        