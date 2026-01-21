"""
使用真实图片测试Grounding DINO检测器

运行方式：
    cd /workspace/PartKep
    python test_real_image.py
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from modules import GroundingDINODetector


def visualize_detections(image, results, save_path="detection_result.jpg"):
    """
    可视化检测结果，将bbox画在图像上
    
    Args:
        image: PIL.Image对象或numpy array
        results: detect()返回的检测结果列表
        save_path: 保存可视化结果的路径
    """
    # 确保image是PIL Image格式
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image.copy()
    
    # 创建绘图对象
    draw = ImageDraw.Draw(image_pil)
    
    # 获取图像尺寸（用于将归一化坐标转换为像素坐标）
    img_width, img_height = image_pil.size
    
    # 定义颜色（可以根据类别选择不同颜色）
    colors = {
        'cup': '#00FF00',      # 绿色
        'bottle': '#FF0000',   # 红色
        'handle': '#0000FF',   # 蓝色
        'default': '#FFFF00'   # 黄色（默认）
    }
    
    # 尝试加载字体（如果失败则使用默认字体）
    try:
        # 尝试使用更大的字体以便看清
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        # 如果找不到字体文件，使用默认字体
        font = ImageFont.load_default()
    
    print(f"\n📊 可视化 {len(results)} 个检测结果:")
    
    # 遍历每个检测结果
    for i, result in enumerate(results):
        bbox_norm = result['bbox']  # 归一化坐标 [x1, y1, x2, y2]
        label = result['label']
        score = result['score']
        
        # 转换为像素坐标
        x1 = int(bbox_norm[0] * img_width)
        y1 = int(bbox_norm[1] * img_height)
        x2 = int(bbox_norm[2] * img_width)
        y2 = int(bbox_norm[3] * img_height)
        
        print(f"  [{i+1}] {label}: score={score:.3f}")
        print(f"      归一化坐标: [{bbox_norm[0]:.3f}, {bbox_norm[1]:.3f}, {bbox_norm[2]:.3f}, {bbox_norm[3]:.3f}]")
        print(f"      像素坐标: [{x1}, {y1}, {x2}, {y2}]")
        print(f"      尺寸: {x2-x1} x {y2-y1} 像素")
        
        # 选择颜色
        color = colors.get(label, colors['default'])
        
        # 画边界框（加粗线条）
        line_width = 4
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        # 准备标签文本
        text = f"{label}: {score:.2f}"
        
        # 获取文本边界框
        try:
            # 新版PIL
            bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # 旧版PIL
            text_width, text_height = draw.textsize(text, font=font)
        
        # 画文本背景（半透明效果）
        text_bg_box = [x1, y1 - text_height - 10, x1 + text_width + 10, y1]
        draw.rectangle(text_bg_box, fill=color)
        
        # 画文本（黑色，更清晰）
        draw.text((x1 + 5, y1 - text_height - 5), text, fill='black', font=font)
    
    # 保存结果
    image_pil.save(save_path)
    print(f"\n💾 可视化结果已保存到: {save_path}")
    print(f"   图像尺寸: {img_width} x {img_height}")
    
    return image_pil


def test_real_image():
    """使用真实图片测试检测功能"""
    
    print("=" * 60)
    print("测试 Grounding DINO 检测器 - 真实图片")
    print("=" * 60)
    print()
    
    # 初始化检测器
    print("正在初始化检测器...")
    detector = GroundingDINODetector()
    print()
    
    # ==================== 测试：检测杯子 ====================
    print("【测试】使用真实图片检测杯子")
    print("-" * 60)
    
    # 读取测试图片
    image_path = "images/cup3.jpg"
    print(f"📁 读取图片: {image_path}")
    
    try:
        # 方式1：使用PIL读取
        image = Image.open(image_path)
        print(f"✓ 图片加载成功: {image.size[0]} x {image.size[1]}")
        
        # 确保是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"✓ 转换为RGB格式")
        
    except FileNotFoundError:
        print(f"❌ 图片文件不存在: {image_path}")
        print("   请确保 images/cup3.jpg 文件存在")
        return
    except Exception as e:
        print(f"❌ 图片加载失败: {e}")
        return
    
    print()
    
    # 执行检测
    try:
        print("🔍 开始检测...")
        results = detector.detect(
            image=image,
            text_prompt="a cup",
            box_threshold=0.35,
            text_threshold=0.25
        )
        
        print()
        print("=" * 60)
        print("✅ 检测完成！")
        print("=" * 60)
        print(f"检测到 {len(results)} 个物体:")
        for i, result in enumerate(results):
            print(f"  [{i+1}] {result['label']}: score={result['score']:.3f}")
        print()
        
        # 可视化检测结果
        if len(results) > 0:
            print("🎨 正在可视化检测结果...")
            visualize_detections(
                image=image,
                results=results,
                save_path="images/cup3_detection.jpg"
            )
            print()
        else:
            print("⚠️  没有检测结果，跳过可视化")
        
    except Exception as e:
        print(f"❌ 检测失败: {e}")
        import traceback
        traceback.print_exc()


def test_with_numpy_array():
    """测试numpy array输入（模拟相机流）"""
    
    print("\n" + "=" * 60)
    print("【测试】numpy array输入（模拟从相机读取）")
    print("=" * 60)
    print()
    
    # 初始化检测器
    detector = GroundingDINODetector()
    print()
    
    # 读取图片并转换为numpy array
    image_path = "images/cup3.jpg"
    print(f"📁 读取图片: {image_path}")
    
    try:
        image_pil = Image.open(image_path).convert('RGB')
        image_np = np.array(image_pil)  # 转换为numpy array
        
        print(f"✓ 图片转换为numpy array: {image_np.shape}")
        print()
        
        # 执行检测
        print("🔍 开始检测...")
        results = detector.detect(
            image=image_np,  # 使用numpy array
            text_prompt="cup"
        )
        
        print()
        print("=" * 60)
        print("✅ numpy array输入检测完成！")
        print("=" * 60)
        print(f"检测结果: {results}")
        print()
        
    except Exception as e:
        print(f"❌ 检测失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 测试1：PIL Image输入
    test_real_image()
    
    # 测试2：numpy array输入
    # test_with_numpy_array()  # 取消注释来测试