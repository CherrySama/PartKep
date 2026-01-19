"""
使用真实图片测试Grounding DINO检测器

运行方式：
    cd /workspace/PartKep
    python test_real_image.py
"""

import numpy as np
from PIL import Image
from modules import GroundingDINODetector


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
            text_prompt="cup",
            box_threshold=0.35,
            text_threshold=0.25
        )
        
        print()
        print("=" * 60)
        print("✅ 检测完成！")
        print("=" * 60)
        print(f"检测结果: {results}")
        print()
        
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