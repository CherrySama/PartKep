"""
Created by Yinghao Ho on 2026-1-24
"""

import os
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime

import numpy as np
from PIL import Image


class ImageProcessor:
    """
    图像处理器
    
    功能：
        - 根据检测结果裁剪物体区域
        - 保存裁剪后的图像
        - 支持批量处理
        - 自动创建输出目录
    
    使用场景：
        在Grounding DINO检测后，将检测到的物体裁剪出来，
        避免SAM3在全图中混淆不同物体的部件
    """
    
    def __init__(self, output_dir: str = "images/objectlist"):
        """
        初始化图像处理器
        
        Args:
            output_dir: 裁剪图像的保存目录
        """
        self.output_dir = Path(output_dir)
        
        # 创建输出目录（如果不存在）
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ ImageProcessor 初始化完成")
        print(f"  输出目录: {self.output_dir}")
    
    def crop_object(self,
                    image: Union[np.ndarray, Image.Image],
                    bbox: List[float],
                    label: str,
                    score: float = 0.0,
                    object_id: Optional[int] = None,
                    padding: int = 0) -> Dict:
        """
        裁剪单个检测到的物体
        
        Args:
            image: 原始图像（PIL Image或numpy array）
            bbox: 归一化坐标 [x1, y1, x2, y2]，范围[0, 1]
            label: 物体类别（如"cup", "bottle"）
            score: 检测置信度
            object_id: 物体ID（用于文件命名，如果为None则自动生成）
            padding: 裁剪时的边距（像素），默认0
        
        Returns:
            Dict: 裁剪结果
            {
                'label': 'cup',
                'bbox': [x1, y1, x2, y2],  # 归一化坐标
                'bbox_pixel': [x1, y1, x2, y2],  # 像素坐标
                'score': 0.95,
                'cropped_image': PIL.Image,  # 裁剪后的图像
                'save_path': 'images/objectlist/cup_0.jpg',
                'crop_size': (width, height)  # 裁剪图像尺寸
            }
        
        Raises:
            ValueError: 如果bbox坐标无效
        """
        # 转换为PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image_pil = image
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")
        
        # 确保是RGB格式
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        # 获取图像尺寸
        img_width, img_height = image_pil.size
        
        # 验证bbox
        if len(bbox) != 4:
            raise ValueError(f"bbox必须包含4个值 [x1, y1, x2, y2]，当前: {bbox}")
        
        # 转换归一化坐标为像素坐标
        x1_norm, y1_norm, x2_norm, y2_norm = bbox
        x1 = int(x1_norm * img_width)
        y1 = int(y1_norm * img_height)
        x2 = int(x2_norm * img_width)
        y2 = int(y2_norm * img_height)
        
        # 应用padding（确保不超出图像边界）
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_width, x2 + padding)
        y2 = min(img_height, y2 + padding)
        
        # 验证裁剪区域
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"无效的裁剪区域: ({x1}, {y1}, {x2}, {y2}), "
                f"图像尺寸: {img_width}x{img_height}"
            )
        
        # 裁剪图像
        cropped_image = image_pil.crop((x1, y1, x2, y2))
        crop_width, crop_height = cropped_image.size
        
        # 生成保存文件名
        if object_id is None:
            # 使用时间戳生成唯一ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{label}_{timestamp}.jpg"
        else:
            filename = f"{label}_{object_id}.jpg"
        
        save_path = self.output_dir / filename
        
        # 保存裁剪后的图像
        cropped_image.save(save_path, quality=95)
        
        # 返回裁剪结果
        return {
            'label': label,
            'bbox': bbox,  # 归一化坐标
            'bbox_pixel': [x1, y1, x2, y2],  # 像素坐标
            'score': score,
            'cropped_image': cropped_image,
            'save_path': str(save_path),
            'crop_size': (crop_width, crop_height)
        }
    
    def crop_objects_batch(self,
                          image: Union[np.ndarray, Image.Image],
                          detection_results: List[Dict],
                          padding: int = 0) -> List[Dict]:
        """
        批量裁剪多个检测结果
        
        Args:
            image: 原始图像
            detection_results: Grounding DINO的detect()返回的结果列表
                [
                    {'bbox': [...], 'label': 'cup', 'score': 0.95},
                    {'bbox': [...], 'label': 'bottle', 'score': 0.87}
                ]
            padding: 裁剪时的边距（像素），默认0
        
        Returns:
            List[Dict]: 裁剪结果列表
        """
        cropped_results = []
        
        print(f"\n🔪 批量裁剪 {len(detection_results)} 个检测结果...")
        
        for idx, detection in enumerate(detection_results):
            try:
                crop_result = self.crop_object(
                    image=image,
                    bbox=detection['bbox'],
                    label=detection['label'],
                    score=detection.get('score', 0.0),
                    object_id=idx,
                    padding=padding
                )
                
                cropped_results.append(crop_result)
                
                print(f"  [{idx+1}/{len(detection_results)}] {detection['label']}: "
                      f"裁剪尺寸 {crop_result['crop_size'][0]}x{crop_result['crop_size'][1]}, "
                      f"保存到 {crop_result['save_path']}")
                
            except Exception as e:
                print(f"  ❌ 裁剪失败 [{idx+1}]: {e}")
                continue
        
        print(f"✓ 批量裁剪完成，成功 {len(cropped_results)}/{len(detection_results)}")
        
        return cropped_results
    
    def clear_output_dir(self):
        """
        清空输出目录（慎用！）
        用于清理之前的裁剪结果
        """
        import shutil
        
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ 已清空输出目录: {self.output_dir}")
        else:
            print(f"⚠️  输出目录不存在: {self.output_dir}")
    
    def get_crop_info(self, crop_result: Dict) -> str:
        """
        获取裁剪结果的格式化信息（用于日志）
        
        Args:
            crop_result: crop_object()返回的结果
        
        Returns:
            str: 格式化的信息字符串
        """
        info = (
            f"类别: {crop_result['label']}\n"
            f"置信度: {crop_result['score']:.3f}\n"
            f"归一化坐标: {crop_result['bbox']}\n"
            f"像素坐标: {crop_result['bbox_pixel']}\n"
            f"裁剪尺寸: {crop_result['crop_size'][0]}x{crop_result['crop_size'][1]}\n"
            f"保存路径: {crop_result['save_path']}"
        )
        return info


# ==================== 模块测试代码 ====================
if __name__ == "__main__":
    """
    测试ImageProcessor
    运行方式: python modules/image_processor.py
    """
    import sys
    
    print("=" * 60)
    print("测试 ImageProcessor - 图像裁剪功能")
    print("=" * 60)
    print()
    
    # 创建处理器
    processor = ImageProcessor(output_dir="images/objectlist")
    print()
    
    # 创建一个测试图像
    print("📷 创建测试图像...")
    test_image = Image.new('RGB', (800, 600), color='white')
    print(f"  图像尺寸: {test_image.size}")
    print()
    
    # 模拟检测结果
    print("🎯 模拟检测结果...")
    fake_detections = [
        {'bbox': [0.1, 0.2, 0.4, 0.6], 'label': 'cup', 'score': 0.95},
        {'bbox': [0.5, 0.3, 0.8, 0.7], 'label': 'bottle', 'score': 0.87}
    ]
    print(f"  检测数量: {len(fake_detections)}")
    print()
    
    # 测试批量裁剪
    print("🔪 测试批量裁剪...")
    crop_results = processor.crop_objects_batch(
        image=test_image,
        detection_results=fake_detections,
        padding=10
    )
    print()
    
    # 显示裁剪结果信息
    print("📊 裁剪结果详情:")
    print("-" * 60)
    for i, result in enumerate(crop_results):
        print(f"\n[{i+1}]")
        print(processor.get_crop_info(result))
    
    print()
    print("=" * 60)
    print("✅ ImageProcessor 测试完成！")
    print("=" * 60)
    