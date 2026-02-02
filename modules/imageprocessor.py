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
    图像处理器（简化坐标系统版本）
    
    坐标系统简化 (2026-2-2)：
        - 直接接收绝对像素坐标（来自GroundingDINO HF版本）
        - 去掉了不必要的归一化→反归一化转换
        - 流程更简洁：绝对坐标 → padding → 裁剪
    
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
            bbox: 绝对像素坐标 [x1, y1, x2, y2]（浮点数）
            label: 物体类别（如"cup", "bottle"）
            score: 检测置信度
            object_id: 物体ID（用于文件命名，如果为None则自动生成）
            padding: 裁剪时的边距（像素），默认0
        
        Returns:
            Dict: 裁剪结果
            {
                'label': 'cup',
                'bbox': [x1, y1, x2, y2],  # 绝对像素坐标（浮点，原始）
                'bbox_pixel': [x1, y1, x2, y2],  # 像素坐标（整数，用于裁剪）
                'bbox_float': [x1, y1, x2, y2],  # 像素坐标（浮点，高精度，应用padding后）
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
        
        # ==================== 坐标处理 ====================
        x1_float, y1_float, x2_float, y2_float = bbox
        
        # 应用padding到浮点坐标（padding保持整数）
        x1_float_padded = x1_float - padding
        y1_float_padded = y1_float - padding
        x2_float_padded = x2_float + padding
        y2_float_padded = y2_float + padding
        
        # 浮点边界检查（在浮点阶段检查边界，保留更多精度）
        x1_float_padded = max(0.0, min(x1_float_padded, float(img_width)))
        y1_float_padded = max(0.0, min(y1_float_padded, float(img_height)))
        x2_float_padded = max(0.0, min(x2_float_padded, float(img_width)))
        y2_float_padded = max(0.0, min(y2_float_padded, float(img_height)))
        
        # 取整用于裁剪（只在必须裁剪时才转为整数）
        x1 = int(x1_float_padded)
        y1 = int(y1_float_padded)
        x2 = int(x2_float_padded)
        y2 = int(y2_float_padded)
        
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
            'bbox': bbox,  # 绝对像素坐标（浮点，原始）
            'bbox_pixel': [x1, y1, x2, y2],  # 像素坐标（整数，用于裁剪）
            'bbox_float': [x1_float_padded, y1_float_padded, 
                          x2_float_padded, y2_float_padded],  # 浮点坐标（高精度，应用padding后）
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
            f"绝对坐标（原始）: [{crop_result['bbox'][0]:.2f}, {crop_result['bbox'][1]:.2f}, "
            f"{crop_result['bbox'][2]:.2f}, {crop_result['bbox'][3]:.2f}]\n"
            f"像素坐标（整数）: {crop_result['bbox_pixel']}\n"
            f"像素坐标（浮点，padding后）: [{crop_result['bbox_float'][0]:.4f}, "
            f"{crop_result['bbox_float'][1]:.4f}, "
            f"{crop_result['bbox_float'][2]:.4f}, "
            f"{crop_result['bbox_float'][3]:.4f}]\n"
            f"裁剪尺寸: {crop_result['crop_size'][0]}x{crop_result['crop_size'][1]}\n"
            f"保存路径: {crop_result['save_path']}"
        )
        return info