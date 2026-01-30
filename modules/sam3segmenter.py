"""
Created by Yinghao Ho on 2026-1-26
"""

from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from configs.part_config import PartConfig


class SAM3Segmenter:
    """
    SAM3部件分割器
    
    使用SAM3模型分割物体的各个部件，并为每个部件提取一个关键点。
    关键点提取使用质心+最近点投影方法，确保关键点落在mask表面上。
    
    工作流程：
        1. 根据物体label从PartConfig获取部件列表
        2. 使用SAM3逐个分割部件（feature reuse优化）
        3. 为每个部件提取单个关键点
        4. 将关键点坐标转换到原始图像坐标系
    
    使用示例：
        >>> segmenter = SAM3Segmenter()
        >>> results = segmenter.segment_parts(
        ...     cropped_image=cup_image,
        ...     label="cup",
        ...     crop_bbox=[128, 144, 512, 432]
        ... )
        >>> for result in results:
        ...     print(f"{result['part_name']}: {result['keypoint']}")
    """
    
    def __init__(self, 
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda"):
        """
        初始化SAM3分割器
        
        Args:
            checkpoint_path: SAM3模型权重路径，如果为None则使用默认路径
            device: 运行设备，"cuda"或"cpu"
        
        Raises:
            FileNotFoundError: 如果模型文件不存在
            RuntimeError: 如果模型加载失败
        """
        print("=" * 60)
        print("初始化 SAM3 分割器")
        print("=" * 60)
        
        # 验证设备
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  警告: CUDA不可用，自动切换到CPU模式")
            device = "cpu"
        
        self.device = device
        print(f"✓ 运行设备: {self.device}")
        
        # 设置默认模型路径
        if checkpoint_path is None:
            checkpoint_path = "/workspace/PartKep/models/SAM3/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
            print(f"✓ 使用默认模型路径")
        
        self.checkpoint_path = checkpoint_path
        print(f"  路径: {self.checkpoint_path}")
        
        # 验证模型文件存在
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"SAM3模型文件不存在: {checkpoint_path}\n"
                f"请确保已下载模型到正确位置"
            )
        
        # 加载SAM3模型
        print("\n正在加载 SAM3 模型...")
        try:
            self.model = build_sam3_image_model(checkpoint_path=checkpoint_path)
            self.processor = Sam3Processor(self.model)
            print("✅ SAM3 模型加载成功！")
        except Exception as e:
            raise RuntimeError(f"SAM3模型加载失败: {e}")
        
        print("\n" + "=" * 60)
        print("✅ SAM3 分割器初始化完成！")
        print("=" * 60)
        print()
    
    def segment_parts(self,
                     cropped_image: Union[np.ndarray, Image.Image],
                     label: str,
                     crop_bbox: List[int]) -> List[Dict]:
        """
        分割物体的所有部件并提取关键点
        
        Args:
            cropped_image: 裁剪后的物体图像
            label: 物体类别（如"cup", "bottle"）
            crop_bbox: 裁剪框在原图的像素坐标 [x1, y1, x2, y2]
        
        Returns:
            List[Dict]: 每个部件的分割结果
            [
                {
                    'part_name': 'handle',
                    'keypoint': (x, y),  # 原图坐标（浮点数）
                    'score': 0.95,       # SAM3置信度
                    'mask': np.ndarray   # 二值mask (H, W)
                },
                ...
            ]
        
        注意：
            - 如果某个部件未检测到，会跳过并打印警告
            - 关键点坐标是浮点数，保持亚像素精度
            - mask是相对于裁剪图的，关键点是相对于原图的
        """
        print(f"\n🔍 开始分割物体部件: {label}")
        print(f"  裁剪框位置: {crop_bbox}")
        
        # 1. 转换图像格式
        if isinstance(cropped_image, np.ndarray):
            image_pil = Image.fromarray(cropped_image)
        elif isinstance(cropped_image, Image.Image):
            image_pil = cropped_image
        else:
            raise ValueError(
                f"不支持的图像类型: {type(cropped_image)}，"
                f"必须是 numpy.ndarray 或 PIL.Image.Image"
            )
        
        # 确保是RGB格式
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        print(f"  裁剪图像尺寸: {image_pil.size[0]}x{image_pil.size[1]}")
        
        # 2. 从PartConfig获取部件列表
        parts = PartConfig.get_parts(label)
        
        if len(parts) == 0:
            print(f"⚠️  警告: 物体 '{label}' 没有预定义的部件配置")
            return []
        
        print(f"  部件列表: {parts}")
        
        # 3. 设置图像（feature reuse优化：只调用一次）
        print("\n📸 设置图像（提取backbone features）...")
        inference_state = self.processor.set_image(image_pil)
        print("✓ 图像特征提取完成")
        
        # 4. 循环处理每个部件
        results = []
        
        for idx, part_name in enumerate(parts):
            print(f"\n[{idx+1}/{len(parts)}] 处理部件: {part_name}")
            
            try:
                # 4.1 使用SAM3分割部件
                print(f"  → 调用SAM3分割...")
                output = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=part_name
                )
                
                # 检查是否有检测结果
                if len(output['masks']) == 0:
                    print(f"  ⚠️  未检测到部件 '{part_name}'，跳过")
                    continue
                
                # 取第一个检测结果（置信度最高）
                mask_tensor = output['masks'][0]  # shape=(1, H, W), bool
                score = float(output['scores'][0])
                
                print(f"  ✓ 检测成功，置信度: {score:.3f}")
                
                # 4.2 转换mask为numpy格式
                mask_np = mask_tensor.cpu().numpy().squeeze().astype(np.uint8) * 255
                print(f"  ✓ Mask shape: {mask_np.shape}")
                
                # 4.3 提取关键点（相对于裁剪图）
                print(f"  → 提取关键点...")
                keypoint_crop = self.extract_single_keypoint(mask_np)
                
                if keypoint_crop is None:
                    print(f"  ⚠️  关键点提取失败，跳过")
                    continue
                
                print(f"  ✓ 关键点（裁剪图坐标）: ({keypoint_crop[0]:.2f}, {keypoint_crop[1]:.2f})")
                
                # 4.4 转换到原图坐标
                keypoint_orig = self.transform_to_original_coords(
                    [keypoint_crop],
                    crop_bbox
                )[0]
                
                print(f"  ✓ 关键点（原图坐标）: ({keypoint_orig[0]:.2f}, {keypoint_orig[1]:.2f})")
                
                # 4.5 保存结果
                results.append({
                    'part_name': part_name,
                    'keypoint': keypoint_orig,
                    'score': score,
                    'mask': mask_np
                })
                
                print(f"  ✅ 部件 '{part_name}' 处理完成")
                
            except Exception as e:
                print(f"  ❌ 处理部件 '{part_name}' 时出错: {e}")
                continue
        
        # 5. 输出总结
        print("\n" + "=" * 60)
        print(f"✅ 部件分割完成！")
        print(f"  物体: {label}")
        print(f"  成功分割: {len(results)}/{len(parts)} 个部件")
        for result in results:
            print(f"    - {result['part_name']}: "
                  f"keypoint=({result['keypoint'][0]:.1f}, {result['keypoint'][1]:.1f}), "
                  f"score={result['score']:.3f}")
        print("=" * 60)
        
        return results
    
    @staticmethod
    def extract_single_keypoint(mask_np: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        从mask提取单个关键点（质心+最近点投影方法）
        
        策略：
            1. 计算mask的质心
            2. 检查质心是否在mask上
            3. 如果质心在mask上，直接返回
            4. 如果质心在空洞中（如handle），找mask上距离质心最近的点
        
        Args:
            mask_np: 二值mask, shape=(H, W), dtype=uint8，值为0或255
        
        Returns:
            (x, y): 关键点坐标（浮点数），相对于mask
                   如果提取失败返回None
        
        Examples:
            >>> mask = np.zeros((100, 100), dtype=np.uint8)
            >>> mask[30:70, 30:70] = 255  # 正方形
            >>> keypoint = SAM3Segmenter.extract_single_keypoint(mask)
            >>> print(keypoint)  # 应该接近 (50.0, 50.0)
        """
        # 1. 提取轮廓
        contours, _ = cv2.findContours(
            mask_np, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return None
        
        # 选择最大的轮廓（面积最大）
        contour = max(contours, key=cv2.contourArea)
        
        # 2. 计算质心
        M = cv2.moments(contour)
        
        if M["m00"] == 0:
            # 面积为0，无法计算质心
            return None
        
        cx = M["m10"] / M["m00"]  # 质心x坐标
        cy = M["m01"] / M["m00"]  # 质心y坐标
        
        # 3. 检查质心是否在mask上
        # 需要先检查坐标是否在图像范围内
        cx_int = int(round(cx))
        cy_int = int(round(cy))
        
        if (0 <= cy_int < mask_np.shape[0] and 
            0 <= cx_int < mask_np.shape[1] and
            mask_np[cy_int, cx_int] > 0):
            # 质心在mask上，直接返回
            return (cx, cy)
        
        # 4. 质心在空洞中，找mask上距离质心最近的点
        # 获取所有mask上的点
        mask_points = np.column_stack(np.where(mask_np > 0))  # shape=(N, 2), [row, col]
        
        if len(mask_points) == 0:
            return None
        
        # 计算每个点到质心的距离
        distances = np.sqrt(
            (mask_points[:, 1] - cx) ** 2 +  # mask_points[:, 1] 是 col (x)
            (mask_points[:, 0] - cy) ** 2     # mask_points[:, 0] 是 row (y)
        )
        
        # 找到最近的点
        closest_idx = np.argmin(distances)
        closest_point = mask_points[closest_idx]
        
        # 返回 (x, y) 格式：(col, row)
        return (float(closest_point[1]), float(closest_point[0]))
    
    @staticmethod
    def transform_to_original_coords(
        keypoints: List[Tuple[float, float]], 
        crop_bbox: List[int]
    ) -> List[Tuple[float, float]]:
        """
        将裁剪图坐标转换为原始图像坐标
        
        转换公式：
            x_orig = x_crop + crop_bbox[0]  # crop_bbox[0] 是 x1
            y_orig = y_crop + crop_bbox[1]  # crop_bbox[1] 是 y1
        
        Args:
            keypoints: 相对于裁剪图的关键点列表 [(x1, y1), (x2, y2), ...]
            crop_bbox: 裁剪框在原图的像素坐标 [x1, y1, x2, y2]
        
        Returns:
            List[Tuple[float, float]]: 原图坐标系的关键点
        
        Examples:
            >>> crop_keypoints = [(10.5, 20.3), (30.0, 40.0)]
            >>> crop_bbox = [100, 200, 300, 400]
            >>> orig_keypoints = SAM3Segmenter.transform_to_original_coords(
            ...     crop_keypoints, crop_bbox
            ... )
            >>> print(orig_keypoints)
            [(110.5, 220.3), (130.0, 240.0)]
        """
        # 提取偏移量
        x_offset = crop_bbox[0]  # x1
        y_offset = crop_bbox[1]  # y1
        
        # 转换每个关键点
        transformed_keypoints = [
            (x + x_offset, y + y_offset)
            for x, y in keypoints
        ]
        
        return transformed_keypoints
    
    def __repr__(self) -> str:
        """返回分割器的字符串表示"""
        return (
            f"SAM3Segmenter(\n"
            f"  device={self.device},\n"
            f"  checkpoint={self.checkpoint_path}\n"
            f")"
        )


# ==================== 模块测试代码 ====================
if __name__ == "__main__":
    """
    测试SAM3Segmenter初始化
    运行方式: python sam3segmenter.py
    """
    try:
        print("开始测试 SAM3Segmenter 初始化...\n")
        
        # 测试初始化
        segmenter = SAM3Segmenter(device="cuda")
        
        # 打印分割器信息
        print("\n分割器信息:")
        print(segmenter)
        
        print("\n✅ 初始化测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()