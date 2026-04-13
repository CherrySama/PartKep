"""
Created by Yinghao Ho on 2026-1-31
"""

from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image
from configs.sam3_cfg import SAM3Config
from transformers import Sam3Model, Sam3Processor
from configs.part_config import PartConfig
from pathlib import Path


class SAM3Segmenter:
    """
    SAM3部件分割器 (Hugging Face Transformers版本)

    使用SAM3模型分割物体的各个部件，并为每个部件提取一个关键点。
    关键点提取使用质心+最近点投影方法，确保关键点落在mask表面上。

    工作流程：
        1. 根据物体 label 从 PartConfig 获取 PartEntry 列表
        2. 预计算图像的 vision embeddings（一次）
        3. 复用 embeddings，用每条 entry["prompt"] 逐个分割部件
        4. 为每个部件提取单个关键点
        5. 将关键点坐标转换到原始图像坐标系
        6. 结果中的 part_name 使用 entry["part_name"]，与 SAP 知识库对应
    """

    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda"):

        print("=" * 60)
        print("初始化 SAM3 分割器 (Hugging Face版本)")
        print("=" * 60)

        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  警告: CUDA不可用，自动切换到CPU模式")
            device = "cpu"

        self.device = device
        print(f"✓ 运行设备: {self.device}")

        if checkpoint_path is None:
            model_path = SAM3Config.get_model_path()
            print(f"✓ 使用模型: {model_path}")
        else:
            model_path = checkpoint_path
            print(f"✓ 使用指定模型: {model_path}")

        print("\n正在加载 SAM3 模型...")
        try:
            self.model = Sam3Model.from_pretrained(model_path)
            self.processor = Sam3Processor.from_pretrained(model_path)
            self.model = self.model.to(device)
            self.model.eval()
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
            label: 物体类别（如 "cup", "bottle"）
            crop_bbox: 裁剪框在原图的像素坐标 [x1, y1, x2, y2]

        Returns:
            List[Dict]: 每个部件的分割结果
            [
                {
                    'part_name': 'handle',        # SAP 知识库 key
                    'prompt':    'cup handle',     # 实际使用的 SAM3 prompt
                    'keypoint':  (x, y),           # 原图坐标（浮点数）
                    'score':     0.95,             # SAM3 置信度
                    'mask':      np.ndarray        # 二值 mask (H, W)
                },
                ...
            ]
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

        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        print(f"  裁剪图像尺寸: {image_pil.size[0]}x{image_pil.size[1]}")

        # 2. 从 PartConfig 获取部件条目列表（包含 part_name 和 prompt）
        #    旧版：parts = PartConfig.get_parts(label)  → List[str]
        #    新版：entries = PartConfig.get_parts(label) → List[PartEntry]
        entries = PartConfig.get_parts(label)

        if len(entries) == 0:
            print(f"⚠️  警告: 物体 '{label}' 没有预定义的部件配置")
            return []

        # 打印 part_name 和 prompt 的对应关系，便于调试
        print(f"  部件配置:")
        for e in entries:
            print(f"    {e['part_name']:8s} → prompt: '{e['prompt']}'")

        # 3. 预计算图像的 vision embeddings（一次性，所有部件复用）
        print("\n📸 预计算图像特征（一次性）...")
        try:
            img_inputs = self.processor(
                images=image_pil,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                vision_embeds = self.model.get_vision_features(
                    pixel_values=img_inputs.pixel_values
                )
            print("✓ 图像特征提取完成")

        except Exception as e:
            raise RuntimeError(f"图像特征提取失败: {e}")

        # 4. 循环处理每个部件（复用 vision embeddings）
        results = []

        for idx, entry in enumerate(entries):
            # 关键变化：
            #   part_name → SAP 知识库 key，存入结果
            #   prompt    → 传给 SAM3 的实际文本
            part_name = entry["part_name"]
            prompt    = entry["prompt"]

            print(f"\n[{idx+1}/{len(entries)}] 处理部件: "
                  f"{part_name} (prompt='{prompt}')")

            try:
                # 4.1 用 prompt（而非 part_name）准备文本输入
                print(f"  → 调用SAM3分割...")
                text_inputs = self.processor(
                    text=prompt,          # ← 使用 prompt 字段
                    return_tensors="pt"
                ).to(self.device)

                # 4.2 复用 vision embeddings
                with torch.no_grad():
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        **text_inputs
                    )

                # 4.3 后处理
                results_raw = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=img_inputs.get("original_sizes").tolist()
                )

                if len(results_raw) == 0 or len(results_raw[0]['masks']) == 0:
                    print(f"  ⚠️  未检测到部件 '{part_name}'（prompt='{prompt}'），跳过")
                    continue

                result = results_raw[0]
                mask_tensor = result['masks'][0]
                score = float(result['scores'][0])

                print(f"  ✓ 检测成功，置信度: {score:.3f}")

                # 4.4 转换 mask 为 numpy uint8
                mask_np = mask_tensor.cpu().numpy()
                if mask_np.dtype == bool:
                    mask_np = mask_np.astype(np.uint8) * 255
                elif mask_np.dtype in [np.int64, np.int32]:
                    mask_np = (mask_np * 255).astype(np.uint8)
                elif mask_np.dtype in [np.float32, np.float64]:
                    mask_np = (mask_np * 255).astype(np.uint8)

                print(f"  ✓ Mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")

                # 4.5 提取关键点（相对于裁剪图）
                print(f"  → 提取关键点...")
                keypoint_crop = self.extract_single_keypoint(mask_np)

                if keypoint_crop is None:
                    print(f"  ⚠️  关键点提取失败，跳过")
                    continue

                print(f"  ✓ 关键点（裁剪图坐标）: "
                      f"({keypoint_crop[0]:.2f}, {keypoint_crop[1]:.2f})")

                # 4.6 转换到原图坐标
                keypoint_orig = self.transform_to_original_coords(
                    [keypoint_crop],
                    crop_bbox
                )[0]

                print(f"  ✓ 关键点（原图坐标）: "
                      f"({keypoint_orig[0]:.2f}, {keypoint_orig[1]:.2f})")

                # 4.7 保存结果
                #   part_name：SAP 知识库 key（供后续约束生成使用）
                #   prompt：记录实际使用的分割文本（供调试）
                results.append({
                    'part_name': part_name,   # SAP key，如 "handle"
                    'prompt':    prompt,       # SAM3文本，如 "cup handle"
                    'keypoint':  keypoint_orig,
                    'score':     score,
                    'mask':      mask_np
                })

                print(f"  ✅ 部件 '{part_name}' 处理完成")

            except Exception as e:
                print(f"  ❌ 处理部件 '{part_name}' 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 5. 输出总结
        print("\n" + "=" * 60)
        print(f"✅ 部件分割完成！")
        print(f"  物体: {label}")
        print(f"  成功分割: {len(results)}/{len(entries)} 个部件")
        for r in results:
            print(f"    - {r['part_name']:8s} ('{r['prompt']}'): "
                  f"keypoint=({r['keypoint'][0]:.1f}, {r['keypoint'][1]:.1f}), "
                  f"score={r['score']:.3f}")
        print("=" * 60)

        return results

    @staticmethod
    def extract_single_keypoint(mask_np: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        从 mask 提取单个关键点（质心 + 最近点投影方法）

        策略：
            1. 计算 mask 的质心
            2. 若质心在 mask 上，直接返回
            3. 若质心在空洞中（如 handle），找 mask 上距离质心最近的点

        Args:
            mask_np: 二值 mask, shape=(H, W), dtype=uint8，值为 0 或 255

        Returns:
            (x, y): 关键点坐标（浮点数），相对于 mask
                    提取失败返回 None
        """
        if mask_np.dtype != np.uint8:
            if mask_np.dtype == bool:
                mask_np = mask_np.astype(np.uint8) * 255
            elif mask_np.dtype in [np.int64, np.int32]:
                mask_np = (mask_np * 255).astype(np.uint8)
            elif mask_np.dtype in [np.float32, np.float64]:
                mask_np = (mask_np * 255).astype(np.uint8)

        # 1. 计算质心
        M = cv2.moments(mask_np)
        if M["m00"] == 0:
            return None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # 2. 检查质心是否在 mask 上
        cx_int = int(round(cx))
        cy_int = int(round(cy))
        h, w = mask_np.shape

        cx_int = max(0, min(cx_int, w - 1))
        cy_int = max(0, min(cy_int, h - 1))

        if mask_np[cy_int, cx_int] > 0:
            return (cx, cy)

        # 3. 质心在空洞中，找 mask 上最近的点
        ys, xs = np.where(mask_np > 0)
        if len(xs) == 0:
            return None

        distances = (xs - cx) ** 2 + (ys - cy) ** 2
        nearest_idx = np.argmin(distances)

        return (float(xs[nearest_idx]), float(ys[nearest_idx]))

    def transform_to_original_coords(self,
                                     keypoints: List[Tuple[float, float]],
                                     crop_bbox: List[int]
                                     ) -> List[Tuple[float, float]]:
        """
        将裁剪图坐标转换到原图坐标

        Args:
            keypoints: 裁剪图坐标列表 [(x, y), ...]
            crop_bbox: 裁剪框 [x1, y1, x2, y2]（原图像素坐标）

        Returns:
            原图坐标列表 [(x, y), ...]
        """
        x1, y1 = crop_bbox[0], crop_bbox[1]
        return [(kp[0] + x1, kp[1] + y1) for kp in keypoints]
        