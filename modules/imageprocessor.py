"""
Created by Yinghao Ho on 2026-1-24
"""

from pathlib import Path
from typing import List, Dict, Union, Optional
from datetime import datetime

import numpy as np
from PIL import Image


class ImageProcessor:
    """
    图像处理器

    核心功能：根据 bbox + padding 从原图裁剪物体区域。

    设计原则：
        - 裁剪是纯内存操作，不强制写盘
        - save_image=True 时才做磁盘 I/O（调试用）
        - 坐标系统：直接接收绝对像素坐标（来自 GroundingDINO）

    使用场景：
        GroundingDINO 检测后裁剪物体，避免 SAM3 在全图中混淆部件。
    """

    def __init__(self, output_dir: str = "images/objectlist"):
        """
        Args:
            output_dir: 调试保存目录（仅 save_image=True 时使用）
        """
        self.output_dir = Path(output_dir)
        # 不在 __init__ 里创建目录，等到真正需要保存时再创建
        print(f"✓ ImageProcessor 初始化完成")
        print(f"  调试输出目录: {self.output_dir}（save_image=True 时生效）")

    def crop_object(self,
                    image: Union[np.ndarray, Image.Image],
                    bbox: List[float],
                    label: str,
                    score: float = 0.0,
                    object_id: Optional[int] = None,
                    padding: int = 0,
                    save_image: bool = False) -> Dict:
        """
        裁剪单个检测到的物体

        Args:
            image:       原始图像（PIL Image 或 numpy array）
            bbox:        绝对像素坐标 [x1, y1, x2, y2]（浮点数）
            label:       物体类别（如 "cup"）
            score:       检测置信度
            object_id:   物体ID（文件命名用，None 时自动生成时间戳）
            padding:     裁剪时的边距（像素），默认 0
            save_image:  是否将裁剪图写入磁盘，默认 False
                         推理时保持 False；调试可视化时设为 True

        Returns:
            Dict:
            {
                'label':         'cup',
                'bbox_pixel':    [x1, y1, x2, y2],  # 整数，含 padding，用于坐标转换
                'score':         0.95,
                'cropped_image': PIL.Image,
                'crop_size':     (width, height),
                'save_path':     str | None          # 仅 save_image=True 时非 None
            }
        """
        # 1. 统一转为 PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image_pil = image
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")

        img_w, img_h = image_pil.size

        # 2. 解析 bbox，应用 padding，clamp 到图像边界
        x1_raw, y1_raw, x2_raw, y2_raw = float(bbox[0]), float(bbox[1]), \
                                          float(bbox[2]), float(bbox[3])

        x1 = max(0, int(x1_raw) - padding)
        y1 = max(0, int(y1_raw) - padding)
        x2 = min(img_w, int(x2_raw) + padding)
        y2 = min(img_h, int(y2_raw) + padding)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"无效的裁剪区域: bbox_pixel=[{x1},{y1},{x2},{y2}]，"
                f"图像尺寸: {img_w}x{img_h}"
            )

        # 3. 裁剪（纯内存操作）
        cropped_image = image_pil.crop((x1, y1, x2, y2))
        crop_width  = x2 - x1
        crop_height = y2 - y1

        # 4. 可选：写盘（调试用）
        save_path = None
        if save_image:
            # 懒创建目录
            self.output_dir.mkdir(parents=True, exist_ok=True)

            if object_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{label}_{timestamp}.jpg"
            else:
                filename = f"{label}_{object_id}.jpg"

            save_path = self.output_dir / filename
            cropped_image.save(save_path, quality=95)
            print(f"  💾 调试图像已保存: {save_path}")

        return {
            'label':         label,
            'bbox_pixel':    [x1, y1, x2, y2],   # SAM3Segmenter 用这个做坐标转换
            'score':         score,
            'cropped_image': cropped_image,
            'crop_size':     (crop_width, crop_height),
            'save_path':     str(save_path) if save_path else None,
        }

    def crop_objects_batch(self,
                           image: Union[np.ndarray, Image.Image],
                           detection_results: List[Dict],
                           padding: int = 0,
                           save_image: bool = False) -> List[Dict]:
        """
        批量裁剪多个检测结果

        Args:
            image:             原始图像
            detection_results: GroundingDINO detect() 返回的结果列表
                               [{'bbox': [...], 'label': 'cup', 'score': 0.95}, ...]
            padding:           裁剪边距（像素），默认 0
            save_image:        是否写盘，默认 False，透传给 crop_object

        Returns:
            List[Dict]: 裁剪结果列表，格式同 crop_object 返回值
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
                    padding=padding,
                    save_image=save_image,     # ← 透传
                )
                cropped_results.append(crop_result)

                print(f"  [{idx+1}/{len(detection_results)}] "
                      f"{detection['label']}: "
                      f"裁剪尺寸 {crop_result['crop_size'][0]}x"
                      f"{crop_result['crop_size'][1]}")

            except Exception as e:
                print(f"  ❌ 裁剪失败 [{idx+1}]: {e}")
                continue

        print(f"✓ 批量裁剪完成，成功 {len(cropped_results)}/{len(detection_results)}")
        return cropped_results

    def get_crop_info(self, crop_result: Dict) -> str:
        """
        获取裁剪结果的格式化信息（调试用）

        Args:
            crop_result: crop_object() 返回的结果

        Returns:
            str: 格式化信息
        """
        save_info = crop_result['save_path'] if crop_result['save_path'] else "未保存（save_image=False）"
        info = (
            f"类别:     {crop_result['label']}\n"
            f"置信度:   {crop_result['score']:.3f}\n"
            f"bbox_pixel: {crop_result['bbox_pixel']}\n"
            f"裁剪尺寸: {crop_result['crop_size'][0]}x{crop_result['crop_size'][1]}\n"
            f"保存路径: {save_info}"
        )
        return info

    def clear_output_dir(self):
        """清空调试输出目录（慎用）"""
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ 已清空输出目录: {self.output_dir}")
        else:
            print(f"⚠️  输出目录不存在: {self.output_dir}")


# ==================== 模块测试 ====================
if __name__ == "__main__":
    import numpy as np

    print("测试 ImageProcessor\n")

    # 创建一张测试图（640x480 随机图）
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    processor = ImageProcessor()

    # 测试1：不保存（默认）
    result = processor.crop_object(
        image=dummy_image,
        bbox=[100.5, 80.3, 300.7, 250.9],
        label="cup",
        score=0.92,
        padding=10,
        save_image=False
    )
    assert result['save_path'] is None
    assert result['bbox_pixel'] == [90, 70, 311, 261]
    assert 'bbox' not in result          # 已去掉冗余字段
    assert 'bbox_float' not in result    # 已去掉冗余字段
    print(f"✅ 不保存模式正常，bbox_pixel={result['bbox_pixel']}")
    print(processor.get_crop_info(result))

    # 测试2：保存（调试模式）
    result_saved = processor.crop_object(
        image=dummy_image,
        bbox=[100.5, 80.3, 300.7, 250.9],
        label="cup",
        score=0.92,
        object_id=0,
        padding=10,
        save_image=True
    )
    assert result_saved['save_path'] is not None
    assert Path(result_saved['save_path']).exists()
    print(f"\n✅ 保存模式正常，文件: {result_saved['save_path']}")

    # 测试3：批量裁剪，默认不保存
    detections = [
        {'bbox': [50.0, 60.0, 200.0, 300.0], 'label': 'cup',    'score': 0.95},
        {'bbox': [250.0, 100.0, 400.0, 400.0], 'label': 'bottle', 'score': 0.88},
    ]
    results = processor.crop_objects_batch(
        image=dummy_image,
        detection_results=detections,
        padding=5,
        save_image=False
    )
    assert len(results) == 2
    assert all(r['save_path'] is None for r in results)
    print(f"\n✅ 批量裁剪正常，共 {len(results)} 个结果，均未写盘")

    print("\n✅ 所有测试通过！")
    