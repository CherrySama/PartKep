"""
Created by Yinghao Ho on 2026-1-19
"""

from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.groundingdino_cfg import GroundingDINOConfig


class GroundingDINODetector:
    """
    Grounding DINO物体检测器 (Hugging Face版本 - 简化坐标系统)
    
    坐标系统简化 (2026-2-2)：
        - HF版本直接返回绝对像素坐标
        - 去掉了不必要的归一化→反归一化往返转换
        - 流程更简洁：HF返回绝对坐标 → 直接传给ImageProcessor
    
    功能：
        - 基于文本prompt检测图像中的物体
        - 支持开放词汇检测（不限于预定义类别）
        - 支持空间关系推理（如"the leftmost cup"）
        - 每个类别只返回置信度最高的一个检测结果
    
    输入：
        - RGB图像 (H, W, 3) - PIL.Image 或 numpy array
        - 文本prompt（如"cup", "cup . bottle"）
    
    输出：
        - 检测结果列表，包含bbox（绝对像素坐标）、label、score
    
    使用示例：
        >>> detector = GroundingDINODetector()
        >>> results = detector.detect(image, "the leftmost cup")
        >>> print(results)
        [{'bbox': [150.0, 200.0, 450.0, 600.0], 'label': 'cup', 'score': 0.95}]
    
    后续工作流程：
        1. GroundingDINO检测 → 返回绝对像素坐标（浮点）
        2. ImageProcessor裁剪 → 使用绝对坐标，保存ROI图像和offset
        3. SAM3处理ROI → 提取关键点（ROI坐标系）
        4. 坐标转换 → 转换回原图坐标系（一次转换）
    """
    
    def __init__(self,
                 device: str = "cuda",
                 model_id: Optional[str] = None):
        """
        初始化Grounding DINO检测器（Hugging Face版本）
        
        Args:
            device (str): 运行设备，可选"cuda"或"cpu"，默认为"cuda"
            model_id (str, optional): Hugging Face模型ID
        """
        print("=" * 60)
        print("初始化 Grounding DINO 检测器 (HF版本 - 简化坐标系统)")
        print("=" * 60)
        
        # ==================== 1. 参数验证 ====================
        if device not in ["cuda", "cpu"]:
            raise ValueError(f"无效的设备类型: {device}，必须是 'cuda' 或 'cpu'")
        
        # 检查CUDA是否可用
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  警告: CUDA不可用，自动切换到CPU模式")
            device = "cpu"
        
        self.device = device
        print(f"✓ 运行设备: {self.device}")
        
        # ==================== 2. 确定模型路径 ====================
        if model_id is None:
            model_path = GroundingDINOConfig.get_model_path()
            print(f"✓ 使用默认模型: {model_path}")
        else:
            model_path = model_id
            print(f"✓ 使用指定模型: {model_path}")
        
        self.model_id = model_path
        
        # 获取缓存目录配置
        cache_dir = GroundingDINOConfig.CACHE_DIR
        if cache_dir:
            print(f"✓ 缓存目录: {cache_dir}")
        else:
            print(f"✓ 缓存目录: ~/.cache/huggingface (默认)")
        
        # ==================== 3. 加载检测参数 ====================
        self.box_threshold = GroundingDINOConfig.BOX_THRESHOLD
        self.text_threshold = GroundingDINOConfig.TEXT_THRESHOLD
        self.nms_threshold = GroundingDINOConfig.NMS_THRESHOLD
        
        print(f"\n✓ 检测参数:")
        print(f"  - BOX_THRESHOLD: {self.box_threshold}")
        print(f"  - TEXT_THRESHOLD: {self.text_threshold}")
        print(f"  - NMS_THRESHOLD: {self.nms_threshold}")
        
        # ==================== 4. 导入 Hugging Face 库 ====================
        print("\n正在导入 transformers 库...")
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            print("✓ transformers 导入成功")
        except ImportError as e:
            raise ImportError(
                "❌ 无法导入 transformers 库\n"
                "📥 请安装: pip install transformers>=4.35.0\n"
                f"错误详情: {e}"
            )
        
        # ==================== 5. 加载 Processor ====================
        print("\n正在加载 Processor...")
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                cache_dir=cache_dir
            )
            print("✓ Processor 加载成功")
        except Exception as e:
            raise RuntimeError(f"❌ Processor 加载失败: {e}")
        
        # ==================== 6. 加载 Model ====================
        print("\n正在加载模型...")
        
        try:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_path,
                cache_dir=cache_dir
            )
            
            # 移动到指定设备
            self.model = self.model.to(self.device)
            
            print(f"✓ 模型加载成功")
            print(f"  设备: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"❌ 模型加载失败: {e}")
        
        # ==================== 7. 设置模型为评估模式 ====================
        self.model.eval()
        
        print("\n" + "=" * 60)
        print("✅ Grounding DINO 检测器初始化完成！")
        print("=" * 60)
        print()
    
    def detect(self,
               image: Union[np.ndarray, Image.Image],
               text_prompt: str,
               box_threshold: Optional[float] = None,
               text_threshold: Optional[float] = None) -> List[Dict]:
        """
        检测图像中的物体
        
        Args:
            image: RGB图像，shape=(H, W, 3)，dtype=uint8 或 PIL.Image对象
            text_prompt: 文本提示，如 "cup" 或 "cup . bottle"
            box_threshold: 边界框阈值（可选，默认使用配置文件中的值）
            text_threshold: 文本阈值（可选，默认使用配置文件中的值）
        
        Returns:
            List[Dict]: 检测结果列表，每个类别只保留置信度最高的一个
            [
                {
                    'bbox': [x1, y1, x2, y2],  # 绝对像素坐标（浮点数）
                    'label': 'cup',
                    'score': 0.95
                },
                ...
            ]
            
        Raises:
            ValueError: 如果图像格式不正确
            RuntimeError: 如果检测过程出错
        
        注意：
            - 返回的bbox是绝对像素坐标（浮点数），直接可用于裁剪
            - HF版本直接返回像素坐标，无需归一化往返转换
            - ImageProcessor直接使用这些坐标进行裁剪和padding
        """
        # ==================== 第1步：输入验证 ====================
        
        # 1.1 验证text_prompt
        if not text_prompt or not isinstance(text_prompt, str):
            raise ValueError("text_prompt必须是非空字符串")
        
        # 1.2 设置阈值
        if box_threshold is None:
            box_threshold = self.box_threshold
        if text_threshold is None:
            text_threshold = self.text_threshold
        
        # 验证阈值范围
        if not 0.0 <= box_threshold <= 1.0:
            raise ValueError(f"box_threshold必须在[0, 1]范围内，当前值：{box_threshold}")
        if not 0.0 <= text_threshold <= 1.0:
            raise ValueError(f"text_threshold必须在[0, 1]范围内，当前值：{text_threshold}")
        
        # 1.3 验证图像格式并转换为PIL Image
        if isinstance(image, np.ndarray):
            # numpy array输入
            if image.ndim != 3:
                raise ValueError(f"图像必须是3维数组 (H, W, 3)，当前维度：{image.ndim}")
            if image.shape[2] != 3:
                raise ValueError(f"图像必须是3通道RGB格式，当前通道数：{image.shape[2]}")
            
            # 检查数据类型
            if image.dtype != np.uint8:
                if image.dtype in [np.float32, np.float64]:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            image_pil = Image.fromarray(image)
            
        elif isinstance(image, Image.Image):
            image_pil = image
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
        else:
            raise ValueError(
                f"image必须是numpy.ndarray或PIL.Image.Image类型，"
                f"当前类型：{type(image)}"
            )
        
        # 保存原始图像尺寸
        image_width, image_height = image_pil.size
        
        # ==================== 格式化 text_prompt ====================
        text_prompt_formatted = text_prompt.lower()
        if not text_prompt_formatted.endswith('.'):
            text_prompt_formatted = text_prompt_formatted + '.'
        
        print(f"📷 输入图像尺寸: {image_width} x {image_height}")
        print(f"🎯 文本提示: '{text_prompt}' → '{text_prompt_formatted}'")
        print(f"⚙️  检测阈值: box={box_threshold:.2f}, text={text_threshold:.2f}")
        
        # ==================== 第2步：Hugging Face 预处理 ====================
        print("🔄 正在预处理图像...")
        
        try:
            inputs = self.processor(
                images=image_pil,
                text=text_prompt_formatted,
                return_tensors="pt"
            ).to(self.device)
            
            print(f"✓ 图像预处理完成")
            
        except Exception as e:
            raise RuntimeError(f"图像预处理失败: {e}")
        
        # ==================== 第3步：模型推理 ====================
        print("🚀 正在运行模型推理...")
        
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            print(f"✓ 模型推理完成")
            
        except Exception as e:
            raise RuntimeError(f"模型推理失败: {e}")
        
        # ==================== 第4步：Hugging Face 后处理 ====================
        print("🔧 正在后处理检测结果...")
        
        try:
            results_hf = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(image_height, image_width)]
            )[0]
            
            print(f"✓ HF后处理完成")
            
        except Exception as e:
            raise RuntimeError(f"后处理失败: {e}")
        
        # ==================== 第5步：提取数据 ====================
        if len(results_hf['boxes']) == 0:
            print("⚠️  未检测到任何物体")
            return []
        
        # 提取数据 - HF直接返回绝对像素坐标
        boxes_abs = results_hf['boxes'].cpu().numpy()    # (N, 4) [x1,y1,x2,y2] 绝对像素坐标
        scores_np = results_hf['scores'].cpu().numpy()   # (N,)
        labels_list = results_hf['labels']               # List[str]
        
        print(f"  检测到 {len(boxes_abs)} 个候选框")
        
        # ==================== 第6步：NMS 去重 ====================
        boxes_tensor = torch.from_numpy(boxes_abs).float()
        scores_tensor = torch.from_numpy(scores_np).float()
        keep_indices = nms(boxes_tensor, scores_tensor, self.nms_threshold)
        keep_indices = keep_indices.numpy()
        
        # 过滤结果
        boxes_abs = boxes_abs[keep_indices]
        scores_np = scores_np[keep_indices]
        labels_list = [labels_list[i] for i in keep_indices]
        
        print(f"  NMS后保留: {len(boxes_abs)} 个框")
        
        # ==================== 第7步：每类去重 ====================
        results_dict = {}
        
        for i in range(len(labels_list)):
            label = labels_list[i]
            score = float(scores_np[i])
            bbox = boxes_abs[i].tolist() 
            
            if label not in results_dict or score > results_dict[label]['score']:
                results_dict[label] = {
                    'bbox': bbox,      
                    'label': label,
                    'score': score
                }
        
        # ==================== 第9步：返回结果 ====================
        results = list(results_dict.values())
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"✓ 后处理完成")
        print(f"  最终结果数: {len(results)}")
        for result in results:
            print(f"    - {result['label']}: score={result['score']:.3f}, "
                  f"bbox=[{result['bbox'][0]:.2f}, {result['bbox'][1]:.2f}, "
                  f"{result['bbox'][2]:.2f}, {result['bbox'][3]:.2f}]")
        
        return results
    
    def __repr__(self) -> str:
        """返回检测器的字符串表示"""
        return (
            f"GroundingDINODetector(\n"
            f"  model_id={self.model_id},\n"
            f"  device={self.device},\n"
            f"  box_threshold={self.box_threshold},\n"
            f"  text_threshold={self.text_threshold},\n"
            f"  nms_threshold={self.nms_threshold}\n"
            f")"
        )