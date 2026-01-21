"""
Created by Yinghao Ho on 2026-1-19
"""

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch
from configs.groundingdino_cfg import GroundingDINOConfig  
from PIL import Image
from groundingdino.util.inference import predict
import groundingdino.datasets.transforms as T
from torchvision.ops import nms


class GroundingDINODetector:
    """
    Grounding DINO物体检测器
    
    功能：
        - 基于文本prompt检测图像中的物体
        - 支持开放词汇检测（不限于预定义类别）
        - 支持空间关系推理（如"the leftmost cup"）
        - 每个类别只返回置信度最高的一个检测结果
    
    输入：
        - RGB图像 (H, W, 3)
        - 文本prompt（如"cup", "cup . bottle"）
    
    输出：
        - 检测结果列表，包含bbox（归一化坐标）、label、score
    
    使用示例：
        >>> detector = GroundingDINODetector()
        >>> results = detector.detect(image, "the leftmost cup")
        >>> print(results)
        [{'bbox': [0.1, 0.2, 0.5, 0.8], 'label': 'cup', 'score': 0.95}]
    """
    
    def __init__(self,
                 config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda"):
        """
        初始化Grounding DINO检测器
        
        Args:
            config_path (str, optional): 模型配置文件路径。
                如果为None，则从GroundingDINOConfig读取默认路径。
            checkpoint_path (str, optional): 模型权重文件路径。
                如果为None，则从GroundingDINOConfig读取默认路径。
            device (str): 运行设备，可选"cuda"或"cpu"。
                默认为"cuda"。
        
        Raises:
            FileNotFoundError: 如果模型文件不存在
            RuntimeError: 如果模型加载失败
            ValueError: 如果device参数无效
        
        注意：
            - 首次初始化会加载约1.5GB的模型权重，可能需要几秒时间
            - 如果使用GPU，需要确保CUDA可用
        """
        print("=" * 60)
        print("初始化 Grounding DINO 检测器")
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
        
        # ==================== 2. 读取配置路径 ====================
        # 如果用户没有指定路径，使用配置文件的默认路径
        if config_path is None:
            config_path = GroundingDINOConfig.MODEL_CONFIG_PATH
            print(f"✓ 使用默认配置路径: {config_path}")
        else:
            print(f"✓ 使用指定配置路径: {config_path}")
        
        if checkpoint_path is None:
            checkpoint_path = GroundingDINOConfig.MODEL_CHECKPOINT_PATH
            print(f"✓ 使用默认权重路径: {checkpoint_path}")
        else:
            print(f"✓ 使用指定权重路径: {checkpoint_path}")
        
        # 保存路径供后续使用
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # ==================== 3. 验证文件存在 ====================
        print("\n正在验证模型文件...")
        try:
            GroundingDINOConfig.validate_paths()
        except FileNotFoundError as e:
            print(f"\n❌ 模型文件验证失败:")
            raise e
        
        # ==================== 4. 加载检测参数 ====================
        # 从配置文件读取默认阈值
        self.box_threshold = GroundingDINOConfig.BOX_THRESHOLD
        self.text_threshold = GroundingDINOConfig.TEXT_THRESHOLD
        self.nms_threshold = GroundingDINOConfig.NMS_THRESHOLD
        
        print(f"\n✓ 检测参数:")
        print(f"  - BOX_THRESHOLD: {self.box_threshold}")
        print(f"  - TEXT_THRESHOLD: {self.text_threshold}")
        print(f"  - NMS_THRESHOLD: {self.nms_threshold}")
        
        # ==================== 5. 加载Grounding DINO模型 ====================
        print("\n正在加载 Grounding DINO 模型...")
        print("（首次加载可能需要10-30秒，请耐心等待）")
        
        try:
            # 导入Grounding DINO的模型加载函数
            from groundingdino.util.inference import load_model
            
            # 加载模型
            self.model = load_model(
                model_config_path=self.config_path,
                model_checkpoint_path=self.checkpoint_path,
                device=self.device
            )
            
            print("✅ 模型加载成功！")
            
        except ImportError as e:
            raise ImportError(
                "无法导入 groundingdino 库。请确保已正确安装:\n"
                "  git clone https://github.com/IDEA-Research/GroundingDINO.git\n"
                "  cd GroundingDINO\n"
                "  pip install -e .\n"
                f"错误详情: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"模型加载失败，请检查模型文件是否完整。\n"
                f"错误详情: {e}"
            )
        
        # ==================== 6. 设置模型为评估模式 ====================
        self.model.eval()  # 关闭dropout等训练相关层
        
        print("\n" + "=" * 60)
        print("✅ Grounding DINO 检测器初始化完成！")
        print("=" * 60)
        print()
    
    def detect(self,
               image: np.ndarray,
               text_prompt: str,
               box_threshold: Optional[float] = None,
               text_threshold: Optional[float] = None) -> List[Dict]:
        """
        检测图像中的物体
        
        Args:
            image: RGB图像，shape=(H, W, 3)，dtype=uint8
                   或者PIL.Image对象
            text_prompt: 文本提示，如 "cup" 或 "cup . bottle"
                        注意：多个物体用空格和点号分隔，如 "cup . bottle"
            box_threshold: 边界框阈值（可选，默认使用配置文件中的值）
            text_threshold: 文本阈值（可选，默认使用配置文件中的值）
        
        Returns:
            List[Dict]: 检测结果列表，每个类别只保留置信度最高的一个
            [
                {
                    'bbox': [x1, y1, x2, y2],  # 归一化坐标 [0, 1]
                    'label': 'cup',
                    'score': 0.95
                },
                ...
            ]
            
        Raises:
            ValueError: 如果图像格式不正确
            RuntimeError: 如果检测过程出错
        """
        # ==================== 第1步：输入验证 ====================
        
        # 1.1 验证text_prompt
        if not text_prompt or not isinstance(text_prompt, str):
            raise ValueError("text_prompt必须是非空字符串")
        
        # 1.2 设置阈值（如果未指定，使用配置文件的默认值）
        if box_threshold is None:
            box_threshold = self.box_threshold
        if text_threshold is None:
            text_threshold = self.text_threshold
        
        # 验证阈值范围
        if not 0.0 <= box_threshold <= 1.0:
            raise ValueError(f"box_threshold必须在[0, 1]范围内，当前值：{box_threshold}")
        if not 0.0 <= text_threshold <= 1.0:
            raise ValueError(f"text_threshold必须在[0, 1]范围内，当前值：{text_threshold}")
        
        # 1.3 验证图像格式
        # 导入PIL用于图像处理
        # 检查输入类型并转换为PIL Image
        if isinstance(image, np.ndarray):
            # numpy array输入（来自OpenCV或相机）
            
            # 检查维度
            if image.ndim != 3:
                raise ValueError(
                    f"图像必须是3维数组 (H, W, 3)，当前维度：{image.ndim}"
                )
            
            # 检查通道数
            if image.shape[2] != 3:
                raise ValueError(
                    f"图像必须是3通道RGB格式，当前通道数：{image.shape[2]}"
                )
            
            # 检查数据类型
            if image.dtype != np.uint8:
                # 尝试转换
                if image.dtype in [np.float32, np.float64]:
                    # 如果是浮点数且在[0,1]范围，转换为[0,255]
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # 转换为PIL Image（假设已经是RGB格式）
            image_pil = Image.fromarray(image)
            
        elif isinstance(image, Image.Image):
            # 已经是PIL Image
            image_pil = image
            
            # 确保是RGB模式
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
        
        else:
            raise ValueError(
                f"image必须是numpy.ndarray或PIL.Image.Image类型，"
                f"当前类型：{type(image)}"
            )
        
        # 保存原始图像尺寸（用于后续可能的可视化）
        image_width, image_height = image_pil.size
        
        print(f"📷 输入图像尺寸: {image_width} x {image_height}")
        print(f"🎯 文本提示: '{text_prompt}'")
        print(f"⚙️  检测阈值: box={box_threshold:.2f}, text={text_threshold:.2f}")
        
        # ==================== 第2步：图像预处理 ====================
        print("🔄 正在预处理图像...")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),  # 调整图像大小
            T.ToTensor(),                           # 转为tensor
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 归一化
        ])
        
        # 应用变换，得到模型输入
        image_transformed, _ = transform(image_pil, None)
        
        print(f"✓ 图像预处理完成，tensor shape: {image_transformed.shape}")
        # ==================== 第3步：模型推理 ====================
        print("🚀 正在运行模型推理...")
        
        try:
            # 调用Grounding DINO的predict函数
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_transformed,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )
            
            print(f"✓ 模型推理完成")
            print(f"  检测到 {len(boxes)} 个候选框")
            
        except Exception as e:
            raise RuntimeError(f"模型推理失败: {e}")
        
        # 检查是否有检测结果
        if len(boxes) == 0:
            print("⚠️  未检测到任何物体")
            return []
        
        # ==================== 第4步：后处理 ====================
        print("🔧 正在后处理检测结果...")
        
        # 4.1 转换为numpy数组（方便处理）
        boxes_np = boxes.cpu().numpy()  # shape: (N, 4), 格式: [cx, cy, w, h], 归一化
        scores_np = logits.cpu().numpy()  # shape: (N,)
        labels_list = phrases  # List[str]
        
        # 4.2 坐标格式转换: [cx, cy, w, h] -> [x1, y1, x2, y2]
        # 保持归一化坐标 [0, 1]
        boxes_xyxy = np.zeros_like(boxes_np)
        boxes_xyxy[:, 0] = boxes_np[:, 0] - boxes_np[:, 2] / 2  # x1 = cx - w/2
        boxes_xyxy[:, 1] = boxes_np[:, 1] - boxes_np[:, 3] / 2  # y1 = cy - h/2
        boxes_xyxy[:, 2] = boxes_np[:, 0] + boxes_np[:, 2] / 2  # x2 = cx + w/2
        boxes_xyxy[:, 3] = boxes_np[:, 1] + boxes_np[:, 3] / 2  # y2 = cy + h/2
        
        # 确保坐标在[0, 1]范围内
        boxes_xyxy = np.clip(boxes_xyxy, 0.0, 1.0)
        
        # 转换为绝对坐标用于 NMS
        boxes_abs = boxes_xyxy * np.array([image_width, image_height, 
                                        image_width, image_height])
        boxes_tensor = torch.from_numpy(boxes_abs).float()
        scores_tensor = torch.from_numpy(scores_np).float()
        keep_indices = nms(boxes_tensor, scores_tensor, self.nms_threshold)
        keep_indices = keep_indices.numpy()
        
        # 过滤结果
        boxes_xyxy = boxes_xyxy[keep_indices]
        scores_np = scores_np[keep_indices]
        phrases = [phrases[i] for i in keep_indices]
        
        print(f"  NMS 前: {len(boxes_np)} 个框, NMS 后: {len(boxes_xyxy)} 个框")
        
        # 4.3 按类别分组，每类只保留置信度最高的一个
        results_dict = {}  # {label: (bbox, score)}
        
        for i in range(len(labels_list)):
            label = labels_list[i]
            score = float(scores_np[i])
            bbox = boxes_xyxy[i].tolist()  # [x1, y1, x2, y2]
            
            # 如果这个类别还没有记录，或者当前分数更高，则更新
            if label not in results_dict or score > results_dict[label]['score']:
                results_dict[label] = {
                    'bbox': bbox,
                    'label': label,
                    'score': score
                }
        
        # 4.4 转换为列表并按score降序排列
        results = list(results_dict.values())
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"✓ 后处理完成")
        print(f"  原始检测数: {len(boxes_np)}")
        print(f"  去重后结果数: {len(results)}")
        for result in results:
            print(f"    - {result['label']}: score={result['score']:.3f}, "
                  f"bbox=[{result['bbox'][0]:.3f}, {result['bbox'][1]:.3f}, "
                  f"{result['bbox'][2]:.3f}, {result['bbox'][3]:.3f}]")
        
        # ==================== 第5步：返回结果 ====================
        return results
    
    def __repr__(self) -> str:
        """返回检测器的字符串表示"""
        return (
            f"GroundingDINODetector(\n"
            f"  device={self.device},\n"
            f"  box_threshold={self.box_threshold},\n"
            f"  text_threshold={self.text_threshold}\n"
            f")"
        )


# ==================== 模块测试代码 ====================
if __name__ == "__main__":
    """
    测试检测器初始化
    运行方式: python modules/detection/grounding_dino_detector.py
    """
    try:
        print("开始测试 GroundingDINODetector 初始化...\n")
        
        # 测试初始化
        detector = GroundingDINODetector(device="cuda")
        
        # 打印检测器信息
        print("\n检测器信息:")
        print(detector)
        
        print("\n✅ 初始化测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        