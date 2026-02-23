"""
Created by Yinghao Ho on 2026-1-16
"""

import os
from pathlib import Path


class GroundingDINOConfig:
    """
    Grounding DINO检测器配置类 (Hugging Face版本)
    包含模型ID、检测阈值、设备配置等
    
    主要变化：
        - 使用 Hugging Face Hub 管理模型，无需手动下载权重
        - 自动处理模型缓存
        - 支持离线模式（通过LOCAL_MODEL_PATH）
    """
    
    # ==================== Hugging Face 模型配置 ====================
    # 使用 Hugging Face 上的官方模型
    # 可选模型：
    #   - "IDEA-Research/grounding-dino-base" (推荐，对应SwinB)
    #   - "IDEA-Research/grounding-dino-tiny" (更快但精度稍低)
    MODEL_ID = "IDEA-Research/grounding-dino-base"
    
    # 模型缓存目录（None表示使用HF默认缓存：~/.cache/huggingface）
    # 如果需要自定义缓存位置，设置为路径字符串，例如："/data/models"
    CACHE_DIR = None
    
    # 本地模型路径（用于离线模式）
    # 如果已经下载模型到本地，设置此路径可以离线使用
    # 例如："/path/to/downloaded/grounding-dino-base"
    # 设置为None时使用在线模型
    LOCAL_MODEL_PATH = None
    
    # ==================== 检测参数配置 ====================
    # 边界框置信度阈值 [0.0, 1.0]
    BOX_THRESHOLD = 0.50
    # 文本匹配置信度阈值 [0.0, 1.0]
    TEXT_THRESHOLD = 0.35
    # NMS（非极大值抑制）IoU阈值 [0.0, 1.0]
    NMS_THRESHOLD = 0.8

    # 计算设备："cuda"（GPU）或 "cpu"
    DEVICE = "cuda"
    
    # 图像输入格式："RGB" 或 "BGR"
    # 注意：Grounding DINO期望RGB格式，如果用OpenCV读取需要转换
    IMAGE_FORMAT = "RGB"
    
    # 边界框坐标格式："xyxy" (x1,y1,x2,y2) 或 "xywh" (x,y,w,h)
    BBOX_FORMAT = "xyxy"
    
    # ==================== 验证方法 ====================
    @classmethod
    def validate_environment(cls):
        """
        验证运行环境（替代原来的validate_paths）
        
        检查项：
            1. transformers 库是否安装
            2. torch 库是否安装
            3. CUDA 是否可用（如果设备设为cuda）
            4. 本地模型路径是否有效（如果指定了LOCAL_MODEL_PATH）
        
        Returns:
            bool: 环境验证通过返回True
            
        Raises:
            ImportError: 如果必要的库未安装
            FileNotFoundError: 如果本地模型路径无效
            RuntimeError: 如果环境配置有问题
        """
        print("🔍 正在验证运行环境...")
        
        # 1. 检查 transformers 库
        try:
            import transformers
            print(f"  ✓ transformers 版本: {transformers.__version__}")
            
            # 检查版本是否足够新
            version_parts = transformers.__version__.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            if major < 4 or (major == 4 and minor < 35):
                print(f"  ⚠️  警告: transformers 版本较旧 ({transformers.__version__})")
                print(f"     推荐版本: >=4.35.0")
                print(f"     升级命令: pip install --upgrade transformers")
        except ImportError:
            raise ImportError(
                "❌ 未安装 transformers 库\n"
                "📥 请安装: pip install transformers\n"
                "   推荐版本: transformers>=4.35.0"
            )
        
        # 2. 检查 torch
        try:
            import torch
            print(f"  ✓ torch 版本: {torch.__version__}")
        except ImportError:
            raise ImportError(
                "❌ 未安装 torch 库\n"
                "📥 请安装: pip install torch torchvision"
            )
        
        # 3. 检查 CUDA（如果配置为cuda）
        if cls.DEVICE == "cuda":
            import torch
            if not torch.cuda.is_available():
                print("  ⚠️  警告: DEVICE设置为'cuda'但CUDA不可用")
                print("     将在运行时自动切换到CPU模式")
                print("     如需使用GPU，请检查CUDA安装")
            else:
                cuda_device = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                print(f"  ✓ CUDA 可用: {cuda_device}")
                print(f"  ✓ CUDA 版本: {cuda_version}")
        
        # 4. 检查模型配置
        if cls.LOCAL_MODEL_PATH is not None:
            # 离线模式：检查本地路径
            local_path = Path(cls.LOCAL_MODEL_PATH)
            if not local_path.exists():
                raise FileNotFoundError(
                    f"❌ 本地模型路径不存在: {cls.LOCAL_MODEL_PATH}\n"
                    f"   请确保路径正确，或设置 LOCAL_MODEL_PATH = None 使用在线模型"
                )
            print(f"  ✓ 本地模型路径: {cls.LOCAL_MODEL_PATH}")
            print(f"    (离线模式)")
        else:
            # 在线模式：提示首次下载
            print(f"  ✓ 将使用 Hugging Face 模型: {cls.MODEL_ID}")
            if cls.CACHE_DIR:
                print(f"    缓存目录: {cls.CACHE_DIR}")
            else:
                print(f"    缓存目录: ~/.cache/huggingface (默认)")
            print(f"    💡 首次运行会自动下载模型（约1.5GB），请确保网络连接")
        
        print("✅ 环境验证通过\n")
        return True
    
    @classmethod
    def get_model_path(cls):
        """
        获取模型路径（本地路径或HF模型ID）
        
        Returns:
            str: 本地路径（离线模式）或 Hugging Face 模型ID（在线模式）
        
        Example:
            >>> path = GroundingDINOConfig.get_model_path()
            >>> # 返回 "IDEA-Research/grounding-dino-base" 或本地路径
        """
        if cls.LOCAL_MODEL_PATH is not None:
            return cls.LOCAL_MODEL_PATH
        else:
            return cls.MODEL_ID
    
    @classmethod
    def get_config_dict(cls):
        """
        获取配置信息的字典表示（用于日志记录或调试）
        
        Returns:
            dict: 包含所有配置项的字典
        """
        return {
            "model_id": cls.MODEL_ID,
            "local_model_path": cls.LOCAL_MODEL_PATH,
            "cache_dir": cls.CACHE_DIR,
            "box_threshold": cls.BOX_THRESHOLD,
            "text_threshold": cls.TEXT_THRESHOLD,
            "nms_threshold": cls.NMS_THRESHOLD,
            "device": cls.DEVICE,
            "image_format": cls.IMAGE_FORMAT,
            "bbox_format": cls.BBOX_FORMAT
        }
    
    @classmethod
    def print_config(cls):
        """打印当前配置（便于调试）"""
        print("=" * 60)
        print("Grounding DINO 配置信息 (Hugging Face版本)")
        print("=" * 60)
        config = cls.get_config_dict()
        for key, value in config.items():
            print(f"{key:20s}: {value}")
        print("=" * 60)


if __name__ == "__main__":
    """
    测试配置文件是否正确
    运行方式: python configs/groundingdino_cfg.py
    """
    try:
        # 打印配置信息
        GroundingDINOConfig.print_config()
        
        # 验证环境
        GroundingDINOConfig.validate_environment()
        
        print("\n✅ 配置文件测试通过！")
        
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        print(f"\n❌ 配置验证失败:\n{e}")
    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")
        