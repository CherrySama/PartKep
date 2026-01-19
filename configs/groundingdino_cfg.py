"""
Created by Yinghao Ho on 2026-1-16
"""

import os
from pathlib import Path


class GroundingDINOConfig:
    """
    Grounding DINO检测器配置类
    包含模型文件路径、检测阈值、设备配置等
    """

    PROJECT_ROOT = Path(__file__).parent.parent
    MODEL_DIR = PROJECT_ROOT / "models" / "GroundingDINO"
    MODEL_CONFIG_PATH = str(MODEL_DIR / "GroundingDINO_SwinB_cfg.py")
    MODEL_CHECKPOINT_PATH = str(MODEL_DIR / "groundingdino_swinb_cogcoor.pth")
    
    # 边界框置信度阈值 [0.0, 1.0]
    BOX_THRESHOLD = 0.35
    # 文本匹配置信度阈值 [0.0, 1.0]
    TEXT_THRESHOLD = 0.25
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
    def validate_paths(cls):
        """
        验证必要的文件路径是否存在
        
        Returns:
            bool: 所有文件存在返回True
            
        Raises:
            FileNotFoundError: 如果关键文件不存在
        """
        # 检查模型配置文件
        if not os.path.exists(cls.MODEL_CONFIG_PATH):
            raise FileNotFoundError(
                f"❌ 模型配置文件不存在: {cls.MODEL_CONFIG_PATH}\n"
                f"📥 请从以下链接下载:\n"
                f"   https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py\n"
                f"📁 保存到: {cls.MODEL_DIR}"
            )
        
        # 检查模型权重文件
        if not os.path.exists(cls.MODEL_CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"❌ 模型权重文件不存在: {cls.MODEL_CHECKPOINT_PATH}\n"
                f"📥 请从以下链接下载 (约1.5GB):\n"
                f"   方法1: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth\n"
                f"   方法2: https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth\n"
                f"📁 保存到: {cls.MODEL_DIR}"
            )
        
        print(f"✅ 模型文件验证通过")
        print(f"   配置文件: {cls.MODEL_CONFIG_PATH}")
        print(f"   权重文件: {cls.MODEL_CHECKPOINT_PATH}")
        return True
    
    @classmethod
    def get_config_dict(cls):
        """
        获取配置信息的字典表示（用于日志记录或调试）
        
        Returns:
            dict: 包含所有配置项的字典
        """
        return {
            "model_config": cls.MODEL_CONFIG_PATH,
            "model_checkpoint": cls.MODEL_CHECKPOINT_PATH,
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
        print("Grounding DINO 配置信息")
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
        
        # 验证文件路径
        GroundingDINOConfig.validate_paths()
        
        print("\n✅ 配置文件测试通过！")
        
    except FileNotFoundError as e:
        print(f"\n❌ 配置验证失败:\n{e}")
    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")
        