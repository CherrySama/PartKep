"""
Created by Yinghao Ho on 2026-4-10
"""

from pathlib import Path


class SAM3Config:
    """
    SAM3 分割器配置类 (Hugging Face 版本)
    包含模型路径、设备配置等

    与 GroundingDINOConfig 保持一致的管理风格：
        - 优先使用本地模型（LOCAL_MODEL_PATH）
        - 本地不存在时回退到 HF 在线模型（MODEL_ID）
        - 通过 get_model_path() 统一获取路径
    """

    # ==================== Hugging Face 模型配置 ====================
    MODEL_ID = "facebook/sam3"

    # 模型缓存目录（None 表示使用 HF 默认缓存：~/.cache/huggingface）
    CACHE_DIR = None

    # 本地模型路径（运行 loadmodels.py 后模型保存在此）
    # 设置为 None 时使用在线模型
    LOCAL_MODEL_PATH = "models/sam3"

    # ==================== 设备配置 ====================
    # 计算设备："cuda"（GPU）或 "cpu"
    DEVICE = "cuda"

    # ==================== 验证方法 ====================
    @classmethod
    def validate_environment(cls):
        """
        验证运行环境

        检查项：
            1. transformers 库是否安装且版本 >= 5.0.0（Sam3Model 要求）
            2. torch 库是否安装
            3. CUDA 是否可用（如果设备设为 cuda）
            4. 本地模型路径是否有效（如果指定了 LOCAL_MODEL_PATH）

        Returns:
            bool: 环境验证通过返回 True

        Raises:
            ImportError: 如果必要的库未安装
            FileNotFoundError: 如果本地模型路径无效
            RuntimeError: 如果环境配置有问题
        """
        print("🔍 正在验证 SAM3 运行环境...")

        # 1. 检查 transformers 库
        try:
            import transformers
            print(f"  ✓ transformers 版本: {transformers.__version__}")

            version_parts = transformers.__version__.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            if major < 5:
                print(f"  ⚠️  警告: transformers 版本较旧 ({transformers.__version__})")
                print(f"     Sam3Model 要求版本: >=5.0.0")
                print(f"     升级命令: pip install --upgrade transformers")
        except ImportError:
            raise ImportError(
                "❌ 未安装 transformers 库\n"
                "📥 请安装: pip install transformers>=5.0.0"
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

        # 3. 检查 CUDA（如果配置为 cuda）
        if cls.DEVICE == "cuda":
            import torch
            if not torch.cuda.is_available():
                print("  ⚠️  警告: DEVICE 设置为 'cuda' 但 CUDA 不可用")
                print("     将在运行时自动切换到 CPU 模式")
            else:
                cuda_device = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                print(f"  ✓ CUDA 可用: {cuda_device}")
                print(f"  ✓ CUDA 版本: {cuda_version}")

        # 4. 检查本地模型路径
        if cls.LOCAL_MODEL_PATH is not None:
            local_path = Path(cls.LOCAL_MODEL_PATH)
            if not local_path.exists():
                print(f"  ⚠️  本地模型路径不存在: {cls.LOCAL_MODEL_PATH}")
                print(f"     将回退到 HF 在线模型: {cls.MODEL_ID}")
                print(f"     如需离线使用，请先运行: python loadmodels.py")
            else:
                print(f"  ✓ 本地模型路径: {cls.LOCAL_MODEL_PATH}（离线模式）")
        else:
            print(f"  ✓ 将使用 HuggingFace 模型: {cls.MODEL_ID}")
            if cls.CACHE_DIR:
                print(f"    缓存目录: {cls.CACHE_DIR}")
            else:
                print(f"    缓存目录: ~/.cache/huggingface（默认）")
            print(f"    💡 首次运行会自动下载模型（约 3.5GB），请确保网络连接")

        print("✅ SAM3 环境验证通过\n")
        return True

    @classmethod
    def get_model_path(cls) -> str:
        """
        获取模型路径（本地路径优先，否则返回 HF 模型 ID）

        Returns:
            str: 本地路径（离线模式）或 HuggingFace 模型 ID（在线模式）
        """
        if cls.LOCAL_MODEL_PATH is not None:
            local_path = Path(cls.LOCAL_MODEL_PATH)
            if local_path.exists():
                return cls.LOCAL_MODEL_PATH
        return cls.MODEL_ID

    @classmethod
    def get_config_dict(cls) -> dict:
        """
        获取配置信息的字典表示（用于日志记录或调试）
        """
        return {
            "model_id":        cls.MODEL_ID,
            "local_model_path": cls.LOCAL_MODEL_PATH,
            "cache_dir":       cls.CACHE_DIR,
            "device":          cls.DEVICE,
        }

    @classmethod
    def print_config(cls):
        """打印当前配置（便于调试）"""
        print("=" * 60)
        print("SAM3 配置信息 (Hugging Face 版本)")
        print("=" * 60)
        for key, value in cls.get_config_dict().items():
            print(f"{key:20s}: {value}")
        print("=" * 60)


if __name__ == "__main__":
    try:
        SAM3Config.print_config()
        SAM3Config.validate_environment()
        print("✅ 配置文件测试通过！")
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        print(f"\n❌ 配置验证失败:\n{e}")
    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")
        