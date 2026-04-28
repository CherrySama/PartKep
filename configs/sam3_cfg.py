"""
SAM3 segmenter configuration.
Created by Yinghao Ho on 2026-4-10
"""

from pathlib import Path


class SAM3Config:

    MODEL_ID         = "facebook/sam3"
    CACHE_DIR        = None
    LOCAL_MODEL_PATH = "models/sam3"

    DEVICE = "cuda"

    @classmethod
    def get_model_path(cls) -> str:
        """Return local model path if it exists, else fall back to HF model ID."""
        if cls.LOCAL_MODEL_PATH and Path(cls.LOCAL_MODEL_PATH).exists():
            return cls.LOCAL_MODEL_PATH
        return cls.MODEL_ID

    @classmethod
    def validate_environment(cls):
        """Check that required libraries are installed and CUDA is available if configured."""
        try:
            import transformers
            major = int(transformers.__version__.split(".")[0])
            if major < 5:
                print(f"WARNING: transformers {transformers.__version__} may be too old "
                      f"(Sam3Model requires >=5.0.0)")
        except ImportError:
            raise ImportError("transformers not installed: pip install transformers>=5.0.0")

        try:
            import torch
        except ImportError:
            raise ImportError("torch not installed: pip install torch torchvision")

        if cls.DEVICE == "cuda":
            import torch
            if not torch.cuda.is_available():
                print("WARNING: DEVICE=cuda but CUDA not available; will fall back to CPU")

        if cls.LOCAL_MODEL_PATH and not Path(cls.LOCAL_MODEL_PATH).exists():
            print(f"WARNING: LOCAL_MODEL_PATH '{cls.LOCAL_MODEL_PATH}' not found; "
                  f"will download from HF")

        return True