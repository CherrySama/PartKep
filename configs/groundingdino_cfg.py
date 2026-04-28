"""
GroundingDINO detector configuration.
Created by Yinghao Ho on 2026-1-16
"""

import os
from pathlib import Path


class GroundingDINOConfig:

    MODEL_ID         = "IDEA-Research/grounding-dino-base"
    CACHE_DIR        = None
    LOCAL_MODEL_PATH = "models/grounding-dino-base"

    BOX_THRESHOLD  = 0.50
    TEXT_THRESHOLD = 0.35
    NMS_THRESHOLD  = 0.8

    DEVICE       = "cuda"
    IMAGE_FORMAT = "RGB"
    BBOX_FORMAT  = "xyxy"

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
            major, minor = map(int, transformers.__version__.split(".")[:2])
            if major < 4 or (major == 4 and minor < 35):
                print(f"WARNING: transformers {transformers.__version__} may be too old (need >=4.35)")
        except ImportError:
            raise ImportError("transformers not installed: pip install transformers>=4.35.0")

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