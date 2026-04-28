"""
GroundingDINO object detector (HuggingFace version)
Created by Yinghao Ho on 2026-1-19
"""

from pathlib import Path
from typing import List, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.groundingdino_cfg import GroundingDINOConfig


class GroundingDINODetector:

    def __init__(self, device: str = "cuda", model_id: Optional[str] = None):
        if device not in ["cuda", "cpu"]:
            raise ValueError(f"device must be 'cuda' or 'cpu', got '{device}'")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        model_path = model_id if model_id else GroundingDINOConfig.get_model_path()
        self.model_id = model_path
        cache_dir = GroundingDINOConfig.CACHE_DIR

        self.box_threshold  = GroundingDINOConfig.BOX_THRESHOLD
        self.text_threshold = GroundingDINOConfig.TEXT_THRESHOLD
        self.nms_threshold  = GroundingDINOConfig.NMS_THRESHOLD

        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path, cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()

        print(f"GroundingDINO loaded  model={model_path}  device={self.device}")

    def detect(
        self,
        image: Union[np.ndarray, Image.Image],
        text_prompt: str,
        box_threshold:  Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Detect objects matching text_prompt in image.

        Returns a list of dicts sorted by score descending.
        Each dict: {'bbox': [x1,y1,x2,y2], 'label': str, 'score': float}
        """
        box_threshold  = box_threshold  if box_threshold  is not None else self.box_threshold
        text_threshold = text_threshold if text_threshold is not None else self.text_threshold

        # convert to PIL RGB
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image_pil = image
        else:
            raise ValueError(f"unsupported image type: {type(image)}")
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        image_width, image_height = image_pil.size

        # GroundingDINO needs the prompt to end with '.'
        text_prompt_formatted = text_prompt.strip()
        if not text_prompt_formatted.endswith("."):
            text_prompt_formatted += "."

        # preprocess
        inputs = self.processor(
            images=image_pil,
            text=text_prompt_formatted,
            return_tensors="pt",
        ).to(self.device)

        # inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # HF post-processing: returns absolute pixel coords directly
        results_hf = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(image_height, image_width)],
        )[0]

        if len(results_hf["boxes"]) == 0:
            return []

        boxes_abs  = results_hf["boxes"].cpu().numpy()   # (N, 4) [x1,y1,x2,y2]
        scores_np  = results_hf["scores"].cpu().numpy()  # (N,)
        labels_list = results_hf["labels"]               # List[str]

        # NMS to remove overlapping boxes
        keep = nms(
            torch.from_numpy(boxes_abs).float(),
            torch.from_numpy(scores_np).float(),
            self.nms_threshold,
        ).numpy()
        boxes_abs   = boxes_abs[keep]
        scores_np   = scores_np[keep]
        labels_list = [labels_list[i] for i in keep]

        # keep only the highest-scoring box per label
        results_dict = {}
        for i in range(len(labels_list)):
            label = labels_list[i]
            score = float(scores_np[i])
            bbox  = boxes_abs[i].tolist()
            if label not in results_dict or score > results_dict[label]["score"]:
                results_dict[label] = {"bbox": bbox, "label": label, "score": score}

        results = sorted(results_dict.values(), key=lambda x: x["score"], reverse=True)
        return results

    def __repr__(self) -> str:
        return (f"GroundingDINODetector(model={self.model_id}, device={self.device}, "
                f"box_thr={self.box_threshold}, text_thr={self.text_threshold})")