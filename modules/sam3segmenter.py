"""
SAM3 part segmenter (HuggingFace Transformers version)
Created by Yinghao Ho on 2026-1-31
"""

from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.sam3_cfg import SAM3Config
from configs.part_config import PartConfig
from transformers import Sam3Model, Sam3Processor


class SAM3Segmenter:

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        model_path = checkpoint_path if checkpoint_path else SAM3Config.get_model_path()

        self.model = Sam3Model.from_pretrained(model_path).to(device)
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.model.eval()

        print(f"SAM3 loaded  model={model_path}  device={device}")

    def segment_parts(
        self,
        cropped_image: Union[np.ndarray, Image.Image],
        label:         str,
        crop_bbox:     List[int],
    ) -> List[Dict]:
        """Segment all parts of an object and extract one keypoint per part.

        Args:
            cropped_image: cropped object region (PIL or numpy)
            label:         object class, e.g. 'cup' -- used to look up PartConfig
            crop_bbox:     [x1,y1,x2,y2] of the crop in original image coordinates

        Returns list of dicts:
            part_name  -- SAP knowledge base key (e.g. 'handle')
            prompt     -- actual SAM3 text used (e.g. 'cup handle')
            keypoint   -- (x, y) in original image coordinates
            score      -- SAM3 confidence
            mask       -- binary mask as uint8 ndarray (H, W)
        """
        if isinstance(cropped_image, np.ndarray):
            image_pil = Image.fromarray(cropped_image)
        elif isinstance(cropped_image, Image.Image):
            image_pil = cropped_image
        else:
            raise ValueError(f"unsupported image type: {type(cropped_image)}")

        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        entries = PartConfig.get_parts(label)
        if not entries:
            print(f"  WARNING: no part config for '{label}'")
            return []

        # compute vision embeddings once; reuse for all parts
        img_inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_embeds = self.model.get_vision_features(
                pixel_values=img_inputs.pixel_values
            )

        results = []
        for entry in entries:
            part_name = entry["part_name"]
            prompt    = entry["prompt"]
            try:
                text_inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(vision_embeds=vision_embeds, **text_inputs)

                results_raw = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=img_inputs.get("original_sizes").tolist(),
                )

                if not results_raw or not results_raw[0]["masks"].numel():
                    continue

                result      = results_raw[0]
                mask_tensor = result["masks"][0]
                score       = float(result["scores"][0])

                # convert mask to uint8
                mask_np = mask_tensor.cpu().numpy()
                if mask_np.dtype == bool:
                    mask_np = mask_np.astype(np.uint8) * 255
                elif mask_np.dtype in [np.int64, np.int32]:
                    mask_np = (mask_np * 255).astype(np.uint8)
                elif mask_np.dtype in [np.float32, np.float64]:
                    mask_np = (mask_np * 255).astype(np.uint8)

                keypoint_crop = self.extract_single_keypoint(mask_np)
                if keypoint_crop is None:
                    continue

                keypoint_orig = self.transform_to_original_coords(
                    [keypoint_crop], crop_bbox
                )[0]

                results.append({
                    "part_name": part_name,
                    "prompt":    prompt,
                    "keypoint":  keypoint_orig,
                    "score":     score,
                    "mask":      mask_np,
                })

            except Exception as e:
                print(f"  segment failed '{part_name}': {e}")
                continue

        print(f"  segmented {len(results)}/{len(entries)} parts for '{label}'")
        return results

    @staticmethod
    def extract_single_keypoint(mask_np: np.ndarray) -> Optional[Tuple[float, float]]:
        """Extract one keypoint from a binary mask.

        Strategy: use the centroid if it lies on the mask, otherwise snap to
        the nearest mask pixel (handles concave parts like a handle loop).
        """
        if mask_np.dtype != np.uint8:
            if mask_np.dtype == bool:
                mask_np = mask_np.astype(np.uint8) * 255
            elif mask_np.dtype in [np.int64, np.int32]:
                mask_np = (mask_np * 255).astype(np.uint8)
            elif mask_np.dtype in [np.float32, np.float64]:
                mask_np = (mask_np * 255).astype(np.uint8)

        M = cv2.moments(mask_np)
        if M["m00"] == 0:
            return None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        h, w = mask_np.shape
        cx_i = max(0, min(int(round(cx)), w - 1))
        cy_i = max(0, min(int(round(cy)), h - 1))

        if mask_np[cy_i, cx_i] > 0:
            return (cx, cy)

        # centroid is inside a hole -- find nearest mask pixel
        ys, xs = np.where(mask_np > 0)
        if len(xs) == 0:
            return None
        distances  = (xs - cx) ** 2 + (ys - cy) ** 2
        nearest    = np.argmin(distances)
        return (float(xs[nearest]), float(ys[nearest]))

    @staticmethod
    def transform_to_original_coords(
        keypoints: List[Tuple[float, float]],
        crop_bbox: List[int],
    ) -> List[Tuple[float, float]]:
        """Translate keypoints from crop coordinates to original image coordinates."""
        x1, y1 = crop_bbox[0], crop_bbox[1]
        return [(kp[0] + x1, kp[1] + y1) for kp in keypoints]