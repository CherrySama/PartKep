"""
Image processor for cropping detected object regions.
Created by Yinghao Ho on 2026-1-24
"""

from pathlib import Path
from typing import List, Dict, Union, Optional
from datetime import datetime

import numpy as np
from PIL import Image


class ImageProcessor:

    def __init__(self, output_dir: str = "images/objectlist"):
        # output_dir is only used when save_image=True
        self.output_dir = Path(output_dir)

    def crop_object(
        self,
        image:     Union[np.ndarray, Image.Image],
        bbox:      List[float],
        label:     str,
        score:     float = 0.0,
        object_id: Optional[int] = None,
        padding:   int = 0,
        save_image: bool = False,
    ) -> Dict:
        """Crop a single detection region from image.

        Args:
            image:      original image (PIL or numpy)
            bbox:       [x1, y1, x2, y2] in absolute pixel coordinates (float)
            label:      object class label
            score:      detection confidence
            object_id:  used in the saved filename; auto timestamp if None
            padding:    extra pixels added around the bbox
            save_image: write crop to disk (debug only)

        Returns dict with keys:
            label, bbox_pixel, score, cropped_image, crop_size, save_path
        """
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image_pil = image
        else:
            raise ValueError(f"unsupported image type: {type(image)}")

        img_w, img_h = image_pil.size

        x1 = max(0, int(bbox[0]) - padding)
        y1 = max(0, int(bbox[1]) - padding)
        x2 = min(img_w, int(bbox[2]) + padding)
        y2 = min(img_h, int(bbox[3]) + padding)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"invalid crop region [{x1},{y1},{x2},{y2}] for image {img_w}x{img_h}"
            )

        cropped_image = image_pil.crop((x1, y1, x2, y2))

        save_path = None
        if save_image:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if object_id is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{label}_{ts}.jpg"
            else:
                filename = f"{label}_{object_id}.jpg"
            save_path = self.output_dir / filename
            cropped_image.save(save_path, quality=95)

        return {
            "label":         label,
            "bbox_pixel":    [x1, y1, x2, y2],
            "score":         score,
            "cropped_image": cropped_image,
            "crop_size":     (x2 - x1, y2 - y1),
            "save_path":     str(save_path) if save_path else None,
        }

    def crop_objects_batch(
        self,
        image:             Union[np.ndarray, Image.Image],
        detection_results: List[Dict],
        padding:           int = 0,
        save_image:        bool = False,
    ) -> List[Dict]:
        """Crop all detections in batch. Returns list of crop dicts."""
        cropped_results = []
        for idx, det in enumerate(detection_results):
            try:
                crop = self.crop_object(
                    image=image,
                    bbox=det["bbox"],
                    label=det["label"],
                    score=det.get("score", 0.0),
                    object_id=idx,
                    padding=padding,
                    save_image=save_image,
                )
                cropped_results.append(crop)
            except Exception as e:
                print(f"  crop failed [{idx}] {det.get('label','?')}: {e}")
        return cropped_results

    def get_crop_info(self, crop_result: Dict) -> str:
        save_info = crop_result["save_path"] or "not saved"
        return (
            f"label:    {crop_result['label']}\n"
            f"score:    {crop_result['score']:.3f}\n"
            f"bbox:     {crop_result['bbox_pixel']}\n"
            f"size:     {crop_result['crop_size'][0]}x{crop_result['crop_size'][1]}\n"
            f"saved to: {save_info}"
        )

    def clear_output_dir(self):
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)