"""
VLM constraint decision module.
Created by Yinghao He on 2026-03-20

Loads Qwen3.5-9B locally and decides constraint weights for the current task.
The VLM only does semantic reasoning -- all geometric computation stays in the SAP layer.

VLM decides:
    w_grasp_axis  -- gripper Y-axis alignment with SAP grasp_axis (C3, pick only)
    w_safety      -- safety distance penalty weight (C4, pick + place)

Fixed constraints (always active, not decided by VLM):
    C1 approach_pos -- end-effector position aligned to contact point
    C2 approach_rot -- gripper Z-axis aligned to approach direction

Falls back to rule-based defaults when the model is unavailable or returns invalid JSON.
"""

import json
import base64
import logging
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.SAP import get_sap_descriptions

logger = logging.getLogger(__name__)


# ── VLMDecision dataclass ─────────────────────────────────────────────────────

@dataclass
class VLMDecision:
    """Constraint weight decision returned by VLMDecider.decide().

    Passed directly to ConstraintInstantiator.instantiate().
    """
    w_grasp_axis: float   # C3 weight, gripper Y-axis alignment (pick only, >= 0)
    w_safety:     float   # C4 weight, safety distance penalty (>= 0)
    confidence:   float   # VLM confidence [0, 1]; below threshold triggers fallback
    reasoning:    str     # one-sentence explanation from VLM (debug only)
    is_fallback:  bool    # True if this came from rule-based fallback

    def __post_init__(self):
        for name, val in [("w_grasp_axis", self.w_grasp_axis),
                           ("w_safety",     self.w_safety)]:
            if val < 0:
                raise ValueError(f"VLMDecision.{name} must be >= 0, got {val}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"VLMDecision.confidence must be in [0,1], got {self.confidence}"
            )

    def __repr__(self) -> str:
        src = "fallback" if self.is_fallback else "VLM"
        return (f"VLMDecision(src={src}, w_grasp={self.w_grasp_axis:.2f}, "
                f"w_safety={self.w_safety:.2f}, conf={self.confidence:.2f})")


# confidence below this threshold forces fallback
CONFIDENCE_THRESHOLD = 0.3


# ── annotated image ───────────────────────────────────────────────────────────

# fixed colours per part name for consistent visualisation
_PART_COLORS: Dict[str, str] = {
    "handle":  "#FF4444",
    "rim":     "#4444FF",
    "body":    "#44AA44",
    "neck":    "#FF8800",
    "cap":     "#AA44AA",
    "surface": "#00AAAA",
}
_DEFAULT_COLOR = "#FFFF00"


def build_annotated_image(
    rgb_image:    Image.Image,
    keypoints_2d: Dict[str, Tuple[float, float]],
) -> Image.Image:
    """Overlay coloured keypoint dots and part labels onto the RGB image."""
    annotated = rgb_image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
    except Exception:
        font = ImageFont.load_default()

    for part_name, (x, y) in keypoints_2d.items():
        color  = _PART_COLORS.get(part_name, _DEFAULT_COLOR)
        radius = 8
        # white outline for contrast
        draw.ellipse([x-radius-2, y-radius-2, x+radius+2, y+radius+2], fill="white")
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
        draw.text((x+radius+4, y-8), part_name, fill=color, font=font)

    return annotated


def _image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── prompt builder ────────────────────────────────────────────────────────────

def _build_vlm_prompt(
    task_instruction: str,
    detected_parts:   List[str],
    mode:             str,
) -> str:
    """Build the text prompt sent to the VLM alongside the annotated image."""
    all_descriptions = get_sap_descriptions()
    parts_desc_block = ""
    for part in detected_parts:
        if part in all_descriptions:
            parts_desc_block += f"  [{part}]: {all_descriptions[part]}\n"

    pick_constraints = """
  - w_grasp_axis (float, 0.0~3.0):
      Weight for gripper Y-axis alignment with the SAP grasp axis.
      Higher value enforces stricter finger orientation relative to the part geometry.
      Set higher when grasping a structured part like a handle or neck.
  - w_safety (float, 0.0~3.0):
      Weight for safety distance penalty from avoid-mode parts (e.g. rim).
      Higher value keeps the gripper further from fragile regions."""

    place_constraints = """
  - w_safety (float, 0.0~3.0):
      Weight for safety distance penalty from avoid-mode parts.
      w_grasp_axis is fixed at 0.0 for place mode (not needed)."""

    constraints_block = pick_constraints if mode == "pick" else place_constraints

    return f"""You are a robot manipulation constraint advisor.
Your role is to decide constraint weights for the current task based on the scene image and task instruction.
You must NOT perform any geometric reasoning. Only make semantic decisions.

=== TASK INSTRUCTION ===
{task_instruction}

=== CURRENT MODE ===
{mode.upper()} mode

=== DETECTED PARTS IN SCENE ===
{parts_desc_block}
=== CONSTRAINT WEIGHTS TO DECIDE ===
{constraints_block}
=== OUTPUT FORMAT ===
Respond with a single JSON object only. No explanation outside the JSON.
{{
  "w_grasp_axis": <float>,
  "w_safety": <float>,
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining your decision>"
}}
"""


# ── VLMDecider ────────────────────────────────────────────────────────────────

class VLMDecider:
    """Loads Qwen3.5-9B locally and decides constraint weights from scene + instruction.

    Usage:
        decider = VLMDecider(model_path="models/Qwen3.5-9B")
        decision = decider.decide(
            rgb_image=pil_image,
            keypoints_2d={"handle": (320, 240), "rim": (400, 100)},
            task_instruction="pick up the cup",
            mode="pick",
        )
    """

    def __init__(
        self,
        model_path:           Optional[str] = "models/Qwen3.5-9B",
        load_in_4bit:         bool  = True,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        """
        Args:
            model_path:           path to Qwen3.5-9B weights (local directory)
            load_in_4bit:         4-bit quantisation -- recommended for 12 GB VRAM
            confidence_threshold: fallback if VLM confidence is below this
        """
        self.confidence_threshold  = confidence_threshold
        self.model                 = None
        self.processor             = None
        self.last_inference_meta   = {}   # populated after each _infer_local() call

        if model_path is None:
            print("VLMDecider: no model_path -- rule-based fallback only")
            return

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            kwargs = {
                "device_map":        {"": 0},   # force all layers onto GPU 0
                "low_cpu_mem_usage": True,
            }
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                kwargs["torch_dtype"] = torch.bfloat16

            t0 = time.time()
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(model_path)
            print(f"VLMDecider: Qwen3.5-9B loaded  "
                  f"path={model_path}  4bit={load_in_4bit}  "
                  f"time={time.time()-t0:.1f}s")

        except Exception as e:
            logger.warning(f"VLMDecider: model load failed ({e}) -- fallback only")
            self.model = None

    # ── public interface ──────────────────────────────────────────────────────

    def decide(
        self,
        rgb_image:        Image.Image,
        keypoints_2d:     Dict[str, Tuple[float, float]],
        task_instruction: str,
        mode:             str,
    ) -> VLMDecision:
        """Run VLM inference and return constraint weights.

        Falls back to rule-based defaults if model unavailable or output invalid.
        """
        assert mode in ("pick", "place"), \
            f"mode must be 'pick' or 'place', got '{mode}'"

        if self.model is None:
            return self._rule_based_fallback(mode)

        annotated      = build_annotated_image(rgb_image, keypoints_2d)
        detected_parts = list(keypoints_2d.keys())
        prompt         = _build_vlm_prompt(task_instruction, detected_parts, mode)

        try:
            decision = self._infer_local(annotated, prompt, mode)
        except Exception as e:
            logger.warning(f"VLM inference failed: {e} -- using fallback")
            return self._rule_based_fallback(mode)

        if decision.confidence < self.confidence_threshold:
            logger.warning(f"VLM confidence {decision.confidence:.2f} < "
                           f"{self.confidence_threshold:.2f} -- using fallback")
            return self._rule_based_fallback(mode)

        return decision

    def generate_annotated_image(
        self,
        rgb_image:    Image.Image,
        keypoints_2d: Dict[str, Tuple[float, float]],
    ) -> Image.Image:
        return build_annotated_image(rgb_image, keypoints_2d)

    # ── private: local inference ──────────────────────────────────────────────

    def _infer_local(
        self,
        annotated_image: Image.Image,
        prompt:          str,
        mode:            str,
    ) -> VLMDecision:
        """Run Qwen3.5-9B inference locally and parse the JSON output.

        Qwen3.5 uses early fusion -- image tokens are embedded directly into
        input_ids. The correct call pattern is:
            1. apply_chat_template(..., tokenize=False) -> text string with placeholders
            2. processor(text=text, images=[pil]) -> fused input_ids
            3. model.generate(**inputs) -- no separate pixel_values needed
        """
        import torch

        # build messages with image placeholder (PIL passed separately to processor)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # step 1: get text string with image token placeholders
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        # step 2: processor fuses text + PIL image into input_ids
        inputs = self.processor(
            text=text,
            images=[annotated_image],
            return_tensors="pt",
        ).to(self.model.device)

        # step 3: generate -- image is already in input_ids, nothing extra needed
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # decode only newly generated tokens
        new_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_text = self.processor.decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        self.last_inference_meta = {
            "input_tokens":  int(inputs["input_ids"].shape[1]),
            "output_tokens": int(new_ids.shape[0]),
            "raw_response":  raw_text,
        }

        return self._parse_vlm_response(raw_text, mode)

    # ── private: response parsing ─────────────────────────────────────────────

    def _parse_vlm_response(self, raw_text: str, mode: str) -> VLMDecision:
        """Parse raw VLM text into a VLMDecision.

        Tries direct JSON parse first, then extracts the first {...} block.
        Raises ValueError if neither works (triggers fallback upstream).
        """
        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            pass

        if data is None:
            start = raw_text.find("{")
            end   = raw_text.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(raw_text[start:end+1])
                except json.JSONDecodeError:
                    pass

        if data is None:
            raise ValueError(f"cannot extract JSON from VLM output: {raw_text[:200]}")

        try:
            w_grasp = float(data.get("w_grasp_axis", 1.0))
            w_safe  = float(data.get("w_safety",     2.0))
            conf    = float(data.get("confidence",   0.5))
            reason  = str(data.get("reasoning",      ""))
        except (TypeError, ValueError) as e:
            raise ValueError(f"VLM JSON field type error: {e}")

        if mode == "place":
            w_grasp = 0.0   # grasp axis not used in place mode

        return VLMDecision(
            w_grasp_axis = float(np.clip(w_grasp, 0.0, 3.0)),
            w_safety     = float(np.clip(w_safe,  0.0, 3.0)),
            confidence   = float(np.clip(conf,    0.0, 1.0)),
            reasoning    = reason,
            is_fallback  = False,
        )

    # ── private: fallback ─────────────────────────────────────────────────────

    def _rule_based_fallback(self, mode: str) -> VLMDecision:
        """Conservative default weights derived from SAP contact modes."""
        if mode == "place":
            return VLMDecision(
                w_grasp_axis=0.0, w_safety=2.0, confidence=0.0,
                reasoning="Rule-based fallback (place mode): no grasp axis needed.",
                is_fallback=True,
            )
        return VLMDecision(
            w_grasp_axis=1.0, w_safety=2.0, confidence=0.0,
            reasoning="Rule-based fallback (pick mode): conservative SAP defaults.",
            is_fallback=True,
        )


# ── module test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing VLMDecider (fallback mode, no model load)")

    # VLMDecision validation
    d = VLMDecision(w_grasp_axis=1.0, w_safety=2.0,
                    confidence=0.85, reasoning="test", is_fallback=False)
    assert d.w_grasp_axis == 1.0 and not d.is_fallback
    assert "w_flip" not in VLMDecision.__dataclass_fields__
    print(f"  VLMDecision OK: {d}")

    # invalid weight
    try:
        VLMDecision(w_grasp_axis=-0.1, w_safety=1.0,
                    confidence=0.5, reasoning="", is_fallback=False)
        print("  ERROR: should have raised")
    except ValueError:
        print("  negative weight raises ValueError OK")

    # fallback-only mode
    decider = VLMDecider(model_path=None)
    decision = decider.decide(
        rgb_image=Image.new("RGB", (640, 480)),
        keypoints_2d={"handle": (100.0, 200.0), "rim": (300.0, 100.0)},
        task_instruction="pick up the cup",
        mode="pick",
    )
    assert decision.is_fallback
    print(f"  fallback decision OK: {decision}")

    print("All tests passed.")