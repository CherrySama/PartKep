"""
Download HF models to local directory.

Target directory: /workspace/PartKep/models/
Usage: python download_models.py
"""

from transformers import (
    Sam3Model,
    Sam3Processor,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor
)

SAM3_PATH           = "models/sam3"
GROUNDING_DINO_PATH = "models/grounding-dino-base"

# ── SAM3 (~3.5 GB) ────────────────────────────────────────────────────────────
print("[1/2] downloading SAM3  (facebook/sam3) -> {SAM3_PATH}")
try:
    model     = Sam3Model.from_pretrained("facebook/sam3")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.save_pretrained(SAM3_PATH)
    processor.save_pretrained(SAM3_PATH)
    print(f"[1/2] SAM3 saved to {SAM3_PATH}")
except Exception as e:
    print(f"[1/2] SAM3 failed: {e}")
    exit(1)

# ── GroundingDINO (~1.5 GB) ───────────────────────────────────────────────────
print(f"[2/2] downloading GroundingDINO  (IDEA-Research/grounding-dino-base) -> {GROUNDING_DINO_PATH}")
try:
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    )
    processor = AutoProcessor.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    )
    model.save_pretrained(GROUNDING_DINO_PATH)
    processor.save_pretrained(GROUNDING_DINO_PATH)
    print(f"[2/2] GroundingDINO saved to {GROUNDING_DINO_PATH}")
except Exception as e:
    print(f"[2/2] GroundingDINO failed: {e}")
    exit(1)