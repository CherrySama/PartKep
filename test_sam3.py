import os
# 可以移除这个，因为用本地路径后不再依赖 hub 缓存（可选）
# os.environ['HF_HUB_CACHE'] = '/workspace/PartKep/models/SAM3'

from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import time
import torch

# print("Loading SAM 3 model...")

# # 直接指定本地权重路径（替换成你 find 出来的路径）
# checkpoint_path = "/workspace/PartKep/models/SAM3/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"

# model = build_sam3_image_model(checkpoint_path=checkpoint_path)
# processor = Sam3Processor(model)

# # 加载图像
# image = Image.open("images/cup3.jpg")
# inference_state = processor.set_image(image)

# # 使用文本提示
# output = processor.set_text_prompt(
#     state=inference_state,
#     prompt="handle"
# )

# print(f"Found {len(output['boxes'])} objects")
# print(f"Scores: {output['scores']}")
print("Loading SAM 3 model...")
start = time.time()
model = build_sam3_image_model(checkpoint_path="/workspace/PartKep/models/SAM3/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt")  # 用本地更快
print(f"Model load time: {time.time() - start:.2f}s")

processor = Sam3Processor(model)

image = Image.open("images/cup3.jpg")
start = time.time()
inference_state = processor.set_image(image)
print(f"Set image time: {time.time() - start:.2f}s")

start = time.time()
output = processor.set_text_prompt(state=inference_state, prompt="handle")
print(f"Inference time: {time.time() - start:.2f}s")

print(f"Found {len(output['boxes'])} objects")
print(f"Scores: {output['scores']}")