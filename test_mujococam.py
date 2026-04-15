"""
测试 scene_cam 相机渲染图像
运行方式（从项目根目录）：
    python test_camera.py
输出：camera_view.png
"""

import mujoco
import numpy as np
from PIL import Image

SCENE_XML = "assets/franka_emika_panda/scene.xml"
WIDTH, HEIGHT = 640, 480

model = mujoco.MjModel.from_xml_path(SCENE_XML)
data  = mujoco.MjData(model)

# 重置到 home 关键帧（索引 0，对应 panda.xml 里的 <key name="home">）
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# 渲染
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
renderer.update_scene(data, camera="scene_cam")
pixels = renderer.render()   # shape: (H, W, 3), uint8

img = Image.fromarray(pixels)
img.save("camera_view.png")
print(f"已保存 camera_view.png  ({WIDTH}x{HEIGHT})")
renderer.close()