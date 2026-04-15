"""
Created by Yinghao Ho on 2026-03-22
PartKep — sim 模式完整链路 + MuJoCo 可视化测试

流程：
    手动写死 sim_keypoints
    → Pipeline.run(sim 模式)
    → pick_q, place_q
    → MuJoCoEnv: home → pick → place → home
"""

import sys
import numpy as np
from PIL import Image

sys.path.insert(0, '.')

from pipeline import Pipeline
from modules.mujocoLoader import MuJoCoEnv
from configs.camera_config import CameraConfig

# ==================== 写死的仿真关键点 ====================
SIM_KEYPOINTS = {
    "cup": {
        "handle": np.array([0.45,  0.07,  0.10]),
        "rim":    np.array([0.45,  0.00,  0.16]),
        "body":   np.array([0.45,  0.00,  0.08]),
    },
    "tray": {
        "surface": np.array([0.65,  0.10,  0.02]),
        "rim":     np.array([0.65,  0.10,  0.05]),
    },
}

SCENE_XML = "../mujoco_menagerie/franka_emika_panda/scene.xml"

def main():
    print("=" * 55)
    print("PartKep — sim 模式 + MuJoCo 可视化测试")
    print("=" * 55)

    # ── Step 1: Pipeline 规划 ──
    print("\n【Step 1】Pipeline 规划（sim 模式）")
    pipeline = Pipeline(
        camera_config = CameraConfig.create_identity(),
        vlm_endpoint  = None,   # fallback 模式
        verbose       = False,
    )

    fake_rgb = Image.new("RGB", (640, 480), (128, 128, 128))
    result = pipeline.run(
        instruction   = "pick up the cup and place it on the tray",
        rgb_image     = fake_rgb,
        depth_map     = None,
        sim_keypoints = SIM_KEYPOINTS,
    )

    if not result.success:
        print(f"❌ Pipeline 失败: {result}")
        return

    pick_q  = result.pick_q
    place_q = result.place_q
    print(f"✅ pick_q  : {np.round(pick_q,  4)}")
    print(f"✅ place_q : {np.round(place_q, 4)}")

    # ── Step 2: MuJoCo 执行 ──
    print("\n【Step 2】MuJoCo 执行")
    env = MuJoCoEnv(scene_xml=SCENE_XML)

    input("\n按 Enter 开始运动...")

    env.move_to(pick_q,  label="pick")
    env.move_to(place_q, label="place")
    env.reset()

    input("\n按 Enter 关闭...")
    env.close()

    print("\n✅ 测试完成")

if __name__ == "__main__":
    main()