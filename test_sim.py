"""
test_sim.py
端到端仿真测试：MuJoCoEnv + Pipeline sim 模式

运行方式（从项目根目录）：
    python test_sim.py

验证目标：
    1. MuJoCoEnv 正常初始化，get_keypoints() 返回正确坐标
    2. Pipeline sim 模式正常规划出 pick_q / place_q
    3. 状态机完整走完 HOME→...→DONE
    4. viewer 实时显示机械臂动作
"""

import numpy as np
import mujoco
import mujoco.viewer

from simulation.mujoco import MuJoCoEnv
from pipeline import Pipeline
from configs.camera_config import CameraConfig

SCENE_XML   = "assets/franka_emika_panda/scene.xml"
INSTRUCTION = "pick up the cup and place it on the tray"


def main():
    print("=" * 60)
    print("PartKep 端到端仿真测试（sim 模式）")
    print("=" * 60)

    # ── 1. 初始化环境 ──
    env = MuJoCoEnv(SCENE_XML, verbose=True)

    # ── 2. 读取关键点（从 data.site_xpos 直接读，不走视觉） ──
    sim_kps = env.get_keypoints()
    print("\n📍 关键点世界坐标：")
    for obj, parts in sim_kps.items():
        for part, xyz in parts.items():
            print(f"   {obj}/{part}: {np.round(xyz, 4)}")

    # ── 3. 获取仿真图像（供 VLMDecider 生成标注图） ──
    rgb_image = env.get_rgb()

    # ── 4. 初始化 Pipeline（sim 模式，无 VLM endpoint 使用 fallback） ──
    cam_cfg  = CameraConfig.create_identity()
    pipeline = Pipeline(
        camera_config = cam_cfg,
        vlm_endpoint  = None,
        verbose       = True,
    )

    # ── 5. 规划 ──
    print(f"\n🤖 任务指令：{INSTRUCTION}")
    result = pipeline.run(
        instruction   = INSTRUCTION,
        rgb_image     = rgb_image,
        depth_map     = None,
        sim_keypoints = sim_kps,
    )

    if not result.success:
        print("❌ Pipeline 规划失败，终止测试")
        return

    print(f"\n✅ 规划成功")
    print(f"   pick_q  = {np.round(result.pick_q, 3)}")
    print(f"   place_q = {np.round(result.place_q, 3)}")

    # ── 6. 设置规划结果到环境 ──
    env.set_plan(result.pick_q, result.place_q)

    # ── 7. 启动 viewer，步进执行状态机 ──
    print("\n🎬 启动仿真 viewer，开始执行动作...")
    print("   （关闭 viewer 窗口可中止）")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while not env.is_done() and viewer.is_running():
            env.step()
            viewer.sync()

    print(f"\n最终状态：{env.state.value}")
    env.close()
    print("=" * 60)
    print("✅ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()