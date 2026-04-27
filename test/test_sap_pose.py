"""
test/test_sap_pose.py

SAP → PoseSolver → MotionPlanner → MuJoCo 完整 pick-place 验证

运行方式（从项目根目录）：
    python test/test_sap_pose.py
"""

import sys
import numpy as np
import mujoco

sys.path.insert(0, '.')

from modules.constraintsInst import ConstraintInstantiator, FINGER_LENGTH
from modules.poseSolver       import PoseSolver
from modules.IKSolver         import IKSolver, Q_HOME
from modules.vlmDecider       import VLMDecision

SCENE_XML = "assets/franka_emika_panda/scene.xml"

# 提前加载 model 供 IKSolver 使用，与 MuJoCoEnv 内部 model 独立但同源
_model = mujoco.MjModel.from_xml_path(SCENE_XML)

CUP_KEYPOINTS = {
    "handle": np.array([0.57,  0.05,  0.45 ]),
    "body":   np.array([0.50,  0.008, 0.45 ]),
    "rim":    np.array([0.50,  0.05,  0.502]),
}

TRAY_KEYPOINTS = {
    "surface": np.array([0.50, -0.20, 0.412]),
}


def check_geometry(T: np.ndarray, mode: str, meta: dict):
    pos       = T[:3, 3]
    gripper_z = T[:3, 2]
    gripper_y = T[:3, 1]
    approach  = meta.get('approach_direction', np.array([0, 0, -1]))
    kp_3d     = meta.get('keypoint_3d')

    print(f"\n  ── 几何检查 [{mode}] ──")
    print(f"  末端位置     : {np.round(pos, 4)}")
    if kp_3d is not None:
        dist = np.linalg.norm(pos - kp_3d)
        print(f"  关键点坐标   : {np.round(kp_3d, 4)}")
        print(f"  末端到关键点 : {dist*100:.1f} cm  （应约 {FINGER_LENGTH*100:.1f}cm）")
    dot_z = float(np.dot(gripper_z, approach))
    print(f"  接近方向 SAP : {np.round(approach, 3)}")
    print(f"  夹爪 Z 轴    : {np.round(gripper_z, 3)}")
    print(f"  Z 轴对齐度   : {dot_z:.4f}  （应接近 ±1.0）")
    if mode == "pick" and 'grasp_axis_target' in meta:
        grasp_ax = meta['grasp_axis_target']
        dot_y    = float(np.dot(gripper_y, grasp_ax))
        print(f"  抓取轴 SAP   : {np.round(grasp_ax, 3)}")
        print(f"  夹爪 Y 轴    : {np.round(gripper_y, 3)}")
        print(f"  Y 轴对齐度   : {dot_y:.4f}  （应接近 ±1.0）")


def main():
    print("=" * 60)
    print("SAP → PoseSolver → MotionPlanner → MuJoCo pick-place 验证")
    print("=" * 60)

    inst        = ConstraintInstantiator(verbose=True)
    pose_solver = PoseSolver(max_iter=200, tol=1e-6, verbose=True)
    ik          = IKSolver(_model, verbose=False)

    T_current = ik.forward_kinematics(Q_HOME)
    print(f"\n  T_current (Q_HOME FK): {np.round(T_current[:3,3], 4)}")

    # ── PICK ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("【PICK】cup 抓取")
    print("=" * 60)

    pick_decision = VLMDecision(
        w_grasp_axis=1.0, w_safety=2.0,
        confidence=0.0, reasoning="test: pick", is_fallback=True
    )
    cost_fn_pick, x0_pick, meta_pick = inst.instantiate(
        keypoints_3d=CUP_KEYPOINTS,
        vlm_decision=pick_decision,
        T_current=T_current,
    )
    pose_pick = pose_solver.solve(cost_fn_pick, x0_pick, meta_pick)
    T_pick    = pose_pick['T']
    check_geometry(T_pick, "pick", meta_pick)
    print(f"\n  PoseSolver: {'✅ 收敛' if pose_pick['success'] else '⚠️  未收敛'}")

    # ── PLACE ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("【PLACE】tray 放置")
    print("=" * 60)

    place_decision = VLMDecision(
        w_grasp_axis=0.0, w_safety=2.0,
        confidence=0.0, reasoning="fallback", is_fallback=True
    )
    cost_fn_place, x0_place, meta_place = inst.instantiate(
        keypoints_3d=TRAY_KEYPOINTS,
        vlm_decision=place_decision,
        T_current=T_current,
    )
    pose_place = pose_solver.solve(cost_fn_place, x0_place, meta_place)
    T_place    = pose_place['T']
    check_geometry(T_place, "place", meta_place)
    print(f"\n  PoseSolver: {'✅ 收敛' if pose_place['success'] else '⚠️  未收敛'}")

    # ── MuJoCo ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("【MuJoCo 初始化 + site 验证】")
    print("=" * 60)

    from simulation.mujoco_env import MuJoCoEnv
    from modules.motionPlanner import MotionPlanner

    env = MuJoCoEnv(scene_xml=SCENE_XML)

    print("\n  ── site 验证 ──")
    for site_name, expected in [
        ("kp_handle",       CUP_KEYPOINTS["handle"]),
        ("kp_body",         CUP_KEYPOINTS["body"]),
        ("kp_rim",          CUP_KEYPOINTS["rim"]),
        ("kp_tray_surface", TRAY_KEYPOINTS["surface"]),
    ]:
        actual = env.get_site_xpos(site_name)
        err    = np.linalg.norm(actual - expected)
        print(f"  {'✅' if err < 0.002 else '⚠️ '} {site_name:18s}: "
              f"{np.round(actual,4)}  diff={err*1000:.1f}mm")

    # ── 轨迹规划 + 执行 ───────────────────────────────
    print("\n" + "=" * 60)
    print("【轨迹规划 + 执行】")
    print("=" * 60)

    planner  = MotionPlanner(env.model, verbose=True)
    segments = planner.plan_pick_place(T_pick, T_place)

    total = sum(len(s['waypoints']) for s in segments)
    input(f"\n  共 {total} waypoints（{len(segments)} 段），按 Enter 开始执行...")

    env.execute_pick_place(segments)

    input("\n  观察结果，按 Enter 关闭...")
    env.close()
    print("\n✅ 测试完成")


if __name__ == "__main__":
    main()