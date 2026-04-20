"""
tests/test_sap_pose.py

Task 1 & 2 合并测试：SAP → PoseSolver → IKSolver → MuJoCo

测试链路：
    keypoints_3d（从 scene.xml site 手动读取）
        → ConstraintInstantiator（FALLBACK_DECISION）
        → PoseSolver
        → 几何合理性检查（打印）
        → IKSolver
        → MuJoCo 可视化

运行方式（从项目根目录）：
    python tests/test_sap_pose.py

测试目标：
    1. pick 模式：SAP 能否根据 cup 关键点给出合理的 T_pick（手 link 位置/朝向）
    2. place 模式：SAP 能否根据 tray 关键点给出合理的 T_place
    3. IKSolver 能否将 T_pick / T_place 转换为合法关节角
    4. MuJoCo 动画中机械臂姿态是否视觉合理
"""

import sys
import numpy as np

sys.path.insert(0, '.')

from modules.constraintsInst import ConstraintInstantiator
from modules.poseSolver       import PoseSolver
from modules.IKSolver         import IKSolver, Q_HOME
from modules.vlmDecider       import FALLBACK_DECISION

SCENE_XML = "assets/franka_emika_panda/scene.xml"

# ── 从 scene.xml site 读取的世界坐标 ──────────────────────────────────────────
# cup body pos="0.5 0.05 0.45"，各 site 加局部偏移后的世界坐标：
#   kp_handle: pos=(0.07, 0,      0)    → world=(0.57,  0.05,  0.45)
#   kp_body:   pos=(0,    -0.042, 0)    → world=(0.50,  0.008, 0.45)
#   kp_rim:    pos=(0,    0,      0.052)→ world=(0.50,  0.05,  0.502)
# tray body pos="0.5 -0.2 0.405"：
#   kp_tray_surface: pos=(0,0,0.007)    → world=(0.50, -0.20,  0.412)

CUP_KEYPOINTS = {
    "handle": np.array([0.57,  0.05,  0.45 ]),
    "body":   np.array([0.50,  0.008, 0.45 ]),
    "rim":    np.array([0.50,  0.05,  0.502]),
}

TRAY_KEYPOINTS = {
    "surface": np.array([0.50, -0.20, 0.412]),
}

# ── 几何合理性检查辅助函数 ────────────────────────────────────────────────────

def check_geometry(T: np.ndarray, mode: str, meta: dict):
    """
    打印 T 的几何解读，便于人工判断是否合理

    检查项：
        - 末端位置是否在关键点附近（约 APPROACH_OFFSET=0.05m）
        - 夹爪 Z 轴（接近方向）是否与 SAP approach_direction 一致
        - 夹爪 Y 轴（张开方向）是否与 SAP grasp_axis 一致（仅 pick）
    """
    pos        = T[:3, 3]
    gripper_z  = T[:3, 2]   # 夹爪接近轴（世界系）
    gripper_y  = T[:3, 1]   # 夹爪张开轴（世界系）

    approach   = meta.get('approach_direction', np.array([0, 0, -1]))
    contact_pt = meta.get('contact_point')
    if contact_pt is None:
        contact_pt = meta.get('target_point')
    kp_3d      = meta.get('keypoint_3d')

    print(f"\n  ── 几何检查 [{mode}] ──")
    print(f"  末端位置     : {np.round(pos, 4)}")

    if kp_3d is not None:
        dist = np.linalg.norm(pos - kp_3d)
        print(f"  关键点坐标   : {np.round(kp_3d, 4)}")
        print(f"  末端到关键点 : {dist*100:.1f} cm  （应约 5cm APPROACH_OFFSET）")

    if contact_pt is not None:
        err = np.linalg.norm(pos - contact_pt)
        print(f"  接触点坐标   : {np.round(contact_pt, 4)}")
        print(f"  位置对齐误差 : {err*1000:.2f} mm  （优化后应接近 0）")

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


def _ik_with_restart(ik, target_T: np.ndarray, q_init: np.ndarray,
                     n_restarts: int = 5) -> dict:
    """
    带多起点重启的 IK 求解，从多个初始猜测中选出最优合法解。

    策略：
        - 起点 0：传入的 q_init（通常是 Q_HOME 或上一步的解）
        - 起点 1~n：在 q_init 基础上加随机扰动，同时强制 joint4 在合法范围内
          （joint4 限位 [-3.07, -0.07]，home 值 -1.5708，扰动后仍须为负）
        - 从所有 within_limits=True 的解中取 position_error 最小的
        - 若全部超限，取 position_error 最小的（并标记失败，让调用方决定）
    """
    rng = np.random.default_rng(seed=42)
    candidates = [q_init.copy()]

    for _ in range(n_restarts):
        q_noisy = q_init.copy()
        q_noisy += rng.normal(0, 0.3, size=7)
        # joint4 必须为负，强制 clamp
        q_noisy[3] = np.clip(q_noisy[3], -3.0, -0.1)
        candidates.append(q_noisy)

    best = None
    for q_start in candidates:
        res = ik.solve(target_T, q_init=q_start)
        if best is None:
            best = res
            continue
        # 优先选在限位内的解；同等条件下取误差更小的
        if res['within_limits'] and not best['within_limits']:
            best = res
        elif res['within_limits'] == best['within_limits']:
            if res['position_error'] < best['position_error']:
                best = res
    return best


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SAP → PoseSolver → IKSolver → MuJoCo 测试")
    print("=" * 60)

    # ── 初始化模块 ──
    inst        = ConstraintInstantiator(verbose=True)
    pose_solver = PoseSolver(max_iter=200, tol=1e-6, verbose=True)
    ik          = IKSolver(verbose=True)

    # ConstraintInstantiator.instantiate() 需要 T_current（当前末端位姿）
    T_current = ik.forward_kinematics(Q_HOME)
    print(f"\n  T_current (Q_HOME FK) 末端位置: {np.round(T_current[:3,3], 4)}")

    # ══════════════════════════════════════════════════
    # PICK：cup 抓取
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("【PICK】cup 抓取")
    print("=" * 60)

    # pick 模式：w_flip=1.0 防止 wrist flip（C5 惩罚夹爪与接近方向反向）
    # 注：原 w_upright 已改为 w_flip，语义不同：
    #   旧 w_upright 惩罚末端偏离竖直，侧向抓取时需设为 0
    #   新 w_flip    惩罚末端方向与 approach_dir 反向，始终应激活
    from modules.vlmDecider import VLMDecision
    pick_decision = VLMDecision(
        w_grasp_axis=1.0, w_safety=2.0, w_flip=1.0,
        confidence=0.0, reasoning="test: pick with wrist flip penalty",
        is_fallback=True
    )

    cost_fn_pick, x0_pick, meta_pick = inst.instantiate(
        keypoints_3d = CUP_KEYPOINTS,
        vlm_decision = pick_decision,
        T_current    = T_current,
    )

    pose_pick = pose_solver.solve(cost_fn_pick, x0_pick, meta_pick)
    T_pick    = pose_pick['T']

    check_geometry(T_pick, "pick", meta_pick)

    print(f"\n  PoseSolver 状态: {'✅ 收敛' if pose_pick['success'] else '⚠️  未收敛'}")
    print(f"  最终代价       : {pose_pick['final_cost']:.6f}")

    # ══════════════════════════════════════════════════
    # PLACE：tray 放置
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("【PLACE】tray 放置")
    print("=" * 60)

    # place 模式的 VLMDecision（w_grasp_axis=0，w_flip 防止放置时 wrist flip）
    from modules.vlmDecider import VLMDecision
    place_decision = VLMDecision(
        w_grasp_axis=0.0, w_safety=2.0, w_flip=1.5,
        confidence=0.0, reasoning="fallback", is_fallback=True
    )

    cost_fn_place, x0_place, meta_place = inst.instantiate(
        keypoints_3d = TRAY_KEYPOINTS,
        vlm_decision = place_decision,
        T_current    = T_current,
    )

    pose_place = pose_solver.solve(cost_fn_place, x0_place, meta_place)
    T_place    = pose_place['T']

    check_geometry(T_place, "place", meta_place)

    print(f"\n  PoseSolver 状态: {'✅ 收敛' if pose_place['success'] else '⚠️  未收敛'}")
    print(f"  最终代价       : {pose_place['final_cost']:.6f}")

    # ══════════════════════════════════════════════════
    # IK 求解
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("【IK 求解】")
    print("=" * 60)

    ik_pick  = _ik_with_restart(ik, T_pick,  Q_HOME)
    ik_place = _ik_with_restart(ik, T_place, ik_pick['q'])

    pick_err  = ik_pick['position_error']
    place_err = ik_place['position_error']

    print(f"\n  pick  IK: {'✅' if ik_pick['success']  else '⚠️ '} "
          f"err={pick_err*1000:.2f}mm  limits={ik_pick['within_limits']}")
    print(f"  place IK: {'✅' if ik_place['success'] else '⚠️ '} "
          f"err={place_err*1000:.2f}mm  limits={ik_place['within_limits']}")

    # ── 护栏：位置误差 > 5cm 或关节超限，均终止 ──
    failed = False
    if pick_err > 0.05:
        print("  ❌ pick 位置误差超过 5cm")
        failed = True
    if not ik_pick['within_limits']:
        print("  ❌ pick IK 关节角超出限位")
        failed = True
    if place_err > 0.05:
        print("  ❌ place 位置误差超过 5cm")
        failed = True
    if not ik_place['within_limits']:
        print("  ❌ place IK 关节角超出限位")
        failed = True
    if failed:
        print("\n  终止 MuJoCo 可视化")
        print("  可能原因：目标位姿超出工作空间，或 IK 需要更好的初始猜测")
        return

    q_pick  = ik_pick['q']
    q_place = ik_place['q']

    # ══════════════════════════════════════════════════
    # MuJoCo 可视化
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("【MuJoCo 可视化】")
    print("=" * 60)
    print("  提示：观察每个姿态下夹爪的位置和朝向是否合理")

    from simulation.mujoco_env import MuJoCoEnv
    from modules.motionPlanner import MotionPlanner

    env = MuJoCoEnv(scene_xml=SCENE_XML)

    # 验证 site 坐标
    print("\n  ── 从仿真读取 site 世界坐标（验证用）──")
    for site_name, expected in [
        ("kp_handle",       CUP_KEYPOINTS["handle"]),
        ("kp_body",         CUP_KEYPOINTS["body"]),
        ("kp_rim",          CUP_KEYPOINTS["rim"]),
        ("kp_tray_surface", TRAY_KEYPOINTS["surface"]),
    ]:
        actual = env.get_site_xpos(site_name)
        err    = np.linalg.norm(actual - expected)
        status = "✅" if err < 0.002 else "⚠️ "
        print(f"  {status} {site_name:18s}: 仿真={np.round(actual,4)}  "
              f"手写={np.round(expected,4)}  diff={err*1000:.1f}mm")

    # MotionPlanner 规划完整轨迹
    print("\n  ── MotionPlanner 规划轨迹 ──")
    planner   = MotionPlanner(env.model, verbose=True)
    waypoints = planner.plan_pick_place(T_pick, T_place)

    input(f"\n  共 {len(waypoints)} 个 waypoints，按 Enter 开始执行...")

    env.execute_trajectory(waypoints)

    input("  观察运动是否合理，按 Enter 关闭...")
    env.close()

    print("\n✅ 测试完成")


if __name__ == "__main__":
    main()