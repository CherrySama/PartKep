"""
VLM-driven pick pipeline in MuJoCo
Created by Yinghao Ho on 2026-04-29
"""

import json
import sys
import time
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

import mujoco

from modules.vlmDecider      import VLMDecision
from modules.constraintsInst import ConstraintInstantiator, compute_place_pose
from modules.poseSolver      import PoseSolver
from modules.IKSolver        import IKSolver, Q_HOME
from simulation.mujoco_env   import MuJoCoEnv, GRIPPER_CLOSE
from modules.motionPlanner   import MotionPlanner

# ── config ────────────────────────────────────────────────────────────────────

SCENE_XML    = "assets/franka_emika_panda/scene.xml"
VLM_JSON     = Path("results/vlm_results.json")
INSTRUCTION  = "pick up the cup"

RETREAT_DIST = 0.05   # approach 方向退后距离（pick_above）
SAFE_Z       = 0.62   # 水平移动时最低安全高度
CONTACT_FWD  = 0.04   # pick 到 contact 的前进距离
LIFT_DIST          = 0.10   # 抬起高度
PLACE_RETREAT_DIST = 0.05   # place_above Z 方向上抬距离
PAUSE              = 0.3    # 段间停顿（秒）


# ── helpers ───────────────────────────────────────────────────────────────────

def load_vlm_decision(json_path: Path, instruction: str) -> VLMDecision:
    with open(json_path) as f:
        records = json.load(f)
    for rec in records:
        if rec["instruction"] == instruction and rec["mode"] == "pick":
            d = rec["decision"]
            decision = VLMDecision(
                w_grasp_axis = d["w_grasp_axis"],
                w_safety     = d["w_safety"],
                confidence   = d["confidence"],
                reasoning    = d["reasoning"],
                is_fallback  = d["is_fallback"],
            )
            print(f"  VLM 决策: {decision}")
            print(f"  reasoning: {d['reasoning']}")
            return decision
    raise ValueError(f"未找到 instruction='{instruction}' mode='pick' 的记录")


def read_cup_keypoints(model, data) -> dict:
    site_map = {"handle": "kp_handle", "body": "kp_body", "rim": "kp_rim"}
    kps = {}
    for key, site in site_map.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        kps[key] = data.site_xpos[sid].copy()
    return kps


def read_tray_keypoint(model, data) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "kp_tray_surface")
    return data.site_xpos[sid].copy()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PartKep demo_mujoco  [pick only]")
    print("=" * 60)

    # ── Step 1: VLM 决策 ──────────────────────────────────────────────────────
    print("\n【Step 1】读取 VLM 决策")
    decision = load_vlm_decision(VLM_JSON, INSTRUCTION)

    # ── Step 2: 场景初始化 + 关键点 ───────────────────────────────────────────
    print("\n【Step 2】MuJoCo 场景初始化")
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    cup_kps      = read_cup_keypoints(model, data)
    surface_point = read_tray_keypoint(model, data)
    for name, pt in cup_kps.items():
        print(f"  {name:6s}: {np.round(pt, 4)}")
    print(f"  tray  : {np.round(surface_point, 4)}")

    # ── Step 3: Pick 姿态求解 ─────────────────────────────────────────────────
    print("\n【Step 3】Pick 姿态求解")
    ik        = IKSolver(model, verbose=True)
    T_current = ik.forward_kinematics(Q_HOME)

    inst               = ConstraintInstantiator(verbose=True)
    cost_fn, x0, meta  = inst.instantiate(cup_kps, decision, T_current)

    solver      = PoseSolver(max_iter=200, tol=1e-6, verbose=True)
    pick_result = solver.solve(cost_fn, x0, meta)
    T_pick      = pick_result['T']
    approach    = meta['approach_direction']

    print(f"  grasp target : {meta['grasp_target']}")
    print(f"  T_pick pos   : {np.round(T_pick[:3, 3], 4)}")
    print(f"  approach dir : {np.round(approach, 3)}")

    # ── Step 4: 中间位姿 IK 求解 ──────────────────────────────────────────────
    print("\n【Step 4】中间位姿 IK 求解")

    T_pick_above        = T_pick.copy()
    T_pick_above[:3, 3] = T_pick[:3, 3] - approach * RETREAT_DIST

    T_safe_above        = T_pick_above.copy()
    T_safe_above[2, 3]  = max(T_pick_above[2, 3], SAFE_Z)

    T_pick_contact        = T_pick.copy()
    T_pick_contact[:3, 3] += approach * CONTACT_FWD

    ik_pick    = ik.solve(T_pick,         q_init=Q_HOME,       n_restarts=10)
    ik_safe    = ik.solve(T_safe_above,   q_init=Q_HOME,       n_restarts=10)
    ik_above   = ik.solve(T_pick_above,   q_init=ik_safe['q'], n_restarts=10)
    ik_contact = ik.solve(T_pick_contact, q_init=ik_pick['q'], n_restarts=10)

    for label, res in [("pick", ik_pick), ("safe_above", ik_safe),
                       ("pick_above", ik_above), ("contact", ik_contact)]:
        status = "ok" if res['success'] else "FAILED"
        print(f"  [{label}] {status}  err={res['position_error']*1000:.2f}mm")

    if not ik_pick['success']:
        print("  ❌ Pick IK 失败，退出")
        return

    # ── Step 5: 规划 + 执行 ───────────────────────────────────────────────────
    print("\n【Step 5】轨迹规划 + 执行")
    env     = MuJoCoEnv(scene_xml=SCENE_XML)
    planner = MotionPlanner(env.model, verbose=True)

    print("\n── HOME → safe_above ──")
    wps_0 = planner.plan_to_pose(Q_HOME,    T_safe_above,   q_target=ik_safe['q'])
    print("── safe_above → pick_above ──")
    wps_1 = planner.plan_to_pose(wps_0[-1], T_pick_above,   q_target=ik_above['q'])
    print("── pick_above → pick ──")
    wps_2 = planner.plan_to_pose(wps_1[-1], T_pick,         q_target=ik_pick['q'])
    print("── pick → contact ──")
    wps_3 = planner.plan_to_pose(wps_2[-1], T_pick_contact, q_target=ik_contact['q'])

    total = len(wps_0) + len(wps_1) + len(wps_2) + len(wps_3)
    input(f"\n  规划完成，共 {total} waypoints，按 Enter 开始全流程...")

    env.execute_trajectory(wps_0)
    time.sleep(PAUSE)
    env.execute_trajectory(wps_1)
    time.sleep(PAUSE)
    env.execute_trajectory(wps_2)
    time.sleep(PAUSE)
    env.execute_trajectory(wps_3)

    env._set_gripper(open=False)
    print("  夹爪已关闭")
    env._set_weld(active=True)
    print("  cup_weld 已激活")
    env._settle(50)
    time.sleep(PAUSE)

    # ── lift ──────────────────────────────────────────────────────────────────
    T_lift       = T_pick_contact.copy()
    T_lift[2, 3] += LIFT_DIST
    ik_lift      = ik.solve(T_lift, q_init=ik_contact['q'], n_restarts=10)

    print("\n── contact → lift ──")
    wps_lift = planner.plan_to_pose(wps_3[-1], T_lift, q_target=ik_lift['q'])

    env.execute_trajectory(wps_lift, gripper_ctrl=GRIPPER_CLOSE)
    time.sleep(PAUSE)

    # ── place_above ───────────────────────────────────────────────────────────
    print("\n【Step 6】Place 姿态推导 + 移动到 place_above")
    T_place = compute_place_pose(T_pick, surface_point)
    gz_place = T_place[:3, 2]

    T_place_above        = T_place.copy()
    T_place_above[2, 3] += PLACE_RETREAT_DIST   # 世界 Z 正上方，而非沿侧向 gz 退后

    ik_place_above = ik.solve(T_place_above, q_init=Q_HOME, n_restarts=10)

    print(f"  T_place pos       : {np.round(T_place[:3, 3], 4)}")
    print(f"  T_place_above pos : {np.round(T_place_above[:3, 3], 4)}")
    status = "ok" if ik_place_above['success'] else "FAILED"
    print(f"  [place_above] {status}  err={ik_place_above['position_error']*1000:.2f}mm")

    q_lift_end = wps_lift[-1]
    print("\n── lift → place_above ──")
    wps_pa = planner.plan_to_pose(q_lift_end, T_place_above, q_target=ik_place_above['q'])

    env.execute_trajectory(wps_pa, gripper_ctrl=GRIPPER_CLOSE)
    time.sleep(PAUSE)

    # ── place ─────────────────────────────────────────────────────────────────
    print("\n── place_above → place ──")
    ik_place = ik.solve(T_place, q_init=ik_place_above['q'], n_restarts=10)
    status   = "ok" if ik_place['success'] else "FAILED"
    print(f"  [place] {status}  err={ik_place['position_error']*1000:.2f}mm")

    wps_place = planner.plan_to_pose(wps_pa[-1], T_place, q_target=ik_place['q'])

    env.execute_trajectory(wps_place, gripper_ctrl=GRIPPER_CLOSE)
    time.sleep(PAUSE)

    env._set_weld(active=False)
    print("  cup_weld 已释放")
    env._settle(50)
    env._set_gripper(open=True)
    print("  夹爪已张开")
    env._settle(50)

    # ── 结果验证 ──────────────────────────────────────────────────────────────
    hand_id    = env.model.body("hand").id
    pos_actual = env.data.xpos[hand_id].copy()
    gz_actual  = env.data.xmat[hand_id].reshape(3, 3)[:, 2].copy()
    print(f"\n  [目标] place pos : {np.round(T_place[:3, 3], 4)}")
    print(f"  [实际] hand pos  : {np.round(pos_actual, 4)}")
    print(f"  [目标] gz        : {np.round(T_place[:3, 2], 3)}")
    print(f"  [实际] gz        : {np.round(gz_actual, 3)}")

    # 仿真持续运行，直到用户按 Enter
    print("\n  仿真运行中，按 Enter 关闭...")
    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
    while not stop.is_set():
        mujoco.mj_step(env.model, env.data)
        env.viewer.sync()
        time.sleep(0.005)

    env.close()
    print("\n✅ demo pick-place 完成")


if __name__ == "__main__":
    main()