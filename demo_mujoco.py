"""
Pipeline:
    results/vlm_results.json (VLM decision pre-saved from server)
        -> VLMDecision
    MuJoCo sites -> 3D keypoints (cup + tray)
        -> ConstraintInstantiator + PoseSolver -> T_pick
        -> compute_place_pose -> T_place
        -> MotionPlanner -> MuJoCoEnv execution
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

# config
SCENE_XML          = "assets/franka_emika_panda/scene.xml"
VLM_JSON           = Path("results/vlm_results.json")
INSTRUCTION        = "pick up the cup"

RETREAT_DIST       = 0.05   # retreat along approach dir for pick_above (m)
SAFE_Z             = 0.62   # minimum Z during horizontal traversal (m)
CONTACT_FWD        = 0.04   # forward offset from pick to contact point (m)
LIFT_DIST          = 0.10   # lift height after grasp (m)
PLACE_RETREAT_DIST = 0.05   # Z offset above place target for place_above (m)
PAUSE              = 0.0    # pause between trajectory segments (s)


def load_vlm_decision(json_path: Path, instruction: str) -> VLMDecision:
    """Load the first matching VLM decision from vlm_results.json."""
    with open(json_path) as f:
        records = json.load(f)
    for rec in records:
        if rec["instruction"] == instruction and rec["mode"] == "pick":
            d = rec["decision"]
            return VLMDecision(
                w_grasp_axis = d["w_grasp_axis"],
                w_safety     = d["w_safety"],
                confidence   = d["confidence"],
                reasoning    = d["reasoning"],
                is_fallback  = d["is_fallback"],
            )
    raise ValueError(f"no record found for instruction='{instruction}' mode='pick'")


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


def main():
    # load VLM decision
    decision = load_vlm_decision(VLM_JSON, INSTRUCTION)
    print(f"[vlm]   w_grasp={decision.w_grasp_axis:.2f}  w_safety={decision.w_safety:.2f}"
          f"  conf={decision.confidence:.2f}  fallback={decision.is_fallback}")
    print(f"        {decision.reasoning}")

    # scene init
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    cup_kps       = read_cup_keypoints(model, data)
    surface_point = read_tray_keypoint(model, data)
    for name, pt in cup_kps.items():
        print(f"[kp]    {name:6s} {np.round(pt, 4)}")
    print(f"[kp]    tray   {np.round(surface_point, 4)}")

    # pick pose optimisation
    ik        = IKSolver(model, verbose=False)
    T_current = ik.forward_kinematics(Q_HOME)

    t0                = time.perf_counter()
    inst              = ConstraintInstantiator(verbose=False)
    cost_fn, x0, meta = inst.instantiate(cup_kps, decision, T_current)
    solver            = PoseSolver(max_iter=200, tol=1e-6, verbose=False)
    pick_result       = solver.solve(cost_fn, x0, meta)
    T_pick            = pick_result['T']
    approach          = meta['approach_direction']
    t_pose            = time.perf_counter() - t0

    print(f"[pick]  target={meta['grasp_target']}  cost={pick_result['final_cost']:.4f}"
          f"  t={t_pose*1000:.1f}ms")
    print(f"        pos={np.round(T_pick[:3,3], 4)}  approach={np.round(approach, 3)}")

    # intermediate pose IK
    T_pick_above          = T_pick.copy()
    T_pick_above[:3, 3]   = T_pick[:3, 3] - approach * RETREAT_DIST
    T_safe_above          = T_pick_above.copy()
    T_safe_above[2, 3]    = max(T_pick_above[2, 3], SAFE_Z)
    T_pick_contact        = T_pick.copy()
    T_pick_contact[:3, 3] += approach * CONTACT_FWD
    T_lift                = T_pick_contact.copy()
    T_lift[2, 3]          += LIFT_DIST

    ik_pick    = ik.solve(T_pick,         q_init=Q_HOME,       n_restarts=10)
    ik_safe    = ik.solve(T_safe_above,   q_init=Q_HOME,       n_restarts=10)
    ik_above   = ik.solve(T_pick_above,   q_init=ik_safe['q'], n_restarts=10)
    ik_contact = ik.solve(T_pick_contact, q_init=ik_pick['q'], n_restarts=10)
    ik_lift    = ik.solve(T_lift,         q_init=ik_contact['q'], n_restarts=10)

    if not ik_pick['success']:
        print("[error] pick IK failed, abort")
        return

    # place pose
    T_place              = compute_place_pose(T_pick, surface_point)
    T_place_above        = T_place.copy()
    T_place_above[2, 3] += PLACE_RETREAT_DIST

    ik_place_above = ik.solve(T_place_above, q_init=Q_HOME,              n_restarts=10)
    ik_place       = ik.solve(T_place,       q_init=ik_place_above['q'], n_restarts=10)

    print(f"[place] pos={np.round(T_place[:3,3], 4)}")

    # report IK errors for all key poses
    for label, res in [("pick", ik_pick), ("safe_above", ik_safe), ("pick_above", ik_above),
                       ("contact", ik_contact), ("lift", ik_lift),
                       ("place_above", ik_place_above), ("place", ik_place)]:
        status = "ok" if res['success'] else "FAILED"
        print(f"[ik]    {label:<14} {status}  err={res['position_error']*1000:.2f}mm")

    # trajectory planning
    env     = MuJoCoEnv(scene_xml=SCENE_XML)
    planner = MotionPlanner(env.model, verbose=False)

    t0        = time.perf_counter()
    wps_0     = planner.plan_to_pose(Q_HOME,        T_safe_above,   q_target=ik_safe['q'])
    wps_1     = planner.plan_to_pose(wps_0[-1],     T_pick_above,   q_target=ik_above['q'])
    wps_2     = planner.plan_to_pose(wps_1[-1],     T_pick,         q_target=ik_pick['q'])
    wps_3     = planner.plan_to_pose(wps_2[-1],     T_pick_contact, q_target=ik_contact['q'])
    wps_lift  = planner.plan_to_pose(wps_3[-1],     T_lift,         q_target=ik_lift['q'])
    wps_pa    = planner.plan_to_pose(wps_lift[-1],  T_place_above,  q_target=ik_place_above['q'])
    wps_place = planner.plan_to_pose(wps_pa[-1],    T_place,        q_target=ik_place['q'])
    t_plan    = time.perf_counter() - t0

    total = sum(len(w) for w in [wps_0, wps_1, wps_2, wps_3, wps_lift, wps_pa, wps_place])
    print(f"[plan]  {total} waypoints  t={t_plan*1000:.0f}ms")

    input("\nPress Enter to start execution...")

    # execution
    for wps in [wps_0, wps_1, wps_2, wps_3]:
        env.execute_trajectory(wps)
        time.sleep(PAUSE)

    env._set_gripper(open=False)
    env._set_weld(active=True)
    env._settle(50)
    time.sleep(PAUSE)

    env.execute_trajectory(wps_lift,  gripper_ctrl=GRIPPER_CLOSE)
    time.sleep(PAUSE)
    env.execute_trajectory(wps_pa,    gripper_ctrl=GRIPPER_CLOSE)
    time.sleep(PAUSE)
    env.execute_trajectory(wps_place, gripper_ctrl=GRIPPER_CLOSE)
    time.sleep(PAUSE)

    env._set_weld(active=False)
    env._settle(50)
    env._set_gripper(open=True)
    env._settle(50)

    # result
    hand_id    = env.model.body("hand").id
    pos_actual = env.data.xpos[hand_id].copy()
    gz_actual  = env.data.xmat[hand_id].reshape(3, 3)[:, 2].copy()
    print(f"[result] target={np.round(T_place[:3,3], 4)}  actual={np.round(pos_actual, 4)}")
    print(f"         gz target={np.round(T_place[:3,2], 3)}  actual={np.round(gz_actual, 3)}")

    # keep simulation running until Enter
    print("\nSimulation running. Press Enter to close...")
    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
    while not stop.is_set():
        mujoco.mj_step(env.model, env.data)
        env.viewer.sync()
        time.sleep(0.005)

    env.close()


if __name__ == "__main__":
    main()