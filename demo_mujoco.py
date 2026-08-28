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

import argparse
import json
import sys
import time
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

import mujoco

from modules.vlmDecider      import VLMDecision
from modules.taskParser      import TaskParser
from modules.constraintsInst import (
    ConstraintInstantiator,
    compute_object_hand_transform,
    compute_object_place_pose,
    compute_place_pose,
)
from modules.poseSolver      import PoseSolver
from modules.IKSolver        import IKSolver, Q_HOME
from simulation.mujoco_env   import MuJoCoEnv, GRIPPER_CLOSE
from modules.motionPlanner   import MotionPlanner

# config
SCENE_XML          = "assets/franka_emika_panda/scene.xml"
VLM_JSON           = Path("results/vlm_results.json")
INSTRUCTION        = "pick up the bottle"

RETREAT_DIST       = 0.05   # retreat along approach dir for pick_above (m)
SAFE_Z             = 0.62   # minimum Z during horizontal traversal (m)
CONTACT_FWD        = 0.01   # forward offset from pick to contact point (m)
LIFT_DIST          = 0.10   # lift height after grasp (m)
MIN_LIFT_HEIGHT    = 0.05   # headless pick acceptance threshold (m)
PLACE_RETREAT_DIST = 0.05   # Z offset above place target for place_above (m)
BOTTLE_SUPPORT_LOCAL = np.array([0.0, 0.0, -0.065])
MAX_PLACE_XY_ERROR = 0.02
MAX_PLACE_TILT     = np.deg2rad(5.0)
MAX_SUPPORT_Z_ERROR = 0.015
PLACE_SETTLE_STEPS = 500
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


def read_object_keypoints(model, data) -> dict:
    from configs.SAP import SAP_KNOWLEDGE_BASE
    result = {}
    for part_name in SAP_KNOWLEDGE_BASE:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"kp_{part_name}")
        if site_id >= 0:
            result[part_name] = data.site_xpos[site_id].copy()
    return result


def read_tray_keypoint(model, data) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "kp_tray_surface")
    return data.site_xpos[sid].copy()


def read_body_pose(model, data, body_name: str) -> np.ndarray:
    """Return a MuJoCo body pose as a base-frame homogeneous transform."""
    mujoco.mj_forward(model, data)
    body_id = model.body(body_name).id
    T = np.eye(4)
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    T[:3, 3] = data.xpos[body_id]
    return T


def _require_success(label: str, result: dict) -> bool:
    """Report an IK result and return whether it is safe to use downstream."""
    if result['success']:
        return True
    print(f"[error] {label} IK failed, abort before trajectory planning")
    print(f"        pos_err={result['position_error'] * 1000:.2f}mm "
          f"rot_err={result['rotation_error']:.4f}")
    return False


def main(plan_only: bool = False, pick_only: bool = False, headless: bool = False) -> int:
    task_spec = TaskParser().parse(INSTRUCTION)

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

    pick_kps = read_object_keypoints(model, data)
    surface_point = read_tray_keypoint(model, data)
    for name, pt in pick_kps.items():
        print(f"[kp]    {name:6s} {np.round(pt, 4)}")
    print(f"[kp]    tray   {np.round(surface_point, 4)}")

    # pick pose optimisation
    ik        = IKSolver(model, verbose=False)
    T_current = ik.forward_kinematics(Q_HOME)

    t0                = time.perf_counter()
    inst              = ConstraintInstantiator(
        object_class=task_spec.object_label,
        verbose=False,
    )
    cost_fn, x0, meta = inst.instantiate(pick_kps, decision, T_current)
    solver            = PoseSolver(max_iter=200, tol=1e-6, verbose=False)
    pick_result       = solver.solve(cost_fn, x0, meta)

    if not pick_result['success']:
        print("[error] pick pose solver failed, abort before IK or environment creation")
        print(f"        message={pick_result['message']}  iterations={pick_result['n_iter']} "
              f"cost={pick_result['final_cost']:.6f}")
        return 1

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
    if not _require_success("pick", ik_pick) or not _require_success("safe_above", ik_safe):
        return 1

    ik_above   = ik.solve(T_pick_above,   q_init=ik_safe['q'], n_restarts=10)
    ik_contact = ik.solve(T_pick_contact, q_init=ik_pick['q'], n_restarts=10)
    if not _require_success("pick_above", ik_above) or not _require_success("contact", ik_contact):
        return 1

    ik_lift    = ik.solve(T_lift,         q_init=ik_contact['q'], n_restarts=10)
    if not _require_success("lift", ik_lift):
        return 1

    ik_results = [("pick", ik_pick), ("safe_above", ik_safe), ("pick_above", ik_above),
                  ("contact", ik_contact), ("lift", ik_lift)]

    if not pick_only:
        # Preserve the grasp transform while moving the bottle itself upright
        # onto the tray support point.
        T_base_object_pick = read_body_pose(model, data, "bottle")
        T_object_hand = compute_object_hand_transform(
            T_base_object_pick, T_pick_contact
        )
        T_base_object_place = compute_object_place_pose(
            surface_point,
            BOTTLE_SUPPORT_LOCAL,
            R_base_object_place=np.eye(3),
        )
        T_place              = compute_place_pose(T_base_object_place, T_object_hand)
        T_place_above        = T_place.copy()
        T_place_above[2, 3] += PLACE_RETREAT_DIST

        ik_place_above = ik.solve(T_place_above, q_init=ik_lift['q'], n_restarts=10)
        if not _require_success("place_above", ik_place_above):
            return 1

        ik_place = ik.solve(T_place, q_init=ik_place_above['q'], n_restarts=10)
        if not _require_success("place", ik_place):
            return 1

        print(f"[place] pos={np.round(T_place[:3,3], 4)}")
        ik_results.extend([("place_above", ik_place_above), ("place", ik_place)])

    # report IK errors for all key poses
    for label, res in ik_results:
        status = "ok" if res['success'] else "FAILED"
        print(f"[ik]    {label:<14} {status}  err={res['position_error']*1000:.2f}mm")

    # trajectory planning
    planner = MotionPlanner(model, verbose=False)

    t0        = time.perf_counter()
    wps_0     = planner.plan_to_pose(Q_HOME,        T_safe_above,   q_target=ik_safe['q'])
    wps_1     = planner.plan_to_pose(wps_0[-1],     T_pick_above,   q_target=ik_above['q'])
    wps_2     = planner.plan_to_pose(wps_1[-1],     T_pick,         q_target=ik_pick['q'])
    wps_3     = planner.plan_to_pose(wps_2[-1],     T_pick_contact, q_target=ik_contact['q'])
    wps_lift  = planner.plan_to_pose(wps_3[-1],     T_lift,         q_target=ik_lift['q'])
    trajectories = [wps_0, wps_1, wps_2, wps_3, wps_lift]
    if not pick_only:
        wps_pa    = planner.plan_to_pose(wps_lift[-1], T_place_above, q_target=ik_place_above['q'])
        wps_place = planner.plan_to_pose(wps_pa[-1],   T_place,       q_target=ik_place['q'])
        wps_retract = planner.plan_to_pose(
            wps_place[-1], T_place_above, q_target=ik_place_above['q']
        )
        trajectories.extend([wps_pa, wps_place, wps_retract])
    t_plan    = time.perf_counter() - t0

    total = sum(len(w) for w in trajectories)
    print(f"[plan]  {total} waypoints  t={t_plan*1000:.0f}ms")

    if plan_only:
        print("[plan]  validation complete; execution skipped (--plan-only)")
        return 0

    env = MuJoCoEnv(scene_xml=SCENE_XML, render=not headless, realtime=not headless)

    bottle_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    if bottle_id < 0:
        print("[error] manipulated body 'bottle' is missing from the execution scene")
        env.close()
        return 1
    bottle_initial = env.data.xpos[bottle_id].copy()

    if not env.has_hand_weld():
        print("[error] no weld constraint involving 'hand' in the execution scene")
        env.close()
        return 1

    if not headless:
        input("\nPress Enter to start execution...")

    # execution
    for wps in [wps_0, wps_1, wps_2, wps_3]:
        env.execute_trajectory(wps)
        time.sleep(PAUSE)

    env._set_gripper(open=False)
    weld_activated = env._set_weld(active=True)
    if not weld_activated:
        print("[error] weld activation failed, abort before lift")
        env.close()
        return 1
    env._settle(50)
    time.sleep(PAUSE)

    env.execute_trajectory(wps_lift,  gripper_ctrl=GRIPPER_CLOSE)
    time.sleep(PAUSE)

    if not pick_only:
        env.execute_trajectory(wps_pa,    gripper_ctrl=GRIPPER_CLOSE)
        time.sleep(PAUSE)
        env.execute_trajectory(wps_place, gripper_ctrl=GRIPPER_CLOSE)
        time.sleep(PAUSE)

        if not env._set_weld(active=False):
            print("[error] weld deactivation failed")
            env.close()
            return 1
        env._settle(50)
        env._set_gripper(open=True)
        env.execute_trajectory(wps_retract)
        env._settle(PLACE_SETTLE_STEPS)

    # result
    hand_id    = env.model.body("hand").id
    pos_actual = env.data.xpos[hand_id].copy()
    gz_actual  = env.data.xmat[hand_id].reshape(3, 3)[:, 2].copy()
    T_result = T_lift if pick_only else T_place_above
    print(f"[result] target={np.round(T_result[:3,3], 4)}  actual={np.round(pos_actual, 4)}")
    print(f"         gz target={np.round(T_result[:3,2], 3)}  actual={np.round(gz_actual, 3)}")

    bottle_final = env.data.xpos[bottle_id].copy()
    lift_height = float(bottle_final[2] - bottle_initial[2])
    displacement_label = "lift" if pick_only else "delta_z"
    print(
        f"[object] bottle={np.round(bottle_final, 4)}  "
        f"{displacement_label}={lift_height * 1000:.1f}mm"
    )

    if pick_only and lift_height < MIN_LIFT_HEIGHT:
        print(f"[error] weld-assisted lift too small: {lift_height * 1000:.1f}mm "
              f"< {MIN_LIFT_HEIGHT * 1000:.1f}mm")
        env.close()
        return 1

    if not pick_only:
        tray_surface = env.get_site_xpos("kp_tray_surface")
        R_bottle = env.data.xmat[bottle_id].reshape(3, 3).copy()
        bottle_axis = R_bottle[:, 2]
        tilt = float(np.arccos(np.clip(np.dot(bottle_axis, [0.0, 0.0, 1.0]), -1.0, 1.0)))
        xy_error = float(np.linalg.norm(bottle_final[:2] - tray_surface[:2]))
        support_world = bottle_final + R_bottle @ BOTTLE_SUPPORT_LOCAL
        support_z_error = float(abs(support_world[2] - tray_surface[2]))
        weld_released = not env.is_hand_weld_active()
        print(
            f"[place-check] xy_err={xy_error * 1000:.1f}mm  "
            f"tilt={np.rad2deg(tilt):.2f}deg  "
            f"support_z_err={support_z_error * 1000:.1f}mm  "
            f"released={weld_released}"
        )
        if (
            xy_error > MAX_PLACE_XY_ERROR
            or tilt > MAX_PLACE_TILT
            or support_z_error > MAX_SUPPORT_Z_ERROR
            or not weld_released
        ):
            print("[error] bottle placement did not meet the headless acceptance limits")
            env.close()
            return 1

    if headless:
        env.close()
        return 0

    # keep simulation running until Enter
    print("\nSimulation running. Press Enter to close...")
    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
    while not stop.is_set():
        mujoco.mj_step(env.model, env.data)
        env.viewer.sync()
        time.sleep(0.005)

    env.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PartKep MuJoCo pick-and-place demo.")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="run pose/IK and waypoint-generation smoke test without opening the viewer or executing",
    )
    parser.add_argument(
        "--pick-only",
        action="store_true",
        help="stop after grasping and lifting the object instead of planning a placement",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="execute without a viewer or real-time sleeps; implies no interactive prompts",
    )
    args = parser.parse_args()
    raise SystemExit(main(plan_only=args.plan_only, pick_only=args.pick_only, headless=args.headless))
