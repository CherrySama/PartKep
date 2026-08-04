"""
keypoints -> SAP -> PoseSolver -> IK -> trajectory
Created by Yinghao Ho on 2026-04-16
"""

import sys
import time
import numpy as np
sys.path.insert(0, '.')

import mujoco
import mujoco.viewer

from modules.constraintsInst import ConstraintInstantiator, FINGER_LENGTH
from modules.poseSolver      import PoseSolver
from modules.vlmDecider      import VLMDecision
from modules.IKSolver        import IKSolver, Q_HOME
from simulation.mujoco_env   import MuJoCoEnv, GRIPPER_CLOSE
from modules.motionPlanner   import MotionPlanner

SCENE_XML = "assets/franka_emika_panda/scene.xml"

RETREAT_DIST = 0.05
SAFE_Z       = 0.62
CONTACT_FWD  = 0.04
LIFT_DIST    = 0.10


def read_object_keypoints(model, data) -> dict:
    from configs.SAP import SAP_KNOWLEDGE_BASE
    result = {}
    for part_name in SAP_KNOWLEDGE_BASE:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"kp_{part_name}")
        if site_id >= 0:
            result[part_name] = data.site_xpos[site_id].copy()
    return result


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    kps = read_object_keypoints(model, data)
    for name, pt in kps.items():
        print(f"[kp]    {name:6s} {np.round(pt, 4)}")

    # use fallback VLM decision for isolated pick test
    decision = VLMDecision(
        w_grasp_axis=1.0, w_safety=2.0,
        confidence=0.0, reasoning="fallback", is_fallback=True
    )

    ik        = IKSolver(model, verbose=False)
    T_current = ik.forward_kinematics(Q_HOME)

    # pose optimisation
    t0                 = time.perf_counter()
    inst               = ConstraintInstantiator(object_class="bottle", verbose=True)
    cost_fn, x0, meta  = inst.instantiate(kps, decision, T_current)
    solver             = PoseSolver(max_iter=200, tol=1e-6, verbose=True)
    result             = solver.solve(cost_fn, x0, meta)
    T_pick             = result['T']
    approach           = meta['approach_direction']
    t_pose             = time.perf_counter() - t0

    print(f"[pick]  target={meta['grasp_target']}  cost={result['final_cost']:.6f}"
          f"  t={t_pose*1000:.1f}ms")
    print(f"        pos={np.round(T_pick[:3,3], 4)}  gz={np.round(T_pick[:3,2], 3)}"
          f"  approach={np.round(approach, 3)}")
    if meta.get('grasp_axis_target') is not None:
        gy    = T_pick[:3, 1]
        dot_y = float(np.dot(gy, meta['grasp_axis_target']))
        print(f"        grasp_axis={np.round(meta['grasp_axis_target'],3)}"
              f"  gy={np.round(gy,3)}  dot={dot_y:.4f}")

    # IK
    ik_res = ik.solve(T_pick, q_init=Q_HOME, n_restarts=10)
    print(f"[ik]    {'ok' if ik_res['success'] else 'FAILED'}"
          f"  pos_err={ik_res['position_error']*1000:.2f}mm")

    if not ik_res['success']:
        return

    T_achieved = ik.forward_kinematics(ik_res['q'])
    rot_err    = np.linalg.norm(T_achieved[:3, :3] - T_pick[:3, :3], 'fro')
    print(f"        rot_err={rot_err:.4f}"
          f"  gz_target={np.round(T_pick[:3,2],3)}"
          f"  gz_actual={np.round(T_achieved[:3,2],3)}")

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance  = 1.5
    viewer.cam.azimuth   = 180
    viewer.cam.elevation = -20
    data.ctrl[:7] = ik_res['q']
    data.ctrl[7]  = 255.0
    for _ in range(2000):
        mujoco.mj_step(model, data)
    viewer.sync()
    input("\nPress Enter to continue to trajectory execution...")
    viewer.close()

    # trajectory planning
    T_pick_above          = T_pick.copy()
    T_pick_above[:3, 3]   = T_pick[:3, 3] - approach * RETREAT_DIST
    T_safe_above          = T_pick_above.copy()
    T_safe_above[2, 3]    = max(T_pick_above[2, 3], SAFE_Z)
    T_pick_contact        = T_pick.copy()
    T_pick_contact[:3, 3] += approach * CONTACT_FWD
    T_lift                = T_pick_contact.copy()
    T_lift[2, 3]          += LIFT_DIST

    ik_safe    = ik.solve(T_safe_above,   q_init=Q_HOME,          n_restarts=10)
    ik_above   = ik.solve(T_pick_above,   q_init=ik_safe['q'],    n_restarts=10)
    ik_contact = ik.solve(T_pick_contact, q_init=ik_res['q'],     n_restarts=10)
    ik_lift    = ik.solve(T_lift,         q_init=ik_contact['q'], n_restarts=10)

    env     = MuJoCoEnv(scene_xml=SCENE_XML)
    planner = MotionPlanner(env.model, verbose=False)

    wps_0 = planner.plan_to_pose(Q_HOME,    T_safe_above,   q_target=ik_safe['q'])
    wps_1 = planner.plan_to_pose(wps_0[-1], T_pick_above,   q_target=ik_above['q'])
    wps_2 = planner.plan_to_pose(wps_1[-1], T_pick,         q_target=ik_res['q'])
    wps_3 = planner.plan_to_pose(wps_2[-1], T_pick_contact, q_target=ik_contact['q'])
    wps_l = planner.plan_to_pose(wps_3[-1], T_lift,         q_target=ik_lift['q'])

    total = len(wps_0) + len(wps_1) + len(wps_2) + len(wps_3) + len(wps_l)
    input(f"\n{total} waypoints. Press Enter to execute...")

    for wps in [wps_0, wps_1, wps_2, wps_3]:
        env.execute_trajectory(wps)

    env._set_gripper(open=False)
    env._set_weld(active=True)
    env._settle(50)

    env.execute_trajectory(wps_l, gripper_ctrl=GRIPPER_CLOSE)

    hand_id    = env.model.body("hand").id
    pos_actual = env.data.xpos[hand_id].copy()
    gz_actual  = env.data.xmat[hand_id].reshape(3, 3)[:, 2].copy()
    print(f"[result] target={np.round(T_pick[:3,3], 4)}  actual={np.round(pos_actual, 4)}")
    print(f"         gz target={np.round(T_pick[:3,2], 3)}  actual={np.round(gz_actual, 3)}")

    input("\nPress Enter to close...")
    env.close()


if __name__ == "__main__":
    main()
