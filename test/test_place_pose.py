"""
test_place_pose.py  —  tray keypoint -> compute_place_pose -> IK -> trajectory
Created by Yinghao Ho on 2026-04-19
"""

import sys
import numpy as np
sys.path.insert(0, '.')

import mujoco
import mujoco.viewer

from modules.constraintsInst import compute_place_pose, FINGER_LENGTH
from modules.IKSolver        import IKSolver, Q_HOME
from simulation.mujoco_env   import MuJoCoEnv
from modules.motionPlanner   import MotionPlanner

SCENE_XML = "assets/franka_emika_panda/scene.xml"

PLACE_RETREAT_DIST = 0.10   # Z offset above place target (m)
SAFE_Z             = 0.62


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    ik = IKSolver(model, verbose=True)

    # reconstruct T_pick from a previously validated pick joint config
    q_pick = np.array([0.436, 0.2518, 0.2324, -1.8182, -1.2628, 0.9733, -0.1775])
    T_pick = ik.forward_kinematics(q_pick)
    print(f"[pick]  pos={np.round(T_pick[:3,3], 4)}  gz={np.round(T_pick[:3,2], 3)}")

    # read tray surface keypoint
    sid           = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "kp_tray_surface")
    surface_point = data.site_xpos[sid].copy()
    print(f"[kp]    tray   {np.round(surface_point, 4)}")

    # compute place pose
    T_place = compute_place_pose(T_pick, surface_point)
    print(f"[place] pos={np.round(T_place[:3,3], 4)}  gz={np.round(T_place[:3,2], 3)}")

    # IK
    ik_res = ik.solve(T_place, q_init=Q_HOME, n_restarts=10)
    print(f"[ik]    {'ok' if ik_res['success'] else 'FAILED'}"
          f"  pos_err={ik_res['position_error']*1000:.2f}mm")
    if not ik_res['success']:
        return

    T_achieved = ik.forward_kinematics(ik_res['q'])
    rot_err    = np.linalg.norm(T_achieved[:3, :3] - T_place[:3, :3], 'fro')
    print(f"        rot_err={rot_err:.4f}"
          f"  gz_target={np.round(T_place[:3,2],3)}"
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
    T_place_above        = T_place.copy()
    T_place_above[2, 3] += PLACE_RETREAT_DIST

    ik_above = ik.solve(T_place_above, q_init=Q_HOME, n_restarts=10)

    env     = MuJoCoEnv(scene_xml=SCENE_XML)
    planner = MotionPlanner(env.model, verbose=False)

    wps_0 = planner.plan_to_pose(Q_HOME,      T_place_above, q_target=ik_above['q'])
    wps_1 = planner.plan_to_pose(wps_0[-1],   T_place,       q_target=ik_res['q'])

    total = len(wps_0) + len(wps_1)
    input(f"\n{total} waypoints. Press Enter to execute...")

    env.execute_trajectory(wps_0)
    env.execute_trajectory(wps_1)

    hand_id    = env.model.body("hand").id
    pos_actual = env.data.xpos[hand_id].copy()
    gz_actual  = env.data.xmat[hand_id].reshape(3, 3)[:, 2].copy()
    print(f"[result] target={np.round(T_place[:3,3], 4)}  actual={np.round(pos_actual, 4)}")
    print(f"         gz target={np.round(T_place[:3,2], 3)}  actual={np.round(gz_actual, 3)}")

    input("\nPress Enter to close...")
    env.close()


if __name__ == "__main__":
    main()