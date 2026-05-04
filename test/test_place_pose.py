"""
test/test_place_pose.py  —  tray keypoint -> SAP -> PoseSolver -> IK -> static viz
与 test_pick_pose.py 前半段完全对称，用于独立验证放置姿态。
run from project root: python test/test_place_pose.py
"""

import sys
import numpy as np
sys.path.insert(0, '.')

import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize as scipy_minimize

from modules.constraintsInst import ConstraintInstantiator, FINGER_LENGTH
from modules.poseSolver       import PoseSolver
from modules.vlmDecider       import VLMDecision
from modules.IKSolver         import IKSolver, Q_HOME, PANDA_JOINT_LIMITS
from modules.constraintsInst import compute_place_pose, FINGER_LENGTH

SCENE_XML = "assets/franka_emika_panda/scene.xml"

_p_home   = np.array([0.5545, 0.0, 0.6245])
_R_home   = R.from_rotvec([np.pi, 0, 0]).as_matrix()
T_current = np.eye(4)
T_current[:3, :3] = _R_home
T_current[:3,  3] = _p_home


def read_tray_keypoints(model, data) -> dict:
    site_map = {"surface": "kp_tray_surface"}
    return {
        key: data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)].copy()
        for key, site in site_map.items()
    }


def print_geometry(T: np.ndarray, meta: dict):
    pos      = T[:3, 3]
    gz       = T[:3, 2]
    kp       = meta['keypoint_3d']
    approach = meta['approach_direction']

    dist  = np.linalg.norm(pos - kp)
    dot_z = float(np.dot(gz, approach))
    print(f"\n  target pos  : {np.round(pos, 4)}  (dist to kp: {dist*100:.1f} cm)")
    print(f"  approach    : {np.round(approach, 3)}  gz: {np.round(gz, 3)}  dot: {dot_z:.4f}")


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # ── 从上次 pick 结果重建 T_pick（FK 反推）──
    ik     = IKSolver(model, verbose=True)
    q_pick = np.array([0.436, 0.2518, 0.2324, -1.8182, -1.2628, 0.9733, -0.1775])
    T_pick = ik.forward_kinematics(q_pick)
    print(f"  T_pick pos : {np.round(T_pick[:3,3], 4)}")
    print(f"  T_pick gz  : {np.round(T_pick[:3,2], 3)}")

    # ── 读取托盘 surface keypoint ──
    kps = read_tray_keypoints(model, data)
    surface_point = kps['surface']
    print(f"  surface    : {np.round(surface_point, 4)}")

    # ── 计算 place 姿态 ──
    T_place = compute_place_pose(T_pick, surface_point)
    print(f"\n  T_place pos : {np.round(T_place[:3,3], 4)}")
    print(f"  T_place gz  : {np.round(T_place[:3,2], 3)}")

    # ── IK ──
    ik_res = ik.solve(T_place, q_init=Q_HOME, n_restarts=10)
    print(f"\n  IK: {'ok' if ik_res['success'] else 'FAILED'}  "
          f"pos_err={ik_res['position_error']*1000:.2f}mm  q={np.round(ik_res['q'], 4)}")
    if not ik_res['success']:
        return

    T_achieved = ik.forward_kinematics(ik_res['q'])
    rot_err = np.linalg.norm(T_achieved[:3,:3] - T_place[:3,:3], 'fro')
    print(f"  rot_err={rot_err:.4f}  gz_target={np.round(T_place[:3,2],3)}  "
          f"gz_actual={np.round(T_achieved[:3,2],3)}")

    # ── 静态 viewer ──
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance  = 1.5
    viewer.cam.azimuth   = 180
    viewer.cam.elevation = -20
    data.ctrl[:7] = ik_res['q']
    data.ctrl[7]  = 255.0
    for _ in range(2000):
        mujoco.mj_step(model, data)
    viewer.sync()
    input("\npress Enter to close...")
    viewer.close()

    # ── 轨迹规划 + 执行（HOME → place_above → place）──
    from simulation.mujoco_env import MuJoCoEnv
    from modules.motionPlanner import MotionPlanner

    RETREAT_DIST  = 0.15
    SAFE_Z        = 0.62
    gz            = T_place[:3, 2]          # 夹爪接近方向（侧向）
    T_place_above = T_place.copy()
    T_place_above[:3, 3] = T_place[:3, 3] - gz * RETREAT_DIST
    T_safe_above  = T_place_above.copy()
    T_safe_above[2, 3] = max(T_place_above[2, 3], SAFE_Z)

    ik_safe       = ik.solve(T_safe_above,  q_init=Q_HOME,       n_restarts=10)
    ik_above      = ik.solve(T_place_above, q_init=ik_safe['q'], n_restarts=10)

    env     = MuJoCoEnv(scene_xml=SCENE_XML)
    planner = MotionPlanner(env.model, verbose=True)

    print("\n── 规划 HOME → safe_above ──")
    wps_0 = planner.plan_to_pose(Q_HOME,    T_safe_above,  q_target=ik_safe['q'])
    print("\n── 规划 safe_above → place_above ──")
    wps_1 = planner.plan_to_pose(wps_0[-1], T_place_above, q_target=ik_above['q'])
    print("\n── 规划 place_above → place ──")
    wps_2 = planner.plan_to_pose(wps_1[-1], T_place,       q_target=ik_res['q'])

    input(f"\n  共 {len(wps_0)+len(wps_1)+len(wps_2)} waypoints，按 Enter 开始执行...")
    env.execute_trajectory(wps_0)
    env.execute_trajectory(wps_1)
    env.execute_trajectory(wps_2)

    hand_id    = env.model.body("hand").id
    pos_actual = env.data.xpos[hand_id].copy()
    gz_actual  = env.data.xmat[hand_id].reshape(3,3)[:,2].copy()
    print(f"\n  [物理实际] place 位置 : {np.round(pos_actual, 4)}")
    print(f"  [物理实际] place gz   : {np.round(gz_actual,  3)}")
    print(f"  [目标]     place 位置 : {np.round(T_place[:3,3], 4)}")
    print(f"  [目标]     place gz   : {np.round(T_place[:3,2], 3)}")

    input("\n  观察轨迹，按 Enter 关闭...")
    env.close()


if __name__ == "__main__":
    main()