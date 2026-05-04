"""
test/test_pick_pose.py  —  keypoints -> SAP -> PoseSolver -> IK -> static viz
run from project root: python test/test_pick_pose.py
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
from simulation.mujoco_env import GRIPPER_CLOSE

SCENE_XML = "assets/franka_emika_panda/scene.xml"

_p_home   = np.array([0.5545, 0.0, 0.6245])
_R_home   = R.from_rotvec([np.pi, 0, 0]).as_matrix()
T_current = np.eye(4)
T_current[:3, :3] = _R_home
T_current[:3,  3] = _p_home


def read_cup_keypoints(model, data) -> dict:
    site_map = {"handle": "kp_handle", "body": "kp_body", "rim": "kp_rim"}
    return {
        key: data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)].copy()
        for key, site in site_map.items()
    }


def print_geometry(T: np.ndarray, meta: dict):
    pos      = T[:3, 3]
    gz       = T[:3, 2]
    gy       = T[:3, 1]
    kp       = meta['keypoint_3d']
    approach = meta['approach_direction']
    grasp_ax = meta.get('grasp_axis_target')

    dist  = np.linalg.norm(pos - kp)
    dot_z = float(np.dot(gz, approach))
    print(f"\n  target pos  : {np.round(pos, 4)}  (dist to kp: {dist*100:.1f} cm)")
    print(f"  approach    : {np.round(approach, 3)}  gz: {np.round(gz, 3)}  dot: {dot_z:.4f}")
    if grasp_ax is not None:
        dot_y = float(np.dot(gy, grasp_ax))
        print(f"  grasp_axis  : {np.round(grasp_ax, 3)}  gy: {np.round(gy, 3)}  dot: {dot_y:.4f}")


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    kps = read_cup_keypoints(model, data)
    # kps.pop('handle')
    for name, pt in kps.items():
        print(f"  {name:8s}: {np.round(pt, 4)}")

    decision = VLMDecision(
        w_grasp_axis=1.0, w_safety=2.0,
        confidence=0.0, reasoning="test", is_fallback=True
    )

    inst    = ConstraintInstantiator(verbose=True)
    cost_fn, x0, meta = inst.instantiate(kps, decision, T_current)

    solver  = PoseSolver(max_iter=200, tol=1e-6, verbose=True)
    result  = solver.solve(cost_fn, x0, meta)
    T_pick  = result['T']
    print(f"  PoseSolver: {'ok' if result['success'] else 'FAILED'}  cost={result['final_cost']:.6f}")
    print_geometry(T_pick, meta)

    # position-only IK sanity check
    diag_data = mujoco.MjData(model)
    hand_id   = model.body("hand").id
    p_target  = T_pick[:3, 3]
    bounds    = [(lo, hi) for lo, hi in PANDA_JOINT_LIMITS]

    def pos_cost(q):
        diag_data.qpos[:7] = q
        mujoco.mj_forward(model, diag_data)
        return float(np.sum((diag_data.xpos[hand_id] - p_target) ** 2))

    print("\n  -- position-only IK --")
    for label, q0 in [("Q_HOME", Q_HOME),
                      ("q_ext ", np.array([0.12, 0.5, 0.0, -0.5, 0.0, 1.0, 0.0]))]:
        res = scipy_minimize(pos_cost, q0, method="L-BFGS-B", bounds=bounds,
                             options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10})
        diag_data.qpos[:7] = res.x
        mujoco.mj_forward(model, diag_data)
        err = np.linalg.norm(diag_data.xpos[hand_id] - p_target) * 1000
        print(f"  [{label}] err={err:.2f}mm  q={np.round(res.x, 4)}")

    # full IK
    ik     = IKSolver(model, verbose=True)
    ik_res = ik.solve(T_pick, q_init=Q_HOME, n_restarts=10)
    print(f"\n  IK: {'ok' if ik_res['success'] else 'FAILED'}  "
          f"pos_err={ik_res['position_error']*1000:.2f}mm  q={np.round(ik_res['q'], 4)}")

    if not ik_res['success']:
        return

    # rotation error
    T_achieved = ik.forward_kinematics(ik_res['q'])
    rot_err    = np.linalg.norm(T_achieved[:3, :3] - T_pick[:3, :3], 'fro')
    print(f"  rot_err={rot_err:.4f}  gz_target={np.round(T_pick[:3,2],3)}  "
          f"gz_actual={np.round(T_achieved[:3,2],3)}")

    # static visualization
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

    # ── 轨迹规划 + 执行（HOME → pick_above → pick）───────────────
    from simulation.mujoco_env import MuJoCoEnv
    from modules.motionPlanner import MotionPlanner

    RETREAT_DIST = 0.05  # retreat distance along approach direction for pre-pose
    SAFE_Z       = 0.62  # minimum Z height during horizontal traversal

    approach     = meta['approach_direction']
    T_pick_above = T_pick.copy()
    T_pick_above[:3, 3] = T_pick[:3, 3] - approach * RETREAT_DIST

    # safe_above: pick_above but Z raised to SAFE_Z if needed.
    # HOME -> safe_above stays high, then safe_above -> pick_above descends outside the object.
    T_safe_above = T_pick_above.copy()
    T_safe_above[2, 3] = max(T_pick_above[2, 3], SAFE_Z)

    ik_safe      = ik.solve(T_safe_above, q_init=Q_HOME,       n_restarts=10)
    ik_above     = ik.solve(T_pick_above, q_init=ik_safe['q'], n_restarts=10)
    q_safe_above = ik_safe['q']
    q_pick_above = ik_above['q']

    env     = MuJoCoEnv(scene_xml=SCENE_XML)
    planner = MotionPlanner(env.model, verbose=True)

    print("\n── 规划 HOME → safe_above ──")
    wps_0 = planner.plan_to_pose(Q_HOME,        T_safe_above, q_target=q_safe_above)
    print("\n── 规划 safe_above → pick_above ──")
    wps_1 = planner.plan_to_pose(wps_0[-1],     T_pick_above, q_target=q_pick_above)
    print("\n── 规划 pick_above → pick ──")
    wps_2 = planner.plan_to_pose(wps_1[-1],     T_pick,       q_target=ik_res['q'])

    # 沿 approach 方向前进 3cm（接触阶段）
    T_pick_contact        = T_pick.copy()
    T_pick_contact[:3, 3] += approach * 0.04

    ik_contact = ik.solve(T_pick_contact, q_init=ik_res['q'], n_restarts=10)
    q_contact  = ik_contact['q']

    print("\n── 规划 pick → contact ──")
    wps_3 = planner.plan_to_pose(wps_2[-1], T_pick_contact, q_target=q_contact)

    input(f"\n  共 {len(wps_0)+len(wps_1)+len(wps_2)+len(wps_3)} waypoints，按 Enter 开始执行...")

    env.execute_trajectory(wps_0)
    env.execute_trajectory(wps_1)
    env.execute_trajectory(wps_2)
    env.execute_trajectory(wps_3)
    env._set_gripper(open=False)   # ← 新增：到达 contact 后关闭夹爪
    print("  夹爪已关闭")
    env._set_weld(active=True)
    print("  cup_weld 已激活")
    env._settle(50) 

    LIFT_DIST    = 0.10
    T_lift       = T_pick_contact.copy()
    T_lift[2, 3] += LIFT_DIST

    ik_lift  = ik.solve(T_lift, q_init=q_contact, n_restarts=10)
    q_lift   = ik_lift['q']

    print("\n── 规划 contact → lift ──")
    wps_lift = planner.plan_to_pose(wps_3[-1], T_lift, q_target=q_lift)

    input(f"\n  抬起段 {len(wps_lift)} waypoints，按 Enter 开始执行...")
    env.execute_trajectory(wps_lift, gripper_ctrl=GRIPPER_CLOSE)

    hand_id  = env.model.body("hand").id
    pos_actual = env.data.xpos[hand_id].copy()
    gz_actual  = env.data.xmat[hand_id].reshape(3, 3)[:, 2].copy()
    print(f"\n  ik_above['q'] : {np.round(ik_above['q'], 4)}")
    print(f"  ik_res['q']   : {np.round(ik_res['q'], 4)}")
    print(f"\n  [物理实际] pick 位置 : {np.round(pos_actual, 4)}")
    print(f"  [物理实际] pick gz   : {np.round(gz_actual,  3)}")
    print(f"  [目标]     pick 位置 : {np.round(T_pick[:3, 3], 4)}")
    print(f"  [目标]     pick gz   : {np.round(T_pick[:3, 2], 3)}")

    input("\n  观察轨迹，按 Enter 关闭...")
    env.close()


if __name__ == "__main__":
    main()