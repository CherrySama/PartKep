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
    for name, pt in kps.items():
        print(f"  {name:8s}: {np.round(pt, 4)}")

    # temporary: force body-only to test IK feasibility
    # kps.pop('handle')

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


if __name__ == "__main__":
    main()