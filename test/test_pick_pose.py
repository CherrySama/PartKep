"""
test/test_pick_pose.py

第一步：site 实时关键点 → SAP → PoseSolver → T_pick 几何验证

不涉及：MotionPlanner
目标：确认 T_pick 的位置和旋转在数学上正确
    - 末端位置距 handle 约 FINGER_LENGTH（≈10.3cm）
    - 夹爪 Z 轴（接近方向）与 approach_direction 对齐
    - 夹爪 Y 轴（张开方向）与 grasp_axis 对齐

运行方式（从项目根目录）：
    python test/test_pick_pose.py
"""

import sys
import numpy as np
sys.path.insert(0, '.')

from modules.constraintsInst import ConstraintInstantiator, FINGER_LENGTH
from modules.poseSolver       import PoseSolver
from modules.vlmDecider       import VLMDecision
from scipy.spatial.transform  import Rotation as R

# Q_HOME FK 结果（IKSolver 初始化时打印过，此处直接硬编码供 T_current 用）
# hand body 在 Q_HOME 时的世界坐标：[0.5545, 0.0, 0.6245]
# 旋转：Q_HOME 时夹爪 Z ≈ [0, 0, -1]（手心朝下）
_p_home   = np.array([0.5545, 0.0, 0.6245])
_R_home   = R.from_rotvec([np.pi, 0, 0]).as_matrix()  # 夹爪Z朝下
T_current = np.eye(4)
T_current[:3, :3] = _R_home
T_current[:3,  3] = _p_home

SCENE_XML = "assets/franka_emika_panda/scene.xml"


def _read_cup_keypoints(model) -> dict:
    """
    从 model 结构读取杯子关键点的初始世界坐标。

    原理：
        cup body 父节点是 worldbody，故：
            model.body_pos[cup_id] = XML pos="0.5 0.05 0.452" = 杯子初始世界位置
            model.site_pos[sid]    = site 在杯子局部坐标系的偏移
        初始状态杯子无旋转，故：site 世界坐标 = body_pos + site_pos

    注意：仅在杯子处于初始位置时正确。
    实际 pipeline 中关键点来自视觉模块，与此函数无关。
    """
    import mujoco
    cup_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    cup_pos = model.body_pos[cup_id].copy()   # [0.5, 0.05, 0.452]

    site_map = {"handle": "kp_handle", "body": "kp_body", "rim": "kp_rim"}
    result = {}
    for key, site_name in site_map.items():
        sid         = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        local_pos   = model.site_pos[sid].copy()
        result[key] = cup_pos + local_pos
    return result


def print_geometry(T: np.ndarray, meta: dict):
    """打印 T_pick 的几何解读，便于人工判断"""
    pos       = T[:3, 3]
    gz        = T[:3, 2]   # 夹爪 Z 轴（接近方向）
    gy        = T[:3, 1]   # 夹爪 Y 轴（张开方向）
    kp        = meta['keypoint_3d']
    approach  = meta['approach_direction']
    grasp_ax  = meta.get('grasp_axis_target')

    print("\n" + "=" * 50)
    print("【几何检查】")
    print(f"  hand body 目标位置 : {np.round(pos, 4)}")
    print(f"  handle 关键点坐标  : {np.round(kp, 4)}")

    dist = np.linalg.norm(pos - kp)
    print(f"  hand→handle 距离   : {dist*100:.1f} cm  "
          f"（应约 {FINGER_LENGTH*100:.1f} cm）")

    dot_z = float(np.dot(gz, approach))
    print(f"\n  SAP approach 方向  : {np.round(approach, 3)}")
    print(f"  夹爪 Z 轴          : {np.round(gz, 3)}")
    print(f"  Z 轴对齐度         : {dot_z:.4f}  （应接近 +1.0）")

    if grasp_ax is not None:
        dot_y = float(np.dot(gy, grasp_ax))
        print(f"\n  SAP grasp_axis     : {np.round(grasp_ax, 3)}")
        print(f"  夹爪 Y 轴          : {np.round(gy, 3)}")
        print(f"  Y 轴对齐度         : {dot_y:.4f}  （应接近 +1.0）")

    print("=" * 50)


def main():
    print("=" * 50)
    print("Step 1: SAP → PoseSolver → T_pick 几何 + 静态可视化")
    print("=" * 50)

    # ── 0. 提前初始化 MuJoCo（site 读取依赖 mj_forward） ──────
    import mujoco
    import mujoco.viewer
    from modules.IKSolver import IKSolver, Q_HOME

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # ── 1. 从 site 实时读取关键点 ─────────────────────────────
    CUP_KEYPOINTS = _read_cup_keypoints(model)
    print(f"\n  kp_handle : {np.round(CUP_KEYPOINTS['handle'], 4)}")
    print(f"  kp_body   : {np.round(CUP_KEYPOINTS['body'],   4)}")
    print(f"  kp_rim    : {np.round(CUP_KEYPOINTS['rim'],    4)}")

    # ── 2. SAP → PoseSolver ──────────────────────────────────
    decision = VLMDecision(
        w_grasp_axis=1.0, w_safety=2.0,
        confidence=0.0, reasoning="hardcoded test", is_fallback=True
    )

    inst = ConstraintInstantiator(verbose=True)
    cost_fn, x0, meta = inst.instantiate(
        keypoints_3d=CUP_KEYPOINTS,
        vlm_decision=decision,
        T_current=T_current,
    )

    solver = PoseSolver(max_iter=200, tol=1e-6, verbose=True)
    result = solver.solve(cost_fn, x0, meta)
    T_pick = result['T']

    print(f"\n  PoseSolver 收敛: {'✅' if result['success'] else '⚠️  未收敛'}")
    print(f"  最终代价: {result['final_cost']:.6f}")
    print_geometry(T_pick, meta)

    # ── 3a. 诊断：仅位置 IK（排除旋转干扰，确认目标点可达性）──
    from scipy.optimize import minimize as _minimize
    from modules.IKSolver import PANDA_JOINT_LIMITS
    _diag_data = mujoco.MjData(model)
    _hand_id   = model.body("hand").id
    _p_target  = T_pick[:3, 3]
    _bounds    = [(lo, hi) for lo, hi in PANDA_JOINT_LIMITS]

    def _pos_only_cost(q7):
        _diag_data.qpos[:7] = q7
        mujoco.mj_forward(model, _diag_data)
        return float(np.sum((_diag_data.xpos[_hand_id] - _p_target) ** 2))

    print("\n── 仅位置 IK 诊断（忽略旋转）──")
    for label, q0 in [
        ("Q_HOME    ", Q_HOME),
        ("q_extended", np.array([0.12, 0.5, 0.0, -0.5, 0.0, 1.0, 0.0])),
    ]:
        _res = _minimize(_pos_only_cost, q0, method="L-BFGS-B", bounds=_bounds,
                         options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10})
        _diag_data.qpos[:7] = _res.x
        mujoco.mj_forward(model, _diag_data)
        _p_got = _diag_data.xpos[_hand_id].copy()
        _err   = np.linalg.norm(_p_got - _p_target) * 1000
        print(f"  [{label}] 误差: {_err:.2f} mm  到达: {np.round(_p_got, 4)}")
        print(f"            q : {np.round(_res.x, 4)}")

    # ── 3. IKSolver → q_pick ─────────────────────────────────
    ik     = IKSolver(model, verbose=True)
    ik_res = ik.solve(T_pick, q_init=Q_HOME, n_restarts=10)

    print(f"\n  IK 结果: {'✅ 成功' if ik_res['success'] else '⚠️  失败'}")
    print(f"  位置误差: {ik_res['position_error']*1000:.2f} mm")
    print(f"  q_pick  : {np.round(ik_res['q'], 4)}")

    if not ik_res['success']:
        print("  ⚠️  IK 未收敛，中止可视化")
        return

    # ── 4. MuJoCo 静态展示 ────────────────────────────────────
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance  = 1.5
    viewer.cam.azimuth   = 180
    viewer.cam.elevation = -20

    # 直接设置关节角，不做插值（静态验证）
    data.ctrl[:7] = ik_res['q']
    data.ctrl[7]  = 255.0         # 夹爪张开
    for _ in range(200):
        mujoco.mj_step(model, data)
    viewer.sync()

    # ── 旋转诊断 ──────────────────────────────────────────────
    T_achieved = ik.forward_kinematics(ik_res['q'])
    R_target   = T_pick[:3, :3]
    R_achieved = T_achieved[:3, :3]
    rot_err    = np.linalg.norm(R_achieved - R_target, 'fro')
    print(f"\n  旋转 Frobenius 误差: {rot_err:.4f}  （0=完美，最大≈5.66）")
    print(f"  目标 夹爪Z轴: {np.round(T_pick[:3, 2], 3)}")
    print(f"  实际 夹爪Z轴: {np.round(T_achieved[:3, 2], 3)}")
    print(f"  目标 夹爪Y轴: {np.round(T_pick[:3, 1], 3)}")
    print(f"  实际 夹爪Y轴: {np.round(T_achieved[:3, 1], 3)}")

    input("\n  👀 请目视检查：夹爪是否从侧面水平对准 handle？\n"
          "  确认后按 Enter 关闭...")
    viewer.close()
    print("✅ Step 1 完成")


if __name__ == "__main__":
    main()