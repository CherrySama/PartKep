"""
test_step5_pipeline.py
PartKep — Step 5 完整链路计时测试

测试链路：
    模拟关键点 (keypoints_3d)
        → ConstraintInstantiator.instantiate()   Stage A
        → PoseSolver.solve()                     Stage B
        → IKSolver.solve()                       Stage C

测试物体：cup / bottle / bowl（每种各一组模拟 3D 关键点）

运行方式：
    cd /workspace/PartKep
    python test_step5_pipeline.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, '.')

# ==================== 颜色辅助（终端输出） ====================
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def info(msg): print(f"  {CYAN}   {msg}{RESET}")


# ==================== 模拟关键点定义 ====================
# 坐标单位：米，世界坐标系
# 物体放置在机器人正前方桌面上（x≈0.45，z≈桌高+物体高度）

SIMULATED_SCENES = {
    "cup": {
        # 杯子：handle 在右侧（+Y 方向偏移），典型侧向抓取
        "object_label": "cup",
        "keypoints_3d": {
            "handle": np.array([0.45,  0.07,  0.10]),  # 把手，偏右
            "rim":    np.array([0.45,  0.00,  0.16]),  # 杯口，正上方
            "body":   np.array([0.45,  0.00,  0.08]),  # 杯身，中央
        },
        "description": "桌上杯子，handle 朝右，侧向抓取"
    },

    "bottle": {
        # 瓶子：neck 是最优抓取点，cap 在顶部，body 最宽
        "object_label": "bottle",
        "keypoints_3d": {
            "neck":   np.array([0.50,  0.00,  0.20]),  # 瓶颈，细处
            "cap":    np.array([0.50,  0.00,  0.26]),  # 瓶盖，顶部
            "body":   np.array([0.50,  0.00,  0.10]),  # 瓶身，下部
        },
        "description": "桌上瓶子，neck 为主抓取目标"
    },

    "bowl": {
        # 碗：无 handle，body 为唯一可抓取部件，rim 需回避
        "object_label": "bowl",
        "keypoints_3d": {
            "rim":    np.array([0.40,  0.00,  0.08]),  # 碗口，avoid
            "body":   np.array([0.40,  0.00,  0.04]),  # 碗身，grasp
        },
        "description": "桌上碗，body 从上方抓取，rim 需回避"
    },
}


# ==================== 单场景测试函数 ====================

def run_scene(scene_name: str, scene: dict,
              inst_cls, pose_cls, ik_cls, q_home, 
              fallback_decision=None) -> dict:
    """
    对单个物体场景运行完整 Step 5 链路并计时

    Returns:
        result_summary: dict，包含各阶段耗时和最终状态
    """
    label       = scene["object_label"]
    kps         = scene["keypoints_3d"]
    description = scene["description"]

    print(f"\n{'='*65}")
    print(f"{BOLD}  场景：{scene_name.upper()}  —  {description}{RESET}")
    print(f"{'='*65}")
    info(f"输入关键点：")
    for pname, coord in kps.items():
        info(f"  {pname:8s}: {np.round(coord, 4)}")

    summary = {
        "scene":          scene_name,
        "t_instantiate":  None,
        "t_pose_solver":  None,
        "t_ik_solver":    None,
        "t_total":        None,
        "pose_success":   False,
        "ik_success":     False,
        "grasp_target":   None,
        "ee_position":    None,
        "joint_angles":   None,
        "pos_error_mm":   None,
        "within_limits":  False,
        "cost_breakdown": None,
    }

    t_start_total = time.perf_counter()

    # ────────────────────────────────────────────
    # Stage A：ConstraintInstantiator
    # ────────────────────────────────────────────
    print(f"\n  {CYAN}[ Stage A ]  ConstraintInstantiator{RESET}")
    t0 = time.perf_counter()
    try:
        instantiator      = inst_cls(verbose=False)
        cost_fn, x0, meta = instantiator.instantiate(kps, fallback_decision)
        t_inst          = time.perf_counter() - t0
        summary["t_instantiate"] = t_inst
        summary["grasp_target"]  = meta["grasp_target"]
        ok(f"代价函数实例化完成  [{t_inst*1000:.2f} ms]")
        info(f"抓取目标部件 : {meta['grasp_target']}")
        info(f"回避目标部件 : {meta['avoid_targets']}")
        info(f"接近方向     : {np.round(meta['approach_direction'], 3)}")
        info(f"初始代价 x0  : {cost_fn(x0):.6f}")
    except Exception as e:
        t_inst = time.perf_counter() - t0
        fail(f"Stage A 失败 [{t_inst*1000:.2f} ms]: {e}")
        summary["t_instantiate"] = t_inst
        return summary

    # ────────────────────────────────────────────
    # Stage B：PoseSolver（SLSQP）
    # ────────────────────────────────────────────
    print(f"\n  {CYAN}[ Stage B ]  PoseSolver  (SLSQP){RESET}")
    t0 = time.perf_counter()
    try:
        pose_solver = pose_cls(verbose=False)
        pose_result = pose_solver.solve(cost_fn, x0, meta)
        t_pose      = time.perf_counter() - t0
        summary["t_pose_solver"]  = t_pose
        summary["pose_success"]   = pose_result["success"]
        summary["ee_position"]    = pose_result["position"]
        summary["cost_breakdown"] = pose_result["cost_breakdown"]

        status_str = "收敛 ✅" if pose_result["success"] else "未收敛 ⚠️"
        print(f"    状态        : {status_str}  [{t_pose*1000:.2f} ms]")
        info(f"迭代次数     : {pose_result['n_iter']}")
        info(f"最终代价     : {pose_result['final_cost']:.6f}")
        bd = pose_result["cost_breakdown"]
        info(f"代价分解     :")
        info(f"  approach = {bd['approach']:.6f}")
        info(f"  grasp    = {bd['grasp']:.6f}")
        info(f"  safety   = {bd['safety']:.6f}")
        info(f"末端位置(m)  : {np.round(pose_result['position'], 4)}")
    except Exception as e:
        t_pose = time.perf_counter() - t0
        fail(f"Stage B 失败 [{t_pose*1000:.2f} ms]: {e}")
        summary["t_pose_solver"] = t_pose
        return summary

    # ────────────────────────────────────────────
    # Stage C：IKSolver
    # ────────────────────────────────────────────
    print(f"\n  {CYAN}[ Stage C ]  IKSolver  (ikpy){RESET}")
    t0 = time.perf_counter()
    try:
        ik_solver  = ik_cls(verbose=False)
        ik_result  = ik_solver.solve(pose_result["T"], q_init=q_home)
        t_ik       = time.perf_counter() - t0
        summary["t_ik_solver"]   = t_ik
        summary["ik_success"]    = ik_result["success"]
        summary["joint_angles"]  = ik_result["q"]
        summary["pos_error_mm"]  = ik_result["position_error"] * 1000
        summary["within_limits"] = ik_result["within_limits"]

        status_str = "成功 ✅" if ik_result["success"] else "失败 ⚠️"
        print(f"    状态        : {status_str}  [{t_ik*1000:.2f} ms]")
        info(f"位置误差     : {ik_result['position_error']*1000:.3f} mm")
        info(f"关节限位内   : {ik_result['within_limits']}")
        info(f"关节角 (rad) : {np.round(ik_result['q'], 4)}")
    except Exception as e:
        t_ik = time.perf_counter() - t0
        fail(f"Stage C 失败 [{t_ik*1000:.2f} ms]: {e}")
        summary["t_ik_solver"] = t_ik
        return summary

    summary["t_total"] = time.perf_counter() - t_start_total
    return summary


# ==================== 汇总打印 ====================

def print_summary(results: list):
    print(f"\n\n{'='*65}")
    print(f"{BOLD}  Step 5 Pipeline 测试汇总{RESET}")
    print(f"{'='*65}")

    header = f"  {'场景':<8} {'Stage A':>10} {'Stage B':>10} {'Stage C':>10} {'总计':>10}  {'结果':<10}"
    print(header)
    print(f"  {'-'*62}")

    all_pass = True
    for r in results:
        t_a   = f"{r['t_instantiate']*1000:.1f}ms" if r['t_instantiate'] else "—"
        t_b   = f"{r['t_pose_solver']*1000:.1f}ms"  if r['t_pose_solver']  else "—"
        t_c   = f"{r['t_ik_solver']*1000:.1f}ms"    if r['t_ik_solver']    else "—"
        t_tot = f"{r['t_total']*1000:.1f}ms"         if r['t_total']        else "—"

        passed = r["pose_success"] and r["ik_success"]
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        if not passed:
            all_pass = False

        print(f"  {r['scene']:<8} {t_a:>10} {t_b:>10} {t_c:>10} {t_tot:>10}  {status}")

    print(f"  {'-'*62}")

    # 位置误差汇总
    print(f"\n  IK 位置误差（FK 验证）：")
    for r in results:
        if r["pos_error_mm"] is not None:
            err_str = f"{r['pos_error_mm']:.3f} mm"
            within  = "限位内 ✅" if r["within_limits"] else "超限位 ⚠️"
            print(f"    {r['scene']:<8}: {err_str}  {within}")

    print()
    if all_pass:
        print(f"  {GREEN}{BOLD}全部通过 ✅{RESET}")
    else:
        print(f"  {RED}{BOLD}存在失败项 ❌{RESET}")
    print(f"{'='*65}\n")


# ==================== 主入口 ====================

def main():
    print(f"\n{BOLD}{'='*65}")
    print(f"  PartKep  —  Step 5 Pipeline 完整链路测试")
    print(f"{'='*65}{RESET}")

    # ----- 导入模块 -----
    print("\n[导入模块]")
    try:
        from modules.constraintsInst import ConstraintInstantiator
        from modules.poseSolver      import PoseSolver
        from modules.IKSolver        import IKSolver, Q_HOME
        from modules.vlmDecider      import FALLBACK_DECISION   # ← 新增
        ok("模块导入成功")
    except ImportError as e:
        fail(f"模块导入失败: {e}")
        sys.exit(1)

    # ----- 逐场景运行 -----
    results = []
    for scene_name, scene in SIMULATED_SCENES.items():
        result = run_scene(
            scene_name=scene_name,
            scene=scene,
            inst_cls=ConstraintInstantiator,
            pose_cls=PoseSolver,
            ik_cls=IKSolver,
            q_home=Q_HOME,
            fallback_decision=FALLBACK_DECISION
        )
        results.append(result)

    # ----- 汇总 -----
    print_summary(results)


if __name__ == "__main__":
    main()
    