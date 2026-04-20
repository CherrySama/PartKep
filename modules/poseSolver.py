"""
Created by Yinghao Ho on 2026-2-23

SLSQP 求解器封装模块
职责：
    接收 ConstraintInstantiator 输出的代价函数和初始猜测，
    调用 SLSQP 求解末端执行器最优位姿（6D → position + R → 4×4 T）。

设计选择：
    - 软约束已全部编码进 cost_fn，求解器无需额外约束/边界
    - x0 来自 ConstraintInstantiator（由 q_current → FK 初始化），
      PoseSolver 直接使用，不再覆盖
    - 求解失败时不抛异常，在结果中标记 success=False，由调用方决定处理策略
"""

import numpy as np
from typing import Dict, Callable
from scipy.optimize import minimize, OptimizeResult

from utils import CoordinateTransformer


class PoseSolver:
    """
    末端执行器位姿求解器

    封装 SLSQP 优化，将 ConstraintInstantiator 输出的代价函数
    求解为最优末端执行器位姿，输出 4×4 齐次变换矩阵 T 供 IKSolver 使用。

    使用示例：
        >>> inst    = ConstraintInstantiator()
        >>> cost_fn, x0, meta = inst.instantiate(keypoints_3d, vlm_decision)
        >>>
        >>> solver  = PoseSolver()
        >>> result  = solver.solve(cost_fn, x0, meta)
        >>>
        >>> if result['success']:
        ...     print(result['position'])   # 末端位置
        ...     print(result['T'])          # 4×4 齐次变换矩阵，传给 IKSolver
    """

    def __init__(self,
                 max_iter: int   = 200,
                 tol:      float = 1e-6,
                 verbose:  bool  = True):
        """
        Args:
            max_iter : SLSQP 最大迭代次数
            tol      : 收敛容差
            verbose  : 是否打印求解过程和结果
        """
        self.max_iter = max_iter
        self.tol      = tol
        self.verbose  = verbose

    def solve(self,
              cost_fn: Callable[[np.ndarray], float],
              x0:      np.ndarray,
              meta:    Dict) -> Dict:
        """
        求解最优末端执行器位姿

        Args:
            cost_fn : 目标函数，输入 x shape=(6,)，返回标量代价
                      直接来自 ConstraintInstantiator.instantiate()
            x0      : shape=(6,) 优化初始猜测
                      直接来自 ConstraintInstantiator.instantiate()
                      由 q_current → FK 初始化，不会被覆盖
            meta    : 调试信息字典，必须包含 'cost_breakdown_fn' 键
                      直接来自 ConstraintInstantiator.instantiate()

        Returns:
            Dict，包含：
                'success'         : bool，是否成功收敛
                'position'        : np.ndarray shape=(3,)，末端位置（米）
                'rotation_matrix' : np.ndarray shape=(3,3)，末端旋转矩阵
                'rvec'            : np.ndarray shape=(3,)，轴角旋转向量
                'T'               : np.ndarray shape=(4,4)，齐次变换矩阵
                'x_opt'           : np.ndarray shape=(6,)，完整优化结果
                'final_cost'      : float，最终代价值
                'cost_breakdown'  : Dict，五约束各项代价分解
                'message'         : str，求解器状态信息
                'n_iter'          : int，实际迭代次数
                'x0'              : np.ndarray shape=(6,)，初始猜测（对比用）
        """
        if self.verbose:
            print(f"\n🔧 SLSQP 精细求解开始")
            print(f"  初始代价: {cost_fn(x0):.6f}")
            print(f"  初始位置: {np.round(x0[:3], 4)}")
            print(f"  最大迭代: {self.max_iter}，容差: {self.tol}")

        # ── 调用 SLSQP ──
        opt_result: OptimizeResult = minimize(
            fun=cost_fn,
            x0=x0,
            method='SLSQP',
            options={
                'maxiter': self.max_iter,
                'ftol':    self.tol,
                'disp':    False,
            }
        )

        # ── 提取结果 ──
        x_opt    = opt_result.x
        position = x_opt[:3]
        rvec     = x_opt[3:]
        R        = CoordinateTransformer.rodrigues(rvec)

        final_cost     = float(opt_result.fun)
        cost_breakdown = meta['cost_breakdown_fn'](x_opt)

        # 4×4 齐次变换矩阵，IKSolver 直接使用
        T = CoordinateTransformer.rotation_to_matrix4x4(position, R)

        result = {
            'success':         bool(opt_result.success),
            'position':        position.copy(),
            'rotation_matrix': R.copy(),
            'rvec':            rvec.copy(),
            'T':               T,
            'x_opt':           x_opt.copy(),
            'final_cost':      final_cost,
            'cost_breakdown':  cost_breakdown,
            'message':         opt_result.message,
            'n_iter':          int(opt_result.nit),
            'x0':              x0.copy(),
        }

        # ── 打印结果摘要 ──
        if self.verbose:
            status = "✅ 收敛" if result['success'] else "⚠️  未收敛"
            print(f"\n  {status}（{opt_result.message}）")
            print(f"  迭代次数:   {result['n_iter']}")
            print(f"  初始代价:   {cost_fn(x0):.6f}")
            print(f"  最终代价:   {final_cost:.6f}")
            print(f"  代价分解:")
            bd = cost_breakdown
            print(f"    C1 approach_pos : {bd.get('approach_pos', 0.0):.6f}")
            print(f"    C2 approach_rot : {bd.get('approach_rot', 0.0):.6f}")
            print(f"    C3 grasp_axis   : {bd.get('grasp_axis',   0.0):.6f}")
            print(f"    C4 safety       : {bd.get('safety',       0.0):.6f}")
            print(f"    C5 flip         : {bd.get('flip',         0.0):.6f}")
            if bd.get('safety_per_part'):
                for part, val in bd['safety_per_part'].items():
                    print(f"      └─ {part}: {val:.6f}")
            print(f"  末端位置:   {np.round(position, 4)}")
            print(f"  轴角旋转:   {np.round(rvec, 4)}")
            print(f"  T[:3,3]:    {np.round(T[:3, 3], 4)}")

        return result

    def solve_with_restart(self,
                           cost_fn:     Callable[[np.ndarray], float],
                           x0:          np.ndarray,
                           meta:        Dict,
                           n_restarts:  int   = 3,
                           noise_scale: float = 0.05) -> Dict:
        """
        带随机重启的求解（应对局部最优）

        在初始猜测附近加入随机扰动，多次求解后取代价最低的结果。

        Args:
            cost_fn     : 目标函数
            x0          : 初始猜测（来自 ConstraintInstantiator）
            meta        : 调试信息
            n_restarts  : 随机重启次数（加上原始 x0 共 n_restarts+1 次）
            noise_scale : 位置扰动幅度（米），旋转扰动为 noise_scale×2

        Returns:
            代价最低的求解结果（格式与 solve() 相同）
        """
        if self.verbose:
            print(f"\n🔄 带重启的求解（{n_restarts} 次重启）")

        best_result  = None
        verbose_orig = self.verbose

        # 构建候选初始点列表
        candidates = [x0.copy()]
        rng = np.random.default_rng(seed=42)
        for _ in range(n_restarts):
            x0_noisy = x0.copy()
            x0_noisy[:3] += rng.normal(0, noise_scale,     size=3)
            x0_noisy[3:] += rng.normal(0, noise_scale * 2, size=3)
            candidates.append(x0_noisy)

        self.verbose = False
        for x0_candidate in candidates:
            result = self.solve(cost_fn, x0_candidate, meta)
            if best_result is None or result['final_cost'] < best_result['final_cost']:
                best_result = result
        self.verbose = verbose_orig

        if self.verbose:
            print(f"  最优结果（{n_restarts + 1} 次求解中代价最低）:")
            print(f"  最终代价: {best_result['final_cost']:.6f}")
            print(f"  末端位置: {np.round(best_result['position'], 4)}")
            print(f"  收敛状态: {'✅' if best_result['success'] else '⚠️ '} "
                  f"{best_result['message']}")

        return best_result


# ==================== 模块测试 ====================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from modules.vlmDecider    import FALLBACK_DECISION
    from modules.constraintsInst import ConstraintInstantiator

    print("=" * 70)
    print("测试 PoseSolver（重构版）")
    print("=" * 70)

    def make_cup_problem():
        keypoints_3d = {
            'handle': np.array([0.45,  0.07,  0.10]),
            'rim':    np.array([0.45,  0.00,  0.16]),
            'body':   np.array([0.45,  0.00,  0.08]),
        }
        inst = ConstraintInstantiator(verbose=False)
        return inst.instantiate(keypoints_3d, FALLBACK_DECISION)

    # ── 【1】基本求解 ──
    print("\n【1】基本求解（cup pick 模式）")
    cost_fn, x0, meta = make_cup_problem()
    solver = PoseSolver(max_iter=200, tol=1e-6, verbose=True)
    result = solver.solve(cost_fn, x0, meta)

    required_keys = ['success', 'position', 'rotation_matrix', 'rvec',
                     'T', 'x_opt', 'final_cost', 'cost_breakdown',
                     'message', 'n_iter', 'x0']
    for key in required_keys:
        assert key in result, f"返回值缺少字段: {key}"
    print(f"\n  ✅ 所有字段存在")
    print(f"  ✅ success={result['success']}, cost={result['final_cost']:.6f}")

    # ── 【2】旋转矩阵正交性 ──
    print("\n【2】旋转矩阵正交性验证")
    R = result['rotation_matrix']
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6), "R 不正交！"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6), "det(R) ≠ 1！"
    print("  ✅ R^T R = I，det(R) = 1")

    # ── 【3】T 矩阵一致性 ──
    print("\n【3】T 矩阵一致性验证")
    T = result['T']
    assert np.allclose(T[:3, 3],  result['position'],        atol=1e-10)
    assert np.allclose(T[:3, :3], result['rotation_matrix'], atol=1e-10)
    assert np.allclose(T[3, :],   [0, 0, 0, 1],              atol=1e-10)
    print("  ✅ T[:3,3] == position，T[:3,:3] == rotation_matrix")

    # ── 【4】代价分解字段验证 ──
    print("\n【4】代价分解字段验证（五约束）")
    bd = result['cost_breakdown']
    for field in ['approach_pos', 'approach_rot', 'grasp_axis', 'safety', 'flip']:
        assert field in bd, f"cost_breakdown 缺少字段: {field}"
        print(f"  {field:15s}: {bd[field]:.6f} ✅")

    # ── 【5】代价分解总和一致性 ──
    print("\n【5】代价分解总和一致性")
    assert abs(bd['total'] - result['final_cost']) < 1e-6, \
        f"breakdown total {bd['total']:.6f} ≠ final_cost {result['final_cost']:.6f}"
    print(f"  ✅ breakdown total = final_cost = {result['final_cost']:.6f}")

    # ── 【6】带重启求解 ──
    print("\n【6】带重启求解")
    result_r = solver.solve_with_restart(cost_fn, x0, meta, n_restarts=3)
    assert result_r['final_cost'] <= result['final_cost'] + 1e-5
    print(f"  ✅ 带重启代价 {result_r['final_cost']:.6f} ≤ "
          f"单次代价 {result['final_cost']:.6f}")

    # ── 【7】place 模式 ──
    print("\n【7】place 模式（tray surface）")
    from modules.vlmDecider import VLMDecision
    place_decision = VLMDecision(
        w_grasp_axis=0.0, w_safety=2.0, w_flip=1.5,
        confidence=0.0, reasoning="fallback", is_fallback=True
    )
    inst_t = ConstraintInstantiator(verbose=False)
    cost_fn_t, x0_t, meta_t = inst_t.instantiate(
        keypoints_3d = {
            'surface': np.array([0.60, 0.10, 0.02]),
            'rim':     np.array([0.60, 0.10, 0.05]),
        },
        vlm_decision = place_decision,
    )
    result_t = solver.solve(cost_fn_t, x0_t, meta_t)
    assert result_t['cost_breakdown']['grasp_axis'] == 0.0  # place 无 C3
    print(f"  ✅ place 模式求解成功，C3=0，"
          f"final_cost={result_t['final_cost']:.6f}")

    print("\n" + "=" * 70)
    print("✅ PoseSolver 所有测试通过！")
    print("=" * 70)
    