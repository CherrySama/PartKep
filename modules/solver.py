"""
Created by Yinghao Ho on 2026-2-23

SLSQP 求解器封装模块

职责：
    接收 ConstraintInstantiator 输出的代价函数和初始猜测，
    调用 SLSQP 求解末端执行器最优位姿（6D → SE(3)）。

设计选择：
    - 软约束已全部编码进 cost_fn，求解器无需额外约束/边界
    - 返回结果包含位置、旋转矩阵、诊断信息，方便下游 IK 直接使用
    - 求解失败时不抛异常，而是在结果中标记 success=False，
      由调用方决定如何处理（重试/报警/降级）
"""

import numpy as np
from typing import Dict, Callable, Optional
from scipy.optimize import minimize, OptimizeResult


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    """
    轴角向量 → 3×3 旋转矩阵（Rodrigues 公式）

    与 constraint_instantiation.py 中的实现相同，
    此处独立保留以避免跨模块循环依赖。

    Args:
        rvec: shape=(3,) 轴角向量

    Returns:
        np.ndarray shape=(3, 3)
    """
    rvec = np.array(rvec, dtype=np.float64)
    theta = np.linalg.norm(rvec)

    if theta < 1e-10:
        return np.eye(3)

    k = rvec / theta
    K = np.array([
        [   0.0, -k[2],  k[1]],
        [ k[2],    0.0, -k[0]],
        [-k[1],  k[0],    0.0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


class PoseSolver:
    """
    末端执行器位姿求解器

    封装 SLSQP 优化，将 ConstraintInstantiator 输出的代价函数
    求解为最优末端执行器位姿。

    使用示例：
        >>> from modules.constraint_instantiation import ConstraintInstantiator
        >>> from modules.solver import PoseSolver
        >>>
        >>> instantiator = ConstraintInstantiator(object_label='cup')
        >>> cost_fn, x0, meta = instantiator.instantiate(keypoints_3d)
        >>>
        >>> solver = PoseSolver()
        >>> result = solver.solve(cost_fn, x0, meta)
        >>>
        >>> if result['success']:
        ...     print(result['position'])        # 末端执行器位置
        ...     print(result['rotation_matrix']) # 末端执行器朝向
    """

    def __init__(self,
                 max_iter: int = 200,
                 tol: float = 1e-6,
                 verbose: bool = True):
        """
        Args:
            max_iter: SLSQP 最大迭代次数
            tol: 收敛容差
            verbose: 是否打印求解过程和结果
        """
        self.max_iter = max_iter
        self.tol      = tol
        self.verbose  = verbose

    def solve(self,
              cost_fn: Callable[[np.ndarray], float],
              x0: np.ndarray,
              meta: Dict) -> Dict:
        """
        求解最优末端执行器位姿

        Args:
            cost_fn: 目标函数，输入 x shape=(6,)，返回标量代价
                     直接来自 ConstraintInstantiator.instantiate()
            x0: shape=(6,) 优化初始猜测
                直接来自 ConstraintInstantiator.instantiate()
            meta: 调试信息字典
                  直接来自 ConstraintInstantiator.instantiate()
                  必须包含 'cost_breakdown_fn' 键

        Returns:
            Dict，包含：
                'success':         bool，求解是否成功收敛
                'position':        np.ndarray shape=(3,)，末端执行器位置（米）
                'rotation_matrix': np.ndarray shape=(3,3)，末端执行器朝向
                'rvec':            np.ndarray shape=(3,)，轴角旋转向量
                'x_opt':           np.ndarray shape=(6,)，完整优化结果
                'final_cost':      float，最终代价值
                'cost_breakdown':  Dict，各项代价分解
                'message':         str，求解器状态信息
                'n_iter':          int，实际迭代次数
                'x0':              np.ndarray shape=(6,)，记录初始猜测（便于对比）
        """
        if self.verbose:
            print(f"\n🔧 SLSQP 求解开始")
            print(f"  初始代价: {cost_fn(x0):.6f}")
            print(f"  初始位置: {np.round(x0[:3], 4)}")
            print(f"  最大迭代: {self.max_iter}，容差: {self.tol}")

        # ===== 调用 SLSQP =====
        opt_result: OptimizeResult = minimize(
            fun=cost_fn,
            x0=x0,
            method='SLSQP',
            options={
                'maxiter': self.max_iter,
                'ftol':    self.tol,
                'disp':    False       # 不让 scipy 自己打印，由我们控制输出
            }
        )

        # ===== 提取结果 =====
        x_opt    = opt_result.x
        position = x_opt[:3]
        rvec     = x_opt[3:]
        R        = _rodrigues(rvec)

        final_cost     = float(opt_result.fun)
        cost_breakdown = meta['cost_breakdown_fn'](x_opt)

        result = {
            'success':         bool(opt_result.success),
            'position':        position.copy(),
            'rotation_matrix': R.copy(),
            'rvec':            rvec.copy(),
            'x_opt':           x_opt.copy(),
            'final_cost':      final_cost,
            'cost_breakdown':  cost_breakdown,
            'message':         opt_result.message,
            'n_iter':          int(opt_result.nit),
            'x0':              x0.copy()
        }

        # ===== 打印结果摘要 =====
        if self.verbose:
            status = "✅ 收敛" if result['success'] else "⚠️  未收敛"
            print(f"\n  {status}（{opt_result.message}）")
            print(f"  迭代次数:   {result['n_iter']}")
            print(f"  初始代价:   {cost_fn(x0):.6f}")
            print(f"  最终代价:   {final_cost:.6f}")
            print(f"  代价分解:")
            bd = cost_breakdown
            print(f"    approach: {bd['approach']:.6f}")
            print(f"    grasp:    {bd['grasp']:.6f}")
            print(f"    safety:   {bd['safety']:.6f}")
            if bd['safety_per_part']:
                for part, val in bd['safety_per_part'].items():
                    print(f"      └─ {part}: {val:.6f}")
            print(f"  末端位置:   {np.round(position, 4)}")
            print(f"  轴角旋转:   {np.round(rvec, 4)}")

        return result

    def solve_with_restart(self,
                           cost_fn: Callable[[np.ndarray], float],
                           x0: np.ndarray,
                           meta: Dict,
                           n_restarts: int = 3,
                           noise_scale: float = 0.05) -> Dict:
        """
        带随机重启的求解（应对局部最优）

        在初始猜测附近加入随机扰动，多次求解后取代价最低的结果。

        Args:
            cost_fn: 目标函数
            x0: 初始猜测
            meta: 调试信息
            n_restarts: 随机重启次数（加上原始 x0 共 n_restarts+1 次求解）
            noise_scale: 位置扰动幅度（米），旋转扰动固定为 noise_scale * 2

        Returns:
            代价最低的求解结果（格式与 solve() 相同）
        """
        if self.verbose:
            print(f"\n🔄 带重启的求解（{n_restarts} 次重启）")

        best_result = None
        verbose_orig = self.verbose

        # 第一次用原始 x0
        candidates = [x0.copy()]

        # 后续用扰动后的 x0
        rng = np.random.default_rng(seed=42)
        for _ in range(n_restarts):
            x0_noisy = x0.copy()
            x0_noisy[:3] += rng.normal(0, noise_scale, size=3)
            x0_noisy[3:] += rng.normal(0, noise_scale * 2, size=3)
            candidates.append(x0_noisy)

        # 静默跑所有候选，最后只打印最优结果
        self.verbose = False
        for i, x0_candidate in enumerate(candidates):
            result = self.solve(cost_fn, x0_candidate, meta)
            if best_result is None or result['final_cost'] < best_result['final_cost']:
                best_result = result

        self.verbose = verbose_orig

        if self.verbose:
            print(f"  最优结果（{n_restarts+1} 次求解中代价最低）:")
            print(f"  最终代价: {best_result['final_cost']:.6f}")
            print(f"  末端位置: {np.round(best_result['position'], 4)}")
            print(f"  收敛状态: {'✅' if best_result['success'] else '⚠️ '} "
                  f"{best_result['message']}")

        return best_result


# ==================== 模块测试代码 ====================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    # 需要从上层目录导入
    from configs.sap_knowledge import get_sap_strict
    from constraint_instantiation import ConstraintInstantiator

    print("=" * 70)
    print("测试 PoseSolver")
    print("=" * 70)

    # ---------- 公共：构造代价函数 ----------
    def make_cup_problem():
        keypoints_3d = {
            'handle': np.array([0.08,  0.00, 0.06]),
            'rim':    np.array([0.00,  0.00, 0.10]),
            'body':   np.array([0.00,  0.00, 0.05]),
        }
        inst = ConstraintInstantiator(object_label='cup')
        return inst.instantiate(keypoints_3d)

    # ---------- 测试1：基本求解 ----------
    print("\n【测试1】基本求解（cup handle）")
    print("-" * 70)

    cost_fn, x0, meta = make_cup_problem()
    solver = PoseSolver(max_iter=200, tol=1e-6, verbose=True)
    result = solver.solve(cost_fn, x0, meta)

    # 验证返回字段完整性
    required_keys = ['success', 'position', 'rotation_matrix', 'rvec',
                     'x_opt', 'final_cost', 'cost_breakdown',
                     'message', 'n_iter', 'x0']
    for key in required_keys:
        assert key in result, f"缺少字段: {key}"
    print(f"\n  ✅ 返回字段完整性验证通过")

    # 验证旋转矩阵正交性
    R = result['rotation_matrix']
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6), "旋转矩阵不正交！"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6), "行列式不为1！"
    print(f"  ✅ 旋转矩阵正交性验证通过")

    # 验证最终代价低于初始代价
    initial_cost = cost_fn(x0)
    assert result['final_cost'] < initial_cost, "优化后代价应低于初始代价！"
    print(f"  ✅ 代价下降: {initial_cost:.4f} → {result['final_cost']:.6f}")

    # ---------- 测试2：bowl（含安全约束）----------
    print("\n【测试2】bowl 求解（含 rim 安全约束）")
    print("-" * 70)

    keypoints_bowl = {
        'rim':  np.array([0.00, 0.00, 0.08]),
        'body': np.array([0.00, 0.00, 0.04]),
    }
    inst_bowl = ConstraintInstantiator(object_label='bowl')
    cost_fn_b, x0_b, meta_b = inst_bowl.instantiate(keypoints_bowl)

    solver_bowl = PoseSolver(verbose=True)
    result_b = solver_bowl.solve(cost_fn_b, x0_b, meta_b)

    # 求解后末端位置应远离 rim（安全距离 0.025m）
    dist_to_rim = np.linalg.norm(
        result_b['position'] - keypoints_bowl['rim']
    )
    print(f"\n  求解后末端到 rim 的距离: {dist_to_rim:.4f}m")
    if dist_to_rim >= 0.025:
        print(f"  ✅ 安全距离满足（>= 0.025m）")
    else:
        print(f"  ⚠️  安全距离未完全满足（可能需要调整权重）")

    # ---------- 测试3：带重启求解 ----------
    print("\n【测试3】带随机重启求解（cup）")
    print("-" * 70)

    cost_fn_r, x0_r, meta_r = make_cup_problem()
    solver_r = PoseSolver(verbose=True)
    result_r = solver_r.solve_with_restart(
        cost_fn_r, x0_r, meta_r, n_restarts=3
    )

    assert result_r['final_cost'] <= cost_fn_r(x0_r), \
        "重启后代价应不高于初始代价！"
    print(f"\n  ✅ 重启求解完成，最终代价={result_r['final_cost']:.6f}")

    # ---------- 测试4：验证 x0 被记录 ----------
    print("\n【测试4】x0 记录验证")
    print("-" * 70)
    cost_fn_t, x0_t, meta_t = make_cup_problem()
    result_t = PoseSolver(verbose=False).solve(cost_fn_t, x0_t, meta_t)
    assert np.allclose(result_t['x0'], x0_t)
    print(f"  ✅ result['x0'] 与传入 x0 一致")

    print()
    print("=" * 70)
    print("✅ PoseSolver 所有测试通过！")
    print("=" * 70)