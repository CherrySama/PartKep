"""
Created by Yinghao Ho on 2026-2-23

SLSQP 求解器封装模块
职责：
    接收 ConstraintInstantiator 输出的代价函数和初始猜测，
    调用 SLSQP 求解末端执行器最优位姿（6D → position + R → SE3）。

设计选择：
    - 软约束已全部编码进 cost_fn，求解器无需额外约束/边界
    - 求解失败时不抛异常，在结果中标记 success=False，由调用方决定处理策略
    - se3 字段在 success=False 时依然填充，方便调试查看当前最优估计
"""

import numpy as np
from typing import Dict, Callable
from scipy.optimize import minimize, OptimizeResult

from utils import CoordinateTransformer   # ← rodrigues / rotation_to_se3 统一来源


class PoseSolver:
    """
    末端执行器位姿求解器

    封装 SLSQP 优化，将 ConstraintInstantiator 输出的代价函数
    求解为最优末端执行器位姿，并直接输出 SE3 供 IKSolver 使用。

    使用示例：
        >>> from modules.constraintsInst import ConstraintInstantiator
        >>> from modules.solver import PoseSolver
        >>>
        >>> inst   = ConstraintInstantiator(object_label='cup')
        >>> cost_fn, x0, meta = inst.instantiate(keypoints_3d)
        >>>
        >>> solver = PoseSolver()
        >>> result = solver.solve(cost_fn, x0, meta)
        >>>
        >>> if result['success']:
        ...     print(result['position'])   # 末端执行器位置
        ...     print(result['se3'])        # 直接传给 IKSolver
    """

    def __init__(self,
                 max_iter: int = 200,
                 tol: float = 1e-6,
                 verbose: bool = True):
        """
        Args:
            max_iter: SLSQP 最大迭代次数
            tol:      收敛容差
            verbose:  是否打印求解过程和结果
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
            x0:      shape=(6,) 优化初始猜测
                     直接来自 ConstraintInstantiator.instantiate()
            meta:    调试信息字典，必须包含 'cost_breakdown_fn' 键
                     直接来自 ConstraintInstantiator.instantiate()

        Returns:
            Dict，包含：
                'success':         bool，是否成功收敛
                'position':        np.ndarray shape=(3,)，末端执行器位置（米）
                'rotation_matrix': np.ndarray shape=(3,3)，末端执行器朝向
                'rvec':            np.ndarray shape=(3,)，轴角旋转向量
                'se3':             spatialmath.SE3，供 IKSolver 直接使用  ← 新增
                'x_opt':           np.ndarray shape=(6,)，完整优化结果
                'final_cost':      float，最终代价值
                'cost_breakdown':  Dict，各项代价分解
                'message':         str，求解器状态信息
                'n_iter':          int，实际迭代次数
                'x0':              np.ndarray shape=(6,)，初始猜测（对比用）
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
                'disp':    False
            }
        )

        # ===== 提取结果 =====
        x_opt    = opt_result.x
        position = x_opt[:3]
        rvec     = x_opt[3:]

        R = CoordinateTransformer.rodrigues(rvec)

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
            if bd.get('safety_per_part'):
                for part, val in bd['safety_per_part'].items():
                    print(f"      └─ {part}: {val:.6f}")
            print(f"  末端位置:   {np.round(position, 4)}")
            print(f"  轴角旋转:   {np.round(rvec, 4)}")
            print(f"  T[:3,3]:    {np.round(T[:3, 3], 4)}")

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
            cost_fn:     目标函数
            x0:          初始猜测
            meta:        调试信息
            n_restarts:  随机重启次数（加上原始 x0 共 n_restarts+1 次求解）
            noise_scale: 位置扰动幅度（米），旋转扰动为 noise_scale*2

        Returns:
            代价最低的求解结果（格式与 solve() 相同）
        """
        if self.verbose:
            print(f"\n🔄 带重启的求解（{n_restarts} 次重启）")

        best_result  = None
        verbose_orig = self.verbose

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
            print(f"  最优结果（{n_restarts+1} 次求解中代价最低）:")
            print(f"  最终代价: {best_result['final_cost']:.6f}")
            print(f"  末端位置: {np.round(best_result['position'], 4)}")
            print(f"  收敛状态: {'✅' if best_result['success'] else '⚠️ '} "
                  f"{best_result['message']}")

        return best_result


# ==================== 模块测试 ====================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from modules.constraintsInst import ConstraintInstantiator

    print("=" * 70)
    print("测试 PoseSolver")
    print("=" * 70)

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
    cost_fn, x0, meta = make_cup_problem()
    solver = PoseSolver(max_iter=200, tol=1e-6, verbose=True)
    result = solver.solve(cost_fn, x0, meta)

    required_keys = ['success', 'position', 'rotation_matrix', 'rvec',
                     'T', 'x_opt', 'final_cost', 'cost_breakdown',
                     'message', 'n_iter', 'x0']
    for key in required_keys:
        assert key in result, f"返回值缺少字段: {key}"
    print(f"\n  ✅ 所有字段存在（含 'T'）")
    print(f"  ✅ success={result['success']}, cost={result['final_cost']:.6f}")

    # 验证 rotation_matrix 正交性
    R = result['rotation_matrix']
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)
    print(f"  ✅ rotation_matrix 正交性验证通过")

    # 验证 T 的平移和旋转与 position/rotation_matrix 一致
    T = result['T']
    assert np.allclose(T[:3, 3],   result['position'],        atol=1e-10)
    assert np.allclose(T[:3, :3],  result['rotation_matrix'], atol=1e-10)
    assert np.allclose(T[3, :],    [0, 0, 0, 1],              atol=1e-10)
    print(f"  ✅ T[:3,3] == position，T[:3,:3] == rotation_matrix")

    # ---------- 测试2：带重启求解 ----------
    print("\n【测试2】带重启求解")
    result_restart = solver.solve_with_restart(cost_fn, x0, meta, n_restarts=3)
    assert result_restart['final_cost'] <= result['final_cost'] + 1e-6
    print(f"  ✅ 带重启代价 ≤ 单次代价: "
          f"{result_restart['final_cost']:.6f} ≤ {result['final_cost']:.6f}")

    print("\n" + "=" * 70)
    print("✅ PoseSolver 所有测试通过！")
    print("=" * 70)
    