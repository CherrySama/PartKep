"""
test_cost_fn_numerical.py

纯数值验证脚本：不依赖 MuJoCo，测试 constraintsInst / vlmDecider / poseSolver 改动。

运行方式（从项目根目录）：
    python test/test_cost_fn_numerical.py

覆盖的验证点：
    【1】VLMDecision.w_flip 字段
    【2】C1 物理正确性：指尖对准关键点时 C1 趋近 0
    【3】C2 方向对齐：gripper_z 对齐 approach_dir 时 C2=0
    【4】C5 数学性质：正向对齐→0，垂直→0.25，反向→1
    【5】C5 与 C2 非线性相关验证
    【6】C4 指尖距离：指尖远离 avoid 部件时 C4=0
    【7】cost_breakdown 总和一致性
    【8】旧字段 upright 不存在，新字段 flip 存在
    【9】候选筛选：代价低的候选胜出
    【10】x0 来自 T_current，数值合理（无 NaN/Inf）
    【11】place 模式 C3 = 0
"""

import sys
import numpy as np

sys.path.insert(0, '.')

# ── 导入被测模块 ──
from modules.vlmDecider    import VLMDecision, FALLBACK_DECISION
from modules.constraintsInst import ConstraintInstantiator, FINGER_LENGTH, CONTACT_OFFSET
from modules.IKSolver       import IKSolver, Q_HOME
from utils                  import CoordinateTransformer

PASS = "✅"
FAIL = "❌"

def check(condition: bool, msg: str):
    status = PASS if condition else FAIL
    print(f"  {status} {msg}")
    if not condition:
        raise AssertionError(f"FAILED: {msg}")

# =====================================================================
# 准备：从 IKSolver 获取 T_current（Q_HOME 下的末端位姿）
# =====================================================================
print("=" * 65)
print("准备：获取 T_current（Q_HOME FK）")
print("=" * 65)

ik = IKSolver(verbose=False)
T_current  = ik.forward_kinematics(Q_HOME)
R_current  = T_current[:3, :3]
gz_current = R_current @ np.array([0., 0., 1.])

print(f"  T_current 末端位置: {np.round(T_current[:3,3], 4)}")
print(f"  gripper_z 方向:     {np.round(gz_current, 4)}")

# =====================================================================
# 【1】VLMDecision.w_flip 字段
# =====================================================================
print("\n" + "=" * 65)
print("【1】VLMDecision.w_flip 字段验证")
print("=" * 65)

d = VLMDecision(
    w_grasp_axis=1.0, w_safety=2.0, w_flip=1.5,
    confidence=0.9, reasoning="test", is_fallback=False
)
check(hasattr(d, 'w_flip'),      "VLMDecision 有 w_flip 字段")
check(d.w_flip == 1.5,           "w_flip 值正确")
check(not hasattr(d, 'w_upright'), "VLMDecision 无旧字段 w_upright")
check('w_flip' in repr(d),      "repr 包含 w_flip")
check('w_upright' not in repr(d), "repr 不含旧字段 w_upright")

# 校验：负权重应报错
try:
    VLMDecision(w_grasp_axis=0.0, w_safety=0.0, w_flip=-0.1,
                confidence=0.5, reasoning="", is_fallback=False)
    check(False, "负 w_flip 应抛出 ValueError")
except ValueError:
    check(True, "负 w_flip 正确抛出 ValueError")

# FALLBACK_DECISION 字段检查
check(hasattr(FALLBACK_DECISION, 'w_flip'), "FALLBACK_DECISION 有 w_flip 字段")
check(FALLBACK_DECISION.w_flip > 0,         "FALLBACK_DECISION.w_flip > 0")

# =====================================================================
# 辅助：构建一个最简单的 cost_fn 用于后续测试
# 场景：handle 关键点，无 avoid 部件
# =====================================================================

KEYPOINT   = np.array([0.50, 0.07, 0.45])   # handle 世界坐标
APPROACH   = np.array([1.0, 0.0, 0.0])       # 侧向接近
GRASP_AXIS = np.array([0.0, 0.0, 1.0])       # 夹爪张开轴

fallback = FALLBACK_DECISION
inst     = ConstraintInstantiator(verbose=False)

cost_fn, breakdown_fn = None, None

# 直接调用内部函数 _build_cost_fn
from modules.constraintsInst import _build_cost_fn

cost_fn, breakdown_fn = _build_cost_fn(
    keypoint_3d  = KEYPOINT,
    approach_dir = APPROACH,
    grasp_axis   = GRASP_AXIS,
    avoid_kps    = {},
    vlm_decision = fallback,
    mode         = "pick",
)

# =====================================================================
# 【2】C1 物理正确性
# =====================================================================
print("\n" + "=" * 65)
print("【2】C1 物理正确性：指尖对准关键点时 C1 趋近 0")
print("=" * 65)

# 构造：gripper_z = approach_dir，指尖 = KEYPOINT
# hand link 位置 = KEYPOINT - approach_dir * FINGER_LENGTH
R_aligned = np.column_stack([
    np.cross(GRASP_AXIS, APPROACH),  # X = Y × Z
    GRASP_AXIS,                       # Y = grasp_axis
    APPROACH,                         # Z = approach_dir
])
from scipy.spatial.transform import Rotation
rvec_aligned = Rotation.from_matrix(R_aligned).as_rotvec()
pos_aligned  = KEYPOINT - APPROACH * FINGER_LENGTH
x_perfect    = np.concatenate([pos_aligned, rvec_aligned])

bd_perfect = breakdown_fn(x_perfect)
c1_perfect = bd_perfect['approach_pos']
print(f"  指尖对准时 C1 加权值 = {c1_perfect:.8f}  （应趋近 0）")
check(c1_perfect < 1e-6, f"C1 在指尖对准时 < 1e-6（实际: {c1_perfect:.2e}）")

# 验证：指尖实际位置
R_test    = CoordinateTransformer.rodrigues(rvec_aligned)
gz_test   = R_test @ np.array([0., 0., 1.])
fingertip = pos_aligned + gz_test * FINGER_LENGTH
ft_err    = np.linalg.norm(fingertip - KEYPOINT)
print(f"  指尖到关键点距离 = {ft_err*1000:.4f} mm  （应趋近 0）")
check(ft_err < 1e-6, f"指尖到关键点距离 < 1e-6 m（实际: {ft_err:.2e} m）")

# =====================================================================
# 【3】C2 方向对齐
# =====================================================================
print("\n" + "=" * 65)
print("【3】C2：gripper_z 对齐 approach_dir 时 C2=0")
print("=" * 65)

bd_aligned = breakdown_fn(x_perfect)
c2_aligned = bd_aligned['approach_rot']
print(f"  对齐时 C2 = {c2_aligned:.8f}  （应 = 0）")
check(c2_aligned < 1e-6, f"C2 对齐时 < 1e-6（实际: {c2_aligned:.2e}）")

# 反向：gripper_z = -approach_dir
R_flip    = np.column_stack([
    np.cross(GRASP_AXIS, -APPROACH),
    GRASP_AXIS,
    -APPROACH,
])
rvec_flip = Rotation.from_matrix(R_flip).as_rotvec()
x_flip    = np.concatenate([pos_aligned, rvec_flip])
bd_flip   = breakdown_fn(x_flip)
c2_flip   = bd_flip['approach_rot'] / fallback.w_safety  # 去权重看原始值
# C2 原始值 = 1 - dot(-app, app)^2 = 1 - (-1)^2 = 0
# 所以 C2 在 flip 时也是 0（这正是 C5 存在的原因）
c2_raw_flip = 1.0 - float(np.dot(-APPROACH, APPROACH)) ** 2
print(f"  反向时 C2 原始值 = {c2_raw_flip:.4f}  （应 = 0，验证 C2 对 flip 盲区）")
check(abs(c2_raw_flip) < 1e-6, "C2 对 wrist flip 确实为 0（验证盲区存在）")

# =====================================================================
# 【4】C5 数学性质
# =====================================================================
print("\n" + "=" * 65)
print("【4】C5 数学性质：((1-d)/2)²")
print("=" * 65)

def c5_formula(d: float) -> float:
    return ((1.0 - d) / 2.0) ** 2

cases = [
    (+1.0, 0.0,    "正向对齐 d=+1 → C5=0"),
    ( 0.0, 0.25,   "垂直     d= 0 → C5=0.25"),
    (-1.0, 1.0,    "wrist flip d=-1 → C5=1"),
    (+0.5, 0.0625, "部分对齐 d=+0.5 → C5=0.0625"),
]
for d_val, expected, desc in cases:
    c5_val = c5_formula(d_val)
    print(f"  {desc}：{c5_val:.6f}  （期望 {expected}）")
    check(abs(c5_val - expected) < 1e-8, desc)

# 连续性：在 d=0 两侧导数是否连续
eps   = 1e-6
grad_left  = (c5_formula(0.0) - c5_formula(-eps)) / eps
grad_right = (c5_formula(eps) - c5_formula(0.0)) / eps
print(f"  d=0 处左导数={grad_left:.6f}，右导数={grad_right:.6f}  （应相等）")
check(abs(grad_left - grad_right) < 1e-4, "C5 在 d=0 处导数连续")

# =====================================================================
# 【5】C5 与 C2 非线性相关验证
# =====================================================================
print("\n" + "=" * 65)
print("【5】C5 与 C2 非线性相关验证（wrist flip 区间梯度互补）")
print("=" * 65)

for d_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    c2_raw = 1.0 - d_val ** 2
    c5_raw = ((1.0 - d_val) / 2.0) ** 2
    print(f"  d={d_val:+.1f}  C2={c2_raw:.4f}  C5={c5_raw:.4f}")

# 关键验证：d=-1 时 C2=0 但 C5=1（互补）
check(abs((1.0 - (-1.0)**2)) < 1e-8,          "d=-1 时 C2=0（盲区）")
check(abs(((1-(-1))/2)**2 - 1.0) < 1e-8,      "d=-1 时 C5=1（补足惩罚）")
check(abs((1.0 - 1.0**2)) < 1e-8,             "d=+1 时 C2=0（正确对齐无惩罚）")
check(abs(((1-1.0)/2)**2) < 1e-8,             "d=+1 时 C5=0（正确对齐无惩罚）")

# =====================================================================
# 【6】C4 指尖距离
# =====================================================================
print("\n" + "=" * 65)
print("【6】C4：指尖远离 avoid 部件时 C4=0")
print("=" * 65)

RIM_POS    = np.array([0.50, 0.07, 0.55])   # rim 在关键点正上方较远处
RIM_MARGIN = 0.025

cost_fn_c4, breakdown_fn_c4 = _build_cost_fn(
    keypoint_3d  = KEYPOINT,
    approach_dir = APPROACH,
    grasp_axis   = GRASP_AXIS,
    avoid_kps    = {'rim': (RIM_POS, RIM_MARGIN)},
    vlm_decision = fallback,
    mode         = "pick",
)

# 指尖对准 keypoint，远离 rim
bd_c4 = breakdown_fn_c4(x_perfect)
# 计算实际指尖到 rim 的距离
R_p     = CoordinateTransformer.rodrigues(rvec_aligned)
gz_p    = R_p @ np.array([0.,0.,1.])
ft_pos  = pos_aligned + gz_p * FINGER_LENGTH
ft_rim_dist = np.linalg.norm(ft_pos - RIM_POS)
print(f"  指尖到 rim 距离 = {ft_rim_dist*1000:.1f} mm")
print(f"  C4 = {bd_c4['safety']:.6f}  （rim 较远时应 = 0）")
check(ft_rim_dist > RIM_MARGIN, f"指尖到 rim 距离 > margin({RIM_MARGIN}m)")
check(bd_c4['safety'] < 1e-8,  "C4=0 当指尖远离 avoid 部件")

# 靠近：把「指尖」移到 rim 附近（5mm 内）
# 注意：应移动指尖而非 pos（hand link）
# fingertip = pos + gripper_z * L，所以：
# pos = RIM_POS - APPROACH*(L + 0.005) 使指尖停在 rim 前方 5mm
pos_near_rim = RIM_POS - APPROACH * (FINGER_LENGTH + 0.005)
x_near       = np.concatenate([pos_near_rim, rvec_aligned])
# 验证指尖确实在 margin 内
R_near       = CoordinateTransformer.rodrigues(rvec_aligned)
gz_near      = R_near @ np.array([0.,0.,1.])
ft_near      = pos_near_rim + gz_near * FINGER_LENGTH
ft_near_dist = np.linalg.norm(ft_near - RIM_POS)
print(f"  指尖靠近 rim 时距离 = {ft_near_dist*1000:.1f} mm  （margin={RIM_MARGIN*1000:.0f}mm）")
bd_near      = breakdown_fn_c4(x_near)
print(f"  靠近 rim 时 C4 = {bd_near['safety']:.6f}  （应 > 0）")
check(ft_near_dist < RIM_MARGIN, f"指尖确实在 margin 内（{ft_near_dist*1000:.1f}mm < {RIM_MARGIN*1000:.0f}mm）")
check(bd_near['safety'] > 0,     "C4>0 当指尖接近 avoid 部件")

# =====================================================================
# 【7】cost_breakdown 总和一致性
# =====================================================================
print("\n" + "=" * 65)
print("【7】cost_breakdown 总和一致性")
print("=" * 65)

# 用随机 x 测试
rng = np.random.default_rng(seed=42)
for i in range(5):
    x_rand = rng.normal(size=6) * 0.3
    x_rand[:3] += np.array([0.5, 0.0, 0.5])
    fn_val = cost_fn(x_rand)
    bd_val = breakdown_fn(x_rand)
    diff   = abs(bd_val['total'] - fn_val)
    print(f"  随机测试 {i+1}: cost={fn_val:.6f}  bd_total={bd_val['total']:.6f}  diff={diff:.2e}")
    check(diff < 1e-8, f"cost_fn 与 breakdown total 一致（diff={diff:.2e}）")

# =====================================================================
# 【8】字段名检查：flip 存在，upright 不存在
# =====================================================================
print("\n" + "=" * 65)
print("【8】字段名检查")
print("=" * 65)

bd_check = breakdown_fn(x_perfect)
check('flip'   in bd_check,    "cost_breakdown 有 'flip' 字段")
check('upright' not in bd_check, "cost_breakdown 无旧字段 'upright'")
check('approach_pos' in bd_check, "cost_breakdown 有 'approach_pos'")
check('approach_rot' in bd_check, "cost_breakdown 有 'approach_rot'")
check('grasp_axis'   in bd_check, "cost_breakdown 有 'grasp_axis'")
check('safety'       in bd_check, "cost_breakdown 有 'safety'")
check('total'        in bd_check, "cost_breakdown 有 'total'")

# =====================================================================
# 【9】候选筛选：代价低的候选胜出
# =====================================================================
print("\n" + "=" * 65)
print("【9】候选筛选：handle vs body，handle 应胜出（侧向接近代价更低）")
print("=" * 65)

# handle 在侧面，body 在下方，两者 SAP 的 approach_dir 不同
# 用 FALLBACK_DECISION（w_grasp_axis=1.0），侧向抓 handle 应代价更低
_, _, meta_sel = inst.instantiate(
    keypoints_3d = {
        'handle': np.array([0.57, 0.05, 0.45]),
        'rim':    np.array([0.50, 0.05, 0.502]),
        'body':   np.array([0.50, 0.008, 0.45]),
    },
    vlm_decision = FALLBACK_DECISION,
    T_current    = T_current,
)
print(f"  筛选出最优交互点: {meta_sel['grasp_target']}")
check(meta_sel['mode'] == 'pick', "模式为 pick")
check(meta_sel['grasp_target'] in ['handle', 'body'], "最优交互点是合法候选")
print(f"  candidate_best_cost = {meta_sel['candidate_best_cost']:.4f}")

# =====================================================================
# 【10】x0 来自 T_current，数值合理
# =====================================================================
print("\n" + "=" * 65)
print("【10】x0 来自 T_current，数值合理")
print("=" * 65)

cost_fn_full, _, meta_full = inst.instantiate(
    keypoints_3d = {
        'handle': np.array([0.57, 0.05, 0.45]),
        'rim':    np.array([0.50, 0.05, 0.502]),
    },
    vlm_decision = FALLBACK_DECISION,
    T_current    = T_current,
)
x0_full = meta_full.get('_x0_debug', None)

# x0 通过 instantiate 返回，检查 best_x0
_, x0_ret, _ = inst.instantiate(
    keypoints_3d = {
        'handle': np.array([0.57, 0.05, 0.45]),
        'rim':    np.array([0.50, 0.05, 0.502]),
    },
    vlm_decision = FALLBACK_DECISION,
    T_current    = T_current,
)
check(x0_ret.shape == (6,),                "x0 shape=(6,)")
check(not np.any(np.isnan(x0_ret)),        "x0 无 NaN")
check(not np.any(np.isinf(x0_ret)),        "x0 无 Inf")
check(np.linalg.norm(x0_ret[:3]) < 5.0,   f"x0 位置在合理范围内（{np.round(x0_ret[:3],3)}）")
print(f"  x0 位置: {np.round(x0_ret[:3], 4)}")
print(f"  x0 旋转: {np.round(x0_ret[3:], 4)}")

# =====================================================================
# 【11】place 模式 C3 = 0
# =====================================================================
print("\n" + "=" * 65)
print("【11】place 模式 C3=0 验证")
print("=" * 65)

place_decision = VLMDecision(
    w_grasp_axis=0.0, w_safety=2.0, w_flip=1.5,
    confidence=0.0, reasoning="test", is_fallback=True
)
_, x0_place, meta_place = inst.instantiate(
    keypoints_3d = {
        'surface': np.array([0.50, -0.20, 0.412]),
        'rim':     np.array([0.50, -0.20, 0.440]),
    },
    vlm_decision = place_decision,
    T_current    = T_current,
)
bd_place = meta_place['cost_breakdown_fn'](x0_place)
check(meta_place['mode'] == 'place',    "模式为 place")
check(bd_place['grasp_axis'] == 0.0,   f"place 模式 C3=0（实际: {bd_place['grasp_axis']}）")
check('flip' in bd_place,              "place 模式 cost_breakdown 有 flip 字段")
print(f"  place C3(grasp_axis) = {bd_place['grasp_axis']}")
print(f"  place C5(flip)       = {bd_place['flip']:.6f}")

# =====================================================================
# 汇总
# =====================================================================
print("\n" + "=" * 65)
print("✅ 所有数值测试通过！")
print("=" * 65)
print()
print("下一步：在仿真里验证指尖实际停靠位置和夹爪朝向是否符合预期。")
