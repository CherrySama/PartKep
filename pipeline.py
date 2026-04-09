"""
Created by Yinghao Ho on 2026-03-22

Pipeline 编排器
核心职责：
    将所有模块串联为完整的端到端 pick-and-place 规划链路。
    接收任务指令 + 场景图像，输出 pick/place 末端位姿和关节角。

支持两种执行模式：
    "vision" 模式（完整视觉 pipeline）：
        RGB 图像 + 深度图
        → GroundingDINO 检测
        → SAM3 分割 + 2D 关键点提取
        → 深度反投影 → 3D 关键点
        → VLMDecider → ConstraintInstantiator → PoseSolver → IKSolver

    "sim" 模式（仿真调试，跳过视觉模块）：
        直接接受 MuJoCo 世界坐标关键点
        → VLMDecider → ConstraintInstantiator → PoseSolver → IKSolver
        用于在 MuJoCo 中验证规划链路，视觉模块单独测试

完整流程（vision 模式）：
    1. TaskParser          ：解析任务指令
    2. GroundingDINO       ：检测目标物体/放置目标 bbox
    3. SAM3Segmenter       ：分割各部件，提取 2D 关键点
    4. 2D → 3D            ：深度反投影到世界坐标
    5. VLMDecider (pick)   ：生成标注图，调用 VLM，输出 pick VLMDecision
    6. ConstraintInstantiator：实例化 pick 代价函数，选最优交互点
    7. PoseSolver          ：SLSQP 精细求解 pick 位姿
    8. IKSolver            ：求解 pick 关节角
    9. （pick_and_place 时）重复 2-8 对 place 目标执行 place 规划
   10. 返回 PipelineResult

输出数据（PipelineResult）：
    - pick_T      : 4×4 pick 末端位姿矩阵
    - place_T     : 4×4 place 末端位姿矩阵（pick_only 时为 None）
    - pick_q      : pick 关节角 shape=(7,)
    - place_q     : place 关节角（pick_only 时为 None）
    - success     : 整体是否成功
    - debug       : 完整中间结果（供评估和失败分析）
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from configs.camera_config import CameraConfig
from configs.part_config import PartConfig
from modules.constraintsInst import ConstraintInstantiator
from modules.IKSolver import IKSolver, Q_HOME
from modules.poseSolver import PoseSolver
from modules.taskParser import TaskParser, TaskSpec
from modules.vlmDecider import VLMDecider, VLMDecision, FALLBACK_DECISION
from utils import CoordinateTransformer

logger = logging.getLogger(__name__)


# ==================== 数据结构 ====================

@dataclass
class PhaseResult:
    """单个规划阶段（pick 或 place）的结果"""
    success:        bool
    T:              Optional[np.ndarray]        # 4×4 末端位姿
    q:              Optional[np.ndarray]        # 关节角 shape=(7,)
    keypoints_2d:   Dict[str, Tuple]            # {part → (x,y)}
    keypoints_3d:   Dict[str, np.ndarray]       # {part → xyz}
    vlm_decision:   Optional[VLMDecision]
    pose_meta:      Optional[Dict]
    final_cost:     Optional[float]
    ik_error_mm:    Optional[float]
    failure_stage:  Optional[str]               # 失败发生在哪个阶段
    failure_reason: Optional[str]


@dataclass
class PipelineResult:
    """Pipeline 完整运行结果"""
    success:        bool
    task_spec:      Optional[TaskSpec]

    pick:           Optional[PhaseResult]       # pick 阶段结果
    place:          Optional[PhaseResult]       # place 阶段结果（pick_only 时为 None）

    # 便捷属性（供下游直接使用）
    @property
    def pick_T(self) -> Optional[np.ndarray]:
        return self.pick.T if self.pick else None

    @property
    def place_T(self) -> Optional[np.ndarray]:
        return self.place.T if self.place else None

    @property
    def pick_q(self) -> Optional[np.ndarray]:
        return self.pick.q if self.pick else None

    @property
    def place_q(self) -> Optional[np.ndarray]:
        return self.place.q if self.place else None

    def __repr__(self) -> str:
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        task   = str(self.task_spec) if self.task_spec else "N/A"
        lines  = [f"PipelineResult({status})", f"  task: {task}"]
        if self.pick:
            lines.append(
                f"  pick : {'✅' if self.pick.success else '❌'} "
                f"cost={self.pick.final_cost:.4f if self.pick.final_cost else 'N/A'} "
                f"ik_err={self.pick.ik_error_mm:.2f if self.pick.ik_error_mm else 'N/A'}mm"
            )
        if self.place:
            lines.append(
                f"  place: {'✅' if self.place.success else '❌'} "
                f"cost={self.place.final_cost:.4f if self.place.final_cost else 'N/A'} "
                f"ik_err={self.place.ik_error_mm:.2f if self.place.ik_error_mm else 'N/A'}mm"
            )
        return "\n".join(lines)


# ==================== Pipeline 主类 ====================

class Pipeline:
    """
    PartKep 端到端 pick-and-place 规划 Pipeline

    使用示例（vision 模式）：
        >>> pipeline = Pipeline(
        ...     camera_config = camera_config,
        ...     vlm_endpoint  = "http://<server>:8000/vlm",
        ... )
        >>> result = pipeline.run(
        ...     instruction = "pick up the cup and place it on the tray",
        ...     rgb_image   = pil_image,
        ...     depth_map   = depth_np,           # (H,W) 毫米
        ... )
        >>> if result.success:
        ...     print(result.pick_T)
        ...     print(result.place_T)

    使用示例（sim 模式，跳过视觉）：
        >>> result = pipeline.run(
        ...     instruction      = "pick up the cup and place it on the tray",
        ...     rgb_image        = pil_image,      # 仍需提供，供 VLM 生成标注图
        ...     depth_map        = None,
        ...     sim_keypoints    = {
        ...         "cup":  {"handle": np.array([...]),
        ...                  "rim":    np.array([...]),
        ...                  "body":   np.array([...])},
        ...         "tray": {"surface": np.array([...]),
        ...                  "rim":     np.array([...])},
        ...     }
        ... )
    """

    def __init__(
            self,
            camera_config:        Optional[CameraConfig] = None,
            vlm_endpoint:         Optional[str]  = None,
            ik_q_home:            np.ndarray     = Q_HOME,
            pose_max_iter:        int             = 200,
            pose_tol:             float           = 1e-6,
            ik_position_tol_m:    float           = 0.01,
            verbose:              bool            = True,
    ):
        """
        Args:
            camera_config      : 相机内外参配置，vision 模式必须提供
            vlm_endpoint       : Qwen3-VL FastAPI 地址，None 则使用 fallback
            ik_q_home          : IK 求解初始关节角（Panda home 位形）
            pose_max_iter      : PoseSolver SLSQP 最大迭代次数
            pose_tol           : PoseSolver 收敛容差
            ik_position_tol_m  : IK 位置误差容忍阈值（米）
            verbose            : 是否打印各阶段详情
        """
        self.camera_config     = camera_config
        self.ik_q_home         = ik_q_home
        self.ik_position_tol_m = ik_position_tol_m
        self.verbose           = verbose

        if verbose:
            print("=" * 65)
            print("初始化 PartKep Pipeline")
            print("=" * 65)

        # ── 初始化各模块 ──
        self.task_parser = TaskParser()

        # 视觉模块（延迟初始化，仅 vision 模式需要）
        self._detector  = None
        self._segmenter = None

        self.vlm_decider = VLMDecider(
            vlm_endpoint = vlm_endpoint,
            verbose      = verbose,
        )
        self.constraint_inst = ConstraintInstantiator(verbose=verbose)
        self.pose_solver     = PoseSolver(
            max_iter = pose_max_iter,
            tol      = pose_tol,
            verbose  = verbose,
        )
        self.ik_solver = IKSolver(verbose=verbose)

        if verbose:
            print("✅ Pipeline 初始化完成\n")

    # ==================== 视觉模块延迟初始化 ====================

    def _ensure_vision_modules(self):
        """按需初始化视觉模块（避免每次 Pipeline 实例化都加载大模型）"""
        if self._detector is None:
            from modules.groundingdino import GroundingDINODetector
            from modules.sam3segmenter import SAM3Segmenter
            self._detector  = GroundingDINODetector()
            self._segmenter = SAM3Segmenter()
            if self.verbose:
                print("✅ 视觉模块加载完成（GroundingDINO + SAM3）")

    # ==================== 公开接口 ====================

    def run(
            self,
            instruction:   str,
            rgb_image:     Image.Image,
            depth_map:     Optional[np.ndarray] = None,
            sim_keypoints: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
            q_init:        Optional[np.ndarray] = None,
    ) -> PipelineResult:
        """
        运行完整 pick-and-place 规划 Pipeline

        Args:
            instruction   : 自然语言任务指令
            rgb_image     : 原始 RGB 场景图（PIL Image）
            depth_map     : 深度图（H×W，单位毫米）；sim 模式可传 None
            sim_keypoints : sim 模式关键点字典
                            {object_label: {part_name: xyz_world shape=(3,)}}
                            vision 模式传 None
            q_init        : IK 初始关节角，None 则用 Q_HOME

        Returns:
            PipelineResult
        """
        q_init = q_init if q_init is not None else self.ik_q_home

        self._log_header("Pipeline 开始运行")
        self._log(f"任务指令: {instruction}")

        # ── Step 1：解析任务指令 ──
        self._log_section("Step 1 TaskParser")
        try:
            task_spec = self.task_parser.parse(instruction)
            self._log(f"✅ {task_spec}")
        except ValueError as e:
            logger.error(f"TaskParser 失败: {e}")
            return self._fail_result(None, f"TaskParser 失败: {e}")

        sim_mode = sim_keypoints is not None

        # ── Step 2：Pick 阶段 ──
        self._log_section("Step 2 Pick 阶段")
        pick_result = self._run_phase(
            phase        = "pick",
            object_label = task_spec.object_label,
            instruction  = instruction,
            rgb_image    = rgb_image,
            depth_map    = depth_map,
            sim_keypoints_for_obj = (
                sim_keypoints.get(task_spec.object_label)
                if sim_mode else None
            ),
            sim_mode     = sim_mode,
            q_init       = q_init,
        )

        if not pick_result.success:
            return PipelineResult(
                success   = False,
                task_spec = task_spec,
                pick      = pick_result,
                place     = None,
            )

        # ── Step 3：Place 阶段（仅 pick_and_place）──
        place_result = None
        if task_spec.action == "pick_and_place":
            self._log_section("Step 3 Place 阶段")
            place_result = self._run_phase(
                phase        = "place",
                object_label = task_spec.target_label,
                instruction  = instruction,
                rgb_image    = rgb_image,
                depth_map    = depth_map,
                sim_keypoints_for_obj = (
                    sim_keypoints.get(task_spec.target_label)
                    if sim_mode else None
                ),
                sim_mode     = sim_mode,
                q_init       = pick_result.q,   # 从 pick 结束位形开始
            )

        overall_success = pick_result.success and (
            place_result is None or place_result.success
        )

        result = PipelineResult(
            success   = overall_success,
            task_spec = task_spec,
            pick      = pick_result,
            place     = place_result,
        )

        self._log_header("Pipeline 完成")
        self._log(str(result))
        return result

    # ==================== 私有：单阶段规划 ====================

    def _run_phase(
            self,
            phase:                  str,
            object_label:           str,
            instruction:            str,
            rgb_image:              Image.Image,
            depth_map:              Optional[np.ndarray],
            sim_keypoints_for_obj:  Optional[Dict[str, np.ndarray]],
            sim_mode:               bool,
            q_init:                 np.ndarray,
    ) -> PhaseResult:
        """
        运行单个规划阶段（pick 或 place）

        Args:
            phase                 : "pick" 或 "place"
            object_label          : 操作目标物体标签
            sim_keypoints_for_obj : sim 模式下该物体的关键点字典

        Returns:
            PhaseResult
        """
        # ── A. 获取 3D 关键点 ──
        if sim_mode:
            if sim_keypoints_for_obj is None:
                return self._phase_fail(
                    phase, "KeypointExtraction",
                    f"sim_keypoints 中未找到 '{object_label}'"
                )
            keypoints_3d = {
                k: np.array(v, dtype=np.float64)
                for k, v in sim_keypoints_for_obj.items()
            }
            # 2D 关键点用于生成标注图：投影回图像坐标（sim 模式用物体中心近似）
            keypoints_2d = self._project_3d_to_2d(keypoints_3d)
            self._log(f"[sim] {object_label} 关键点: {list(keypoints_3d.keys())}")
        else:
            # Vision 模式：视觉 pipeline
            kp_result = self._vision_pipeline(
                object_label = object_label,
                rgb_image    = rgb_image,
                depth_map    = depth_map,
            )
            if kp_result is None:
                return self._phase_fail(
                    phase, "VisionPipeline",
                    f"视觉 pipeline 未能提取 '{object_label}' 的关键点"
                )
            keypoints_2d, keypoints_3d = kp_result
            self._log(f"[vision] {object_label} 关键点: {list(keypoints_3d.keys())}")

        # ── B. VLMDecider ──
        self._log(f"[VLM] 模式={phase}")
        try:
            vlm_decision = self.vlm_decider.decide(
                rgb_image        = rgb_image,
                keypoints_2d     = keypoints_2d,
                task_instruction = instruction,
                mode             = phase,
            )
            self._log(f"[VLM] 决策: {vlm_decision}")
        except Exception as e:
            logger.warning(f"VLMDecider 异常，使用 fallback: {e}")
            vlm_decision = FALLBACK_DECISION

        # ── C. ConstraintInstantiator ──
        self._log("[Constraint] 实例化代价函数")
        try:
            cost_fn, x0, meta = self.constraint_inst.instantiate(
                keypoints_3d = keypoints_3d,
                vlm_decision = vlm_decision,
            )
        except (ValueError, KeyError) as e:
            return self._phase_fail(phase, "ConstraintInstantiator", str(e))

        # ── D. PoseSolver ──
        self._log("[PoseSolver] SLSQP 精细求解")
        pose_result = self.pose_solver.solve(cost_fn, x0, meta)
        if not pose_result['success']:
            logger.warning(
                f"PoseSolver 未收敛（{phase}），继续使用当前最优估计"
            )

        T          = pose_result['T']
        final_cost = pose_result['final_cost']
        self._log(f"[PoseSolver] cost={final_cost:.4f}, "
                  f"pos={np.round(T[:3,3], 3)}")

        # ── E. IKSolver ──
        self._log("[IKSolver] 求解关节角")
        ik_result = self.ik_solver.solve(T, q_init=q_init)
        ik_ok      = ik_result['success']
        ik_err_mm  = ik_result['position_error'] * 1000
        q          = ik_result['q']

        if not ik_ok:
            logger.warning(
                f"IKSolver {phase} 失败: err={ik_err_mm:.2f}mm, "
                f"within_limits={ik_result['within_limits']}"
            )

        self._log(f"[IKSolver] {'✅' if ik_ok else '⚠️'} "
                  f"err={ik_err_mm:.2f}mm")

        return PhaseResult(
            success        = ik_ok,
            T              = T,
            q              = q,
            keypoints_2d   = keypoints_2d,
            keypoints_3d   = keypoints_3d,
            vlm_decision   = vlm_decision,
            pose_meta      = meta,
            final_cost     = final_cost,
            ik_error_mm    = ik_err_mm,
            failure_stage  = None if ik_ok else "IKSolver",
            failure_reason = None if ik_ok else (
                f"position_error={ik_err_mm:.2f}mm, "
                f"within_limits={ik_result['within_limits']}"
            ),
        )

    # ==================== 私有：视觉 pipeline ====================

    def _vision_pipeline(
            self,
            object_label: str,
            rgb_image:    Image.Image,
            depth_map:    np.ndarray,
    ) -> Optional[Tuple[Dict, Dict]]:
        """
        GroundingDINO + SAM3 + 深度反投影

        Returns:
            (keypoints_2d, keypoints_3d) 或 None（失败时）
            keypoints_2d: {part_name → (x, y)}
            keypoints_3d: {part_name → np.ndarray shape=(3,)}
        """
        if self.camera_config is None:
            logger.error("vision 模式需要提供 camera_config")
            return None
        if depth_map is None:
            logger.error("vision 模式需要提供 depth_map")
            return None

        self._ensure_vision_modules()

        # ── GroundingDINO 检测物体 ──
        self._log(f"[GDINO] 检测 '{object_label}'")
        detection_results = self._detector.detect(
            image       = rgb_image,
            text_prompt = object_label,
        )
        if not detection_results:
            self._log(f"[GDINO] 未检测到 '{object_label}'")
            return None

        # 取置信度最高的检测结果
        best_det  = max(detection_results, key=lambda d: d['score'])
        crop_bbox = [int(v) for v in best_det['bbox']]  # [x1, y1, x2, y2]
        self._log(f"[GDINO] 检测到 '{object_label}' score={best_det['score']:.3f}")

        # ── SAM3 分割部件，提取 2D 关键点 ──
        self._log(f"[SAM3] 分割 '{object_label}' 各部件")
        seg_results = self._segmenter.segment_parts(
            cropped_image = rgb_image.crop(crop_bbox),
            label         = object_label,
            crop_bbox     = crop_bbox,
        )
        if not seg_results:
            self._log(f"[SAM3] 未分割出任何部件")
            return None

        # ── 2D 关键点 → 3D 世界坐标 ──
        keypoints_2d: Dict[str, Tuple] = {}
        keypoints_3d: Dict[str, np.ndarray] = {}

        for seg in seg_results:
            part_name = seg['part_name']
            x_px, y_px = seg['keypoint']
            keypoints_2d[part_name] = (x_px, y_px)

            try:
                pt_3d = CoordinateTransformer.get_keypoint_3d(
                    mode          = "real",
                    camera_config = self.camera_config,
                    x_pixel       = x_px,
                    y_pixel       = y_px,
                    depth_map     = depth_map,
                )
                keypoints_3d[part_name] = pt_3d
                self._log(f"  {part_name}: 2D=({x_px:.1f},{y_px:.1f}) "
                          f"3D={np.round(pt_3d,3)}")
            except Exception as e:
                logger.warning(f"关键点 '{part_name}' 深度获取失败: {e}，跳过")

        if not keypoints_3d:
            self._log("[Vision] 所有部件深度获取失败")
            return None

        return keypoints_2d, keypoints_3d

    # ==================== 私有：工具方法 ====================

    def _project_3d_to_2d(
            self,
            keypoints_3d: Dict[str, np.ndarray],
    ) -> Dict[str, Tuple[float, float]]:
        """
        sim 模式下将 3D 关键点投影为 2D 坐标（用于生成标注图）
        若无 camera_config，返回虚拟坐标（标注图仍可生成，位置不精确）
        """
        if self.camera_config is None:
            # 无相机配置：均匀分布虚拟坐标
            n = len(keypoints_3d)
            return {
                name: (100.0 + i * 80.0, 240.0)
                for i, name in enumerate(keypoints_3d.keys())
            }

        kps_2d = {}
        fx, fy = self.camera_config.fx, self.camera_config.fy
        cx, cy = self.camera_config.cx, self.camera_config.cy
        T_w2c  = np.linalg.inv(self.camera_config.extrinsic_matrix)

        for name, xyz_world in keypoints_3d.items():
            p_h   = np.append(xyz_world, 1.0)
            p_cam = (T_w2c @ p_h)[:3]
            if p_cam[2] <= 0:
                kps_2d[name] = (cx, cy)
                continue
            x_px = float(fx * p_cam[0] / p_cam[2] + cx)
            y_px = float(fy * p_cam[1] / p_cam[2] + cy)
            kps_2d[name] = (x_px, y_px)

        return kps_2d

    @staticmethod
    def _phase_fail(
            phase: str,
            stage: str,
            reason: str,
    ) -> PhaseResult:
        return PhaseResult(
            success        = False,
            T              = None,
            q              = None,
            keypoints_2d   = {},
            keypoints_3d   = {},
            vlm_decision   = None,
            pose_meta      = None,
            final_cost     = None,
            ik_error_mm    = None,
            failure_stage  = stage,
            failure_reason = reason,
        )

    @staticmethod
    def _fail_result(task_spec, reason: str) -> PipelineResult:
        logger.error(reason)
        return PipelineResult(
            success   = False,
            task_spec = task_spec,
            pick      = None,
            place     = None,
        )

    def _log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")

    def _log_section(self, title: str):
        if self.verbose:
            print(f"\n── {title} {'─' * max(0, 50 - len(title))}")

    def _log_header(self, title: str):
        if self.verbose:
            print(f"\n{'='*65}")
            print(f"  {title}")
            print(f"{'='*65}")


# ==================== 模块测试 ====================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    print("=" * 65)
    print("测试 Pipeline（sim 模式，跳过视觉）")
    print("=" * 65)

    # ── 模拟场景：杯子 pick + 托盘 place ──
    sim_kps = {
        "cup": {
            "handle": np.array([0.45,  0.07,  0.10]),
            "rim":    np.array([0.45,  0.00,  0.16]),
            "body":   np.array([0.45,  0.00,  0.08]),
        },
        "tray": {
            "surface": np.array([0.65,  0.10,  0.02]),
            "rim":     np.array([0.65,  0.10,  0.05]),
        },
    }

    # ── 使用 identity 相机配置（sim 模式无需精确内外参）──
    cam_cfg = CameraConfig.create_identity()

    # ── 初始化 Pipeline（无 VLM endpoint，使用 fallback）──
    pipeline = Pipeline(
        camera_config = cam_cfg,
        vlm_endpoint  = None,       # fallback 模式
        verbose       = True,
    )

    # ── pick_and_place 任务 ──
    print("\n【测试1】pick_and_place（cup → tray）")
    fake_rgb = Image.new("RGB", (640, 480), (128, 128, 128))
    result = pipeline.run(
        instruction   = "pick up the cup and place it on the tray",
        rgb_image     = fake_rgb,
        depth_map     = None,
        sim_keypoints = sim_kps,
    )

    print(f"\n{result}")

    # 验证基本字段
    assert result.task_spec is not None
    assert result.pick is not None
    assert result.place is not None
    assert result.pick_T is not None and result.pick_T.shape == (4, 4)
    assert result.place_T is not None and result.place_T.shape == (4, 4)
    assert result.pick_q is not None and result.pick_q.shape == (7,)
    assert result.place_q is not None and result.place_q.shape == (7,)
    print("\n✅ 所有字段验证通过")

    # ── pick_only 任务 ──
    print("\n【测试2】pick_only（cup）")
    result2 = pipeline.run(
        instruction   = "pick up the cup",
        rgb_image     = fake_rgb,
        depth_map     = None,
        sim_keypoints = sim_kps,
    )
    assert result2.place is None, "pick_only 不应有 place 结果"
    assert result2.pick is not None
    print(f"  ✅ pick_only: place=None, pick_T shape={result2.pick_T.shape}")

    # ── 无效指令 ──
    print("\n【测试3】无效指令 → 应 graceful 失败")
    result3 = pipeline.run(
        instruction   = "grab the robot",
        rgb_image     = fake_rgb,
        depth_map     = None,
        sim_keypoints = sim_kps,
    )
    assert not result3.success
    print(f"  ✅ 无效指令正确返回 success=False")

    print("\n" + "=" * 65)
    print("✅ Pipeline 所有测试通过！")
    print("=" * 65)
    