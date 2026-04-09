"""
Created by Yinghao Ho on 2026-03-20

VLM 约束决策模块
核心职责：
    接收标注场景图 + 任务指令 + SAP 语义描述，
    调用本地 Qwen3-VL 模型，输出 VLMDecision（约束权重集合）。
    VLM 只做语义决策，不做任何几何推理。

VLM 决策的内容（对应论文 3.4 节）：
    - w_grasp_axis : 夹爪 Y 轴对齐 SAP grasp_axis 的权重（C3，pick 专用）
    - w_safety     : 安全距离惩罚的权重（C4，pick + place）
    - w_upright    : 保持末端竖直的权重（C5，pick + place）

VLM 不控制的约束（固定激活，属于 SAP 几何职责）：
    - C1 approach_pos : 末端位置对齐接触点（始终激活）
    - C2 approach_rot : 夹爪 Z 轴对齐接近方向（始终激活）

Fallback 策略：
    VLM 不可用或返回无效 JSON 时，自动切换到 rule-based fallback。
    Fallback 权重根据 SAP contact_mode 派生，保证系统优雅降级。

部署说明：
    Qwen3-VL 运行在 RunPod A40 服务器上，通过 FastAPI 接口调用。
    本地调用时将 vlm_endpoint 设为 None，系统自动使用 fallback。
"""

import json
import base64
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from configs.SAP import get_sap_descriptions

logger = logging.getLogger(__name__)


# ==================== 数据结构 ====================

@dataclass
class VLMDecision:
    """
    VLM 约束决策结果

    由 VLMDecider.decide() 返回，传入 ConstraintInstantiator.instantiate()。

    Attributes:
        w_grasp_axis : C3 权重，夹爪 Y 轴对齐 grasp_axis（pick 专用，≥ 0）
        w_safety     : C4 权重，安全距离惩罚（pick + place，≥ 0）
        w_upright    : C5 权重，末端保持竖直（pick + place，≥ 0）
        confidence   : VLM 决策置信度 [0, 1]，低于阈值时触发 fallback
        reasoning    : VLM 推理说明（调试用，不参与计算）
        is_fallback  : True 表示本次决策来自 rule-based fallback
    """
    w_grasp_axis: float
    w_safety:     float
    w_upright:    float
    confidence:   float
    reasoning:    str
    is_fallback:  bool

    def __post_init__(self):
        """校验权重范围"""
        for name, val in [
            ('w_grasp_axis', self.w_grasp_axis),
            ('w_safety',     self.w_safety),
            ('w_upright',    self.w_upright),
        ]:
            if val < 0:
                raise ValueError(f"VLMDecision.{name} 必须 ≥ 0，当前: {val}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"VLMDecision.confidence 必须在 [0,1]，当前: {self.confidence}"
            )

    def __repr__(self) -> str:
        src = "fallback" if self.is_fallback else "VLM"
        return (
            f"VLMDecision(src={src}, "
            f"w_grasp={self.w_grasp_axis:.2f}, "
            f"w_safety={self.w_safety:.2f}, "
            f"w_upright={self.w_upright:.2f}, "
            f"conf={self.confidence:.2f})"
        )


# ==================== Fallback 默认权重 ====================
# rule-based fallback：当 VLM 不可用或返回无效结果时使用
# 设计原则：保守策略，safety 权重最高，upright 默认激活

FALLBACK_DECISION = VLMDecision(
    w_grasp_axis = 1.0,
    w_safety     = 2.0,
    w_upright    = 1.0,
    confidence   = 0.0,
    reasoning    = "Rule-based fallback: VLM unavailable or returned invalid response.",
    is_fallback  = True,
)

# VLM 置信度低于此阈值时强制切换 fallback
CONFIDENCE_THRESHOLD = 0.3


# ==================== 标注图生成 ====================

# 关键点在标注图上的颜色（按部件名称固定）
_PART_COLORS: Dict[str, str] = {
    "handle":  "#FF4444",   # 红
    "rim":     "#4444FF",   # 蓝
    "body":    "#44AA44",   # 绿
    "neck":    "#FF8800",   # 橙
    "cap":     "#AA44AA",   # 紫
    "surface": "#00AAAA",   # 青
}
_DEFAULT_COLOR = "#FFFF00"  # 未知部件：黄


def build_annotated_image(
        rgb_image: Image.Image,
        keypoints_2d: Dict[str, Tuple[float, float]],
) -> Image.Image:
    """
    在 RGB 图上叠加关键点标注，生成供 VLM 读取的标注图

    每个关键点绘制：
        - 彩色实心圆点（半径 8px）
        - 白色描边（2px，增强可见性）
        - 部件名称文字标签（圆点右侧）

    Args:
        rgb_image    : 原始 RGB 图像（PIL Image）
        keypoints_2d : {part_name → (x, y)} 图像像素坐标

    Returns:
        标注后的 PIL Image（与输入尺寸相同）
    """
    annotated = rgb_image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    # 尝试加载字体，失败则使用 PIL 默认字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for part_name, (x, y) in keypoints_2d.items():
        color  = _PART_COLORS.get(part_name, _DEFAULT_COLOR)
        radius = 8

        # 白色描边（增强对比度）
        draw.ellipse(
            [x - radius - 2, y - radius - 2,
             x + radius + 2, y + radius + 2],
            fill="white"
        )
        # 彩色实心圆
        draw.ellipse(
            [x - radius, y - radius,
             x + radius, y + radius],
            fill=color
        )
        # 部件名称标签
        draw.text((x + radius + 4, y - 8), part_name, fill=color, font=font)

    return annotated


def _image_to_base64(image: Image.Image) -> str:
    """PIL Image → base64 字符串（JPEG 格式）"""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ==================== Prompt 构建 ====================

def _build_vlm_prompt(
        task_instruction: str,
        detected_parts:   List[str],
        mode:             str,
) -> str:
    """
    构建发送给 Qwen3-VL 的文字 prompt

    Args:
        task_instruction : 任务指令，如 "pick up the cup and place it on the tray"
        detected_parts   : 当前场景中检测到的部件列表，如 ["handle", "rim", "body"]
        mode             : "pick" 或 "place"

    Returns:
        完整的系统 prompt 字符串
    """
    # 获取所有检测到部件的 SAP 描述
    all_descriptions = get_sap_descriptions()
    parts_desc_block = ""
    for part in detected_parts:
        if part in all_descriptions:
            parts_desc_block += f"  [{part}]: {all_descriptions[part]}\n"

    # pick 模式的约束说明
    pick_constraints = """
  - w_grasp_axis (float, 0.0~3.0):
      Weight for gripper Y-axis alignment with the SAP grasp axis.
      Higher value enforces stricter finger orientation relative to the part geometry.
      Set higher when grasping a structured part like a handle or neck.
      Set lower when grasping an unstructured surface like body.

  - w_safety (float, 0.0~3.0):
      Weight for safety distance penalty from avoid-mode parts (e.g. rim).
      Higher value enforces stricter avoidance of fragile or sensitive regions.
      Set higher when the object contains liquid or has a delicate opening.

  - w_upright (float, 0.0~3.0):
      Weight for keeping the end-effector upright during pick.
      Higher value enforces more vertical gripper orientation.
      Set higher when the object must not be tilted (e.g. cup with liquid).
      Set to 0.0 when tilting is acceptable or required by the task.
"""

    # place 模式的约束说明（无 w_grasp_axis）
    place_constraints = """
  - w_grasp_axis: NOT USED in place mode. Always set to 0.0.

  - w_safety (float, 0.0~3.0):
      Weight for safety distance penalty from avoid-mode parts (e.g. tray rim).
      Higher value enforces stricter avoidance of placement platform edges.

  - w_upright (float, 0.0~3.0):
      Weight for keeping the end-effector upright during placement.
      Higher value enforces stable, level placement of the object.
      This should generally be high to ensure objects are placed stably.
"""

    constraints_block = pick_constraints if mode == "pick" else place_constraints

    prompt = f"""You are a semantic constraint decision-maker for a robotic manipulation system.
Your role is to decide constraint weights for the current task based on the scene image and task instruction.
You must NOT perform any geometric reasoning. Only make semantic decisions.

=== TASK INSTRUCTION ===
{task_instruction}

=== CURRENT MODE ===
{mode.upper()} mode

=== DETECTED PARTS IN SCENE ===
{parts_desc_block}
=== CONSTRAINT WEIGHTS TO DECIDE ===
{constraints_block}
=== OUTPUT FORMAT ===
Respond with a single JSON object only. No explanation outside the JSON.
{{
  "w_grasp_axis": <float>,
  "w_safety": <float>,
  "w_upright": <float>,
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining your decision>"
}}
"""
    return prompt


# ==================== 主类 ====================

class VLMDecider:
    """
    VLM 约束决策器

    调用 Qwen3-VL 对当前场景进行语义推理，
    输出 VLMDecision 供 ConstraintInstantiator 使用。

    使用示例：
        >>> decider = VLMDecider(vlm_endpoint="http://<server>:8000/vlm")
        >>> decision = decider.decide(
        ...     rgb_image        = pil_image,
        ...     keypoints_2d     = {"handle": (320, 240), "rim": (400, 100)},
        ...     task_instruction = "pick up the cup and place it on the tray",
        ...     mode             = "pick",
        ... )
        >>> print(decision)
    """

    def __init__(
            self,
            vlm_endpoint:         Optional[str] = None,
            confidence_threshold: float = CONFIDENCE_THRESHOLD,
            request_timeout:      int   = 30,
            verbose:              bool  = True,
    ):
        """
        Args:
            vlm_endpoint         : Qwen3-VL FastAPI 服务地址。
                                   None → 直接使用 fallback（离线调试用）
            confidence_threshold : 低于此置信度时强制使用 fallback
            request_timeout      : HTTP 请求超时（秒）
            verbose              : 是否打印决策过程
        """
        self.vlm_endpoint         = vlm_endpoint
        self.confidence_threshold = confidence_threshold
        self.request_timeout      = request_timeout
        self.verbose              = verbose

        if verbose:
            print("=" * 60)
            print("初始化 VLMDecider")
            if vlm_endpoint:
                print(f"  VLM endpoint : {vlm_endpoint}")
            else:
                print("  VLM endpoint : None（将使用 rule-based fallback）")
            print(f"  置信度阈值   : {confidence_threshold}")
            print("=" * 60)

    # ==================== 公开接口 ====================

    def decide(
            self,
            rgb_image:        Image.Image,
            keypoints_2d:     Dict[str, Tuple[float, float]],
            task_instruction: str,
            mode:             str,
    ) -> VLMDecision:
        """
        核心决策接口

        Args:
            rgb_image        : 原始 RGB 场景图（PIL Image）
            keypoints_2d     : {part_name → (x, y)} 图像坐标，来自 SAM3Segmenter
            task_instruction : 自然语言任务指令
            mode             : "pick" 或 "place"

        Returns:
            VLMDecision，包含三个约束权重 + 置信度 + 推理说明
        """
        assert mode in ("pick", "place"), \
            f"mode 必须是 'pick' 或 'place'，当前: '{mode}'"

        if self.verbose:
            print(f"\n🤖 VLMDecider.decide()")
            print(f"  任务指令 : {task_instruction}")
            print(f"  模式     : {mode}")
            print(f"  检测部件 : {list(keypoints_2d.keys())}")

        # ── 无 endpoint，直接 fallback ──
        if self.vlm_endpoint is None:
            if self.verbose:
                print("  ⚠️  无 VLM endpoint，使用 rule-based fallback")
            return self._rule_based_fallback(mode)

        # ── 生成标注图 ──
        annotated = build_annotated_image(rgb_image, keypoints_2d)

        # ── 构建 prompt ──
        detected_parts = list(keypoints_2d.keys())
        prompt = _build_vlm_prompt(task_instruction, detected_parts, mode)

        # ── 调用 VLM ──
        try:
            decision = self._call_vlm(annotated, prompt, mode)
        except Exception as e:
            logger.warning(f"VLM 调用失败: {e}，切换到 fallback")
            if self.verbose:
                print(f"  ⚠️  VLM 调用异常: {e}")
                print("  → 切换到 rule-based fallback")
            return self._rule_based_fallback(mode)

        # ── 置信度检查 ──
        if decision.confidence < self.confidence_threshold:
            if self.verbose:
                print(f"  ⚠️  置信度 {decision.confidence:.2f} < "
                      f"阈值 {self.confidence_threshold:.2f}，切换到 fallback")
            return self._rule_based_fallback(mode)

        if self.verbose:
            print(f"  ✅ VLM 决策成功: {decision}")

        return decision

    # ==================== 私有：VLM 调用 ====================

    def _call_vlm(
            self,
            annotated_image: Image.Image,
            prompt:          str,
            mode:            str,
    ) -> VLMDecision:
        """
        向 FastAPI 服务发送请求，解析 VLM 输出

        FastAPI 服务预期接收：
            {
                "image_b64": "<base64 JPEG>",
                "prompt":    "<text prompt>"
            }
        预期返回：
            {
                "response": "<VLM 生成的 JSON 字符串>"
            }

        Raises:
            requests.RequestException : 网络错误
            ValueError                : VLM 返回内容无法解析为有效 JSON
        """
        payload = {
            "image_b64": _image_to_base64(annotated_image),
            "prompt":    prompt,
        }

        if self.verbose:
            print(f"  → 发送请求到 {self.vlm_endpoint} ...")

        resp = requests.post(
            self.vlm_endpoint,
            json=payload,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()

        raw_text = resp.json().get("response", "")

        if self.verbose:
            preview = raw_text[:200].replace('\n', ' ')
            print(f"  ← VLM 原始返回: {preview}...")

        return self._parse_vlm_response(raw_text, mode)

    def _parse_vlm_response(self, raw_text: str, mode: str) -> VLMDecision:
        """
        解析 VLM 返回的 JSON 字符串为 VLMDecision

        容错策略：
            1. 直接尝试 json.loads
            2. 若失败，尝试提取 { } 之间的内容再解析
            3. 仍失败则抛出 ValueError，由上层触发 fallback

        Raises:
            ValueError: 无法从返回内容中提取有效 JSON
        """
        # 策略 1：直接解析
        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            pass

        # 策略 2：提取 {...} 子串
        if data is None:
            start = raw_text.find('{')
            end   = raw_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(raw_text[start:end + 1])
                except json.JSONDecodeError:
                    pass

        if data is None:
            raise ValueError(
                f"无法从 VLM 返回中提取有效 JSON。\n原始返回: {raw_text[:300]}"
            )

        # ── 提取并校验字段 ──
        try:
            w_grasp = float(data.get("w_grasp_axis", 1.0))
            w_safe  = float(data.get("w_safety",     2.0))
            w_up    = float(data.get("w_upright",     1.0))
            conf    = float(data.get("confidence",    0.5))
            reason  = str(data.get("reasoning",       ""))
        except (TypeError, ValueError) as e:
            raise ValueError(f"VLM JSON 字段类型错误: {e}")

        # place 模式强制 w_grasp_axis = 0
        if mode == "place":
            w_grasp = 0.0

        # 权重截断到合理范围 [0, 3]
        w_grasp = float(np.clip(w_grasp, 0.0, 3.0))
        w_safe  = float(np.clip(w_safe,  0.0, 3.0))
        w_up    = float(np.clip(w_up,    0.0, 3.0))
        conf    = float(np.clip(conf,    0.0, 1.0))

        return VLMDecision(
            w_grasp_axis = w_grasp,
            w_safety     = w_safe,
            w_upright    = w_up,
            confidence   = conf,
            reasoning    = reason,
            is_fallback  = False,
        )

    # ==================== 私有：fallback ====================

    def _rule_based_fallback(self, mode: str) -> VLMDecision:
        """
        Rule-based fallback 决策

        Pick 模式：
            w_grasp_axis = 1.0（标准夹爪对齐）
            w_safety     = 2.0（安全优先）
            w_upright    = 1.0（默认保持竖直）

        Place 模式：
            w_grasp_axis = 0.0（place 不需要）
            w_safety     = 2.0（避开托盘边缘）
            w_upright    = 1.5（放置时更强的竖直约束）
        """
        if mode == "place":
            return VLMDecision(
                w_grasp_axis = 0.0,
                w_safety     = 2.0,
                w_upright    = 1.5,
                confidence   = 0.0,
                reasoning    = (
                    "Rule-based fallback (place mode): "
                    "no grasp axis needed, upright enforced for stable placement."
                ),
                is_fallback  = True,
            )
        else:
            return VLMDecision(
                w_grasp_axis = 1.0,
                w_safety     = 2.0,
                w_upright    = 1.0,
                confidence   = 0.0,
                reasoning    = (
                    "Rule-based fallback (pick mode): "
                    "conservative default weights derived from SAP contact modes."
                ),
                is_fallback  = True,
            )

    def generate_annotated_image(
            self,
            rgb_image:    Image.Image,
            keypoints_2d: Dict[str, Tuple[float, float]],
    ) -> Image.Image:
        """
        公开的标注图生成接口（供 Pipeline 调试可视化使用）

        Args:
            rgb_image    : 原始 RGB 图像
            keypoints_2d : {part_name → (x, y)}

        Returns:
            标注后的 PIL Image
        """
        return build_annotated_image(rgb_image, keypoints_2d)


# ==================== 模块测试 ====================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    print("=" * 65)
    print("测试 VLMDecider")
    print("=" * 65)

    # 【1】VLMDecision dataclass 校验
    print("\n【1】VLMDecision 构造与校验")
    d = VLMDecision(
        w_grasp_axis=1.0, w_safety=2.0, w_upright=1.0,
        confidence=0.85, reasoning="test", is_fallback=False
    )
    assert d.w_grasp_axis == 1.0
    assert not d.is_fallback
    print(f"  ✅ {d}")

    # 【2】非法权重应报错
    print("\n【2】非法权重 → 应抛出 ValueError")
    try:
        VLMDecision(w_grasp_axis=-0.1, w_safety=1.0, w_upright=1.0,
                    confidence=0.5, reasoning="", is_fallback=False)
        print("  ❌ 未报错！")
    except ValueError:
        print("  ✅ 负权重正确报错")

    # 【3】VLMDecider 初始化（无 endpoint，fallback 模式）
    print("\n【3】VLMDecider 初始化（fallback 模式）")
    decider = VLMDecider(vlm_endpoint=None, verbose=True)

    # 【4】fallback 决策
    print("\n【4】fallback 决策（pick 模式）")
    fake_image = Image.new("RGB", (640, 480), color=(128, 128, 128))
    fake_kps   = {"handle": (320.0, 240.0), "rim": (320.0, 100.0)}
    decision   = decider.decide(
        rgb_image        = fake_image,
        keypoints_2d     = fake_kps,
        task_instruction = "pick up the cup and place it on the tray",
        mode             = "pick",
    )
    assert decision.is_fallback
    assert decision.w_safety > 0
    print(f"  ✅ {decision}")

    # 【5】fallback 决策（place 模式）
    print("\n【5】fallback 决策（place 模式）")
    decision_place = decider.decide(
        rgb_image        = fake_image,
        keypoints_2d     = {"surface": (300.0, 350.0), "rim": (300.0, 300.0)},
        task_instruction = "pick up the cup and place it on the tray",
        mode             = "place",
    )
    assert decision_place.is_fallback
    assert decision_place.w_grasp_axis == 0.0   # place 模式无 grasp axis
    assert decision_place.w_upright >= decision.w_upright  # place 竖直约束更强
    print(f"  ✅ {decision_place}")

    # 【6】标注图生成
    print("\n【6】标注图生成")
    annotated = decider.generate_annotated_image(fake_image, fake_kps)
    assert annotated.size == fake_image.size
    print(f"  ✅ 标注图尺寸: {annotated.size}")

    # 【7】JSON 解析（模拟 VLM 返回）
    print("\n【7】JSON 解析测试")
    dummy_decider = VLMDecider(vlm_endpoint="http://dummy", verbose=False)
    raw_valid = json.dumps({
        "w_grasp_axis": 1.5, "w_safety": 2.5, "w_upright": 0.5,
        "confidence": 0.9, "reasoning": "handle detected, upright not critical"
    })
    parsed = dummy_decider._parse_vlm_response(raw_valid, mode="pick")
    assert parsed.w_grasp_axis == 1.5
    assert not parsed.is_fallback
    print(f"  ✅ 正常 JSON 解析: {parsed}")

    # 带 markdown 代码块的返回（常见 VLM 输出格式）
    raw_wrapped = "```json\n" + raw_valid + "\n```"
    parsed2 = dummy_decider._parse_vlm_response(raw_wrapped, mode="pick")
    assert parsed2.w_grasp_axis == 1.5
    print(f"  ✅ 带代码块的 JSON 解析: {parsed2}")

    # place 模式下 w_grasp_axis 强制为 0
    parsed_place = dummy_decider._parse_vlm_response(raw_valid, mode="place")
    assert parsed_place.w_grasp_axis == 0.0
    print(f"  ✅ place 模式 w_grasp_axis 强制为 0: {parsed_place}")

    print("\n" + "=" * 65)
    print("✅ VLMDecider 所有测试通过！")
    print("=" * 65)
    