"""
Created by Yinghao Ho on 2026-2-23

Constraint instantiation module.
Converts semantic keypoints + SAP knowledge base + VLMDecision -> SLSQP-ready cost function.

Constraint system:
    C1  fingertip position alignment to keypoint     fixed weight   pick + place
    C2  gripper Z-axis alignment to approach_dir     fixed weight   pick + place
    C3  gripper Y-axis alignment to SAP grasp_axis   VLM weight     pick only
    C4  safety distance penalty (fingertip to avoid) VLM weight     pick + place
    P1  axis-angle regularisation (eps * ||rvec||^2) fixed weight   pick + place

C1/C2 always active; C2 uses (1-dot)^2 so wrist-flip (dot=-1) costs 4, aligned (dot=+1) costs 0.
C3/C4 weights come from VLMDecision.

Optimisation variable x (6D): [px, py, pz, ax, ay, az]
    (px,py,pz): end-effector position in world frame (metres)
    (ax,ay,az): end-effector rotation as axis-angle vector (rad)

Gripper local frame convention:
    local Z: approach axis
    local Y: grasp axis (gripper opening direction)
    local X: right-hand rule, X = Y x Z

Cost normalisation:
    C1: (||fingertip - keypoint|| / FINGER_LENGTH)^2    -> [0, inf)
    C2: (1 - dot(gripper_Z, approach_dir))^2            -> [0, 4]
    C3: 1 - dot(gripper_Y, grasp_axis)^2                -> [0, 1]
    C4: sum max(0, 1 - ||fingertip - p_avoid|| / margin)^2
    P1: eps * ||rvec||^2

Pick mode candidate selection:
    Each grasp candidate runs a full SLSQP (200 iter); the lowest-cost candidate is selected.
    The best interaction point emerges from optimisation, not from hardcoded priority.
"""

from typing import Dict, Tuple, Callable, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from configs.SAP import get_sap_strict, SAP
from modules.vlmDecider import VLMDecision
from utils import CoordinateTransformer


W_APPROACH_POS = 1.0    # C1 weight (fixed)
W_APPROACH_ROT = 1.0    # C2 weight (fixed; range [0,4] due to (1-dot)^2 formula)
EPS_RVEC_REG   = 1e-6   # P1 weight: prevents Rodrigues singularity at rvec=0

# hand link origin -> fingertip pad centre along hand Z axis
# from panda.xml: finger pos=0.0584 + pad_1 centre=0.0445 = 0.1029m ~ 0.103m
FINGER_LENGTH = 0.103


def _compute_actual_approach(
        part_name:     str,
        sap:           SAP,
        keypoint_3d:   np.ndarray,
        object_center: np.ndarray,
) -> np.ndarray:
    """
    Runtime correction of approach_direction.

    Lateral parts (handle, neck, body): replace SAP reference [1,0,0] with the
    horizontal unit vector from keypoint toward object center, so the gripper
    approaches from outside the object.
    Other parts (cap, surface): use SAP approach_direction directly.

    Returns normalised approach vector, shape=(3,).
    """
    LATERAL_PARTS = {"handle", "neck", "body"}

    if part_name in LATERAL_PARTS:
        # inward direction from keypoint toward object center = approach from outside
        delta_xy = object_center[:2] - keypoint_3d[:2]
        norm = np.linalg.norm(delta_xy)
        if norm < 1e-6:
            return sap.approach_direction.copy()
        return np.array([delta_xy[0] / norm, delta_xy[1] / norm, 0.0])
    else:
        return sap.approach_direction.copy()


def _compute_rvec_init(
        approach_dir: np.ndarray,
        grasp_axis:   np.ndarray,
) -> np.ndarray:
    """
    Compute initial rotation vector via cross product, avoiding the zero-gradient
    dead zone at rvec=[0,0,0].

    Constructs R_init such that:
        R_init @ [0,0,1] = approach_dir  (gripper Z aligned to approach)
        R_init @ [0,1,0] = grasp_axis    (gripper Y aligned to grasp axis)
        R_init @ [1,0,0] = grasp_axis x approach_dir  (right-hand rule)

    Prerequisite: approach_dir perp grasp_axis (guaranteed by SAP design).
    Returns axis-angle rvec, shape=(3,).
    """
    z_col = approach_dir.copy()
    y_col = grasp_axis.copy()
    x_col = np.cross(grasp_axis, approach_dir)

    norm_x = np.linalg.norm(x_col)
    if norm_x < 1e-8:
        return np.zeros(3)
    x_col = x_col / norm_x

    R_init = np.column_stack([x_col, y_col, z_col])

    try:
        rvec = Rotation.from_matrix(R_init).as_rotvec()
    except Exception:
        rvec = _rodrigues_inverse(R_init)

    return rvec


def _rodrigues_inverse(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector. Fallback when scipy is unavailable."""
    trace_val = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = np.arccos(trace_val)
    if theta < 1e-10:
        return np.zeros(3)
    k = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))
    return k * theta


def _build_cost_fn(
        keypoint_3d:  np.ndarray,
        approach_dir: np.ndarray,
        grasp_axis:   Optional[np.ndarray],
        avoid_kps:    Dict[str, Tuple[np.ndarray, float]],
        vlm_decision: VLMDecision,
        mode:         str,
) -> Tuple[Callable, Callable]:
    """
    Build cost_fn and cost_breakdown_fn for a single candidate keypoint.

    All parameters are captured by closure; no mutable state is shared.
    Returns (cost_fn, cost_breakdown_fn).
    """
    _kp   = keypoint_3d.copy()
    _app  = approach_dir.copy()
    _gavt = grasp_axis.copy() if grasp_axis is not None else None
    _avd  = {k: (v[0].copy(), v[1]) for k, v in avoid_kps.items()}
    _L    = FINGER_LENGTH

    _wc1 = W_APPROACH_POS
    _wc2 = W_APPROACH_ROT
    _eps  = EPS_RVEC_REG
    _wc3  = vlm_decision.w_grasp_axis if mode == "pick" else 0.0
    _wc4  = vlm_decision.w_safety

    def _compute_intermediates(x: np.ndarray):
        pos       = x[:3]
        rvec      = x[3:]
        R         = CoordinateTransformer.rodrigues(rvec)
        gripper_z = R @ np.array([0.0, 0.0, 1.0])
        gripper_y = R @ np.array([0.0, 1.0, 0.0])
        fingertip = pos + gripper_z * _L
        return gripper_z, gripper_y, fingertip, rvec

    def _c1_position(fingertip: np.ndarray) -> float:
        return (float(np.linalg.norm(fingertip - _kp)) / _L) ** 2

    def _c2_approach(gripper_z: np.ndarray) -> float:
        # (1-dot)^2: aligned=0, wrist-flip=4
        d = float(np.clip(np.dot(gripper_z, _app), -1.0, 1.0))
        return (1.0 - d) ** 2

    def _c3_grasp_axis(gripper_y: np.ndarray) -> float:
        # dot^2 makes +Y and -Y equivalent (symmetric gripper)
        if _gavt is None:
            return 0.0
        d = float(np.clip(np.dot(gripper_y, _gavt), -1.0, 1.0))
        return 1.0 - d ** 2

    def _c4_safety(fingertip: np.ndarray):
        total    = 0.0
        per_part = {}
        for pname, (p_avoid, margin) in _avd.items():
            dist = float(np.linalg.norm(fingertip - p_avoid))
            v    = max(0.0, 1.0 - dist / margin)
            val  = v * v
            per_part[pname] = val
            total += val
        return total, per_part

    def _p1_rvec_reg(rvec: np.ndarray) -> float:
        return float(np.dot(rvec, rvec))

    def cost_fn(x: np.ndarray) -> float:
        gz, gy, fingertip, rvec = _compute_intermediates(x)
        c1    = _c1_position(fingertip)
        c2    = _c2_approach(gz)
        c3    = _c3_grasp_axis(gy)
        c4, _ = _c4_safety(fingertip)
        p1    = _p1_rvec_reg(rvec)
        return float(_wc1*c1 + _wc2*c2 + _wc3*c3 + _wc4*c4 + _eps*p1)

    def cost_breakdown_fn(x: np.ndarray) -> Dict:
        gz, gy, fingertip, rvec = _compute_intermediates(x)
        c1               = _c1_position(fingertip)
        c2               = _c2_approach(gz)
        c3               = _c3_grasp_axis(gy)
        c4, safety_parts = _c4_safety(fingertip)
        p1               = _p1_rvec_reg(rvec)
        total = _wc1*c1 + _wc2*c2 + _wc3*c3 + _wc4*c4 + _eps*p1
        return {
            'total':           float(total),
            'approach_pos':    float(_wc1 * c1),
            'approach_rot':    float(_wc2 * c2),
            'grasp_axis':      float(_wc3 * c3),
            'safety':          float(_wc4 * c4),
            'safety_per_part': safety_parts,
            'rvec_reg':        float(_eps  * p1),
        }

    return cost_fn, cost_breakdown_fn


class ConstraintInstantiator:
    """
    Builds SLSQP-ready cost functions from keypoints + SAP + VLMDecision.

    Pick mode: runs full SLSQP (200 iter) per grasp candidate; returns the lowest-cost
               candidate's cost_fn and x0 for PoseSolver to refine.
    Place mode: single surface target; builds C1/C2/C4 cost function.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def instantiate(
            self,
            keypoints_3d: Dict[str, np.ndarray],
            vlm_decision: VLMDecision,
            T_current:    np.ndarray,
    ) -> Tuple[Callable, np.ndarray, Dict]:
        """
        Instantiate cost function (auto-selects pick or place mode).

        Args:
            keypoints_3d: {part_name -> world-frame 3D point, shape=(3,)}
            vlm_decision: VLMDecision from VLMDecider.decide()
            T_current:    shape=(4,4) current end-effector transform from FK(q_current)

        Returns:
            (cost_fn, x0, meta)

        Raises:
            ValueError: keypoints_3d contains both grasp and place parts (semantic conflict)
            ValueError: keypoints_3d contains neither grasp nor place parts
            KeyError:   part name not in SAP knowledge base
        """
        grasp_kps: Dict[str, np.ndarray]               = {}
        place_kps: Dict[str, np.ndarray]               = {}
        avoid_kps: Dict[str, Tuple[np.ndarray, float]] = {}

        for part_name, point in keypoints_3d.items():
            sap = get_sap_strict(part_name)
            pt  = np.array(point, dtype=np.float64)
            if sap.contact_mode == 'grasp':
                grasp_kps[part_name] = pt
            elif sap.contact_mode == 'place':
                place_kps[part_name] = pt
            else:
                avoid_kps[part_name] = (pt, sap.safety_margin)

        if grasp_kps and place_kps:
            raise ValueError(
                f"keypoints_3d contains both grasp parts {list(grasp_kps.keys())} "
                f"and place parts {list(place_kps.keys())}. "
                f"Call instantiate() separately for pick and place targets."
            )
        if not grasp_kps and not place_kps:
            raise ValueError(
                f"keypoints_3d has neither grasp nor place parts. "
                f"Parts provided: {list(keypoints_3d.keys())}"
            )

        if grasp_kps:
            return self._instantiate_pick(grasp_kps, avoid_kps, vlm_decision, T_current)
        else:
            return self._instantiate_place(place_kps, avoid_kps, vlm_decision, T_current)

    def _instantiate_pick(
            self,
            grasp_kps:    Dict[str, np.ndarray],
            avoid_kps:    Dict[str, Tuple[np.ndarray, float]],
            vlm_decision: VLMDecision,
            T_current:    np.ndarray,
    ) -> Tuple[Callable, np.ndarray, Dict]:
        """
        Pick mode: run full SLSQP per grasp candidate and return the lowest-cost result.
        """
        avoid_pts = [v[0] for v in avoid_kps.values()]
        if avoid_pts:
            object_center = np.mean(np.stack(avoid_pts), axis=0)
        else:
            object_center = np.mean(np.stack(list(grasp_kps.values())), axis=0)

        if self.verbose:
            print(f"[ConstraintInstantiator] pick  "
                  f"grasp={list(grasp_kps.keys())}  avoid={list(avoid_kps.keys())}  "
                  f"vlm={vlm_decision}")

        best_cost         = float('inf')
        best_cost_fn      = None
        best_breakdown_fn = None
        best_x0           = None
        best_meta_part    = None

        for part_name, keypoint_3d in grasp_kps.items():
            sap          = get_sap_strict(part_name)
            approach_dir = _compute_actual_approach(part_name, sap, keypoint_3d, object_center)
            x0_pos       = keypoint_3d - approach_dir * FINGER_LENGTH
            x0_rot       = _compute_rvec_init(approach_dir, sap.grasp_axis)
            x0           = np.concatenate([x0_pos, x0_rot])

            cost_fn, breakdown_fn = _build_cost_fn(
                keypoint_3d  = keypoint_3d,
                approach_dir = approach_dir,
                grasp_axis   = sap.grasp_axis,
                avoid_kps    = avoid_kps,
                vlm_decision = vlm_decision,
                mode         = "pick",
            )

            result   = minimize(cost_fn, x0, method='SLSQP',
                                options={'maxiter': 200, 'ftol': 1e-6, 'disp': False})
            cost_val = float(result.fun)

            if self.verbose:
                print(f"  [{part_name}] approach={np.round(approach_dir, 3)} cost={cost_val:.4f}")

            if cost_val < best_cost:
                best_cost         = cost_val
                best_cost_fn      = cost_fn
                best_breakdown_fn = breakdown_fn
                best_x0           = result.x
                best_meta_part    = {
                    'part_name':    part_name,
                    'approach_dir': approach_dir,
                    'keypoint_3d':  keypoint_3d,
                    'grasp_axis':   sap.grasp_axis,
                }

        if self.verbose:
            print(f"[ConstraintInstantiator] best candidate: {best_meta_part['part_name']} "
                  f"(cost={best_cost:.4f})")

        meta = {
            'mode':                'pick',
            'grasp_target':        best_meta_part['part_name'],
            'avoid_targets':       list(avoid_kps.keys()),
            'object_center':       object_center,
            'approach_direction':  best_meta_part['approach_dir'],
            'keypoint_3d':         best_meta_part['keypoint_3d'],
            'grasp_axis_target':   best_meta_part['grasp_axis'],
            'vlm_decision':        vlm_decision,
            'cost_breakdown_fn':   best_breakdown_fn,
            'candidate_best_cost': best_cost,
        }

        return best_cost_fn, best_x0, meta

    def _instantiate_place(
            self,
            place_kps:    Dict[str, np.ndarray],
            avoid_kps:    Dict[str, Tuple[np.ndarray, float]],
            vlm_decision: VLMDecision,
            T_current:    np.ndarray,
    ) -> Tuple[Callable, np.ndarray, Dict]:
        """
        Place mode: single surface target, approach fixed to [0,0,-1] (top-down).
        Builds C1/C2/C4 cost function (C3 disabled, grasp_axis=None).
        """
        place_target_name  = next(iter(place_kps))
        place_target_point = place_kps[place_target_name]
        place_sap          = get_sap_strict(place_target_name)

        approach_dir = place_sap.approach_direction.copy()  # [0,0,-1]
        x0_pos       = place_target_point - approach_dir * FINGER_LENGTH
        R_current    = T_current[:3, :3]

        try:
            x0_rot = Rotation.from_matrix(R_current).as_rotvec()
        except Exception:
            x0_rot = _rodrigues_inverse(R_current)

        x0 = np.concatenate([x0_pos, x0_rot])

        cost_fn, breakdown_fn = _build_cost_fn(
            keypoint_3d  = place_target_point,
            approach_dir = approach_dir,
            grasp_axis   = None,
            avoid_kps    = avoid_kps,
            vlm_decision = vlm_decision,
            mode         = "place",
        )

        if self.verbose:
            print(f"[ConstraintInstantiator] place  "
                  f"target={place_target_name}  avoid={list(avoid_kps.keys())}  "
                  f"surface={np.round(place_target_point, 3)}  vlm={vlm_decision}")

        meta = {
            'mode':               'place',
            'place_target':       place_target_name,
            'avoid_targets':      list(avoid_kps.keys()),
            'approach_direction': approach_dir,
            'keypoint_3d':        place_target_point,
            'vlm_decision':       vlm_decision,
            'cost_breakdown_fn':  breakdown_fn,
        }

        return cost_fn, x0, meta