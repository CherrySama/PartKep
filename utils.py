"""
Created by Yinghao Ho on 2026-1-24
"""

import numpy as np
from typing import Tuple, List, Union, Optional, TYPE_CHECKING

# 避免循环导入：仅在类型检查时导入CameraConfig
if TYPE_CHECKING:
    from configs.camera_config import CameraConfig


class CoordinateTransformer:
    """
    坐标变换工具类

    主要功能：
        1.  归一化坐标 → 绝对浮点坐标
        2.  绝对浮点坐标 → 裁剪用整数坐标
        3.  ROI坐标 → 原图坐标（支持单点和批量）
        4.  深度图双线性插值（保留亚像素精度）
        5.  坐标变换验证
        6.  像素坐标 → 相机3D坐标
        7.  相机3D坐标 → 世界3D坐标
        8.  统一关键点3D坐标获取接口（支持仿真/真实两种模式）
        9.  轴角向量 → 旋转矩阵（Rodrigues公式）        ← 新增
        10. 位置 + 旋转矩阵 → SE3齐次变换矩阵           ← 新增

    坐标系统说明：
        - 归一化坐标：[0, 1] 范围，与图像尺寸无关
        - 绝对浮点坐标：像素单位，保留小数精度
        - 裁剪整数坐标：用于数组索引，必须是整数
        - ROI坐标：相对于ROI左上角的坐标
        - 原图坐标：相对于原始图像左上角的坐标
        - 相机3D坐标：以相机光心为原点的3D坐标系（单位：米）
        - 世界3D坐标：固定世界坐标系（单位：米）

    精度保留策略：
        - 全程使用浮点数运算
        - 仅在必须裁剪时才转换为整数
        - 使用双线性插值获取亚像素位置的深度值
    """

    # ==================== 2D 坐标变换 ====================

    @staticmethod
    def normalize_to_absolute(bbox_norm: Union[List[float], np.ndarray],
                              img_w: int,
                              img_h: int) -> Tuple[float, float, float, float]:
        """
        归一化坐标 → 绝对浮点坐标

        Args:
            bbox_norm: [x1, y1, x2, y2] 归一化坐标，范围 [0, 1]
            img_w: 图像宽度（像素）
            img_h: 图像高度（像素）

        Returns:
            (x1_float, y1_float, x2_float, y2_float): 绝对像素坐标（浮点数）
        """
        if len(bbox_norm) != 4:
            raise ValueError(
                f"bbox_norm必须包含4个元素，当前有 {len(bbox_norm)} 个"
            )
        x1_float = float(bbox_norm[0]) * img_w
        y1_float = float(bbox_norm[1]) * img_h
        x2_float = float(bbox_norm[2]) * img_w
        y2_float = float(bbox_norm[3]) * img_h
        return x1_float, y1_float, x2_float, y2_float

    @staticmethod
    def absolute_to_normalize(bbox_abs: Union[List[float], np.ndarray],
                              img_w: int,
                              img_h: int) -> Tuple[float, float, float, float]:
        """
        绝对坐标 → 归一化坐标

        Args:
            bbox_abs: [x1, y1, x2, y2] 绝对像素坐标
            img_w: 图像宽度（像素）
            img_h: 图像高度（像素）

        Returns:
            (x1_norm, y1_norm, x2_norm, y2_norm): 归一化坐标
        """
        if len(bbox_abs) != 4:
            raise ValueError(
                f"bbox_abs必须包含4个元素，当前有 {len(bbox_abs)} 个"
            )
        if img_w <= 0 or img_h <= 0:
            raise ValueError(
                f"图像尺寸必须为正数，当前: width={img_w}, height={img_h}"
            )
        x1_norm = float(bbox_abs[0]) / img_w
        y1_norm = float(bbox_abs[1]) / img_h
        x2_norm = float(bbox_abs[2]) / img_w
        y2_norm = float(bbox_abs[3]) / img_h
        return x1_norm, y1_norm, x2_norm, y2_norm

    @staticmethod
    def get_crop_bbox(x1_float: float, y1_float: float,
                      x2_float: float, y2_float: float,
                      img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """
        获取用于裁剪的整数bbox（带边界检查）

        Args:
            x1_float, y1_float: 左上角坐标（浮点数）
            x2_float, y2_float: 右下角坐标（浮点数）
            img_w: 图像宽度
            img_h: 图像高度

        Returns:
            (x1_crop, y1_crop, x2_crop, y2_crop): 裁剪用整数坐标
        """
        if img_w <= 0 or img_h <= 0:
            raise ValueError(
                f"图像尺寸必须为正数，当前: width={img_w}, height={img_h}"
            )
        x1_crop = max(0, min(int(x1_float), img_w - 1))
        y1_crop = max(0, min(int(y1_float), img_h - 1))
        x2_crop = max(0, min(int(x2_float), img_w))
        y2_crop = max(0, min(int(y2_float), img_h))

        if x2_crop <= x1_crop:
            x2_crop = min(x1_crop + 1, img_w)
        if y2_crop <= y1_crop:
            y2_crop = min(y1_crop + 1, img_h)

        return x1_crop, y1_crop, x2_crop, y2_crop

    @staticmethod
    def transform_point_roi_to_original(
            point_roi: Tuple[float, float],
            roi_offset_x: int,
            roi_offset_y: int) -> Tuple[float, float]:
        """
        单点坐标变换：ROI坐标系 → 原图坐标系

        Args:
            point_roi: (x_roi, y_roi) ROI坐标系下的点坐标
            roi_offset_x: ROI在原图中的x偏移量
            roi_offset_y: ROI在原图中的y偏移量

        Returns:
            (x_original, y_original): 原图坐标系下的点坐标
        """
        x_roi, y_roi = point_roi
        return float(x_roi) + roi_offset_x, float(y_roi) + roi_offset_y

    @staticmethod
    def transform_points_batch(points_roi: np.ndarray,
                               roi_offset_x: int,
                               roi_offset_y: int) -> np.ndarray:
        """
        批量点坐标变换：ROI坐标系 → 原图坐标系

        Args:
            points_roi: (N, 2) numpy array，ROI坐标系下的N个点
            roi_offset_x: ROI在原图中的x偏移量
            roi_offset_y: ROI在原图中的y偏移量

        Returns:
            (N, 2) numpy array，原图坐标系下的N个点
        """
        if not isinstance(points_roi, np.ndarray):
            points_roi = np.array(points_roi)
        if points_roi.ndim != 2 or points_roi.shape[1] != 2:
            raise ValueError(
                f"points_roi必须是 (N, 2) 形状的数组，"
                f"当前形状: {points_roi.shape}"
            )
        points_original = points_roi.copy().astype(np.float64)
        points_original[:, 0] += roi_offset_x
        points_original[:, 1] += roi_offset_y
        return points_original

    @staticmethod
    def get_depth_bilinear(depth_map: np.ndarray,
                           x: float,
                           y: float) -> float:
        """
        双线性插值获取浮点坐标的深度值

        Args:
            depth_map: (H, W) 深度图，单位通常为毫米
            x: x坐标（浮点数），范围 [0, W)
            y: y坐标（浮点数），范围 [0, H)

        Returns:
            插值后的深度值（浮点数），超出边界返回 0.0
        """
        h, w = depth_map.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return 0.0

        x0, x1 = int(np.floor(x)), min(int(np.floor(x)) + 1, w - 1)
        y0, y1 = int(np.floor(y)), min(int(np.floor(y)) + 1, h - 1)
        dx, dy = x - x0, y - y0

        d00 = float(depth_map[y0, x0])
        d01 = float(depth_map[y0, x1])
        d10 = float(depth_map[y1, x0])
        d11 = float(depth_map[y1, x1])

        return (d00 * (1 - dx) * (1 - dy) +
                d01 * dx * (1 - dy) +
                d10 * (1 - dx) * dy +
                d11 * dx * dy)

    @staticmethod
    def verify_transform(bbox_norm: List[float],
                         img_w: int,
                         img_h: int,
                         point_roi: Tuple[float, float],
                         point_original: Tuple[float, float],
                         eps: float = 1e-6) -> bool:
        """坐标变换验证（调试用）"""
        x1_f, y1_f, x2_f, y2_f = CoordinateTransformer.normalize_to_absolute(
            bbox_norm, img_w, img_h
        )
        x1_c, y1_c, x2_c, y2_c = CoordinateTransformer.get_crop_bbox(
            x1_f, y1_f, x2_f, y2_f, img_w, img_h
        )
        x_calc, y_calc = CoordinateTransformer.transform_point_roi_to_original(
            point_roi, x1_c, y1_c
        )
        x_diff = abs(x_calc - point_original[0])
        y_diff = abs(y_calc - point_original[1])
        is_correct = (x_diff < eps and y_diff < eps)
        if not is_correct:
            print(f"⚠️ 坐标变换验证失败: "
                  f"预期{point_original}，计算({x_calc:.6f},{y_calc:.6f})")
        return is_correct

    @staticmethod
    def get_roi_info(bbox_norm: List[float],
                     img_w: int,
                     img_h: int) -> dict:
        """获取ROI的完整信息（调试用）"""
        x1_f, y1_f, x2_f, y2_f = CoordinateTransformer.normalize_to_absolute(
            bbox_norm, img_w, img_h
        )
        x1_c, y1_c, x2_c, y2_c = CoordinateTransformer.get_crop_bbox(
            x1_f, y1_f, x2_f, y2_f, img_w, img_h
        )
        return {
            'bbox_norm':  bbox_norm,
            'bbox_float': (x1_f, y1_f, x2_f, y2_f),
            'bbox_crop':  (x1_c, y1_c, x2_c, y2_c),
            'roi_size':   (x2_c - x1_c, y2_c - y1_c),
            'offset':     (x1_c, y1_c)
        }

    # ==================== 3D 坐标转换 ====================

    @staticmethod
    def pixel_to_camera_3d(x_pixel: float,
                           y_pixel: float,
                           depth_m: float,
                           camera_config: 'CameraConfig') -> np.ndarray:
        """
        像素坐标 + 深度 → 相机坐标系3D点

        针孔相机反投影公式：
            X_cam = (x_pixel - cx) * depth / fx
            Y_cam = (y_pixel - cy) * depth / fy
            Z_cam = depth

        Args:
            x_pixel: 像素x坐标（浮点，支持亚像素精度）
            y_pixel: 像素y坐标
            depth_m: 深度值（单位：米，必须 > 0）
            camera_config: 提供内参 fx/fy/cx/cy

        Returns:
            np.ndarray shape=(3,)：相机坐标系3D点 [X, Y, Z]（米）
        """
        if depth_m <= 0:
            raise ValueError(f"深度值必须大于0，当前: {depth_m}")
        X_cam = (x_pixel - camera_config.cx) * depth_m / camera_config.fx
        Y_cam = (y_pixel - camera_config.cy) * depth_m / camera_config.fy
        return np.array([X_cam, Y_cam, float(depth_m)], dtype=np.float64)

    @staticmethod
    def camera_to_world_3d(point_camera: np.ndarray,
                           camera_config: 'CameraConfig') -> np.ndarray:
        """
        相机坐标系3D点 → 世界坐标系3D点

        P_world = T_cam2world @ [X_cam, Y_cam, Z_cam, 1]^T

        Args:
            point_camera: shape=(3,) 相机坐标系3D点（米）
            camera_config: 提供外参矩阵 extrinsic_matrix (4×4)

        Returns:
            np.ndarray shape=(3,)：世界坐标系3D点（米）
        """
        point_camera = np.array(point_camera, dtype=np.float64).flatten()
        if point_camera.shape != (3,):
            raise ValueError(f"point_camera必须shape=(3,)，当前: {point_camera.shape}")
        point_homogeneous = np.append(point_camera, 1.0)
        point_world_h = camera_config.extrinsic_matrix @ point_homogeneous
        return point_world_h[:3]

    @staticmethod
    def get_keypoint_3d(
            mode: str,
            camera_config: Optional['CameraConfig'] = None,
            x_pixel: Optional[float] = None,
            y_pixel: Optional[float] = None,
            depth_map: Optional[np.ndarray] = None,
            depth_m: Optional[float] = None,
            xyz_world: Optional[Union[np.ndarray, List[float]]] = None
    ) -> np.ndarray:
        """
        统一关键点3D世界坐标获取接口

        【真实模式 mode="real"】
            像素坐标 + 深度图/深度值 → 相机3D → 世界3D

        【仿真模式 mode="sim"】
            直接接收 MuJoCo 提供的世界坐标，只做格式统一：
                xyz = data.body('cup_handle').xpos.copy()
                point = get_keypoint_3d(mode="sim", xyz_world=xyz)

        Args:
            mode:          "real" 或 "sim"
            camera_config: 真实模式必须提供
            x_pixel:       真实模式必须提供
            y_pixel:       真实模式必须提供
            depth_map:     真实模式：(H, W) 深度图（毫米），与 depth_m 二选一
            depth_m:       真实模式：直接提供深度值（米），与 depth_map 二选一
            xyz_world:     仿真模式必须提供，shape=(3,)

        Returns:
            np.ndarray shape=(3,)：世界坐标系3D关键点（米）
        """
        if mode == "sim":
            if xyz_world is None:
                raise ValueError("仿真模式必须提供 xyz_world")
            result = np.array(xyz_world, dtype=np.float64).flatten()
            if result.shape != (3,):
                raise ValueError(f"xyz_world必须是3维向量，当前shape: {result.shape}")
            return result

        elif mode == "real":
            if camera_config is None:
                raise ValueError("真实模式必须提供 camera_config")
            if x_pixel is None or y_pixel is None:
                raise ValueError("真实模式必须提供 x_pixel 和 y_pixel")
            if depth_map is None and depth_m is None:
                raise ValueError("真实模式必须提供 depth_map 或 depth_m 之一")

            if depth_m is None:
                depth_mm = CoordinateTransformer.get_depth_bilinear(
                    depth_map, x_pixel, y_pixel
                )
                if depth_mm <= 0:
                    raise ValueError(
                        f"像素({x_pixel:.1f},{y_pixel:.1f})处深度值无效: {depth_mm}mm"
                    )
                depth_m = depth_mm / 1000.0

            point_camera = CoordinateTransformer.pixel_to_camera_3d(
                x_pixel, y_pixel, depth_m, camera_config
            )
            return CoordinateTransformer.camera_to_world_3d(point_camera, camera_config)

        else:
            raise ValueError(f"mode必须是 'real' 或 'sim'，当前: '{mode}'")

    # ==================== 旋转表示转换（新增） ====================

    @staticmethod
    def rodrigues(rvec: np.ndarray) -> np.ndarray:
        """
        轴角向量 → 3×3 旋转矩阵（Rodrigues 公式）

        这是全项目唯一的 Rodrigues 实现，constraintsInst.py 和
        solver.py 中的同名私有函数均已删除，统一调用此方法。

        公式推导：
            设 θ = ||rvec||，k = rvec / θ（单位旋转轴）
            R = I + sin(θ)·K + (1-cos(θ))·K²
            其中 K 是 k 的反对称矩阵（叉积矩阵）
            零向量 → 单位矩阵（无旋转）

        Args:
            rvec: shape=(3,) 轴角向量，方向为旋转轴，模长为旋转角（rad）

        Returns:
            np.ndarray shape=(3, 3)：旋转矩阵，满足 R^T R = I，det(R) = 1

        Examples:
            >>> CoordinateTransformer.rodrigues(np.zeros(3))
            array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

            >>> R = CoordinateTransformer.rodrigues(np.array([0, 0, np.pi/2]))
            >>> # 绕Z轴旋转90°：x→y，y→-x
        """
        rvec = np.array(rvec, dtype=np.float64)
        theta = np.linalg.norm(rvec)

        if theta < 1e-10:
            return np.eye(3)

        k = rvec / theta
        K = np.array([
            [  0.0, -k[2],  k[1]],
            [ k[2],   0.0, -k[0]],
            [-k[1],  k[0],   0.0]
        ])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    @staticmethod
    def rotation_to_matrix4x4(position: np.ndarray,
                               R: np.ndarray) -> np.ndarray:
        """
        位置向量 + 旋转矩阵 → 4×4 齐次变换矩阵（纯 numpy）

        T = [[R, p],
             [0, 1]]

        Args:
            position: shape=(3,) 末端执行器位置（世界坐标，米）
            R:        shape=(3,3) 末端执行器旋转矩阵

        Returns:
            np.ndarray shape=(4,4)：齐次变换矩阵，供 IKSolver 直接使用

        Example:
            >>> pos = np.array([0.5, 0.0, 0.3])
            >>> R   = CoordinateTransformer.rodrigues(np.zeros(3))
            >>> T   = CoordinateTransformer.rotation_to_matrix4x4(pos, R)
            >>> print(T.shape)  # (4, 4)
        """
        position = np.array(position, dtype=np.float64).flatten()
        R = np.array(R, dtype=np.float64)

        if position.shape != (3,):
            raise ValueError(f"position 必须 shape=(3,)，当前: {position.shape}")
        if R.shape != (3, 3):
            raise ValueError(f"R 必须 shape=(3,3)，当前: {R.shape}")

        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = position
        return T


# ==================== 模块测试 ====================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from configs.camera_config import CameraConfig

    print("=" * 70)
    print("测试 CoordinateTransformer")
    print("=" * 70)

    # ===== 原有功能快速验证 =====
    print("\n【1】2D坐标变换")
    img_w, img_h = 640, 480
    bbox_norm = [0.234567, 0.456789, 0.567890, 0.789012]
    x1_f, y1_f, x2_f, y2_f = CoordinateTransformer.normalize_to_absolute(
        bbox_norm, img_w, img_h
    )
    x1_c, y1_c, x2_c, y2_c = CoordinateTransformer.get_crop_bbox(
        x1_f, y1_f, x2_f, y2_f, img_w, img_h
    )
    point_roi = (106.7823, 79.4512)
    x_orig, y_orig = CoordinateTransformer.transform_point_roi_to_original(
        point_roi, x1_c, y1_c
    )
    ok = CoordinateTransformer.verify_transform(
        bbox_norm, img_w, img_h, point_roi, (x_orig, y_orig)
    )
    print(f"  ✅ 2D坐标变换验证: {ok}")

    # ===== 3D坐标转换快速验证 =====
    print("\n【2】3D坐标转换（仿真模式）")
    mujoco_xpos = np.array([0.156, 0.298, 0.045])
    p3d = CoordinateTransformer.get_keypoint_3d(mode="sim", xyz_world=mujoco_xpos)
    assert np.allclose(p3d, mujoco_xpos)
    print(f"  ✅ 仿真模式: {p3d}")

    # ===== 新增：Rodrigues 测试 =====
    print("\n【3】rodrigues - 轴角 → 旋转矩阵")

    # 零向量 → 单位矩阵
    R0 = CoordinateTransformer.rodrigues(np.zeros(3))
    assert np.allclose(R0, np.eye(3))
    print(f"  ✅ 零向量 → 单位矩阵")

    # 绕Z轴旋转90°
    R_90z = CoordinateTransformer.rodrigues(np.array([0.0, 0.0, np.pi / 2]))
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert np.allclose(R_90z, expected, atol=1e-6)
    print(f"  ✅ 绕Z轴90° → 正确旋转矩阵")

    # 任意旋转：正交性验证
    R_rand = CoordinateTransformer.rodrigues(np.array([0.3, -0.5, 0.8]))
    assert np.allclose(R_rand.T @ R_rand, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R_rand), 1.0, atol=1e-10)
    print(f"  ✅ 任意旋转满足 R^T R=I，det=1")

    # ===== 新增：rotation_to_matrix4x4 测试 =====
    print("\n【4】rotation_to_matrix4x4 - position + R → 4×4 矩阵")
    pos = np.array([0.5, 0.1, 0.3])
    R   = CoordinateTransformer.rodrigues(np.zeros(3))
    T   = CoordinateTransformer.rotation_to_matrix4x4(pos, R)

    assert T.shape == (4, 4)
    assert np.allclose(T[:3, 3], pos, atol=1e-10)
    assert np.allclose(T[:3, :3], R,  atol=1e-10)
    assert np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-10)
    print(f"  ✅ shape={T.shape}")
    print(f"  ✅ 平移部分: {T[:3, 3]}")
    print(f"  ✅ 最后一行: {T[3, :]}")

    print("\n" + "=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)
    