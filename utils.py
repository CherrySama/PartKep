"""
Created by Yinghao Ho on 2026-1-24
Updated on 2026-2-23: 新增3D坐标转换方法（pixel→camera→world）
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
        1. 归一化坐标 → 绝对浮点坐标
        2. 绝对浮点坐标 → 裁剪用整数坐标
        3. ROI坐标 → 原图坐标（支持单点和批量）
        4. 深度图双线性插值（保留亚像素精度）
        5. 坐标变换验证
        6. 像素坐标 → 相机3D坐标（新增）
        7. 相机3D坐标 → 世界3D坐标（新增）
        8. 统一关键点3D坐标获取接口（新增，支持仿真/真实两种模式）
    
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

    # ==================== 原有方法（保持不变）====================

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

        x1_crop = int(x1_float)
        y1_crop = int(y1_float)
        x2_crop = int(x2_float)
        y2_crop = int(y2_float)

        x1_crop = max(0, min(x1_crop, img_w - 1))
        y1_crop = max(0, min(y1_crop, img_h - 1))
        x2_crop = max(0, min(x2_crop, img_w))
        y2_crop = max(0, min(y2_crop, img_h))

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
        x_original = float(x_roi) + roi_offset_x
        y_original = float(y_roi) + roi_offset_y

        return x_original, y_original

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

        x0 = int(np.floor(x))
        x1 = min(x0 + 1, w - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, h - 1)

        dx = x - x0
        dy = y - y0

        d00 = float(depth_map[y0, x0])
        d01 = float(depth_map[y0, x1])
        d10 = float(depth_map[y1, x0])
        d11 = float(depth_map[y1, x1])

        depth = (d00 * (1 - dx) * (1 - dy) +
                 d01 * dx * (1 - dy) +
                 d10 * (1 - dx) * dy +
                 d11 * dx * dy)

        return depth

    @staticmethod
    def verify_transform(bbox_norm: List[float],
                         img_w: int,
                         img_h: int,
                         point_roi: Tuple[float, float],
                         point_original: Tuple[float, float],
                         eps: float = 1e-6) -> bool:
        """
        验证坐标变换是否正确
        
        Args:
            bbox_norm: 归一化bbox [x1, y1, x2, y2]
            img_w: 图像宽度
            img_h: 图像高度
            point_roi: ROI坐标系下的点
            point_original: 预期的原图坐标系下的点
            eps: 允许的浮点误差
        
        Returns:
            True: 坐标变换正确，False: 有误差
        """
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
            print(f"⚠️ 坐标变换验证失败:")
            print(f"  预期原图坐标: "
                  f"({point_original[0]:.6f}, {point_original[1]:.6f})")
            print(f"  计算原图坐标: ({x_calc:.6f}, {y_calc:.6f})")
            print(f"  误差: (Δx={x_diff:.6f}, Δy={y_diff:.6f})")

        return is_correct

    @staticmethod
    def get_roi_info(bbox_norm: List[float],
                     img_w: int,
                     img_h: int) -> dict:
        """
        获取ROI的完整信息（便于调试）
        
        Args:
            bbox_norm: 归一化bbox
            img_w: 图像宽度
            img_h: 图像高度
        
        Returns:
            字典：bbox_norm / bbox_float / bbox_crop / roi_size / offset
        """
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

    # ==================== 新增方法：3D坐标转换 ====================

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
            x_pixel: 像素x坐标（可以是浮点数，支持亚像素精度）
            y_pixel: 像素y坐标
            depth_m: 该像素对应的深度值（单位：米）
                     注意：如果深度图单位是毫米，需要先除以1000
            camera_config: CameraConfig实例，提供内参
        
        Returns:
            np.ndarray shape=(3,)：相机坐标系下的3D点 [X, Y, Z]（单位：米）
        
        Raises:
            ValueError: 如果depth_m <= 0
        
        Example:
            >>> config = CameraConfig.create_identity(fx=615.0, fy=615.0,
            ...                                        cx=320.0, cy=240.0)
            >>> point_3d = CoordinateTransformer.pixel_to_camera_3d(
            ...     x_pixel=320.0, y_pixel=240.0,
            ...     depth_m=1.0,
            ...     camera_config=config
            ... )
            >>> print(point_3d)  # [0. 0. 1.]（光心正前方1米）
        """
        if depth_m <= 0:
            raise ValueError(
                f"深度值必须大于0，当前: {depth_m}。"
                f"如果深度图单位是毫米，请先除以1000。"
            )

        X_cam = (x_pixel - camera_config.cx) * depth_m / camera_config.fx
        Y_cam = (y_pixel - camera_config.cy) * depth_m / camera_config.fy
        Z_cam = float(depth_m)

        return np.array([X_cam, Y_cam, Z_cam], dtype=np.float64)

    @staticmethod
    def camera_to_world_3d(point_camera: np.ndarray,
                           camera_config: 'CameraConfig') -> np.ndarray:
        """
        相机坐标系3D点 → 世界坐标系3D点
        
        利用4×4齐次外参矩阵进行变换：
            P_world = T_cam2world @ [X_cam, Y_cam, Z_cam, 1]^T
        
        Args:
            point_camera: shape=(3,) 相机坐标系下的3D点（单位：米）
            camera_config: CameraConfig实例，提供外参矩阵
        
        Returns:
            np.ndarray shape=(3,)：世界坐标系下的3D点（单位：米）
        
        Example:
            >>> # 外参为单位矩阵时，相机坐标 = 世界坐标
            >>> config = CameraConfig.create_identity()
            >>> p_cam = np.array([0.1, 0.2, 1.0])
            >>> p_world = CoordinateTransformer.camera_to_world_3d(p_cam, config)
            >>> print(p_world)  # [0.1, 0.2, 1.0]
        """
        point_camera = np.array(point_camera, dtype=np.float64).flatten()
        if point_camera.shape != (3,):
            raise ValueError(
                f"point_camera必须是shape=(3,)的数组，"
                f"当前shape: {point_camera.shape}"
            )

        # 转为齐次坐标 [X, Y, Z, 1]
        point_homogeneous = np.append(point_camera, 1.0)

        # 应用外参矩阵
        point_world_homogeneous = camera_config.extrinsic_matrix @ point_homogeneous

        # 返回3D坐标（去掉齐次分量）
        return point_world_homogeneous[:3]

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
        
        支持两种模式：
        
        【真实模式 mode="real"】
            通过深度相机数据计算3D世界坐标：
            像素坐标 + 深度图/深度值 → 相机3D坐标 → 世界3D坐标
        
        【仿真模式 mode="sim"】
            直接接收MuJoCo提供的3D世界坐标，只做格式统一：
                xyz = data.body('cup_handle').xpos.copy()
                point = get_keypoint_3d(mode="sim", xyz_world=xyz)
        
        Args:
            mode: "real" 或 "sim"
            
            --- 真实模式参数（mode="real"时必须提供）---
            camera_config: CameraConfig实例
            x_pixel: 关键点像素x坐标
            y_pixel: 关键点像素y坐标
            depth_map: (H, W) 深度图（毫米单位），与depth_m二选一
            depth_m: 直接提供深度值（米单位），与depth_map二选一
            
            --- 仿真模式参数（mode="sim"时必须提供）---
            xyz_world: MuJoCo提供的3D世界坐标，shape=(3,) 或 list[3]
        
        Returns:
            np.ndarray shape=(3,)：世界坐标系下的3D关键点坐标（单位：米）
        
        Raises:
            ValueError: mode不合法，或必要参数缺失
        
        Example（真实模式）:
            >>> config = CameraConfig.load_from_yaml("configs/camera.yaml")
            >>> depth_map = load_depth_image(...)  # 毫米单位
            >>> point_3d = CoordinateTransformer.get_keypoint_3d(
            ...     mode="real",
            ...     camera_config=config,
            ...     x_pixel=245.7, y_pixel=312.3,
            ...     depth_map=depth_map
            ... )
        
        Example（仿真模式）:
            >>> import mujoco
            >>> xyz = data.body('cup_handle').xpos.copy()
            >>> point_3d = CoordinateTransformer.get_keypoint_3d(
            ...     mode="sim",
            ...     xyz_world=xyz
            ... )
        """
        if mode == "sim":
            # ===== 仿真模式：格式统一，不做任何计算 =====
            if xyz_world is None:
                raise ValueError(
                    "仿真模式（mode='sim'）必须提供 xyz_world 参数。\n"
                    "示例：xyz = data.body('cup_handle').xpos.copy()"
                )
            result = np.array(xyz_world, dtype=np.float64).flatten()
            if result.shape != (3,):
                raise ValueError(
                    f"xyz_world必须是3维向量，当前shape: {result.shape}"
                )
            return result

        elif mode == "real":
            # ===== 真实模式：完整的像素→3D变换链 =====
            if camera_config is None:
                raise ValueError("真实模式（mode='real'）必须提供 camera_config")
            if x_pixel is None or y_pixel is None:
                raise ValueError(
                    "真实模式（mode='real'）必须提供 x_pixel 和 y_pixel"
                )
            if depth_map is None and depth_m is None:
                raise ValueError(
                    "真实模式（mode='real'）必须提供 depth_map 或 depth_m 之一"
                )

            # 获取深度值（米单位）
            if depth_m is None:
                # 从深度图读取（双线性插值，毫米→米）
                depth_mm = CoordinateTransformer.get_depth_bilinear(
                    depth_map, x_pixel, y_pixel
                )
                if depth_mm <= 0:
                    raise ValueError(
                        f"像素 ({x_pixel:.1f}, {y_pixel:.1f}) "
                        f"处深度值无效: {depth_mm}mm。"
                        f"请检查深度图是否有效或关键点是否在有效区域内。"
                    )
                depth_m = depth_mm / 1000.0  # 毫米 → 米

            # 第一步：像素坐标 → 相机3D坐标
            point_camera = CoordinateTransformer.pixel_to_camera_3d(
                x_pixel=x_pixel,
                y_pixel=y_pixel,
                depth_m=depth_m,
                camera_config=camera_config
            )

            # 第二步：相机3D坐标 → 世界3D坐标
            point_world = CoordinateTransformer.camera_to_world_3d(
                point_camera=point_camera,
                camera_config=camera_config
            )

            return point_world

        else:
            raise ValueError(
                f"mode必须是 'real' 或 'sim'，当前: '{mode}'"
            )


# ==================== 模块测试代码 ====================
if __name__ == "__main__":
    """
    测试 CoordinateTransformer（包含新增3D坐标转换方法）
    运行方式: python utils.py
    """
    import sys
    sys.path.insert(0, '.')
    from configs.camera_config import CameraConfig

    print("=" * 70)
    print("测试 CoordinateTransformer（含3D坐标转换）")
    print("=" * 70)

    # ==================== 原有测试（简化版）====================
    print("\n【原有功能】快速验证")
    print("-" * 70)

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
    is_correct = CoordinateTransformer.verify_transform(
        bbox_norm, img_w, img_h, point_roi, (x_orig, y_orig)
    )
    print(f"✅ 原有坐标变换功能正常，验证结果: {is_correct}")

    # ==================== 新增功能测试 ====================

    # 创建测试用相机配置
    # 模拟相机固定安装：相机在世界坐标系 (0, 0, 1.5)m 处，俯视向下
    extrinsic = np.array([
        [1.0,  0.0,  0.0,  0.0],
        [0.0, -1.0,  0.0,  0.0],
        [0.0,  0.0, -1.0,  1.5],
        [0.0,  0.0,  0.0,  1.0]
    ])
    config = CameraConfig(
        fx=615.0, fy=615.0,
        cx=320.0, cy=240.0,
        extrinsic_matrix=extrinsic
    )

    # 测试1：pixel_to_camera_3d
    print("\n【新增测试1】pixel_to_camera_3d")
    print("-" * 70)

    # 主点像素(320, 240)，深度1.0m → 应该在光轴上 [0, 0, 1]
    p_cam = CoordinateTransformer.pixel_to_camera_3d(
        x_pixel=320.0, y_pixel=240.0,
        depth_m=1.0,
        camera_config=config
    )
    print(f"主点像素(320, 240), depth=1.0m → 相机3D: {p_cam}")
    assert np.allclose(p_cam, [0.0, 0.0, 1.0], atol=1e-6), "主点反投影失败！"
    print("✅ 主点反投影正确（光轴方向）")

    # 非主点像素
    p_cam2 = CoordinateTransformer.pixel_to_camera_3d(
        x_pixel=320.0 + 61.5,  # 偏移61.5像素 = fx/10
        y_pixel=240.0,
        depth_m=1.0,
        camera_config=config
    )
    print(f"偏移像素(381.5, 240), depth=1.0m → 相机3D: {p_cam2}")
    assert np.allclose(p_cam2[0], 0.1, atol=1e-6), "X方向反投影失败！"
    print("✅ 偏移像素反投影正确")

    # 测试2：camera_to_world_3d
    print("\n【新增测试2】camera_to_world_3d")
    print("-" * 70)

    # 相机在(0,0,1.5)俯视，相机Z轴对应世界-Z轴
    # 相机坐标[0,0,1]应变换到世界坐标[0,0,0.5]（1.5-1.0=0.5）
    p_world = CoordinateTransformer.camera_to_world_3d(
        point_camera=np.array([0.0, 0.0, 1.0]),
        camera_config=config
    )
    print(f"相机坐标[0,0,1] → 世界坐标: {p_world}")
    assert np.allclose(p_world, [0.0, 0.0, 0.5], atol=1e-6), "世界坐标变换失败！"
    print("✅ 相机→世界变换正确")

    # 测试3：get_keypoint_3d（真实模式，使用depth_map）
    print("\n【新增测试3】get_keypoint_3d - 真实模式（depth_map）")
    print("-" * 70)

    # 创建模拟深度图（640×480，主点深度1000mm=1.0m）
    depth_map_test = np.ones((480, 640), dtype=np.uint16) * 1000  # 1000mm = 1.0m

    point_3d = CoordinateTransformer.get_keypoint_3d(
        mode="real",
        camera_config=config,
        x_pixel=320.0, y_pixel=240.0,
        depth_map=depth_map_test
    )
    print(f"真实模式关键点3D坐标: {point_3d}")
    assert np.allclose(point_3d, [0.0, 0.0, 0.5], atol=1e-3), "真实模式失败！"
    print("✅ 真实模式（depth_map）正确")

    # 测试4：get_keypoint_3d（仿真模式）
    print("\n【新增测试4】get_keypoint_3d - 仿真模式")
    print("-" * 70)

    # 模拟从MuJoCo读取的坐标
    mujoco_xpos = np.array([0.156, 0.298, 0.045])  # 模拟cup handle位置

    point_3d_sim = CoordinateTransformer.get_keypoint_3d(
        mode="sim",
        xyz_world=mujoco_xpos
    )
    print(f"仿真模式关键点3D坐标: {point_3d_sim}")
    assert np.allclose(point_3d_sim, mujoco_xpos), "仿真模式格式转换失败！"
    print("✅ 仿真模式正确（格式统一，数值不变）")

    # 测试5：错误处理
    print("\n【新增测试5】错误处理")
    print("-" * 70)

    try:
        CoordinateTransformer.get_keypoint_3d(mode="invalid")
    except ValueError as e:
        print(f"✅ 非法mode捕获: {e}")

    try:
        CoordinateTransformer.get_keypoint_3d(mode="sim", xyz_world=None)
    except ValueError as e:
        print(f"✅ sim模式缺少xyz_world捕获: {e}")

    try:
        CoordinateTransformer.get_keypoint_3d(
            mode="real", camera_config=config,
            x_pixel=320.0, y_pixel=240.0
            # 故意不提供depth_map或depth_m
        )
    except ValueError as e:
        print(f"✅ real模式缺少深度参数捕获: {e}")

    print()
    print("=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)