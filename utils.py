"""
Created by Yinghao Ho on 2026-1-24
"""


import numpy as np
from typing import Tuple, List, Union, Optional


class CoordinateTransformer:
    """
    坐标变换工具类
    
    主要功能：
        1. 归一化坐标 → 绝对浮点坐标
        2. 绝对浮点坐标 → 裁剪用整数坐标
        3. ROI坐标 → 原图坐标（支持单点和批量）
        4. 深度图双线性插值（保留亚像素精度）
        5. 坐标变换验证
    
    坐标系统说明：
        - 归一化坐标：[0, 1] 范围，与图像尺寸无关
        - 绝对浮点坐标：像素单位，保留小数精度
        - 裁剪整数坐标：用于数组索引，必须是整数
        - ROI坐标：相对于ROI左上角的坐标
        - 原图坐标：相对于原始图像左上角的坐标
    
    精度保留策略：
        - 全程使用浮点数运算
        - 仅在必须裁剪时才转换为整数
        - 使用双线性插值获取亚像素位置的深度值
    """
    
    @staticmethod
    def normalize_to_absolute(bbox_norm: Union[List[float], np.ndarray], 
                             img_w: int, 
                             img_h: int) -> Tuple[float, float, float, float]:
        """
        归一化坐标 → 绝对浮点坐标
        
        将归一化的边界框坐标（范围[0,1]）转换为绝对像素坐标（浮点数）。
        此函数保留完整的浮点精度，不进行取整。
        
        Args:
            bbox_norm: [x1, y1, x2, y2] 归一化坐标，范围 [0, 1]
                - x1, y1: 左上角坐标
                - x2, y2: 右下角坐标
            img_w: 图像宽度（像素）
            img_h: 图像高度（像素）
        
        Returns:
            (x1_float, y1_float, x2_float, y2_float): 绝对像素坐标（浮点数）
        
        Example:
            >>> bbox_norm = [0.25, 0.5, 0.75, 0.9]
            >>> img_w, img_h = 640, 480
            >>> x1, y1, x2, y2 = CoordinateTransformer.normalize_to_absolute(
            ...     bbox_norm, img_w, img_h
            ... )
            >>> print(f"Absolute coords: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
            Absolute coords: (160.00, 240.00, 480.00, 432.00)
        
        Notes:
            - 输出坐标可能超出图像边界，需要后续边界检查
            - 保留浮点精度，不进行任何取整操作
        """
        if len(bbox_norm) != 4:
            raise ValueError(f"bbox_norm必须包含4个元素，当前有 {len(bbox_norm)} 个")
        
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
        
        将绝对像素坐标转换为归一化坐标（范围[0,1]）。
        
        Args:
            bbox_abs: [x1, y1, x2, y2] 绝对像素坐标
            img_w: 图像宽度（像素）
            img_h: 图像高度（像素）
        
        Returns:
            (x1_norm, y1_norm, x2_norm, y2_norm): 归一化坐标
        
        Example:
            >>> bbox_abs = [160.0, 240.0, 480.0, 432.0]
            >>> img_w, img_h = 640, 480
            >>> x1, y1, x2, y2 = CoordinateTransformer.absolute_to_normalize(
            ...     bbox_abs, img_w, img_h
            ... )
            >>> print(f"Normalized: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")
            Normalized: [0.25, 0.50, 0.75, 0.90]
        """
        if len(bbox_abs) != 4:
            raise ValueError(f"bbox_abs必须包含4个元素，当前有 {len(bbox_abs)} 个")
        
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"图像尺寸必须为正数，当前: width={img_w}, height={img_h}")
        
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
        
        将浮点数坐标转换为整数坐标，用于实际的图像裁剪操作。
        包含边界检查，确保坐标在图像范围内。
        
        Args:
            x1_float, y1_float: 左上角坐标（浮点数）
            x2_float, y2_float: 右下角坐标（浮点数）
            img_w: 图像宽度
            img_h: 图像高度
        
        Returns:
            (x1_crop, y1_crop, x2_crop, y2_crop): 裁剪用整数坐标
        
        Notes:
            - 使用 int() 截断小数部分（向零取整）
            - 自动进行边界检查和修正
            - 确保 x2 > x1 且 y2 > y1（至少差1像素）
        
        Example:
            >>> x1_f, y1_f = 150.7823, 219.4512
            >>> x2_f, y2_f = 363.2156, 378.8901
            >>> img_w, img_h = 640, 480
            >>> x1, y1, x2, y2 = CoordinateTransformer.get_crop_bbox(
            ...     x1_f, y1_f, x2_f, y2_f, img_w, img_h
            ... )
            >>> print(f"Crop bbox: [{x1}, {y1}, {x2}, {y2}]")
            Crop bbox: [150, 219, 363, 378]
        
        Raises:
            ValueError: 如果图像尺寸无效
        """
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"图像尺寸必须为正数，当前: width={img_w}, height={img_h}")
        
        # 转换为整数（截断小数部分）
        x1_crop = int(x1_float)
        y1_crop = int(y1_float)
        x2_crop = int(x2_float)
        y2_crop = int(y2_float)
        
        # 边界检查：确保坐标在 [0, img_size] 范围内
        x1_crop = max(0, min(x1_crop, img_w - 1))
        y1_crop = max(0, min(y1_crop, img_h - 1))
        x2_crop = max(0, min(x2_crop, img_w))
        y2_crop = max(0, min(y2_crop, img_h))
        
        # 确保有效的ROI（至少1像素大小）
        if x2_crop <= x1_crop:
            x2_crop = min(x1_crop + 1, img_w)
        if y2_crop <= y1_crop:
            y2_crop = min(y1_crop + 1, img_h)
        
        return x1_crop, y1_crop, x2_crop, y2_crop
    
    @staticmethod
    def transform_point_roi_to_original(point_roi: Tuple[float, float],
                                       roi_offset_x: int,
                                       roi_offset_y: int) -> Tuple[float, float]:
        """
        单点坐标变换：ROI坐标系 → 原图坐标系
        
        将ROI坐标系下的点转换到原图坐标系。
        保留浮点精度，适用于亚像素级别的关键点。
        
        Args:
            point_roi: (x_roi, y_roi) ROI坐标系下的点坐标（浮点数）
            roi_offset_x: ROI在原图中的x偏移量（即 x1_crop）
            roi_offset_y: ROI在原图中的y偏移量（即 y1_crop）
        
        Returns:
            (x_original, y_original): 原图坐标系下的点坐标（浮点数）
        
        Notes:
            - 变换公式：original = roi + offset
            - 完全保留浮点精度
            - offset是整数，但结果仍为浮点数
        
        Example:
            >>> # ROI左上角在原图的 (150, 219)
            >>> point_roi = (106.7823, 79.4512)  # ROI中的点
            >>> roi_offset_x, roi_offset_y = 150, 219
            >>> x_orig, y_orig = CoordinateTransformer.transform_point_roi_to_original(
            ...     point_roi, roi_offset_x, roi_offset_y
            ... )
            >>> print(f"Original coords: ({x_orig:.4f}, {y_orig:.4f})")
            Original coords: (256.7823, 298.4512)
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
        
        批量处理多个点的坐标变换，效率更高。
        
        Args:
            points_roi: (N, 2) numpy array，ROI坐标系下的N个点
                - points_roi[:, 0] 是 x 坐标
                - points_roi[:, 1] 是 y 坐标
            roi_offset_x: ROI在原图中的x偏移量
            roi_offset_y: ROI在原图中的y偏移量
        
        Returns:
            (N, 2) numpy array，原图坐标系下的N个点
        
        Example:
            >>> points_roi = np.array([
            ...     [10.5, 20.3],
            ...     [30.7, 40.1],
            ...     [50.2, 60.9]
            ... ])
            >>> roi_offset_x, roi_offset_y = 100, 200
            >>> points_orig = CoordinateTransformer.transform_points_batch(
            ...     points_roi, roi_offset_x, roi_offset_y
            ... )
            >>> print(points_orig)
            [[110.5 220.3]
             [130.7 240.1]
             [150.2 260.9]]
        
        Raises:
            ValueError: 如果输入数组形状不正确
        """
        if not isinstance(points_roi, np.ndarray):
            points_roi = np.array(points_roi)
        
        if points_roi.ndim != 2 or points_roi.shape[1] != 2:
            raise ValueError(
                f"points_roi必须是 (N, 2) 形状的数组，当前形状: {points_roi.shape}"
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
        
        对于亚像素坐标，使用双线性插值获取深度值，
        比简单的最近邻取整更准确。
        
        Args:
            depth_map: (H, W) 深度图，单位通常为毫米
            x: x坐标（浮点数），范围 [0, W)
            y: y坐标（浮点数），范围 [0, H)
        
        Returns:
            插值后的深度值（浮点数）
            如果坐标超出边界，返回 0.0
        
        Algorithm:
            对于坐标 (x, y)，找到周围四个整数坐标点：
                (x0, y0)  (x1, y0)
                (x0, y1)  (x1, y1)
            
            插值公式：
                depth = d00 * (1-dx) * (1-dy) +
                       d01 * dx * (1-dy) +
                       d10 * (1-dx) * dy +
                       d11 * dx * dy
            
            其中 dx = x - x0, dy = y - y0
        
        Example:
            >>> depth_map = np.array([
            ...     [100, 110, 120],
            ...     [200, 210, 220],
            ...     [300, 310, 320]
            ... ], dtype=np.uint16)
            >>> # 查询 (1.5, 1.5) 的深度
            >>> depth = CoordinateTransformer.get_depth_bilinear(
            ...     depth_map, 1.5, 1.5
            ... )
            >>> print(f"Interpolated depth: {depth:.2f}")
            Interpolated depth: 210.00  # (100+110+200+210)/4 的加权结果
        
        Notes:
            - 对于整数坐标，结果等价于直接索引
            - 处理边界情况：超出范围返回0
            - 适用于16位或32位深度图
        """
        h, w = depth_map.shape
        
        # 边界检查
        if x < 0 or x >= w or y < 0 or y >= h:
            return 0.0
        
        # 计算四个邻近点的整数坐标
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, w - 1)  # 防止超出右边界
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, h - 1)  # 防止超出下边界
        
        # 计算插值系数
        dx = x - x0  # [0, 1)
        dy = y - y0  # [0, 1)
        
        # 获取四个点的深度值
        d00 = float(depth_map[y0, x0])
        d01 = float(depth_map[y0, x1])
        d10 = float(depth_map[y1, x0])
        d11 = float(depth_map[y1, x1])
        
        # 双线性插值
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
        
        通过重新计算来验证给定的坐标变换是否准确。
        
        Args:
            bbox_norm: 归一化bbox [x1, y1, x2, y2]
            img_w: 图像宽度
            img_h: 图像高度
            point_roi: ROI坐标系下的点
            point_original: 预期的原图坐标系下的点
            eps: 允许的浮点误差，默认 1e-6
        
        Returns:
            True: 坐标变换正确
            False: 坐标变换有误差
        
        Example:
            >>> bbox_norm = [0.234567, 0.456789, 0.567890, 0.789012]
            >>> img_w, img_h = 640, 480
            >>> point_roi = (106.7823, 79.4512)
            >>> point_original = (256.7823, 298.4512)
            >>> is_correct = CoordinateTransformer.verify_transform(
            ...     bbox_norm, img_w, img_h, point_roi, point_original
            ... )
            >>> print(f"Transform correct: {is_correct}")
            Transform correct: True
        """
        # 重新计算整个变换过程
        x1_f, y1_f, x2_f, y2_f = CoordinateTransformer.normalize_to_absolute(
            bbox_norm, img_w, img_h
        )
        
        x1_c, y1_c, x2_c, y2_c = CoordinateTransformer.get_crop_bbox(
            x1_f, y1_f, x2_f, y2_f, img_w, img_h
        )
        
        x_calc, y_calc = CoordinateTransformer.transform_point_roi_to_original(
            point_roi, x1_c, y1_c
        )
        
        # 检查是否一致（考虑浮点误差）
        x_diff = abs(x_calc - point_original[0])
        y_diff = abs(y_calc - point_original[1])
        
        is_correct = (x_diff < eps and y_diff < eps)
        
        if not is_correct:
            print(f"⚠️ 坐标变换验证失败:")
            print(f"  预期原图坐标: ({point_original[0]:.6f}, {point_original[1]:.6f})")
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
            字典包含：
                - 'bbox_norm': 归一化坐标
                - 'bbox_float': 绝对浮点坐标
                - 'bbox_crop': 裁剪用整数坐标
                - 'roi_size': (width, height)
                - 'offset': (offset_x, offset_y)
        
        Example:
            >>> bbox_norm = [0.25, 0.5, 0.75, 0.9]
            >>> roi_info = CoordinateTransformer.get_roi_info(bbox_norm, 640, 480)
            >>> print(roi_info)
            {
                'bbox_norm': [0.25, 0.5, 0.75, 0.9],
                'bbox_float': (160.0, 240.0, 480.0, 432.0),
                'bbox_crop': (160, 240, 480, 432),
                'roi_size': (320, 192),
                'offset': (160, 240)
            }
        """
        x1_f, y1_f, x2_f, y2_f = CoordinateTransformer.normalize_to_absolute(
            bbox_norm, img_w, img_h
        )
        
        x1_c, y1_c, x2_c, y2_c = CoordinateTransformer.get_crop_bbox(
            x1_f, y1_f, x2_f, y2_f, img_w, img_h
        )
        
        roi_w = x2_c - x1_c
        roi_h = y2_c - y1_c
        
        return {
            'bbox_norm': bbox_norm,
            'bbox_float': (x1_f, y1_f, x2_f, y2_f),
            'bbox_crop': (x1_c, y1_c, x2_c, y2_c),
            'roi_size': (roi_w, roi_h),
            'offset': (x1_c, y1_c)
        }


# ==================== 模块测试代码 ====================
if __name__ == "__main__":
    """
    测试坐标变换工具
    运行方式: python utils/coordinate_transform.py
    """
    
    print("=" * 70)
    print("测试 CoordinateTransformer")
    print("=" * 70)
    
    # 测试参数
    img_w, img_h = 640, 480
    bbox_norm = [0.234567, 0.456789, 0.567890, 0.789012]
    
    print(f"\n📷 图像尺寸: {img_w} x {img_h}")
    print(f"📦 归一化bbox: {bbox_norm}")
    
    # 测试1：归一化 → 绝对坐标
    print("\n" + "-" * 70)
    print("测试1: 归一化坐标 → 绝对浮点坐标")
    print("-" * 70)
    
    x1_f, y1_f, x2_f, y2_f = CoordinateTransformer.normalize_to_absolute(
        bbox_norm, img_w, img_h
    )
    print(f"绝对浮点坐标: ({x1_f:.6f}, {y1_f:.6f}, {x2_f:.6f}, {y2_f:.6f})")
    
    # 测试2：获取裁剪bbox
    print("\n" + "-" * 70)
    print("测试2: 获取裁剪用整数坐标")
    print("-" * 70)
    
    x1_c, y1_c, x2_c, y2_c = CoordinateTransformer.get_crop_bbox(
        x1_f, y1_f, x2_f, y2_f, img_w, img_h
    )
    print(f"裁剪整数坐标: ({x1_c}, {y1_c}, {x2_c}, {y2_c})")
    print(f"ROI尺寸: {x2_c - x1_c} x {y2_c - y1_c}")
    
    # 测试3：ROI坐标 → 原图坐标
    print("\n" + "-" * 70)
    print("测试3: ROI坐标 → 原图坐标（单点）")
    print("-" * 70)
    
    point_roi = (106.7823, 79.4512)
    print(f"ROI坐标: {point_roi}")
    
    x_orig, y_orig = CoordinateTransformer.transform_point_roi_to_original(
        point_roi, x1_c, y1_c
    )
    print(f"原图坐标: ({x_orig:.4f}, {y_orig:.4f})")
    
    # 测试4：批量变换
    print("\n" + "-" * 70)
    print("测试4: ROI坐标 → 原图坐标（批量）")
    print("-" * 70)
    
    points_roi = np.array([
        [10.5, 20.3],
        [30.7, 40.1],
        [50.2, 60.9]
    ])
    print(f"ROI坐标（批量）:\n{points_roi}")
    
    points_orig = CoordinateTransformer.transform_points_batch(
        points_roi, x1_c, y1_c
    )
    print(f"原图坐标（批量）:\n{points_orig}")
    
    # 测试5：深度插值
    print("\n" + "-" * 70)
    print("测试5: 深度图双线性插值")
    print("-" * 70)
    
    # 创建模拟深度图
    depth_map = np.array([
        [100, 110, 120, 130],
        [200, 210, 220, 230],
        [300, 310, 320, 330],
        [400, 410, 420, 430]
    ], dtype=np.uint16)
    
    print(f"深度图:\n{depth_map}")
    
    # 测试整数坐标
    x_int, y_int = 1, 1
    depth_int = CoordinateTransformer.get_depth_bilinear(depth_map, x_int, y_int)
    print(f"\n整数坐标 ({x_int}, {y_int}) 深度: {depth_int:.2f}")
    print(f"  验证: depth_map[{y_int}, {x_int}] = {depth_map[y_int, x_int]}")
    
    # 测试浮点坐标
    x_float, y_float = 1.5, 1.5
    depth_float = CoordinateTransformer.get_depth_bilinear(depth_map, x_float, y_float)
    print(f"\n浮点坐标 ({x_float}, {y_float}) 深度: {depth_float:.2f}")
    print(f"  预期约: (210 + 220 + 310 + 320) / 4 = 265.0")
    
    # 测试6：坐标变换验证
    print("\n" + "-" * 70)
    print("测试6: 坐标变换验证")
    print("-" * 70)
    
    is_correct = CoordinateTransformer.verify_transform(
        bbox_norm, img_w, img_h, point_roi, (x_orig, y_orig)
    )
    
    if is_correct:
        print("✅ 坐标变换验证通过！")
    else:
        print("❌ 坐标变换验证失败！")
    
    # 测试7：获取ROI信息
    print("\n" + "-" * 70)
    print("测试7: 获取ROI完整信息")
    print("-" * 70)
    
    roi_info = CoordinateTransformer.get_roi_info(bbox_norm, img_w, img_h)
    
    print("ROI信息:")
    for key, value in roi_info.items():
        print(f"  {key}: {value}")
    
    # 测试8：边界情况测试
    print("\n" + "-" * 70)
    print("测试8: 边界情况处理")
    print("-" * 70)
    
    # 超出边界的bbox
    bbox_extreme = [-0.1, -0.1, 1.1, 1.1]
    print(f"极端bbox（超出边界）: {bbox_extreme}")
    
    roi_info_extreme = CoordinateTransformer.get_roi_info(bbox_extreme, img_w, img_h)
    print(f"裁剪后bbox: {roi_info_extreme['bbox_crop']}")
    print(f"ROI尺寸: {roi_info_extreme['roi_size']}")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试完成！")
    print("=" * 70)
    