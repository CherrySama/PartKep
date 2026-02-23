"""
Created by Yinghao Ho on 2026-2-23
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Union


class CameraConfig:
    """
    深度相机配置类（固定安装）
    
    存储内容：
        - 内参：fx, fy, cx, cy（针孔相机模型）
        - 外参：4×4齐次变换矩阵 T_cam2world
          [[r00, r01, r02, tx],
           [r10, r11, r12, ty],
           [r20, r21, r22, tz],
           [0,   0,   0,   1 ]]
    
    坐标变换说明：
        T_cam2world 将相机坐标系下的点变换到世界坐标系：
            P_world = T_cam2world @ [X_cam, Y_cam, Z_cam, 1]^T
    
    使用示例：
        >>> # 手动初始化
        >>> config = CameraConfig(
        ...     fx=615.0, fy=615.0, cx=320.0, cy=240.0,
        ...     extrinsic_matrix=np.eye(4)
        ... )
        
        >>> # 从yaml文件加载
        >>> config = CameraConfig.load_from_yaml("configs/camera.yaml")
        
        >>> # 保存到yaml文件
        >>> config.save_to_yaml("configs/camera.yaml")
    """

    def __init__(self,
                 fx: float,
                 fy: float,
                 cx: float,
                 cy: float,
                 extrinsic_matrix: np.ndarray):
        """
        初始化相机配置

        Args:
            fx: x方向焦距（像素单位）
            fy: y方向焦距（像素单位）
            cx: 主点x坐标（像素单位），通常接近图像宽度的一半
            cy: 主点y坐标（像素单位），通常接近图像高度的一半
            extrinsic_matrix: 4×4齐次变换矩阵（相机坐标系 → 世界坐标系）

        Raises:
            ValueError: 如果extrinsic_matrix形状不是(4, 4)
            ValueError: 如果extrinsic_matrix最后一行不是[0, 0, 0, 1]
        """
        # 验证内参
        if fx <= 0 or fy <= 0:
            raise ValueError(f"焦距必须为正数，当前: fx={fx}, fy={fy}")

        # 验证外参矩阵
        extrinsic_matrix = np.array(extrinsic_matrix, dtype=np.float64)
        if extrinsic_matrix.shape != (4, 4):
            raise ValueError(
                f"外参矩阵必须是 (4, 4) 形状，当前: {extrinsic_matrix.shape}"
            )
        last_row = extrinsic_matrix[3]
        if not np.allclose(last_row, [0, 0, 0, 1], atol=1e-6):
            raise ValueError(
                f"齐次矩阵最后一行必须是 [0, 0, 0, 1]，当前: {last_row}"
            )

        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.extrinsic_matrix = extrinsic_matrix

        print(f"✓ CameraConfig 初始化完成")
        print(f"  内参: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        print(f"  外参矩阵:\n{self.extrinsic_matrix}")

    # ==================== 属性访问 ====================

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """
        返回3×3相机内参矩阵 K
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
        """
        return np.array([
            [self.fx,      0, self.cx],
            [     0, self.fy, self.cy],
            [     0,       0,       1]
        ], dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """从外参矩阵提取3×3旋转矩阵"""
        return self.extrinsic_matrix[:3, :3].copy()

    @property
    def translation_vector(self) -> np.ndarray:
        """从外参矩阵提取3×1平移向量"""
        return self.extrinsic_matrix[:3, 3].copy()

    # ==================== 文件读写 ====================

    def save_to_yaml(self, filepath: Union[str, Path]):
        """
        保存相机配置到yaml文件

        Args:
            filepath: 保存路径（如 "configs/camera.yaml"）

        yaml文件格式：
            intrinsics:
              fx: 615.0
              fy: 615.0
              cx: 320.0
              cy: 240.0
            extrinsic_matrix:
              - [r00, r01, r02, tx]
              - [r10, r11, r12, ty]
              - [r20, r21, r22, tz]
              - [0.0, 0.0, 0.0, 1.0]
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 构造yaml数据结构
        config_dict = {
            'intrinsics': {
                'fx': float(self.fx),
                'fy': float(self.fy),
                'cx': float(self.cx),
                'cy': float(self.cy)
            },
            'extrinsic_matrix': self.extrinsic_matrix.tolist()
        }

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        print(f"✓ 相机配置已保存到: {filepath}")

    @classmethod
    def load_from_yaml(cls, filepath: Union[str, Path]) -> 'CameraConfig':
        """
        从yaml文件加载相机配置

        Args:
            filepath: yaml文件路径

        Returns:
            CameraConfig 实例

        Raises:
            FileNotFoundError: 如果文件不存在
            KeyError: 如果yaml文件缺少必要字段
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"相机配置文件不存在: {filepath}")

        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        # 解析内参
        intrinsics = config_dict['intrinsics']
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']

        # 解析外参矩阵
        extrinsic_matrix = np.array(
            config_dict['extrinsic_matrix'],
            dtype=np.float64
        )

        print(f"✓ 相机配置已从 {filepath} 加载")
        return cls(fx=fx, fy=fy, cx=cx, cy=cy,
                   extrinsic_matrix=extrinsic_matrix)

    @classmethod
    def create_identity(cls,
                        fx: float = 615.0,
                        fy: float = 615.0,
                        cx: float = 320.0,
                        cy: float = 240.0) -> 'CameraConfig':
        """
        创建外参为单位矩阵的配置（相机坐标系 = 世界坐标系）
        
        用途：
            - 快速测试时的占位配置
            - 相机恰好位于世界坐标系原点且朝向一致时
        
        Args:
            fx, fy: 焦距，默认615.0（RealSense D435典型值）
            cx, cy: 主点，默认320.0/240.0（640×480分辨率中心）
        """
        print("⚠️  使用单位外参矩阵，相机坐标系 = 世界坐标系")
        return cls(fx=fx, fy=fy, cx=cx, cy=cy,
                   extrinsic_matrix=np.eye(4))

    def __repr__(self) -> str:
        return (
            f"CameraConfig(\n"
            f"  intrinsics: fx={self.fx}, fy={self.fy}, "
            f"cx={self.cx}, cy={self.cy}\n"
            f"  extrinsic_matrix:\n{self.extrinsic_matrix}\n"
            f")"
        )


# ==================== 模块测试代码 ====================
if __name__ == "__main__":
    """
    测试 CameraConfig
    运行方式: python configs/camera_config.py
    """
    import tempfile
    import os

    print("=" * 60)
    print("测试 CameraConfig")
    print("=" * 60)
    print()

    # 测试1：手动初始化
    print("【测试1】手动初始化")
    print("-" * 60)

    # 模拟 RealSense D435 的典型内参
    extrinsic = np.array([
        [ 0.0, -1.0,  0.0,  0.5],
        [ 0.0,  0.0, -1.0,  1.2],
        [ 1.0,  0.0,  0.0,  0.8],
        [ 0.0,  0.0,  0.0,  1.0]
    ])

    config = CameraConfig(
        fx=615.0, fy=615.0,
        cx=320.0, cy=240.0,
        extrinsic_matrix=extrinsic
    )
    print()
    print("内参矩阵 K:")
    print(config.intrinsic_matrix)
    print()
    print("旋转矩阵 R:")
    print(config.rotation_matrix)
    print()
    print("平移向量 t:")
    print(config.translation_vector)
    print()

    # 测试2：保存和加载yaml
    print("【测试2】保存和加载yaml")
    print("-" * 60)

    with tempfile.NamedTemporaryFile(
        suffix='.yaml', mode='w', delete=False
    ) as f:
        tmp_path = f.name

    config.save_to_yaml(tmp_path)
    config_loaded = CameraConfig.load_from_yaml(tmp_path)

    # 验证读写一致性
    assert np.allclose(config.extrinsic_matrix,
                       config_loaded.extrinsic_matrix), "外参矩阵不一致！"
    assert config.fx == config_loaded.fx, "fx不一致！"
    print("✅ yaml读写一致性验证通过")
    os.unlink(tmp_path)
    print()

    # 测试3：快速创建单位配置
    print("【测试3】create_identity")
    print("-" * 60)
    config_identity = CameraConfig.create_identity()
    assert np.allclose(config_identity.extrinsic_matrix, np.eye(4))
    print("✅ 单位外参验证通过")
    print()

    # 测试4：错误输入验证
    print("【测试4】错误输入验证")
    print("-" * 60)

    try:
        CameraConfig(fx=615.0, fy=615.0, cx=320.0, cy=240.0,
                     extrinsic_matrix=np.eye(3))  # 错误：3×3
    except ValueError as e:
        print(f"✅ 捕获到预期错误: {e}")

    try:
        bad_matrix = np.eye(4)
        bad_matrix[3] = [0, 0, 1, 0]  # 错误的最后一行
        CameraConfig(fx=615.0, fy=615.0, cx=320.0, cy=240.0,
                     extrinsic_matrix=bad_matrix)
    except ValueError as e:
        print(f"✅ 捕获到预期错误: {e}")

    print()
    print("=" * 60)
    print("✅ CameraConfig 所有测试通过！")
    print("=" * 60)