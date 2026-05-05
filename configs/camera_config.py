"""
Created by Yinghao Ho on 2026-2-23
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Union


class CameraConfig:
    """
    Fixed-mount depth camera configuration.

    Stores intrinsics (fx, fy, cx, cy) and a 4×4 extrinsic matrix T_cam2world:
        [[r00, r01, r02, tx],
         [r10, r11, r12, ty],
         [r20, r21, r22, tz],
         [0,   0,   0,   1 ]]

    Coordinate transform:
        P_world = T_cam2world @ [X_cam, Y_cam, Z_cam, 1]^T
    """

    def __init__(self,
                 fx: float,
                 fy: float,
                 cx: float,
                 cy: float,
                 extrinsic_matrix: np.ndarray):
        """
        Args:
            fx, fy           : focal lengths in pixels
            cx, cy           : principal point in pixels
            extrinsic_matrix : 4×4 homogeneous transform, camera -> world
        """
        # validate intrinsics
        if fx <= 0 or fy <= 0:
            raise ValueError(f"focal lengths must be positive: fx={fx}, fy={fy}")

        # validate extrinsic matrix
        extrinsic_matrix = np.array(extrinsic_matrix, dtype=np.float64)
        if extrinsic_matrix.shape != (4, 4):
            raise ValueError(
                f"extrinsic_matrix must be (4, 4), got {extrinsic_matrix.shape}"
            )
        last_row = extrinsic_matrix[3]
        if not np.allclose(last_row, [0, 0, 0, 1], atol=1e-6):
            raise ValueError(
                f"last row of homogeneous matrix must be [0,0,0,1], got {last_row}"
            )

        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.extrinsic_matrix = extrinsic_matrix

        print(f"[CameraConfig] fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """
        3×3 camera intrinsic matrix K:
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
        """3×3 rotation matrix extracted from extrinsic_matrix."""
        return self.extrinsic_matrix[:3, :3].copy()

    @property
    def translation_vector(self) -> np.ndarray:
        """3-element translation vector extracted from extrinsic_matrix."""
        return self.extrinsic_matrix[:3, 3].copy()

    def save_to_yaml(self, filepath: Union[str, Path]):
        """
        Save camera configuration to a yaml file.

        Args:
            filepath: output path (e.g. "configs/camera.yaml")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

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

        print(f"[CameraConfig] saved to {filepath}")

    @classmethod
    def load_from_yaml(cls, filepath: Union[str, Path]) -> 'CameraConfig':
        """
        Load camera configuration from a yaml file.

        Args:
            filepath: path to yaml file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"camera config not found: {filepath}")

        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        # parse intrinsics
        intrinsics = config_dict['intrinsics']
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']

        # parse extrinsic matrix
        extrinsic_matrix = np.array(
            config_dict['extrinsic_matrix'],
            dtype=np.float64
        )

        print(f"[CameraConfig] loaded from {filepath}")
        return cls(fx=fx, fy=fy, cx=cx, cy=cy,
                   extrinsic_matrix=extrinsic_matrix)

    @classmethod
    def create_identity(cls,
                        fx: float = 615.0,
                        fy: float = 615.0,
                        cx: float = 320.0,
                        cy: float = 240.0) -> 'CameraConfig':
        """
        Create a config with identity extrinsic (camera frame == world frame).

        Args:
            fx, fy: focal lengths, default 615.0 (RealSense D435 typical)
            cx, cy: principal point, default 320.0/240.0 (640×480 center)
        """
        print("[CameraConfig] warning: identity extrinsic, camera frame == world frame")
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


if __name__ == "__main__":
    # run with: python configs/camera_config.py
    import tempfile
    import os

    print("=" * 60)
    print("CameraConfig self-test")
    print("=" * 60)

    # test 1: manual init
    print("\n[test 1] manual init")
    print("-" * 60)

    # typical RealSense D435 intrinsics
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
    print("K:", config.intrinsic_matrix)
    print("R:", config.rotation_matrix)
    print("t:", config.translation_vector)

    # test 2: yaml round-trip
    print("\n[test 2] yaml round-trip")
    print("-" * 60)

    with tempfile.NamedTemporaryFile(
        suffix='.yaml', mode='w', delete=False
    ) as f:
        tmp_path = f.name

    config.save_to_yaml(tmp_path)
    config_loaded = CameraConfig.load_from_yaml(tmp_path)

    assert np.allclose(config.extrinsic_matrix,
                       config_loaded.extrinsic_matrix), "extrinsic mismatch"
    assert config.fx == config_loaded.fx, "fx mismatch"
    print("yaml round-trip OK")
    os.unlink(tmp_path)

    # test 3: identity config
    print("\n[test 3] create_identity")
    print("-" * 60)
    config_identity = CameraConfig.create_identity()
    assert np.allclose(config_identity.extrinsic_matrix, np.eye(4))
    print("identity extrinsic OK")

    # test 4: invalid input
    print("\n[test 4] invalid input")
    print("-" * 60)

    try:
        CameraConfig(fx=615.0, fy=615.0, cx=320.0, cy=240.0,
                     extrinsic_matrix=np.eye(3))  # wrong shape: 3×3
    except ValueError as e:
        print(f"caught expected error: {e}")

    try:
        bad_matrix = np.eye(4)
        bad_matrix[3] = [0, 0, 1, 0]  # invalid last row
        CameraConfig(fx=615.0, fy=615.0, cx=320.0, cy=240.0,
                     extrinsic_matrix=bad_matrix)
    except ValueError as e:
        print(f"caught expected error: {e}")

    print("\n" + "=" * 60)
    print("all tests passed")
    print("=" * 60)